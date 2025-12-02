import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from torch.autograd import Function


# ----------------- Gradient Reversal -----------------

class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


class GradReverse(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return grad_reverse(x, self.lambd)


# ----------------- Binary DANN Classifier -----------------

class BinaryDANNClassifier(pl.LightningModule):
    """
    Domain-adversarial binary classifier.

    Input batch (da MergedDatasetDANN):
        embeddings : [B, sample_emb_dim]
        habitats   : [B, habitat_dim] (one-hot o simile)
        labels     : [B] (0/1)
        domains    : [B] (indice habitat 0..num_domains-1)
    """

    def __init__(
        self,
        sample_emb_dim: int,
        habitat_dim: int,
        num_domains: int,
        initial_learning_rate: float = 1e-5,
        pos_weight: float | torch.Tensor | None = None,
        lambda_domain: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_domains   = num_domains
        self.lambda_domain = lambda_domain

        input_dim = sample_emb_dim + habitat_dim

        # Encoder condiviso
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Testa binaria (protezione 0/1) → singolo logit
        self.task_head = nn.Linear(128, 1)

        # GRL + testa dominio (habitat)
        self.grl = GradReverse(lambd=lambda_domain)
        self.domain_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
        )

        # BCEWithLogitsLoss con optional pos_weight
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor([pos_weight], dtype=torch.float)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Metriche per il TASK (binary)
        self.train_accuracy  = Accuracy(task="binary")
        self.val_accuracy    = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision   = Precision(task="binary")
        self.train_recall    = Recall(task="binary")
        self.val_recall      = Recall(task="binary")

        self.train_mae = MeanAbsoluteError()
        self.val_mae   = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        # Metriche per il DOMAIN (habitat, multiclass)
        self.train_domain_acc = Accuracy(task="multiclass", num_classes=num_domains)
        self.val_domain_acc   = Accuracy(task="multiclass", num_classes=num_domains)

        # Per plotting a fine validazione
        self.validation_logits = []
        self.validation_labels = []

    # -------- forward: solo task (logit binario) --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        logit = self.task_head(z).squeeze(1)  # [B]
        return logit

    # -------- training_step --------
    def training_step(self, batch, batch_idx):
        embeddings, habitats, labels, domains = batch
        embeddings = embeddings.to(self.device)
        habitats   = habitats.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        x = torch.cat((embeddings, habitats), dim=1)  # [B, input_dim]
        z = self.encoder(x)

        # Task: protezione binaria
        logits_task = self.task_head(z).squeeze(1)  # [B]
        loss_task = self.loss_fn(logits_task, labels.float())

        probs = torch.sigmoid(logits_task)          # [B]
        preds = (probs > 0.5).long()

        # Domain: habitat (via GRL)
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)     # [B, num_domains]
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        # Loss totale (stessa struttura del multiclass DANN)
        loss = loss_task + loss_domain

        # Metriche task
        acc  = self.train_accuracy(preds, labels)
        prec = self.train_precision(preds, labels)
        rec  = self.train_recall(preds, labels)
        mae  = self.train_mae(probs, labels.float())
        mse  = self.train_mse(probs, labels.float())

        # Metriche dominio
        pred_dom = torch.argmax(logits_domain, dim=1)
        acc_dom  = self.train_domain_acc(pred_dom, domains)

        self.log_dict({
            'train_total_loss':  loss,
            'train_task_loss':   loss_task,
            'train_domain_loss': loss_domain,
            'train_acc':         acc,
            'train_prec':        prec,
            'train_rec':         rec,
            'train_mae':         mae,
            'train_mse':         mse,
            'train_domain_acc':  acc_dom,
        }, on_step=True, on_epoch=True)

        return loss

    # -------- validation_step --------
    def validation_step(self, batch, batch_idx):
        embeddings, habitats, labels, domains = batch
        embeddings = embeddings.to(self.device)
        habitats   = habitats.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        x = torch.cat((embeddings, habitats), dim=1)
        z = self.encoder(x)

        # Task
        logits_task = self.task_head(z).squeeze(1)
        loss_task = self.loss_fn(logits_task, labels.float())

        probs = torch.sigmoid(logits_task)
        preds = (probs > 0.5).long()

        # Domain
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        loss = loss_task + loss_domain

        # Accumulo per visualizzazioni finali
        self.validation_logits.append(logits_task.detach().cpu())
        self.validation_labels.append(labels.detach().cpu())

        # Metriche task
        acc  = self.val_accuracy(preds, labels)
        prec = self.val_precision(preds, labels)
        rec  = self.val_recall(preds, labels)
        mae  = self.val_mae(probs, labels.float())
        mse  = self.val_mse(probs, labels.float())

        # Metriche dominio
        pred_dom = torch.argmax(logits_domain, dim=1)
        acc_dom  = self.val_domain_acc(pred_dom, domains)

        self.log_dict({
            'val_total_loss':  loss,
            'val_task_loss':   loss_task,
            'val_domain_loss': loss_domain,
            'val_acc':         acc,
            'val_prec':        prec,
            'val_rec':         rec,
            'val_mae':         mae,
            'val_mse':         mse,
            'val_domain_acc':  acc_dom,
        }, on_epoch=True)

        return loss

    # -------- visualizzazioni a fine validation epoch --------
    def on_validation_epoch_end(self):
        if len(self.validation_logits) == 0:
            return

        logits = torch.cat(self.validation_logits)
        labels = torch.cat(self.validation_labels)

        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(labels.numpy(), preds)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Matrice di confusione
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predetti')
        axes[0].set_ylabel('Veri')
        axes[0].set_title('Matrice di Confusione (Binary DANN)')

        # Istogramma probabilità
        axes[1].hist(probs, bins=50, edgecolor='black')
        axes[1].set_xlabel('Probabilità positive')
        axes[1].set_ylabel('Frequenza')
        axes[1].set_title('Distribuzione di sigmoid(logit)')

        wandb_logger = self.logger.experiment
        wandb_logger.log({"Validation Visualizations Binary DANN": wandb.Image(fig)})
        plt.close(fig)

        self.validation_logits.clear()
        self.validation_labels.clear()

    # -------- optimizer / scheduler --------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.initial_learning_rate,
            weight_decay=1e-4
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            ),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
