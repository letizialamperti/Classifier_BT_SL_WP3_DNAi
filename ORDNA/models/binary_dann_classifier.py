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
        # gradiente invertito e scalato da lambd
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


class GradReverse(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return grad_reverse(x, self.lambd)


# ----------------- Binary DANN Classifier (allineato al primo modello) -----------------

class BinaryDANNClassifier(pl.LightningModule):
    """
    Domain-adversarial binary classifier.

    Vuole come input del batch:
        embeddings : [B, sample_emb_dim]
        labels     : [B] (0/1)
        domains    : [B] (0..num_domains-1)

    L'habitat NON viene usato come feature di input,
    ma solo come etichetta di dominio per il ramo avversario.

    NOTA IMPORTANTE:
    - Il ramo task è reso il più simile possibile al primo modello:
        embeddings -> Linear(sample_emb_dim, 256) -> BN -> ReLU -> Dropout -> Linear(256,1)
    - Con lambda_domain = 0:
        * il GRL non trasferisce gradiente al encoder
        * il training sul task è (quasi) equivalente al primo modello con soli embedding.
    """

    def __init__(
        self,
        sample_emb_dim: int,
        num_domains: int,
        initial_learning_rate: float = 1e-5,
        pos_weight: float | torch.Tensor | None = None,
        lambda_domain: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_domains   = num_domains
        self.lambda_domain = float(lambda_domain)

        input_dim = sample_emb_dim  # solo embeddings

        # ---- Ramo TASK: architettura allineata al primo modello (solo embeddings) ----
        # Primo modello (solo embedding) era: Linear(d,256) + BN + ReLU + Dropout + Linear(256,1)
        # Qui lo spezziamo in encoder (fino a 256) + task_head (256 -> 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.task_head = nn.Linear(256, 1)

        # ---- Ramo DOMAIN: habitat (multiclasse) con GRL ----
        self.grl = GradReverse(lambd=self.lambda_domain)
        self.domain_head = nn.Sequential(
            nn.Linear(256, 64),
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

        # Metriche per il TASK (binary) – stessi tipi del primo modello
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

    # -------- forward: SOLO task (logit binario) --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Comportamento identico al primo modello (con soli embedding):
        embeddings -> encoder -> task_head -> logit [B]
        """
        z = self.encoder(x)
        logit = self.task_head(z).squeeze(1)  # [B]
        return logit

    # -------- training_step --------
    def training_step(self, batch, batch_idx):
        # Batch: embeddings, labels, domains
        embeddings, labels, domains = batch
        embeddings = embeddings.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        # ---- TASK ----
        z = self.encoder(embeddings)
        logits_task = self.task_head(z).squeeze(1)  # [B]
        loss_task = self.loss_fn(logits_task, labels.float())

        probs = torch.sigmoid(logits_task)          # [B]
        preds = (probs > 0.5).long()

        # ---- DOMAIN (DANN) ----
        # GRL: se lambda_domain = 0, non passa gradiente all'encoder
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)     # [B, num_domains]
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        # Loss totale: task + dominio
        # NB: per il task è importante che il scheduler monitori solo val_loss (= loss_task)
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

        # Log "compatibili" col primo modello per il task
        self.log_dict({
            'train_loss':        loss_task,   # come nel primo modello
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
        embeddings, labels, domains = batch
        embeddings = embeddings.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        # ---- TASK ----
        z = self.encoder(embeddings)
        logits_task = self.task_head(z).squeeze(1)
        loss_task = self.loss_fn(logits_task, labels.float())

        probs = torch.sigmoid(logits_task)
        preds = (probs > 0.5).long()

        # ---- DOMAIN ----
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        # Loss totale
        loss = loss_task + loss_domain

        # Accumulo per visualizzazioni finali (task)
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

        # Log "compatibili" col primo modello per il task
        self.log_dict({
            'val_loss':        loss_task,   # come nel primo modello
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
            # esattamente come il primo modello → monitoriamo la loss del task
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
