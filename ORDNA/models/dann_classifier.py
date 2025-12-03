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

from .classifier import CoralLoss, coral_to_label
from torch.autograd import Function


# ================================================================
# ---------------------- GRADIENT REVERSAL ------------------------
# ================================================================

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


# ================================================================
# ------------------- DANN Classifier (CORAL) ---------------------
# ================================================================

class DANNClassifier(pl.LightningModule):
    def __init__(
        self,
        sample_emb_dim: int,
        num_classes: int,
        num_domains: int,
        initial_learning_rate: float = 1e-5,
        lambda_domain: float = 1.0,
        class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes   = num_classes
        self.num_domains   = num_domains
        self.lambda_domain = lambda_domain

        input_dim = sample_emb_dim 

        # ---------------- Encoder ----------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ---------------- Task head (protection) ----------------
        self.task_head = nn.Linear(128, num_classes - 1)   # CORAL logits

        # ---------------- Domain head ----------------
        self.grl = GradReverse(lambd=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

        # ---------------- Loss ----------------
        self.loss_fn = CoralLoss(num_classes, class_weights, return_raw=False)

        # ---------------- Metrics (task) ----------------
        self.train_accuracy  = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy    = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision   = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall    = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall      = Recall(task="multiclass", num_classes=num_classes)

        self.train_mae = MeanAbsoluteError()
        self.val_mae   = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        # ---------------- Metrics (domain) ----------------
        self.train_domain_acc = Accuracy(task="multiclass", num_classes=num_domains)
        self.val_domain_acc   = Accuracy(task="multiclass", num_classes=num_domains)

        # ---------------- Buffers for plots ----------------
        self.validation_preds   = []
        self.validation_labels  = []
        self.validation_raw_bce = []


    # ================================================================
    # -------------------------- FORWARD ------------------------------
    # ================================================================
    def forward(self, x):
        z = self.encoder(x)
        logits = self.task_head(z)
        return logits


    # ================================================================
    # ------------------------ TRAINING STEP --------------------------
    # ================================================================
    def training_step(self, batch, batch_idx):

        embeddings, labels, domains = batch
        embeddings = embeddings.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        x = embeddings

        # ----- Task -----
        z = self.encoder(x)
        logits_task = self.task_head(z)
        loss_task = self.loss_fn(logits_task, labels)

        pred_labels = coral_to_label(logits_task)

        # ----- Domain -----
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        # ----- Total -----
        loss = loss_task + self.lambda_domain * loss_domain

        # Task metrics
        acc  = self.train_accuracy(pred_labels, labels)
        prec = self.train_precision(pred_labels, labels)
        rec  = self.train_recall(pred_labels, labels)
        mae  = self.train_mae(pred_labels.float(), labels.float())
        mse  = self.train_mse(pred_labels.float(), labels.float())

        # Domain metrics
        pred_dom = torch.argmax(logits_domain, dim=1)
        acc_dom  = self.train_domain_acc(pred_dom, domains)

        # Log
        self.log_dict({
            'train_total_loss':  loss,
            'train_class_loss':  loss_task,
            'train_domain_loss': loss_domain,
            'train_accuracy':    acc,
            'train_precision':   prec,
            'train_recall':      rec,
            'train_mae':         mae,
            'train_mse':         mse,
            'train_domain_acc':  acc_dom
        }, on_step=True, on_epoch=True)

        return loss


    # ================================================================
    # ------------------------ VALIDATION STEP ------------------------
    # ================================================================
    def validation_step(self, batch, batch_idx):

        embeddings, labels, domains = batch
        embeddings = embeddings.to(self.device)
        labels     = labels.to(self.device)
        domains    = domains.to(self.device)

        x = embeddings

        # ----- Task -----
        z = self.encoder(x)
        logits_task = self.task_head(z)
        loss_task, raw_bce = self.loss_fn(logits_task, labels, return_raw=True)

        pred_labels = coral_to_label(logits_task)

        # ----- Domain -----
        z_rev = self.grl(z)
        logits_domain = self.domain_head(z_rev)
        loss_domain = F.cross_entropy(logits_domain, domains.long())

        loss = loss_task + self.lambda_domain * loss_domain

        # Task metrics
        acc  = self.val_accuracy(pred_labels, labels)
        prec = self.val_precision(pred_labels, labels)
        rec  = self.val_recall(pred_labels, labels)
        mae  = self.val_mae(pred_labels.float(), labels.float())
        mse  = self.val_mse(pred_labels.float(), labels.float())

        # Domain metrics
        pred_dom = torch.argmax(logits_domain, dim=1)
        acc_dom  = self.val_domain_acc(pred_dom, domains)

        # Save for plots
        self.validation_preds.append(pred_labels.cpu())
        self.validation_labels.append(labels.cpu())
        self.validation_raw_bce.append(raw_bce.cpu())

        # Log
        self.log_dict({
            'val_total_loss':  loss,
            'val_class_loss':  loss_task,
            'val_domain_loss': loss_domain,
            'val_accuracy':    acc,
            'val_precision':   prec,
            'val_recall':      rec,
            'val_mae':         mae,
            'val_mse':         mse,
            'val_domain_acc':  acc_dom
        }, on_epoch=True)

        return loss


    # ================================================================
    # ----------------- END OF VALIDATION EPOCH -----------------------
    # ================================================================
    
    def on_validation_epoch_end(self):

        if len(self.validation_preds) == 0:
            return

        preds  = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)

        cm = confusion_matrix(labels.numpy(), preds.numpy())
        errors = preds.numpy() - labels.numpy()

        raw_bce_all = torch.cat([b.flatten() for b in self.validation_raw_bce]).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title("Confusion Matrix (Protection)")

        axes[1].hist(errors, bins=range(-self.num_classes, self.num_classes+1), edgecolor="black")
        axes[1].set_title("Error Histogram")

        axes[2].hist(raw_bce_all, bins=50, edgecolor="black")
        axes[2].set_title("Raw BCE Histogram (CORAL)")

        wandb_logger = self.logger.experiment
        wandb_logger.log({"Validation Visualizations DANN": wandb.Image(fig)})

        plt.close(fig)

        self.validation_preds.clear()
        self.validation_labels.clear()
        self.validation_raw_bce.clear()


    # ================================================================
    # --------------------------- OPTIMIZER ----------------------------
    # ================================================================
    def configure_optimizers(self):

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.initial_learning_rate,
            weight_decay=1e-4
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            ),
            'monitor': 'val_class_loss'
        }

        return [optimizer], [scheduler]
