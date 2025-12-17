import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from ORDNA.models.coral_loss_weighted_BCEwithLogits import WeightedCoralLoss


# ------------------------
# Helpers
# ------------------------

def coral_to_label(logits: torch.Tensor) -> torch.Tensor:
    """
    Old heuristic: counts thresholds with sigmoid(logit)>0.5.
    Kept here only for comparison/debug.
    """
    prob = torch.sigmoid(logits)
    return torch.sum(prob > 0.5, dim=1)


def coral_logits_to_class_probs(
    logits: torch.Tensor,
    eps: float = 1e-8,
    renormalize: bool = True,
):
    """
    Decode cumulative logits (B, K-1) into class probabilities (B, K) and argmax class.

    p_k = P(y > k) = sigmoid(logits[:, k])
      P(y=0)   = 1 - p0
      P(y=c)   = p_{c-1} - p_c   for c=1..K-2
      P(y=K-1) = p_{K-2}

    If thresholds cross (non-monotone p), diffs can be negative:
    we clamp to 0 and optionally renormalize.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B, K-1), got shape {logits.shape}")

    B, K_minus_1 = logits.shape
    K = K_minus_1 + 1

    p = torch.sigmoid(logits)  # (B, K-1)

    class_probs = logits.new_zeros((B, K))
    class_probs[:, 0] = 1.0 - p[:, 0]

    if K > 2:
        class_probs[:, 1:-1] = p[:, :-1] - p[:, 1:]

    class_probs[:, -1] = p[:, -1]

    class_probs = torch.clamp(class_probs, min=0.0)

    if renormalize:
        Z = class_probs.sum(dim=1, keepdim=True).clamp(min=eps)
        class_probs = class_probs / Z

    pred = torch.argmax(class_probs, dim=1)
    return class_probs, pred


def levels_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels: (B,) values 0..num_classes-1
    returns: (B, num_classes-1) with levels[i, j] = 1 if labels[i] > j else 0
    """
    B = labels.size(0)
    device = labels.device
    thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0).expand(B, -1)
    labels_expanded = labels.unsqueeze(1).expand_as(thresholds)
    levels = (labels_expanded > thresholds).float()
    return levels


# ------------------------
# Lightning Module
# ------------------------

class Classifier(pl.LightningModule):
    def __init__(
        self,
        sample_emb_dim: int,
        num_classes: int,
        habitat_dim: int,
        initial_learning_rate: float = 1e-5,
        pos_weights: torch.Tensor = None,
        importance_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        input_dim = sample_emb_dim + habitat_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes - 1)  # outputs (K-1) logits
        )

        self.loss_fn = WeightedCoralLoss(reduction='mean')
        self.loss_fn_raw = WeightedCoralLoss(reduction=None)

        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.view(-1))
        else:
            self.pos_weights = None

        if importance_weights is not None:
            self.register_buffer("importance_weights", importance_weights.view(-1))
        else:
            self.importance_weights = None

        # Metrics (Lightning will move them to the right device)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision   = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall   = Recall(task="multiclass", num_classes=num_classes)
        self.train_mae = MeanAbsoluteError()
        self.val_mae   = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        self.validation_preds = []
        self.validation_labels = []
        self.validation_raw_loss = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)  # (B, K-1)
        levels = levels_from_labels(labels, self.num_classes)

        loss = self.loss_fn(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )

        # ✅ Correct decoding (A)
        _, pred = coral_logits_to_class_probs(output)

        accuracy  = self.train_accuracy(pred, labels)
        precision = self.train_precision(pred, labels)
        recall    = self.train_recall(pred, labels)
        mae       = self.train_mae(pred, labels)
        mse       = self.train_mse(pred, labels)

        self.log('train_class_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy',  accuracy,  on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall',    recall,    on_step=True, on_epoch=True)
        self.log('train_mae',       mae,       on_step=True, on_epoch=True)
        self.log('train_mse',       mse,       on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        embeddings, habitats, labels = batch
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)  # (B, K-1)
        levels = levels_from_labels(labels, self.num_classes)

        loss = self.loss_fn(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )

        raw_loss = self.loss_fn_raw(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )  # (B,)

        # ✅ Correct decoding (A)
        _, pred = coral_logits_to_class_probs(output)

        accuracy  = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall    = self.val_recall(pred, labels)
        mae       = self.val_mae(pred, labels)
        mse       = self.val_mse(pred, labels)

        self.validation_preds.append(pred.detach().cpu())
        self.validation_labels.append(labels.detach().cpu())
        self.validation_raw_loss.append(raw_loss.detach().cpu())

        self.log('val_class_loss', loss, on_epoch=True)
        self.log('val_accuracy',   accuracy,  on_epoch=True)
        self.log('val_precision',  precision, on_epoch=True)
        self.log('val_recall',     recall,    on_epoch=True)
        self.log('val_mae',        mae,       on_epoch=True)
        self.log('val_mse',        mse,       on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        preds  = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)

        cm = confusion_matrix(labels.numpy(), preds.numpy())
        errors = preds.numpy() - labels.numpy()

        raw_loss_all = torch.cat(self.validation_raw_loss).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        axes[0].set_title('Confusion Matrix')

        axes[1].hist(errors, bins=range(-self.num_classes, self.num_classes + 1), edgecolor='black')
        axes[1].set_xlabel('Prediction Error (Predicted - True)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Histogram')

        axes[2].hist(raw_loss_all, bins=50, edgecolor='black')
        axes[2].set_xlabel('Per-sample CORAL loss')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Loss Histogram')

        wandb_logger = self.logger.experiment
        wandb_logger.log({"Validation Visualizations": wandb.Image(fig)})

        plt.close(fig)
        self.validation_preds.clear()
        self.validation_labels.clear()
        self.validation_raw_loss.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            ),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
