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

# CORAL Loss for Ordinal Regression with optional raw output
class CoralLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, return_raw=True):
        """
        CORAL Loss for ordinal regression.
        
        Args:
            num_classes (int): Total number of ordinal classes.
            class_weights (Tensor, optional): Weights for each threshold (should be of shape [num_classes-1]).
                                               These weights balance the importance of binary decisions.
            return_raw (bool): If True, the forward pass returns a tuple (mean_loss, raw_bce).
        """
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.return_raw = return_raw

    def forward(self, logits, labels, return_raw=None):
        if return_raw is None:
            return_raw = self.return_raw
        B = labels.size(0)
        # Generate thresholds for each class
        thresholds = torch.arange(self.num_classes - 1, device=labels.device).unsqueeze(0).expand(B, -1)
        labels_expanded = labels.unsqueeze(1).expand_as(thresholds)
        target = (labels_expanded > thresholds).float()  # Shape: [B, num_classes-1]

        # Calculate probabilities using sigmoid
        prob = torch.sigmoid(logits)
        # Calculate BCE for each threshold without reduction
        bce = F.binary_cross_entropy(prob, target, reduction='none')  # [B, num_classes-1]
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            bce = bce * cw.unsqueeze(0)
        loss = bce.mean()
        if return_raw:
            return loss, bce  # Return both mean loss and raw BCE tensor
        return loss

# Function to convert CORAL logits to discrete class labels
def coral_to_label(logits):
    """
    Convert logits from CORAL loss to predicted discrete labels.
    It sums the number of thresholds passed (probability > 0.5).
    """
    prob = torch.sigmoid(logits)
    return torch.sum(prob > 0.5, dim=1)

# Classifier using CORAL Loss
class Classifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, num_classes: int, habitat_dim: int, 
                 initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        input_dim = sample_emb_dim + habitat_dim

        # Classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes - 1)  # CORAL: outputs num_classes-1 logits
        )

        self.loss_fn = CoralLoss(num_classes, class_weights, return_raw=False)
        # For validation, we will call the loss with return_raw=True
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_mae = MeanAbsoluteError().to(self.device)
        self.val_mae = MeanAbsoluteError().to(self.device)
        self.train_mse = MeanSquaredError().to(self.device)
        self.val_mse = MeanSquaredError().to(self.device)
        self.validation_preds = []
        self.validation_labels = []
        self.validation_raw_bce = []  # Store raw BCE tensors for histogram

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        embeddings = embeddings.to(self.device)
        habitats = habitats.to(self.device)
        labels = labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)

        # In training, we do not need raw BCE values
        loss = self.loss_fn(combined_input, labels, return_raw=False)
        output = self(combined_input)
        pred = coral_to_label(output)

        accuracy = self.train_accuracy(pred, labels)
        precision = self.train_precision(pred, labels)
        recall = self.train_recall(pred, labels)
        mae = self.train_mae(pred, labels)
        mse = self.train_mse(pred, labels)

        self.log('train_class_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        embeddings, habitats, labels = batch
        embeddings = embeddings.to(self.device)
        habitats = habitats.to(self.device)
        labels = labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)

        # In validation, get both loss and raw BCE values
        loss, raw_bce = self.loss_fn(combined_input, labels, return_raw=True)
        output = self(combined_input)
        pred = coral_to_label(output)

        accuracy = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall = self.val_recall(pred, labels)
        mae = self.val_mae(pred, labels)
        mse = self.val_mse(pred, labels)

        self.validation_preds.append(pred)
        self.validation_labels.append(labels)
        # Save the raw BCE tensor for histogram plotting later.
        self.validation_raw_bce.append(raw_bce.detach().cpu())

        self.log('val_class_loss', loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # Concatenate predictions and labels for confusion matrix and error histogram
        preds = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        errors = preds.cpu().numpy() - labels.cpu().numpy()

        # Concatenate raw BCE values across all batches, flatten them
        raw_bce_all = torch.cat([bce.flatten() for bce in self.validation_raw_bce]).numpy()

        # Create subplots: confusion matrix, error histogram, and raw BCE histogram
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        axes[0].set_title('Confusion Matrix')

        # Error Histogram (Prediction error = pred - true)
        axes[1].hist(errors, bins=range(-self.num_classes, self.num_classes+1), edgecolor='black')
        axes[1].set_xlabel('Prediction Error (Predicted - True)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Histogram')

        # Raw BCE Histogram (distribution of raw BCE values)
        axes[2].hist(raw_bce_all, bins=50, edgecolor='black')
        axes[2].set_xlabel('Raw BCE Value')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Raw BCE Histogram')

        # Log the combined figure to WandB
        wandb_logger = self.logger.experiment
        wandb_logger.log({"Validation Visualizations": wandb.Image(fig)})

        plt.close(fig)
        # Clear the stored lists for the next epoch
        self.validation_preds.clear()
        self.validation_labels.clear()
        self.validation_raw_bce.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            ),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
