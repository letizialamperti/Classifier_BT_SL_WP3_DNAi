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

# CORAL Loss for Ordinal Regression
class CoralLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        """
        CORAL Loss for ordinal regression.
        
        Args:
            num_classes (int): Total number of ordinal classes.
            class_weights (Tensor, optional): Weights for each threshold (should be of shape [num_classes-1]).
                                               These weights are applied to balance the importance
                                               of binary decisions related to thresholds.
        """
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, labels):
        """
        Calculates the CORAL loss.

        Args:
            logits: Tensor of shape [B, num_classes-1] containing logits for each threshold.
            labels: Tensor of shape [B] with integer labels (0, 1, ..., num_classes-1).

        Returns:
            loss: Scalar value of CORAL loss.
        """
        B = labels.size(0)
        # Generate thresholds for each class
        thresholds = torch.arange(self.num_classes - 1, device=labels.device).unsqueeze(0).expand(B, -1)
        labels_expanded = labels.unsqueeze(1).expand_as(thresholds)
        target = (labels_expanded > thresholds).float()  # Shape: [B, num_classes-1]

        # Calculate probabilities for each threshold using sigmoid
        prob = torch.sigmoid(logits)
        
        # Calculate Binary Cross Entropy for each threshold without reduction
        bce = F.binary_cross_entropy(prob, target, reduction='none')  # [B, num_classes-1]
        
        # Apply class weights if provided
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            bce = bce * cw.unsqueeze(0)  # Multiply each element by the corresponding weight
        # Mean across all thresholds and samples
        loss = bce.mean()
        return loss

# Function to convert CORAL logits to discrete class labels
def coral_to_label(logits):
    """
    Convert logits from CORAL loss to predicted labels.
    """
    prob = torch.sigmoid(logits)
    return torch.sum(prob > 0.5, dim=1)

# Classifier using CORAL Loss
class Classifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, num_classes: int, habitat_dim: int, initial_learning_rate: float = 1e-5, class_weights=None):
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
            nn.Linear(256, num_classes - 1)  # CORAL: num_classes-1 logits
        )

        self.loss_fn = CoralLoss(num_classes, class_weights)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)
        class_loss = self.loss_fn(output, labels)
        
        # Convert logits to labels
        pred = coral_to_label(output)

        # Calculate metrics
        accuracy = self.train_accuracy(pred, labels)
        precision = self.train_precision(pred, labels)
        recall = self.train_recall(pred, labels)
        mae = self.train_mae(pred, labels)
        mse = self.train_mse(pred, labels)
        
        # Log metrics
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        embeddings, habitats, labels = batch
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)

        output = self(combined_input)
        class_loss = self.loss_fn(output, labels)
        
        # Convert logits to labels
        pred = coral_to_label(output)

        # Calculate metrics
        accuracy = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall = self.val_recall(pred, labels)
        mae = self.val_mae(pred, labels)
        mse = self.val_mse(pred, labels)
        
        self.validation_preds.append(pred)
        self.validation_labels.append(labels)

        # Log metrics
        self.log('val_class_loss', class_loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)

        return class_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
