import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device).view(-1)
        labels = labels.view(-1).float()
        if self.class_weights is not None:
            class_weights = self.class_weights[labels.long()].to(labels.device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=class_weights)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss

class BinaryClassifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, habitat_dim: int, initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.classifier = nn.Sequential(
            nn.Linear(sample_emb_dim + habitat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Output layer with a single neuron for binary classification
        )
        
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.loss_fn = BinaryCrossEntropyLoss(self.class_weights)
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy = Accuracy(task="binary").to(self.device)
        self.train_precision = Precision(task="binary").to(self.device)
        self.val_precision = Precision(task="binary").to(self.device)
        self.train_recall = Recall(task="binary").to(self.device)
        self.val_recall = Recall(task="binary").to(self.device)
        self.validation_preds = []
        self.validation_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, habitats, labels = batch
        print(f"DEBUG - Training step embeddings.shape: {embeddings.shape}, habitats.shape: {habitats.shape}, labels.shape: {labels.shape}")
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)
        print(f"DEBUG - Training step combined_input.shape: {combined_input.shape}")

        output = self(combined_input).view(-1)
        class_loss = self.loss_fn(output, labels)
        
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True)
        pred = torch.sigmoid(output) > 0.5
        accuracy = self.train_accuracy(pred, labels)
        precision = self.train_precision(pred, labels)
        recall = self.train_recall(pred, labels)
        
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        embeddings, habitats, labels = batch
        print(f"DEBUG - Validation step embeddings.shape: {embeddings.shape}, habitats.shape: {habitats.shape}, labels.shape: {labels.shape}")
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)
        print(f"DEBUG - Validation step combined_input.shape: {combined_input.shape}")

        output = self(combined_input).view(-1)
        class_loss = self.loss_fn(output, labels)
        
        pred = torch.sigmoid(output) > 0.5
        accuracy = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall = self.val_recall(pred, labels)
        
        self.validation_preds.append(pred)
        self.validation_labels.append(labels)

        self.log('val_class_loss', class_loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)

        return class_loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')

        # Log confusion matrix to Wandb
        wandb_logger = self.logger.experiment
        wandb_logger.log({"confusion_matrix": wandb.Image(fig)})

        plt.close(fig)

        # Clear lists for the next epoch
        self.validation_preds.clear()
        self.validation_labels.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_class_loss'
        }
        return [optimizer], [scheduler]
