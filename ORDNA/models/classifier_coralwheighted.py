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

from ORDNA.models.coral_loss_wheighted import WeightedCoralLoss  

# Helper: convert CORAL logits to discrete class labels
def coral_to_label(logits):
    """
    Convert logits from CORAL loss to predicted discrete labels.
    It sums the number of thresholds passed (where probability > 0.5).
    """
    prob = torch.sigmoid(logits)
    return torch.sum(prob > 0.5, dim=1)


# Helper: costruisce i livelli CORAL (0/1) da etichette intere
def levels_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    labels: (B,) con valori 0..num_classes-1
    return: (B, num_classes-1) con livelli 0/1
            levels[i, j] = 1 se labels[i] > j, altrimenti 0
    """
    B = labels.size(0)
    device = labels.device
    thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0).expand(B, -1)
    labels_expanded = labels.unsqueeze(1).expand_as(thresholds)
    levels = (labels_expanded > thresholds).float()
    return levels


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
        # PRIMA: input_dim = sample_emb_dim + habitat_dim
        input_dim = sample_emb_dim  # <-- ora usiamo solo gli embeddings
        

        # Classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes - 1)  # CORAL: outputs num_classes-1 logits
        )

        # 🔹 Loss CORAL pesata
        # una versione con reduction='mean' per training/val_loss
        self.loss_fn = WeightedCoralLoss(reduction='mean')
        # una versione con reduction=None per avere loss per campione (per istogramma)
        self.loss_fn_raw = WeightedCoralLoss(reduction=None)

        # 🔹 Registra i pesi come buffer, così vanno automaticamente sul device giusto
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.view(-1))
        else:
            self.pos_weights = None

        if importance_weights is not None:
            self.register_buffer("importance_weights", importance_weights.view(-1))
        else:
            self.importance_weights = None

        # Metrics (discrete labels are used for computing these)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_precision   = Precision(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.val_recall   = Recall(task="multiclass", num_classes=num_classes).to(self.device)
        self.train_mae = MeanAbsoluteError().to(self.device)
        self.val_mae   = MeanAbsoluteError().to(self.device)
        self.train_mse = MeanSquaredError().to(self.device)
        self.val_mse   = MeanSquaredError().to(self.device)
        
        self.validation_preds = []
        self.validation_labels = []
        self.validation_raw_loss = []  # <--- CHANGED: ora memorizziamo la loss per campione

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        embeddings, _, labels = batch
        embeddings = embeddings.to(self.device)
        habitats = habitats.to(self.device)
        labels = labels.to(self.device)
        #combined_input = torch.cat((embeddings, habitats), dim=1)
        combined_input = embeddings 

        # Get the logits from the classifier
        output = self(combined_input)  # shape (B, num_classes-1)

        # Costruisci i livelli CORAL (0/1)
        levels = levels_from_labels(labels, self.num_classes)

        # Compute loss con pesi CORAL
        loss = self.loss_fn(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )
        
        # Convert logits to discrete predictions for metrics
        pred = coral_to_label(output)
        
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
        embeddings, _, labels = batch
        embeddings = embeddings.to(self.device)
        habitats = habitats.to(self.device)
        labels = labels.to(self.device)
        #combined_input = torch.cat((embeddings, habitats), dim=1)
        combined_input = embeddings
        

        # Get the logits from the classifier
        output = self(combined_input)  # shape (B, num_classes-1)

        # Costruisci i livelli CORAL
        levels = levels_from_labels(labels, self.num_classes)

        # Loss media per logging
        loss = self.loss_fn(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )

        # Loss per campione (reduction=None) per l'istogramma
        raw_loss = self.loss_fn_raw(
            output,
            levels,
            pos_weights=self.pos_weights,
            importance_weights=self.importance_weights,
        )  # shape (B,)

        # Convert logits to discrete predictions for metrics and further evaluation
        pred = coral_to_label(output)
        
        accuracy  = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall    = self.val_recall(pred, labels)
        mae       = self.val_mae(pred, labels)
        mse       = self.val_mse(pred, labels)
        
        self.validation_preds.append(pred)
        self.validation_labels.append(labels)
        self.validation_raw_loss.append(raw_loss.detach().cpu())
        
        self.log('val_class_loss', loss, on_epoch=True)
        self.log('val_accuracy',   accuracy,  on_epoch=True)
        self.log('val_precision',  precision, on_epoch=True)
        self.log('val_recall',     recall,    on_epoch=True)
        self.log('val_mae',        mae,       on_epoch=True)
        self.log('val_mse',        mse,       on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and labels from the validation epoch
        preds  = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        errors = preds.cpu().numpy() - labels.cpu().numpy()

        # Concatenate all per-sample losses across batches and flatten them
        raw_loss_all = torch.cat(self.validation_raw_loss).numpy()  # shape (N_val,)

        # Create subplots: confusion matrix, error histogram, and loss histogram
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        axes[0].set_title('Confusion Matrix')
        
        # Plot error histogram (difference between predicted and true labels)
        axes[1].hist(errors, bins=range(-self.num_classes, self.num_classes + 1), edgecolor='black')
        axes[1].set_xlabel('Prediction Error (Predicted - True)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Histogram')
        
        # Plot loss histogram (distribution of per-sample CORAL loss)
        axes[2].hist(raw_loss_all, bins=50, edgecolor='black')
        axes[2].set_xlabel('Per-sample CORAL loss')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Loss Histogram')
        
        # Log the combined figure to WandB
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
