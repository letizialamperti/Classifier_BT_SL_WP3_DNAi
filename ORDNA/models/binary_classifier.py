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

class BinaryClassifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, habitat_dim: int, 
                 initial_learning_rate: float = 1e-5, pos_weight: float = None):
        super().__init__()
        self.save_hyperparameters()
        input_dim = sample_emb_dim + habitat_dim

        # Architettura: output singolo logit
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 1 logit per esempio
        )

        # BCEWithLogitsLoss con optional weighting per classe positiva
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics binarie
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy   = Accuracy(task="binary").to(self.device)
        self.train_precision = Precision(task="binary").to(self.device)
        self.val_precision   = Precision(task="binary").to(self.device)
        self.train_recall    = Recall(task="binary").to(self.device)
        self.val_recall      = Recall(task="binary").to(self.device)
        self.train_mae = MeanAbsoluteError().to(self.device)
        self.val_mae   = MeanAbsoluteError().to(self.device)
        self.train_mse = MeanSquaredError().to(self.device)
        self.val_mse   = MeanSquaredError().to(self.device)

        # Per plotting a fine epoca di validazione
        self.validation_logits = []
        self.validation_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(1)  # [B]

    def training_step(self, batch, batch_idx):
        embeddings, habitats, labels = batch
        x = torch.cat((embeddings, habitats), dim=1)
        logits = self(x)  # [B]
        loss = self.loss_fn(logits, labels.float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        # Calcolo metriche
        acc = self.train_accuracy(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)
        mae = self.train_mae(probs, labels.float())
        mse = self.train_mse(probs, labels.float())

        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
            'train_prec': prec,
            'train_rec': rec,
            'train_mae': mae,
            'train_mse': mse
        }, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, habitats, labels = batch
        x = torch.cat((embeddings, habitats), dim=1)
        logits = self(x)
        loss = self.loss_fn(logits, labels.float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        # Accumulo per plot finale
        self.validation_logits.append(logits.detach().cpu())
        self.validation_labels.append(labels.cpu())

        acc = self.val_accuracy(preds, labels)
        prec = self.val_precision(preds, labels)
        rec  = self.val_recall(preds, labels)
        mae = self.val_mae(probs, labels.float())
        mse = self.val_mse(probs, labels.float())

        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_rec': rec,
            'val_mae': mae,
            'val_mse': mse
        }, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # Confusion matrix
        logits = torch.cat(self.validation_logits)
        labels = torch.cat(self.validation_labels)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(labels.numpy(), preds)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # CM
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predetti')
        axes[0].set_ylabel('Veri')
        axes[0].set_title('Matrice di Confusione')

        # Istogramma probabilità
        axes[1].hist(probs, bins=50, edgecolor='black')
        axes[1].set_xlabel('Probabilità positive')
        axes[1].set_ylabel('Frequenza')
        axes[1].set_title('Distribuzione di sigmoid(logit)')

        wandb_logger = self.logger.experiment
        wandb_logger.log({"Validation Visualizations": wandb.Image(fig)})
        plt.close(fig)

        self.validation_logits.clear()
        self.validation_labels.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.initial_learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
