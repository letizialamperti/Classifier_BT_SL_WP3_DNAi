import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, labels):
        logits = logits.to(labels.device)
        logits = logits.view(-1, self.num_classes)
        labels = labels.view(-1)
        cum_probs = torch.sigmoid(logits)
        cum_probs = torch.cat([cum_probs, torch.ones_like(cum_probs[:, :1])], dim=1)
        prob = cum_probs[:, :-1] - cum_probs[:, 1:]
        one_hot_labels = torch.zeros_like(prob).scatter(1, labels.unsqueeze(1), 1)
        epsilon = 1e-9
        prob = torch.clamp(prob, min=epsilon, max=1-epsilon)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(labels.device)[labels].view(-1, 1)
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1) * class_weights
        else:
            loss = - (one_hot_labels * torch.log(prob) + (1 - one_hot_labels) * torch.log(1 - prob)).sum(dim=1)
        return loss.mean()
        

class CoralLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        """
        CORAL Loss per problemi di regressione ordinale.
        
        Args:
            num_classes (int): Numero totale di classi ordinali.
            class_weights (Tensor, optional): Pesi per ogni soglia (dovrebbe essere un tensore di forma [num_classes-1]).
                                               Questi pesi verranno applicati per bilanciare l'importanza
                                               delle decisioni binarie relative alle soglie.
        """
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, labels):
        """
        Calcola la CORAL loss.

        Args:
            logits: Tensor di forma [B, num_classes-1] contenente i logit per ciascuna soglia.
            labels: Tensor di forma [B] con etichette intere (0, 1, ..., num_classes-1).

        Returns:
            loss: Valore scalare della CORAL loss.
        """
        B = labels.size(0)
        # Per ogni soglia i (0,1,...,num_classes-2), il target è 1 se l'etichetta > i, altrimenti 0.
        thresholds = torch.arange(self.num_classes - 1, device=labels.device).unsqueeze(0).expand(B, -1)
        labels_expanded = labels.unsqueeze(1).expand_as(thresholds)
        target = (labels_expanded > thresholds).float()  # Forma: [B, num_classes-1]

        # Calcola le probabilità per ogni soglia tramite la funzione sigmoide
        prob = torch.sigmoid(logits)
        
        # Calcola la Binary Cross Entropy senza riduzione, per ogni soglia
        bce = F.binary_cross_entropy(prob, target, reduction='none')  # [B, num_classes-1]
        
        # Applica i pesi per classe se forniti: ci aspettiamo un tensore di forma [num_classes-1]
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)  # Assicurati che i pesi siano sullo stesso device
            bce = bce * cw.unsqueeze(0)  # Moltiplica ogni elemento per il peso corrispondente
        # Media su tutte le soglie e sui campioni
        loss = bce.mean()
        return loss


class Classifier(pl.LightningModule):
    def __init__(self, sample_emb_dim: int, num_classes: int, habitat_dim: int, initial_learning_rate: float = 1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        input_dim = sample_emb_dim + habitat_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.loss_fn = CoralLoss(num_classes, self.class_weights)
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
        print(f"DEBUG - Training step embeddings.shape: {embeddings.shape}, habitats.shape: {habitats.shape}, labels.shape: {labels.shape}")
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)
        print(f"DEBUG - Training step combined_input.shape: {combined_input.shape}")

        output = self(combined_input)
        class_loss = self.loss_fn(output, labels)
        
        self.log('train_class_loss', class_loss, on_step=True, on_epoch=True)
        pred = torch.argmax(output, dim=1)
        accuracy = self.train_accuracy(pred, labels)
        precision = self.train_precision(pred, labels)
        recall = self.train_recall(pred, labels)
        mae = self.train_mae(pred, labels)
        mse = self.train_mse(pred, labels)
        
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)

        return class_loss

    def validation_step(self, batch, batch_idx: int):
        embeddings, habitats, labels = batch
        print(f"DEBUG - Validation step embeddings.shape: {embeddings.shape}, habitats.shape: {habitats.shape}, labels.shape: {labels.shape}")
        embeddings, habitats, labels = embeddings.to(self.device), habitats.to(self.device), labels.to(self.device)
        combined_input = torch.cat((embeddings, habitats), dim=1)
        print(f"DEBUG - Validation step combined_input.shape: {combined_input.shape}")

        output = self(combined_input)
        class_loss = self.loss_fn(output, labels)
        
        pred = torch.argmax(output, dim=1)
        accuracy = self.val_accuracy(pred, labels)
        precision = self.val_precision(pred, labels)
        recall = self.val_recall(pred, labels)
        mae = self.val_mae(pred, labels)
        mse = self.val_mse(pred, labels)
        
        self.validation_preds.append(pred)
        self.validation_labels.append(labels)

        self.log('val_class_loss', class_loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)

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
