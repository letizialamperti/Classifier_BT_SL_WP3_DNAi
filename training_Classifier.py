import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from merged_dataset import MergedDataModule
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb
import pandas as pd
from pathlib import Path

# Funzione per calcolare i pesi delle classi
def calculate_class_weights_from_csv(protection_file: Path, num_classes: int) -> torch.Tensor:
    labels_df = pd.read_csv(protection_file)
    print(f"DEBUG - protection_file content:\n{labels_df.head()}")
    label_counts = labels_df['protection'].value_counts().sort_index()
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
    return torch.tensor(class_weights.values, dtype=torch.float)

def calculate_class_weights_from_csv_coral(protection_file: Path, num_classes: int) -> torch.Tensor:
    """
    Calcola i pesi per la CORAL loss basati sulla distribuzione delle etichette nel file CSV.

    Args:
        protection_file (Path): Il percorso al file CSV contenente le etichette.
        num_classes (int): Il numero totale di classi ordinali.

    Returns:
        torch.Tensor: Un tensore contenente i pesi per le soglie (num_classes - 1).
    """
    labels_df = pd.read_csv(protection_file)
    label_counts = labels_df['protection'].value_counts().sort_index()

    # Calcolare i pesi in base alla distribuzione delle etichette
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalizza i pesi

    # CORAL: utilizziamo i pesi per le soglie (num_classes - 1)
    # Prendiamo la media dei pesi consecutivi per approssimare i pesi per le soglie
    threshold_weights = [(class_weights[i] + class_weights[i + 1]) / 2 for i in range(num_classes - 1)]
    return torch.tensor(threshold_weights, dtype=torch.float)

def main():
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: {0}] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    print("DEBUG - Reading CSV files...")
    embeddings_df = pd.read_csv(args.embeddings_file)
    protection_df = pd.read_csv(args.protection_file)
    habitat_df = pd.read_csv(args.habitat_file)
    print(f"DEBUG - embeddings_file content:\n{embeddings_df.head()}")
    print(f"DEBUG - protection_file content:\n{protection_df.head()}")
    print(f"DEBUG - habitat_file content:\n{habitat_df.head()}")

    datamodule = MergedDataModule(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file,
        batch_size=args.batch_size
    )
    datamodule.setup()

    print(f"DEBUG - sample_emb_dim: {datamodule.sample_emb_dim}, habitat_dim: {datamodule.num_habitats}")

    class_weights = calculate_class_weights_from_csv(Path(args.protection_file), args.num_classes)
    print(f"DEBUG - class_weights: {class_weights}")

    model = Classifier(
        sample_emb_dim=datamodule.sample_emb_dim,
        num_classes=args.num_classes,
        habitat_dim=datamodule.num_habitats,
        initial_learning_rate=args.initial_learning_rate,
        class_weights=class_weights
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_class_loss',
        dirpath='checkpoints_classifier',
        filename='classifier-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=3,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_class_loss',
        patience=3,
        mode='min'
    )

    wandb_logger = WandbLogger(project='ORDNA_Class_july', save_dir="lightning_logs", config=args, log_model=False)
    wandb_run = wandb.init(project='ORDNA_Class_july', config=args)
    print(f"DEBUG - Wandb run URL: {wandb_run.url}")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10)

    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")
    wandb.finish()

if __name__ == '__main__':
    main()
