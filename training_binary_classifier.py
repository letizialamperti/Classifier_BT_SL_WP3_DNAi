import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from merged_dataset import MergedDataModule
from ORDNA.models.binary_classifier import BinaryClassifier
from ORDNA.utils.argparser import get_args, write_config_file
import wandb
import pandas as pd
from pathlib import Path


def calculate_pos_weight_from_csv(protection_file: Path) -> torch.Tensor:
    """
    Calcola il peso per la classe positiva (1) a partire dalla distribuzione dei label in CSV.
    """
    labels = pd.read_csv(protection_file)['protection']
    counts = labels.value_counts().to_dict()
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)
    if pos == 0:
        raise ValueError("Nessun sample positivo trovato nel file di protezione.")
    # Peso per la classe positiva: num_neg / num_pos
    return torch.tensor([neg / pos], dtype=torch.float)


def main():
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # DEBUG: leggere e mostrare le prime righe dei CSV
    embeddings_df = pd.read_csv(args.embeddings_file)
    protection_df = pd.read_csv(args.protection_file)
    habitat_df = pd.read_csv(args.habitat_file)
    print(f"DEBUG - embeddings_file content:\n{embeddings_df.head()}")
    print(f"DEBUG - protection_file content:\n{protection_df.head()}")
    print(f"DEBUG - habitat_file content:\n{habitat_df.head()}")

    # Inizializzazione del DataModule
    datamodule = MergedDataModule(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file,
        batch_size=args.batch_size
    )
    datamodule.setup()

    print(f"DEBUG - sample_emb_dim: {datamodule.sample_emb_dim}, habitat_dim: {datamodule.num_habitats}")

    # Calcolo del pos_weight
    pos_weight = calculate_pos_weight_from_csv(Path(args.protection_file))
    print(f"DEBUG - Positive class weight: {pos_weight.item():.4f}")

    # Inizializzazione del modello binario
    model = BinaryClassifier(
        sample_emb_dim=datamodule.sample_emb_dim,
        habitat_dim=datamodule.num_habitats,
        initial_learning_rate=args.initial_learning_rate,
        pos_weight=pos_weight
    )

    # Callback: checkpoint sul val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_binary_classifier',
        filename='binary-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # Wandb logger
    wandb_logger = WandbLogger(
        project='ORDNA_Binary',
        save_dir='lightning_logs',
        config=args,
        log_model=False
    )
    wandb_run = wandb.init(project='ORDNA_Binary', config=args)
    print(f"DEBUG - Wandb run URL: {wandb_run.url}")

    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10
    )

    print("Starting binary classification training...")
    trainer.fit(model=model, datamodule=datamodule)

    print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")
    wandb.finish()


if __name__ == '__main__':
    main()
