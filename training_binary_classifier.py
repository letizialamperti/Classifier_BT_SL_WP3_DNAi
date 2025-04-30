import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from merged_dataset import MergedDataset
from torch.utils.data import DataLoader, Subset
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
    # Rimuovi eventuali argomenti vuoti per evitare 'unrecognized arguments:'
    import sys
    sys.argv = [arg for arg in sys.argv if arg.strip()]

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
        # Carica l'intero dataset
    dataset_full = MergedDataset(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file
    )
    # Calcolo delle dimensioni di input dal dataset
    sample_emb_dim = dataset_full.embeddings.shape[1]
    habitat_dim    = dataset_full.habitats.shape[1]

    # Splitting basato sulla colonna 'set' del CSV di cross-validation
    kdf = pd.read_csv(args.k_cross_file, dtype=str)
    # Codici di train e val
    train_codes = kdf.loc[kdf['set'] == 'train', 'spygen_code'].astype(str).tolist()
    val_codes   = kdf.loc[kdf['set'] != 'train', 'spygen_code'].astype(str).tolist()

    print(f"DEBUG - Total samples: {len(dataset_full)}")
    print(f"DEBUG - Train codes count: {len(train_codes)}")
    print(f"DEBUG - Val codes count: {len(val_codes)}")

    # Indici di train e val basati sui codici
    train_indices = [i for i, code in enumerate(dataset_full.codes) if code in train_codes]
    val_indices   = [i for i, code in enumerate(dataset_full.codes) if code in val_codes]

    # Crea Subset e DataLoader
    train_dataset = Subset(dataset_full, train_indices)
    val_dataset   = Subset(dataset_full, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=4)

    print(f"DEBUG - sample_emb_dim: {sample_emb_dim}, habitat_dim: {habitat_dim}")

    # Calcolo del pos_weight
    pos_weight = calculate_pos_weight_from_csv(Path(args.protection_file))
    print(f"DEBUG - Positive class weight: {pos_weight.item():.4f}")

    # Inizializzazione del modello binario
    model = BinaryClassifier(
        sample_emb_dim=sample_emb_dim,
        habitat_dim=habitat_dim,
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
    wandb_run = wandb.init(
        project='ORDNA_Binary',
        config=args
    )
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
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")
    wandb.finish()

    print("Starting binary classification training...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")
    wandb.finish()


if __name__ == '__main__':
    main()
