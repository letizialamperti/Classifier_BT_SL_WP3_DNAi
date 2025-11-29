import os
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from merged_dataset import MergedDatasetDANN
from ORDNA.models.binary_dann_classifier import BinaryDANNClassifier
from ORDNA.utils.argparser import get_args, write_config_file


# Funzione per calcolare il peso della classe positiva
def calculate_pos_weight_from_csv(protection_file: Path) -> torch.Tensor:
    labels = pd.read_csv(protection_file)['protection']
    counts = labels.value_counts().to_dict()
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)
    if pos == 0:
        raise ValueError("Nessun sample positivo trovato nel file di protezione.")
    return torch.tensor([neg / pos], dtype=torch.float)


def main():
    # Pulisce eventuali argomenti vuoti
    sys.argv = [arg for arg in sys.argv if arg.strip()]

    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # Percorsi
    embeddings_file = Path(args.embeddings_file)
    protection_file = Path(args.protection_file)   # binary protection
    habitat_file    = Path(args.habitat_file)
    k_cross_path    = Path(args.k_cross_file)

    # Costruzione cartella output
    output_dir = Path("binary_dann")
    output_dir.mkdir(exist_ok=True)

    # Lista dei file di split
    if k_cross_path.is_dir():
        split_files = sorted(k_cross_path.glob("*.csv"))
    else:
        split_files = [k_cross_path]

    # λ per la loss di dominio (se non presente negli args, default 1.0)
    lambda_domain = getattr(args, "lambda_domain", 1.0)

    for split_file in split_files:
        print(f"=== Processing fold: {split_file.name} ===")

        # Carica dataset completo DANN e calcola dimensioni
        dataset_full = MergedDatasetDANN(
            embeddings_file=str(embeddings_file),
            protection_file=str(protection_file),
            habitat_file=str(habitat_file)
        )
        sample_emb_dim = dataset_full.embeddings.shape[1]
        habitat_dim    = dataset_full.habitats.shape[1]
        num_domains    = dataset_full.num_domains

        print(f"  → samples:     {len(dataset_full)}")
        print(f"  → emb_dim:     {sample_emb_dim}")
        print(f"  → habitat_dim: {habitat_dim}")
        print(f"  → num_domains: {num_domains}")

        # Carica split
        kdf = pd.read_csv(split_file, dtype=str)
        train_codes = kdf.loc[kdf['set'] == 'train', 'spygen_code'].tolist()
        val_codes   = kdf.loc[kdf['set'] != 'train', 'spygen_code'].tolist()

        # Mappa spygen_code → indice nel dataset
        code_to_idx = {code: i for i, code in enumerate(dataset_full.codes)}
        train_indices = [code_to_idx[c] for c in train_codes if c in code_to_idx]
        val_indices   = [code_to_idx[c] for c in val_codes   if c in code_to_idx]

        print(f"  → train size: {len(train_indices)}")
        print(f"  → val size:   {len(val_indices)}")

        # DataLoader
        train_loader = DataLoader(
            Subset(dataset_full, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        val_dataset = Subset(dataset_full, val_indices)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Calcolo del pos_weight
        pos_weight = calculate_pos_weight_from_csv(protection_file)
        print(f"DEBUG - Positive class weight: {pos_weight.item():.4f}")

        # Inizializzazione modello DANN binario
        model = BinaryDANNClassifier(
            sample_emb_dim=sample_emb_dim,
            habitat_dim=habitat_dim,
            num_domains=num_domains,
            initial_learning_rate=args.initial_learning_rate,
            pos_weight=pos_weight,
            lambda_domain=lambda_domain
        )

        # Callback e logger
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss_total',
            dirpath='checkpoints_binary_dann',
            filename=f"binary-dann-{split_file.stem}" + "-{val_acc:.2f}",
            save_top_k=3,
            mode='min',
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss_total',
            patience=3,
            mode='min'
        )
        wandb_logger = WandbLogger(
            project='ORDNA_Binary_DANN',
            save_dir='lightning_logs',
            config=args,
            log_model=False
        )

        # Trainer
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            max_epochs=args.max_epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=10
        )

        # Training
        print("Starting binary DANN classification training...")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")

        # ------------------ Salvataggio predizioni su VAL ------------------
        model.eval()
        preds_list = []
        true_labels = []
        prob_list = []

        val_codes_list = [dataset_full.codes[i] for i in val_indices]

        with torch.no_grad():
            for emb, hab, label, dom in val_loader:
                emb   = emb.to(model.device)
                hab   = hab.to(model.device)
                label = label.to(model.device)

                x = torch.cat((emb, hab), dim=1)
                logit = model(x)  # forward → logit binario
                prob = torch.sigmoid(logit).detach().cpu().numpy()
                preds = (prob > 0.5).astype(int)

                preds_list.extend(preds.tolist())
                true_labels.extend(label.cpu().numpy().tolist())
                prob_list.extend(prob.tolist())

        residuals = [p - t for p, t in zip(preds_list, true_labels)]

        out_df = pd.DataFrame({
            'spygen_code': val_codes_list,
            'label':       true_labels,
            'prediction':  preds_list,
            'residual':    residuals,
            'prob_pos':    prob_list
        })
        csv_out = output_dir / f"predictions_dann_{split_file.stem}.csv"
        out_df.to_csv(csv_out, index=False)
        print(f"Saved predictions CSV: {csv_out}")


if __name__ == '__main__':
    main()
