import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from merged_dataset import MergedDataset
from ORDNA.models.classsifier_cw_nohabitat import Classifier  # <--- nuovo classifier

from ORDNA.utils.argparser import get_args, write_config_file
from ORDNA.utils.utils import compute_coral_weights_from_csv 
from ORDNA.models.coral_loss_wheighted import WeightedCoralLoss 


def calculate_class_weights_from_csv_coral(protection_file: Path, num_classes: int) -> torch.Tensor:
    """
    Calcola i pesi per la CORAL loss basati sulla distribuzione delle etichette,
    assicurando che tutte le classi 0..num_classes-1 compaiano (anche con conteggio 0).
    """
    labels_df = pd.read_csv(protection_file)
    counts = labels_df['protection'].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0)

    eps = 1e-9
    cw = 1.0 / (counts + eps)
    cw = cw / cw.sum() * num_classes

    cw_arr = cw.to_numpy()   # shape == (num_classes,)
    threshold_weights = [(cw_arr[i] + cw_arr[i+1]) / 2 for i in range(num_classes - 1)]

    return torch.tensor(threshold_weights, dtype=torch.float)


class DropHabitatDataset(Dataset):
    """Wrapper che prende un dataset (emb, hab, lab) e restituisce (emb, lab)."""
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        emb, hab, lab = self.base_ds[idx]
        return emb, lab


def main():
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # 1) split CSV
    split_df    = pd.read_csv(args.k_cross_file)
    train_codes = split_df.loc[split_df['set']=='train','spygen_code'].astype(str).tolist()
    val_codes   = split_df.loc[split_df['set']=='validation','spygen_code'].astype(str).tolist()

    # 2) dataset e subset (MergedDataset continua a usare anche habitat_file)
    full_ds     = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )
    code_to_idx = {code: i for i, code in enumerate(full_ds.codes)}
    train_idx   = [code_to_idx[c] for c in train_codes]
    val_idx     = [code_to_idx[c] for c in val_codes]

    train_ds_raw = Subset(full_ds, train_idx)   # (emb, hab, lab)
    val_ds_raw   = Subset(full_ds, val_idx)

    # Wrapper che rimuove habitat: ora i batch sono (emb, lab)
    train_ds = DropHabitatDataset(train_ds_raw)
    val_ds   = DropHabitatDataset(val_ds_raw)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 3) pesi CORAL (pos_weights + importance_weights)
    pos_weights, importance_weights = compute_coral_weights_from_csv(
        Path(args.protection_file),
        args.num_classes
    )

    # 4) modello (usa SOLO gli embeddings)
    model = Classifier(
        sample_emb_dim=full_ds.embeddings.shape[1],
        num_classes=args.num_classes,
        initial_learning_rate=args.initial_learning_rate,
        pos_weights=pos_weights,
        importance_weights=importance_weights,
    )

    # 5) logger e callback
    wandb_logger = WandbLogger(project='CV_Class_nohabitat', log_model=False)
    ckpt_cb      = ModelCheckpoint(
        monitor='val_class_loss',
        save_top_k=1,
        mode='min'
    )
    es_cb        = EarlyStopping(
        monitor='val_class_loss',
        patience=10,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=10
    )

    # 6) fit
    trainer.fit(model, train_loader, val_loader)

    # 7) predict e salvataggio metriche
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for emb, lab in val_loader:                  # <--- niente più hab
            x   = emb.to(model.device)
            out = model(x)
            p   = torch.sum(torch.sigmoid(out) > 0.5, dim=1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(lab.numpy().tolist())

    codes = [full_ds.codes[i] for i in val_idx]
    df_met = pd.DataFrame({
        'spygen_code': codes,
        'label':       labels,
        'prediction':  preds,
        'residual':    [l - p for l, p in zip(labels, preds)]
    })
    out_csv = Path(args.k_cross_file).stem.replace('split','metrics_nohabitat') + '.csv'
    df_met.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")


if __name__ == "__main__":
    main()
