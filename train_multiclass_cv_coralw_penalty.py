import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from merged_dataset import MergedDataset

# ⚠️ usa il nuovo classifier con CORAL pesata + boundary penalty
from ORDNA.models.classifier_coral_weighted_penalty import Classifier

from ORDNA.utils.argparser import get_args, write_config_file
from ORDNA.utils.utils import compute_coral_weights_from_csv


def main():
    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # ==========================
    # 1) Lettura split CSV
    # ==========================
    split_df    = pd.read_csv(args.k_cross_file)
    train_codes = split_df.loc[split_df['set'] == 'train',      'spygen_code'].astype(str).tolist()
    val_codes   = split_df.loc[split_df['set'] == 'validation', 'spygen_code'].astype(str).tolist()

    # ==========================
    # 2) Dataset e subset
    # ==========================
    full_ds = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )

    code_to_idx = {code: i for i, code in enumerate(full_ds.codes)}
    train_idx   = [code_to_idx[c] for c in train_codes]
    val_idx     = [code_to_idx[c] for c in val_codes]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # ==========================
    # 3) Pesi CORAL (pos_weights + importance_weights)
    # ==========================
    # compute_coral_weights_from_csv deve restituire:
    #   pos_weights:        shape (num_classes-1,)
    #   importance_weights: shape (num_classes-1,)
    pos_weights, importance_weights = compute_coral_weights_from_csv(
        Path(args.protection_file),
        args.num_classes
    )

    # ==========================
    # 4) Modello (Classifier CORAL + penalty)
    # ==========================
    # Se non hai ancora definito args.lambda_boundary nell'argparser,
    # questo prende default = 1.0
    lambda_boundary = getattr(args, "lambda_boundary", 1.0)

    model = Classifier(
        sample_emb_dim=full_ds.embeddings.shape[1],
        habitat_dim   =full_ds.habitats.shape[1],
        num_classes   =args.num_classes,
        initial_learning_rate=args.initial_learning_rate,
        pos_weights=pos_weights,
        importance_weights=importance_weights,
        lambda_boundary=lambda_boundary,
    )

    # ==========================
    # 5) Logger e callback
    # ==========================
    wandb_logger = WandbLogger(project='CV_Class', log_model=False)

    ckpt_cb = ModelCheckpoint(
        monitor='val_class_loss',
        save_top_k=1,
        mode='min'
    )
    es_cb = EarlyStopping(
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

    # ==========================
    # 6) Training
    # ==========================
    trainer.fit(model, train_loader, val_loader)

    # ==========================
    # 7) Predizioni e salvataggio metriche
    # ==========================
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for emb, hab, lab in val_loader:
            x   = torch.cat((emb, hab), dim=1).to(model.device)
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
    # ========= SAVE METRICS IN DEDICATED FOLDER ========= #

    metrics_dir = Path("metrics_coralwpenalty")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome file basato sul nome dello split
    metrics_name = Path(args.k_cross_file).stem.replace("split", "metrics") + ".csv"
    
    out_csv = metrics_dir / metrics_name
    df_met.to_csv(out_csv, index=False)
    
    print(f"Saved metrics to {out_csv}")



if __name__ == "__main__":
    main()
