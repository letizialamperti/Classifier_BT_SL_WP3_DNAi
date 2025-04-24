import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from merged_dataset import MergedDataset
from ORDNA.models.classifier import Classifier
from ORDNA.utils.argparser import write_config_file, get_args
from pathlib import Path

def calculate_class_weights_from_csv_coral(protection_file: Path, num_classes: int) -> torch.Tensor:
    labels_df = pd.read_csv(protection_file)
    counts = labels_df['protection'].value_counts().sort_index()
    cw = 1.0 / counts
    cw = cw / cw.sum() * num_classes
    thresh_w = [(cw[i] + cw[i+1]) / 2 for i in range(len(cw)-1)]
    return torch.tensor(thresh_w, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True)
    parser.add_argument('--protection_file', type=str, required=True)
    parser.add_argument('--habitat_file', type=str, required=True)
    parser.add_argument('--k_cross_file', type=str, required=True,
                        help="CSV with columns spygen_code + set=train/validation")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--arg_log', action='store_true')
    args = parser.parse_args()

    if args.arg_log:
        write_config_file(args)

    # 1) split CSV
    split_df = pd.read_csv(args.k_cross_file)
    train_codes = split_df.loc[split_df['set']=='train','spygen_code'].astype(str).tolist()
    val_codes   = split_df.loc[split_df['set']=='validation','spygen_code'].astype(str).tolist()

    # 2) dataset e subset
    full_ds = MergedDataset(args.embeddings_file,
                            args.protection_file,
                            args.habitat_file)
    # mappa code -> indice
    code_to_idx = {code: i for i, code in enumerate(full_ds.codes)}
    train_idx = [code_to_idx[c] for c in train_codes]
    val_idx   = [code_to_idx[c] for c in val_codes]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader   = DataLoader(val_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4)

    # 3) pesi CORAL
    class_weights = calculate_class_weights_from_csv_coral(
        Path(args.protection_file), args.num_classes)

    # 4) modello
    model = Classifier(
        sample_emb_dim=full_ds.embeddings.shape[1],
        habitat_dim=full_ds.habitats.shape[1],
        num_classes=args.num_classes,
        initial_learning_rate=args.initial_learning_rate,
        class_weights=class_weights
    )

    # 5) logger e callback
    wandb_logger = WandbLogger(project='CV_Class', log_model=False)
    ckpt_cb = ModelCheckpoint(monitor='val_class_loss',
                              save_top_k=1, mode='min')
    es_cb   = EarlyStopping(monitor='val_class_loss',
                            patience=3, mode='min')

    trainer = Trainer(
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
    preds, labels, codes = [], [], []
    with torch.no_grad():
        for emb, hab, lab in val_loader:
            x = torch.cat((emb, hab), dim=1).to(model.device)
            out = model(x)
            p = torch.sum(torch.sigmoid(out) > 0.5, dim=1).cpu().numpy()
            preds.extend(p.tolist())
            labels.extend(lab.numpy().tolist())

    codes = [full_ds.codes[i] for i in val_idx]
    df_met = pd.DataFrame({
        'spygen_code': codes,
        'label': labels,
        'prediction': preds,
        'residual': [l - p for l, p in zip(labels, preds)]
    })
    out_csv = Path(args.k_cross_file).stem.replace('split','metrics') + '.csv'
    df_met.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

if __name__ == "__main__":
    main()
