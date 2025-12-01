import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from merged_dataset_dann import MergedDatasetDANN
from ORDNA.models.classifier import coral_to_label
from ORDNA.models.dann_classifier import DANNClassifier
from ORDNA.utils.argparser import get_args, write_config_file


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


def main():
    # Pulisce eventuali argomenti vuoti (bugfix classico)
    sys.argv = [arg for arg in sys.argv if arg.strip()]

    args = get_args()
    if args.arg_log:
        write_config_file(args)

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # Percorsi
    embeddings_file = Path(args.embeddings_file)
    protection_file = Path(args.protection_file)
    habitat_file    = Path(args.habitat_file)
    k_cross_path    = Path(args.k_cross_file)

    # Cartella output per metriche
    output_dir = Path("metrics_dann")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Lista dei file di split (può essere singolo CSV o directory)
    if k_cross_path.is_dir():
        split_files = sorted(k_cross_path.glob("*.csv"))
    else:
        split_files = [k_cross_path]

    # Loop sui fold
    for split_file in split_files:
        print(f"=== Processing fold: {split_file.name} ===")

        # Dataset completo (DANN)
        dataset_full = MergedDatasetDANN(
            embeddings_file=str(embeddings_file),
            protection_file=str(protection_file),
            habitat_file=str(habitat_file)
        )
        sample_emb_dim = dataset_full.embeddings.shape[1]
        habitat_dim    = dataset_full.habitats.shape[1]

        print(f"  → samples:      {len(dataset_full)}")
        print(f"  → emb_dim:      {sample_emb_dim}")
        print(f"  → habitat_dim:  {habitat_dim}")
        print(f"  → num_domains:  {dataset_full.num_domains}")

        # Carica split (train / validation) da CSV
        kdf = pd.read_csv(split_file, dtype=str)
        train_codes = kdf.loc[kdf['set'] == 'train',      'spygen_code'].tolist()
        val_codes   = kdf.loc[kdf['set'] != 'train',      'spygen_code'].tolist()

        # Indici nel dataset_full
        code_to_idx = {code: i for i, code in enumerate(dataset_full.codes)}
        train_indices = train_indices = [code_to_idx[c] for c in train_codes if c in code_to_idx]
        val_indices   = [code_to_idx[c] for c in val_codes   if c in code_to_idx]

        print(f"  → train size: {len(train_indices)}")
        print(f"  → val size:   {len(val_indices)}")

        train_ds = Subset(dataset_full, train_indices)
        val_ds   = Subset(dataset_full, val_indices)

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
            num_workers=4,
            drop_last=False
        )

        # Pesi CORAL
        class_weights = calculate_class_weights_from_csv_coral(
            protection_file,
            args.num_classes
        )

        # λ per la loss di dominio (se non definito negli args, usa 1.0)
        lambda_domain = getattr(args, "lambda_domain", 1.0)

        # Modello DANN
        model = DANNClassifier(
            sample_emb_dim=sample_emb_dim,
            habitat_dim=habitat_dim,
            num_classes=args.num_classes,
            num_domains=dataset_full.num_domains,
            initial_learning_rate=args.initial_learning_rate,
            lambda_domain=lambda_domain,
            class_weights=class_weights
        )

        # Logger W&B
        wandb_logger = WandbLogger(
            project='ORDNA_DANN',
            save_dir='lightning_logs',
            config=args,
            log_model=False
        )

        # Nome della run WandB
        run_name = wandb_logger.experiment.name  # es: "cool-sun-42"

        # Cartella dedicata per i checkpoint di questa run
        ckpt_dir = Path("checkpoints_dann_classifier") / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Salviamo SOLO il best checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='val_class_loss',
            dirpath=str(ckpt_dir),
            filename=f"{split_file.stem}-best",
            save_top_k=1,
            save_last=False,
            mode='min'
        )

        # Early stopping
        early_stopping_callback = EarlyStopping(
            monitor='val_class_loss',
            patience=10,
            mode='min'
        )

        # Trainer
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            max_epochs=args.max_epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=10
        )


        print("Starting DANN training...")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")

        # ------------------ PREDIZIONI SU VAL E METRIC CSV -------------------
        model.eval()
        preds_list        = []
        true_labels       = []
        dom_true_list     = []
        dom_pred_list     = []

        # ordiniamo i codici val in base a val_indices (stesso ordine del DataLoader con shuffle=False)
        val_codes_list = [dataset_full.codes[i] for i in val_indices]

        with torch.no_grad():
            for emb, hab, lab, dom in val_loader:
                emb = emb.to(model.device)
                hab = hab.to(model.device)
                lab = lab.to(model.device)
                dom = dom.to(model.device)

                x      = torch.cat((emb, hab), dim=1)

                # logits task (CORAL)
                logits_task = model(x)
                pred = coral_to_label(logits_task).detach().cpu().numpy()

                # logits dominio (usiamo encoder + domain_head)
                z = model.encoder(x)
                logits_dom = model.domain_head(z)
                dom_pred = torch.argmax(logits_dom, dim=1).detach().cpu().numpy()

                preds_list.extend(pred.tolist())
                true_labels.extend(lab.cpu().numpy().tolist())
                dom_true_list.extend(dom.cpu().numpy().tolist())
                dom_pred_list.extend(dom_pred.tolist())

        residuals = [l - p for l, p in zip(true_labels, preds_list)]
        dom_correct = [int(t == p) for t, p in zip(dom_true_list, dom_pred_list)]
        if len(dom_correct) > 0:
            domain_accuracy = sum(dom_correct) / len(dom_correct)
        else:
            domain_accuracy = float('nan')

        print(f"  → Domain accuracy (val set) for fold {split_file.name}: {domain_accuracy:.4f}")

        out_df = pd.DataFrame({
            'spygen_code':   val_codes_list,
            'label':         true_labels,
            'prediction':    preds_list,
            'residual':      residuals,
            'domain_true':   dom_true_list,
            'domain_pred':   dom_pred_list,
            'domain_correct':dom_correct
        })

        # ---- SALVATAGGIO CSV METRICHE PER FOLD ----
        
        # stringa "file-safe" per lambda, es: 1.0 -> "1_0"
        lambda_str = str(lambda_domain).replace('.', '_')
        
        # cartella dedicata per questo valore di lambda
        lambda_metrics_dir = output_dir / f"lambda_{lambda_str}"
        lambda_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # nome del file CSV (es: dann_metrics_split1.csv)
        csv_out = lambda_metrics_dir / f"dann_metrics_{split_file.stem}.csv"
        
        out_df.to_csv(csv_out, index=False)
        print(f"Saved DANN metrics CSV: {csv_out}")


    print("All DANN folds done!")


if __name__ == "__main__":
    main()
