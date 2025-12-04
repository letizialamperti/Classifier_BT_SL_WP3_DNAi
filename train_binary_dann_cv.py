import os
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from merged_dataset_dann import MergedDatasetDANN
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
    output_dir = Path("metrics_binary_dann")
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
        num_domains    = dataset_full.num_domains

        print(f"  → samples:     {len(dataset_full)}")
        print(f"  → emb_dim:     {sample_emb_dim}")
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

        # Inizializzazione modello DANN binario (nuova firma: niente habitat_dim)
        model = BinaryDANNClassifier(
            sample_emb_dim=sample_emb_dim,
            num_domains=num_domains,
            initial_learning_rate=args.initial_learning_rate,
            pos_weight=pos_weight,
            lambda_domain=lambda_domain
        )

        # Logger W&B
        wandb_logger = WandbLogger(
            project='ORDNA_DANN',
            save_dir='lightning_logs',
            config=args,
            log_model=False
        )

        # Stringa "file-safe" per lambda, es: 1.0 -> "1_0"
        lambda_str = str(lambda_domain).replace('.', '_')

        # Cartella dedicata per questo valore di lambda (UNICA per tutti i fold)
        ckpt_dir = Path("checkpoints_binary_dann_classifier") / f"lambda_{lambda_str}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Salviamo SOLO il best checkpoint per questo fold
        checkpoint_callback = ModelCheckpoint(
            monitor='val_task_loss',
            dirpath=str(ckpt_dir),
            filename=f"{split_file.stem}-best",  # es: fold1-best.ckpt
            save_top_k=1,
            save_last=False,
            mode='min'
        )

        # Early stopping: monitoriamo la stessa metrica
        early_stopping_callback = EarlyStopping(
            monitor='val_task_loss',
            patience=20,
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

        # Training
        print("Starting binary DANN classification training...")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"DEBUG - Early stopping triggered: {trainer.should_stop}")

        # ------------------ Salvataggio predizioni su VAL ------------------
        model.eval()
        preds_list    = []
        true_labels   = []
        prob_list     = []
        dom_true_list = []
        dom_pred_list = []

        val_codes_list = [dataset_full.codes[i] for i in val_indices]

        with torch.no_grad():
            for emb, label, dom in val_loader:   # <-- ora dataset = (emb, label, dom)
                emb   = emb.to(model.device)
                label = label.to(model.device)
                dom   = dom.to(model.device)

                x = emb

                # logit binario (task)
                logit = model(x)  # forward → logit binario [B]
                prob  = torch.sigmoid(logit).detach().cpu().numpy()
                preds = (prob > 0.5).astype(int)

                # dominio (usiamo encoder + domain_head)
                z = model.encoder(x)
                logits_dom = model.domain_head(z)
                dom_pred = torch.argmax(logits_dom, dim=1).detach().cpu().numpy()

                preds_list.extend(preds.tolist())
                true_labels.extend(label.cpu().numpy().tolist())
                prob_list.extend(prob.tolist())
                dom_true_list.extend(dom.cpu().numpy().tolist())
                dom_pred_list.extend(dom_pred.tolist())

        residuals = [p - t for p, t in zip(preds_list, true_labels)]
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
            'prob_pos':      prob_list,
            'domain_true':   dom_true_list,
            'domain_pred':   dom_pred_list,
            'domain_correct':dom_correct
        })

        # ---- SALVATAGGIO CSV METRICHE PER FOLD ----

        lambda_str = str(lambda_domain).replace('.', '_')
        
        # cartella dedicata per questo valore di lambda
        lambda_metrics_dir = output_dir / f"lambda_{lambda_str}"
        lambda_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        csv_out = lambda_metrics_dir / f"dann_metrics_{split_file.stem}.csv"
        out_df.to_csv(csv_out, index=False)
        
        print(f"Saved DANN metrics CSV: {csv_out}")



if __name__ == '__main__':
    main()
