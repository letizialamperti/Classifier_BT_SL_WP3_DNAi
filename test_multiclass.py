import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from merged_dataset import MergedDataset
from ORDNA.models.classifier import Classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained multiclass CORAL classifier on a new dataset"
    )

    parser.add_argument(
        "--embeddings_file", type=str, required=True,
        help="Path to embeddings file for the TEST set"
    )
    parser.add_argument(
        "--protection_file", type=str, required=True,
        help="Path to protection labels CSV for the TEST set"
    )
    parser.add_argument(
        "--habitat_file", type=str, required=True,
        help="Path to habitat labels CSV for the TEST set"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.ckpt)"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device to use (auto/cpu/gpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--output_csv", type=str, default="multiclass_test_predictions.csv",
        help="Output CSV file with per-sample predictions"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[rank: 0] Seed set to {args.seed}")
    pl.seed_everything(args.seed)

    # ------------------------------------------------------------------
    # 1) Carica il dataset di TEST completo
    # ------------------------------------------------------------------
    print("Loading test dataset...")
    test_ds = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )

    sample_emb_dim = test_ds.embeddings.shape[1]
    habitat_dim    = test_ds.habitats.shape[1]

    # numero classi solo per info/log
    prot_df = pd.read_csv(args.protection_file)
    num_classes = len(sorted(prot_df["protection"].unique()))

    print(f"Test samples: {len(test_ds)}")
    print(f"Embedding dim: {sample_emb_dim}, "
          f"habitat dim: {habitat_dim}, "
          f"num_classes (from CSV): {num_classes}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # ------------------------------------------------------------------
    # 2) Carica il modello dal checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading model from checkpoint: {ckpt_path}")

    # Caso semplice: Classifier salva i suoi hyperparams dentro il ckpt
    model: Classifier = Classifier.load_from_checkpoint(str(ckpt_path))
    model.eval()

    # Device
    if args.accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.accelerator == "cpu":
        device = torch.device("cpu")
    else:
        # "auto": usa GPU se disponibile
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 3) Loop di test: predizioni + raccolta etichette
    # ------------------------------------------------------------------
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for emb, hab, lab in test_loader:
            emb = emb.to(device)
            hab = hab.to(device)

            x   = torch.cat((emb, hab), dim=1)
            out = model(x)  # output CORAL: logit per ogni soglia

            # STESSA logica del training script:
            # class = numero di soglie superate
            p = torch.sum(torch.sigmoid(out) > 0.5, dim=1).cpu().numpy()

            all_preds.extend(p.tolist())
            all_labels.extend(lab.numpy().tolist())

    # Codici nell'ordine naturale del dataset (shuffle=False)
    codes = [str(code) for code in test_ds.codes]

    if not (len(codes) == len(all_labels) == len(all_preds)):
        raise RuntimeError(
            f"Length mismatch: len(codes)={len(codes)}, "
            f"len(labels)={len(all_labels)}, len(preds)={len(all_preds)}"
        )

    residuals = [l - p for l, p in zip(all_labels, all_preds)]

    # ------------------------------------------------------------------
    # 4) Salva le predizioni in un CSV
    # ------------------------------------------------------------------
    df_out = pd.DataFrame({
        "spygen_code": codes,
        "label":       all_labels,
        "prediction":  all_preds,
        "residual":    residuals,
    })

    out_path = Path(args.output_csv)
    df_out.to_csv(out_path, index=False)
    print(f"Saved per-sample predictions to: {out_path}")

    # ------------------------------------------------------------------
    # 5) Metriche globali (accuracy, e se c'è sklearn anche F1 etc.)
    # ------------------------------------------------------------------
    correct = sum(int(l == p) for l, p in zip(all_labels, all_preds))
    acc = correct / len(all_labels) if all_labels else float("nan")
    print(f"Accuracy (micro): {acc:.4f}  ({correct}/{len(all_labels)})")

    try:
        from sklearn.metrics import f1_score, classification_report

        f1_macro = f1_score(all_labels, all_preds, average="macro")
        print(f"Macro F1: {f1_macro:.4f}")
        print("\nClassification report:")
        print(classification_report(all_labels, all_preds, digits=3))
    except ImportError:
        print("sklearn not installed → only accuracy printed. "
              "Install with `pip install scikit-learn` for more metrics.")


if __name__ == "__main__":
    main()
