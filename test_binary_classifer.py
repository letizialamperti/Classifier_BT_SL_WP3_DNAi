import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from merged_dataset import MergedDataset
from ORDNA.models.binary_classifier import BinaryClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testare il classificatore binario su una lista di spygen_code"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Percorso al file .ckpt salvato dal training")

    parser.add_argument("--embeddings_file", type=str, required=True,
                        help="File con gli embeddings (es. .csv o .npy)")

    parser.add_argument("--protection_file", type=str, required=True,
                        help="File CSV con le etichette di protezione (colonna: protection)")

    parser.add_argument("--habitat_file", type=str, required=True,
                        help="File con le info di habitat")

    parser.add_argument("--spygen_codes", type=str, nargs="+", required=True,
                        help="Lista di codici spygen da valutare")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per il DataLoader")

    parser.add_argument("--output_csv", type=str, default="test_binary_metrics.csv",
                        help="Nome del file CSV di output")

    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per la riproducibilità")

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Dispositivo su cui eseguire il modello (default: cuda se disponibile, altrimenti cpu)"
    )

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Numero di worker per il DataLoader")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # 1) Ricostruisci il dataset completo
    full_ds = MergedDataset(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file
    )

    # 2) Mappatura spygen_code -> indice nel dataset
    code_to_idx = {str(code): i for i, code in enumerate(full_ds.codes)}
    requested_codes = [str(c) for c in args.spygen_codes]

    missing_codes = [c for c in requested_codes if c not in code_to_idx]
    if missing_codes:
        raise ValueError(
            f"I seguenti spygen_code non sono presenti nel dataset: {missing_codes}"
        )

    selected_indices = [code_to_idx[c] for c in requested_codes]
    test_ds = Subset(full_ds, selected_indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 3) Carica il modello dal checkpoint
    print(f"Carico il modello da: {args.checkpoint}")
    model = BinaryClassifier.load_from_checkpoint(args.checkpoint)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model.to(device)
    model.eval()
    model.freeze()

    # Per ricostruire i codici nell'ordine del subset
    all_codes = [str(full_ds.codes[i]) for i in selected_indices]

    preds = []
    labels = []
    probs = []
    logits = []

    # 4) Inference
    with torch.no_grad():
        for emb, hab, lab in test_loader:
            x = torch.cat((emb, hab), dim=1).to(device)

            logit = model(x)  # shape: (B, 1) o (B,)
            prob = torch.sigmoid(logit)

            # uniforma shape -> (B,)
            logit_flat = logit.view(-1).detach().cpu()
            prob_flat = prob.view(-1).detach().cpu()

            pred = (prob_flat > 0.5).long()

            logits.extend(logit_flat.tolist())
            probs.extend(prob_flat.tolist())
            preds.extend(pred.tolist())
            labels.extend(lab.view(-1).cpu().tolist())

    residuals = [p - t for p, t in zip(preds, labels)]

    # 5) Output CSV
    df_out = pd.DataFrame({
        "spygen_code": all_codes,
        "label": labels,
        "prediction": preds,
        "prob": probs,
        "logit": logits,
        "residual": residuals
    })

    out_path = Path(args.output_csv)
    df_out.to_csv(out_path, index=False)
    print(f"Salvate predizioni in {out_path.resolve()}")


if __name__ == "__main__":
    main()
