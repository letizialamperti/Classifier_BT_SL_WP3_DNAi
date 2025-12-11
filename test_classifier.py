import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from merged_dataset import MergedDataset
from ORDNA.models.classifier_coralwheighted import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Testare il classificatore su una lista di spygen_code")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Percorso al file .ckpt salvato dal training")

    parser.add_argument("--embeddings_file", type=str, required=True,
                        help="File con gli embeddings")

    parser.add_argument("--protection_file", type=str, required=True,
                        help="File CSV con le etichette di protezione")

    parser.add_argument("--habitat_file", type=str, required=True,
                        help="File con le info di habitat")

    parser.add_argument("--spygen_codes", type=str, nargs="+", required=True,
                        help="Lista di codici spygen da valutare")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per il DataLoader")

    parser.add_argument("--output_csv", type=str, default="test_metrics.csv",
                        help="Nome del file CSV di output")

    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per la riproducibilità")

    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(args.seed)

    # 1) Ricostruisci il dataset completo
    full_ds = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )

    # 2) Mappatura spygen_code -> indice nel dataset
    code_to_idx = {code: i for i, code in enumerate(full_ds.codes)}

    missing_codes = [c for c in args.spygen_codes if c not in code_to_idx]
    if missing_codes:
        raise ValueError(
            f"I seguenti spygen_code non sono presenti nel dataset: {missing_codes}"
        )

    selected_indices = [code_to_idx[c] for c in args.spygen_codes]

    test_ds = Subset(full_ds, selected_indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 3) Carica il modello dal checkpoint
    #    (assumendo che il Classifier salvi gli hyperparameters con self.save_hyperparameters())
    print(f"Carico il modello da: {args.checkpoint}")
    model = Classifier.load_from_checkpoint(args.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.freeze()

    preds = []
    labels = []
    all_codes = [full_ds.codes[i] for i in selected_indices]

    # 4) Inference
    with torch.no_grad():
        for emb, hab, lab in test_loader:
            x = torch.cat((emb, hab), dim=1).to(device)
            out = model(x)  # output CORAL (logits per soglia)

            # Stesso criterio del training script per trasformare in classi
            p = torch.sum(torch.sigmoid(out) > 0.5, dim=1).cpu().numpy()

            preds.extend(p.tolist())
            labels.extend(lab.numpy().tolist())

    # 5) Costruisci il DataFrame risultati
    df_out = pd.DataFrame({
        "spygen_code": all_codes,
        "label":       labels,
        "prediction":  preds,
        "residual":    [l - p for l, p in zip(labels, preds)]
    })

    out_path = Path(args.output_csv)
    df_out.to_csv(out_path, index=False)
    print(f"Salvate metriche/predizioni in {out_path.resolve()}")


if __name__ == "__main__":
    main()
