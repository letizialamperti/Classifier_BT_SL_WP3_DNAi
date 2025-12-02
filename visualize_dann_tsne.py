import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

from merged_dataset_dann import MergedDatasetDANN
from ORDNA.models.dann_classifier import DANNClassifier


def extract_latent(model, loader, device):
    """
    Ritorna tutte le rappresentazioni latenti Z,
    le label Y e gli habitat D.
    """
    model.eval()
    all_z = []
    all_y = []
    all_d = []

    with torch.no_grad():
        for emb, hab, y, d in loader:
            emb = emb.to(device)
            hab = hab.to(device)

            z = model.encoder(torch.cat((emb, hab), dim=1))  # → Z

            all_z.append(z.cpu())
            all_y.append(y)
            all_d.append(d)

    Z = torch.cat(all_z).numpy()
    Y = torch.cat(all_y).numpy()
    D = torch.cat(all_d).numpy()

    return Z, Y, D


def plot_tsne(Z_2d, labels, title, wandb_name):
    """
    Plotta e manda il grafico a wandb.
    """
    df = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "label": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="label",
        palette="tab10",
        s=60, alpha=0.8
    )
    plt.title(title)
    plt.legend(title="label", bbox_to_anchor=(1.05, 1), loc='upper left')

    wandb.log({wandb_name: wandb.Image(plt)})
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--protection_file", type=str, required=True)
    parser.add_argument("--habitat_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--lambda_domain", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_prefix", type=str, default="tsne_dann")

    args = parser.parse_args()

    wandb.init(project="DANN_tSNE", config=vars(args))

    # Dataset
    ds = MergedDatasetDANN(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Carica modello
    model = DANNClassifier.load_from_checkpoint(
        args.checkpoint,
        sample_emb_dim=ds.embeddings.shape[1],
        habitat_dim=ds.habitats.shape[1],
        num_classes=args.num_classes,
        lambda_domain=args.lambda_domain,
        strict=False,
    )
    model.to(args.device)

    # === Estrarre Z, Y, D ===
    Z, Y, D = extract_latent(model, loader, args.device)

    # === t-SNE ===
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate="auto", init="random", random_state=42)
    Z_2d = tsne.fit_transform(Z)

    # === Plot protection ===
    plot_tsne(
        Z_2d,
        labels=Y,
        title="t-SNE of Z (colored by protection Y)",
        wandb_name="tsne_protection"
    )

    # === Plot habitat / domain ===
    plot_tsne(
        Z_2d,
        labels=D,
        title="t-SNE of Z (colored by habitat/domain D)",
        wandb_name="tsne_habitat"
    )

    print("t-SNE finished and uploaded to WandB!")


if __name__ == "__main__":
    main()
