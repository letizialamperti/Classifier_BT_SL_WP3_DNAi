import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

from merged_dataset_dann import MergedDatasetDANN
from ORDNA.models.dann_classifier import DANNClassifier


def extract_latent(model, loader, device):
    model.eval()
    all_z, all_y, all_d = [], [], []

    with torch.no_grad():
        for emb, label, dom in loader:
            emb = emb.to(device)

            z = model.encoder(emb)
            all_z.append(z.cpu())
            all_y.append(label)
            all_d.append(dom)

    return (
        torch.cat(all_z).numpy(),
        torch.cat(all_y).numpy(),
        torch.cat(all_d).numpy()
    )


def plot_tsne(Z_2d, labels, title, wandb_name):
    df = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "label": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="x", y="y",
        hue="label", palette="tab10",
        s=60, alpha=0.8
    )
    plt.title(title)
    plt.legend(title="label", bbox_to_anchor=(1.05, 1), loc="upper left")

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
    parser.add_argument("--lambda_domain", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # --- INITIALIZE WANDB RUN ---
    wandb.init(project="DANN_tSNE", config=vars(args))

    # Dataset
    ds = MergedDatasetDANN(
        embeddings_file=args.embeddings_file,
        protection_file=args.protection_file,
        habitat_file=args.habitat_file
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = DANNClassifier.load_from_checkpoint(
        args.checkpoint,
        sample_emb_dim=ds.embeddings.shape[1],
        num_classes=args.num_classes,
        num_domains=ds.num_domains,
        lambda_domain=args.lambda_domain,
        strict=False,
    )
    model.to(args.device)

    # Extract latent space
    Z, Y, D = extract_latent(model, loader, args.device)

    # Run TSNE
    Z_2d = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="random",
        random_state=42
    ).fit_transform(Z)

    plot_tsne(Z_2d, Y, "t-SNE of Z (Protection)", "tsne_protection")
    plot_tsne(Z_2d, D, "t-SNE of Z (Habitat / Domain)", "tsne_domain")

    print("t-SNE visualization complete!")

    # --- CLOSE THE RUN (forces wandb to sync + print link) ---
    wandb.finish()


if __name__ == "__main__":
    main()
