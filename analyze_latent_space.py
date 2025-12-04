import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import wandb


from merged_dataset import MergedDataset
from ORDNA.models.classifier_coralwheighted import Classifier


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisi spazio latente del Classifier (PCA, t-SNE, PCoA) con logging su Weights & Biases."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path al checkpoint .ckpt del modello Lightning.")
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--protection_file", type=str, required=True)
    parser.add_argument("--habitat_file", type=str, required=True)

    parser.add_argument("--habitat_labels_file", type=str, default=None,
                        help="CSV con etichette di habitat (stesso numero di righe del dataset). Deve contenere una colonna 'habitat_label'.")
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_points", type=int, default=2000,
                        help="Numero massimo di punti da usare per t-SNE/PCoA (subsample se necessario).")
    parser.add_argument("--wandb_project", type=str, default="latent_analysis",
                        help="Nome del progetto Weights & Biases.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Entity/utente/team Wandb (opzionale).")
    parser.add_argument("--wandb_run_name", type=str, default="latent_coral_analysis")

    return parser.parse_args()


# ---------- Estrazione spazio latente ----------

@torch.no_grad()
def extract_latent_representations(model, dataset, batch_size=128, device="cuda"):
    """
    Estrae:
      - latent: rappresentazioni 256D dal penultimo layer
      - prot_labels: etichette di protezione (0..K-1)
    NB: le etichette di habitat categoriche le passiamo da un CSV separato.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    model.to(device)

    all_latent = []
    all_prot = []

    for emb, hab, prot in loader:
        emb = emb.to(device)
        hab = hab.to(device)
        prot = prot.to(device)

        x = torch.cat((emb, hab), dim=1)  # [B, sample_emb_dim + habitat_dim]

        # Il tuo classifier:
        # Linear -> BN -> ReLU -> Dropout -> Linear
        # Usiamo tutto tranne l'ultimo Linear: model.classifier[:-1](x)
        latent = model.classifier[:-1](x)  # shape [B, 256]

        all_latent.append(latent.cpu())
        all_prot.append(prot.cpu())

    all_latent = torch.cat(all_latent, dim=0).numpy()   # [N, 256]
    all_prot = torch.cat(all_prot, dim=0).numpy()       # [N]

    return all_latent, all_prot


# ---------- Riduzioni di dimensionalità ----------

def pca_2d(latent):
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    return latent_2d, pca.explained_variance_ratio_


def tsne_2d(latent, perplexity=30, random_state=0):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    latent_2d = tsne.fit_transform(latent)
    return latent_2d


def pcoa_2d(latent):
    """
    PCoA (Principal Coordinates Analysis) = classical MDS su matrice di distanza.
    """
    # 1) Matrice di distanza (euclidea)
    D = pairwise_distances(latent, metric="euclidean")
    # 2) Classical scaling (double centering)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(D ** 2).dot(J)

    # 3) Autovalori/autovettori
    eigvals, eigvecs = np.linalg.eigh(B)  # eigh = simmetrica
    # ordina in valore assoluto decrescente (autovalori potrebbero essere leggermente negativi numericamente)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 4) prendi le prime due componenti con autovalori positivi
    positive = eigvals > 0
    eigvals_pos = eigvals[positive]
    eigvecs_pos = eigvecs[:, positive]

    if eigvals_pos.size < 2:
        # fallback: usa le prime due comunque
        coords = eigvecs[:, :2] * np.sqrt(np.abs(eigvals[:2]))
        explained = eigvals[:2] / np.sum(np.abs(eigvals))
        return coords, explained

    coords = eigvecs_pos[:, :2] * np.sqrt(eigvals_pos[:2])
    explained = eigvals_pos[:2] / eigvals_pos.sum()
    print("PCoA explained variance ratio (first 2):", explained)
    return coords, explained


# ---------- Utils ----------

def maybe_subsample(latent, prot_labels, habitat_labels, max_points, random_state=0):
    """
    Subsample casuale se i punti sono > max_points.
    Ritorna latent_sub, prot_sub, hab_sub.
    """
    n = latent.shape[0]
    if n <= max_points:
        return latent, prot_labels, habitat_labels

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    latent_sub = latent[idx]
    prot_sub = prot_labels[idx]
    hab_sub = habitat_labels[idx]
    return latent_sub, prot_sub, hab_sub


def plot_scatter(latent_2d, labels, title, cmap="viridis"):
    """
    Ritorna una figura matplotlib con scatter 2D colorato per labels.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                    c=labels, cmap=cmap, s=10, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("label")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    return fig


# ---------- Main ----------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Init Weights & Biases
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": vars(args),
    }
    if args.wandb_entity is not None:
        wandb_kwargs["entity"] = args.wandb_entity
    wandb.init(**wandb_kwargs)

    # 2) Carica dataset
    full_ds = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )

    # 3) Carica modello dal checkpoint
    model = Classifier.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        strict=True  # oppure strict=False se stai sperimentando versioni leggermente diverse
    )


    # 4) Estrai spazio latente + etichette protezione
    latent, prot_labels = extract_latent_representations(
        model,
        full_ds,
        batch_size=args.batch_size,
        device=device
    )

    n_samples = latent.shape[0]
    print(f"Numero di campioni nel dataset: {n_samples}")

    # 5) Etichette habitat categoriche (fattorizzate)
    if args.habitat_labels_file is not None:
        hab_df = pd.read_csv(args.habitat_labels_file)
        if len(hab_df) != n_samples:
            raise ValueError(
                f"habitat_labels_file ha {len(hab_df)} righe ma il dataset ha {n_samples} campioni."
            )
        if "habitat_label" not in hab_df.columns:
            raise ValueError("habitat_labels_file deve contenere una colonna 'habitat_label'.")
        hab_str = hab_df["habitat_label"].astype(str).values
        hab_codes, hab_uniques = pd.factorize(hab_str)
        # Logghiamo la mappatura label -> codice
        mapping_table = wandb.Table(columns=["habitat_label", "code"])
        for lbl, code in zip(hab_uniques, range(len(hab_uniques))):
            mapping_table.add_data(lbl, code)
        wandb.log({"habitat_label_mapping": mapping_table})
    else:
        # fallback: fattorizza argmax sul vettore habitat del dataset
        # (ad esempio se full_ds.habitats è un one-hot o simile)
        habitats = full_ds.habitats   # numpy array [N, H] o tensor
        if torch.is_tensor(habitats):
            habitats_np = habitats.numpy()
        else:
            habitats_np = habitats
        hab_idx = habitats_np.argmax(axis=1)
        hab_codes = hab_idx
        hab_uniques = np.unique(hab_codes)
        mapping_table = wandb.Table(columns=["habitat_code"])
        for code in hab_uniques:
            mapping_table.add_data(int(code))
        wandb.log({"habitat_code_mapping": mapping_table})

    # 6) Subsample per metodi costosi (t-SNE, PCoA)
    latent_sub, prot_sub, hab_sub = maybe_subsample(
        latent, prot_labels, hab_codes, args.max_points
    )
    print(f"Useremo {latent_sub.shape[0]} punti per t-SNE e PCoA.")

    # 7) PCA 2D (solo per debug / baseline)
    latent_pca_2d, pca_var = pca_2d(latent_sub)
    wandb.log({
        "pca_var_explained_PC1": float(pca_var[0]),
        "pca_var_explained_PC2": float(pca_var[1]),
    })

    fig_pca_prot = plot_scatter(
        latent_pca_2d, prot_sub,
        title="PCA (2D) - colored by protection"
    )
    fig_pca_hab = plot_scatter(
        latent_pca_2d, hab_sub,
        title="PCA (2D) - colored by habitat"
    )
    wandb.log({
        "PCA_protection": wandb.Image(fig_pca_prot),
        "PCA_habitat": wandb.Image(fig_pca_hab),
    })
    plt.close(fig_pca_prot)
    plt.close(fig_pca_hab)

    # 8) t-SNE 2D
    latent_tsne_2d = tsne_2d(latent_sub, perplexity=min(30, max(5, latent_sub.shape[0] // 10)))
    fig_tsne_prot = plot_scatter(
        latent_tsne_2d, prot_sub,
        title="t-SNE (2D) - colored by protection"
    )
    fig_tsne_hab = plot_scatter(
        latent_tsne_2d, hab_sub,
        title="t-SNE (2D) - colored by habitat"
    )
    wandb.log({
        "tSNE_protection": wandb.Image(fig_tsne_prot),
        "tSNE_habitat": wandb.Image(fig_tsne_hab),
    })
    plt.close(fig_tsne_prot)
    plt.close(fig_tsne_hab)

    # 9) PCoA 2D
    latent_pcoa_2d, pcoa_var = pcoa_2d(latent_sub)
    wandb.log({
        "pcoa_var_explained_PC1": float(pcoa_var[0]),
        "pcoa_var_explained_PC2": float(pcoa_var[1]),
    })

    fig_pcoa_prot = plot_scatter(
        latent_pcoa_2d, prot_sub,
        title="PCoA (2D) - colored by protection"
    )
    fig_pcoa_hab = plot_scatter(
        latent_pcoa_2d, hab_sub,
        title="PCoA (2D) - colored by habitat"
    )
    wandb.log({
        "PCoA_protection": wandb.Image(fig_pcoa_prot),
        "PCoA_habitat": wandb.Image(fig_pcoa_hab),
    })
    plt.close(fig_pcoa_prot)
    plt.close(fig_pcoa_hab)

    wandb.finish()


if __name__ == "__main__":
    main()
