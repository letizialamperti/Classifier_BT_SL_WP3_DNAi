import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import wandb

from merged_dataset import MergedDataset
from ORDNA.models.classifier_coralwheighted import Classifier


# ---------- Estrazione spazio latente ----------

@torch.no_grad()
def extract_latent_representations(model, dataset, batch_size=128, device="cuda"):
    """
    Estrae:
      - latent: rappresentazioni dal penultimo layer del classificatore
      - prot_labels: etichette di protezione (0..K-1)
    NB: le etichette di habitat categoriali arrivano da un CSV separato.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    model.to(device)

    # dimensione di input attesa dal primo Linear del classifier
    expected_in = model.classifier[0].in_features

    all_latent = []
    all_prot = []

    for emb, hab, prot in loader:
        emb = emb.to(device)
        hab = hab.to(device)
        prot = prot.to(device)

        x = torch.cat((emb, hab), dim=1)  # [B, D_current]
        D_current = x.shape[1]

        if D_current > expected_in:
            # taglia le feature in eccesso (usa le prime expected_in)
            x = x[:, :expected_in]
        elif D_current < expected_in:
            raise RuntimeError(
                f"Input dim {D_current} < expected {expected_in}. "
                f"Probabile mismatch tra dataset e modello."
            )

        # Il classifier è: Linear -> BN -> ReLU -> Dropout -> Linear
        # Usiamo tutto tranne l'ultimo Linear:
        latent = model.classifier[:-1](x)  # shape [B, hidden_dim]

        all_latent.append(latent.cpu())
        all_prot.append(prot.cpu())

    all_latent = torch.cat(all_latent, dim=0).numpy()   # [N, H]
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
    # ordina in valore assoluto decrescente (autovalori possono avere piccoli negativi numerici)
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

def maybe_subsample(latent, prot_labels, habitats, max_points, random_state=0):
    """
    Subsample casuale se i punti sono > max_points.
    Ritorna latent_sub, prot_sub, hab_sub.
    """
    n = latent.shape[0]
    if n <= max_points:
        return latent, prot_labels, habitats

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    latent_sub = latent[idx]
    prot_sub = prot_labels[idx]
    hab_sub = habitats[idx]
    return latent_sub, prot_sub, hab_sub


def plot_scatter(latent_2d, labels, title, categories=None, cmap="tab20"):
    """
    Scatter 2D per labels discreti (protection o habitat).

    labels     = array di codici numerici (es. 0,1,2,...)
    categories = lista dei nomi delle categorie (stessa lunghezza dei codici unici)
                 Se None → usa semplicemente i codici come stringhe.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    # Se categories non è fornito, usiamo le stringhe dei codici
    if categories is None:
        categories = [str(u) for u in unique_labels]

    # Colori discreti
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot per categoria (così possiamo costruire la legenda)
    for idx, u in enumerate(unique_labels):
        mask = labels == u
        ax.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            s=10,
            alpha=0.7,
            color=colors[idx],
            label=categories[idx]
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Category", fontsize=8)

    fig.tight_layout()
    return fig


# ---------- Main ----------

def main():

    # ---------- PARSER SEMPLICE ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--protection_file", type=str, required=True)
    parser.add_argument("--habitat_file", type=str, required=True)
    parser.add_argument("--use_raw_embeddings", action="store_true",
                    help="Se attivo, usa direttamente embeddings_file come matrice (no modello).")
    parser.add_argument("--emb_code_col", type=str, default="spygen_code",
                    help="Nome colonna codice nel CSV embeddings (default: spygen_code).")
    parser.add_argument("--emb_drop_cols", type=str, default="spygen_code",
                    help="Colonne da escludere dalla matrice embedding (comma-separated).")


    # CSV con etichette categoriali per visualizzare l'habitat
    parser.add_argument("--habitat_labels_file", type=str, default=None)

    # split file (con colonne: set, spygen_code)
    parser.add_argument("--k_cross_file", type=str, required=True)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_points", type=int, default=2000)

    # wandb
    parser.add_argument("--wandb_project", type=str, default="LatentSpaceCORAL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="coral_latent_analysis_val_only")

    args = parser.parse_args()
    # -------------------------------------

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

    # 2) Carica dataset completo
    full_ds = MergedDataset(
        args.embeddings_file,
        args.protection_file,
        args.habitat_file
    )

    # 3) Split CSV -> val_idx -> val_ds
    split_df = pd.read_csv(args.k_cross_file)
    val_codes_raw = split_df.loc[split_df['set'] == 'validation', 'spygen_code'].astype(str).tolist()

    code_to_idx = {str(code): i for i, code in enumerate(full_ds.codes)}
    valid_codes = set(map(str, full_ds.codes))

    val_codes = [c for c in val_codes_raw if c in valid_codes]
    val_idx = [code_to_idx[c] for c in val_codes]
    val_ds = Subset(full_ds, val_idx)

    if args.use_raw_embeddings:
    emb_df = pd.read_csv(args.embeddings_file)

    # 1) recupera i codici per joinare/filtrare validation
    if args.emb_code_col not in emb_df.columns:
        raise ValueError(
            f"Per --use_raw_embeddings, embeddings_file deve contenere la colonna '{args.emb_code_col}'. "
            f"Colonne trovate: {list(emb_df.columns)}"
        )

        emb_df = emb_df.copy()
        emb_df[args.emb_code_col] = emb_df[args.emb_code_col].astype(str)
    
        # 2) filtra SOLO validation usando i codici del k_cross (val_codes)
        emb_val = emb_df[emb_df[args.emb_code_col].isin(set(val_codes))].copy()
    
        # preserva l'ordine di val_codes (stesso ordine di val_ds)
        emb_val["_order"] = emb_val[args.emb_code_col].map({c: i for i, c in enumerate(val_codes)})
        emb_val = emb_val.sort_values("_order").drop(columns=["_order"])
    
        # 3) costruisci matrice X
        drop_cols = [c.strip() for c in args.emb_drop_cols.split(",") if c.strip()]
        feature_cols = [c for c in emb_val.columns if c not in drop_cols]
        X = emb_val[feature_cols].to_numpy(dtype=np.float32)
    
        latent = X
        n_samples = latent.shape[0]
        print(f"Numero di campioni (validation) da embeddings CSV: {n_samples}")
    
        # 4) prot_labels per colorare (serve join con PROTECTION file se vuoi)
        # Se non vuoi colorare, puoi mettere prot_labels = np.zeros(n_samples)
        prot_df = pd.read_csv(args.protection_file)
        if "spygen_code" not in prot_df.columns:
            raise ValueError("protection_file deve avere colonna 'spygen_code' per joinare in modalità raw embeddings.")
        if "protection" not in prot_df.columns:
            # cambia qui se la tua colonna si chiama diversamente
            raise ValueError("protection_file deve avere colonna 'protection' (o adattare il codice).")
    
        prot_df = prot_df.copy()
        prot_df["spygen_code"] = prot_df["spygen_code"].astype(str)
        code_to_prot = dict(zip(prot_df["spygen_code"].values, prot_df["protection"].values))
    
        missing_p = [c for c in val_codes if c not in code_to_prot]
        if missing_p:
            raise ValueError(f"Mancano protection label per {len(missing_p)} codici. Esempio: {missing_p[:10]}")
    
        prot_labels = np.array([code_to_prot[c] for c in val_codes])
    else:
        # ... percorso attuale: modello + val_ds
        latent, prot_labels = extract_latent_representations(...)


    # 4) Carica modello dal checkpoint
    model = Classifier.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        strict=True
    )

    # 5) Estrai spazio latente + etichette protezione (SOLO validation)
    latent, prot_labels = extract_latent_representations(
        model,
        val_ds,
        batch_size=args.batch_size,
        device=device
    )

    n_samples = latent.shape[0]
    print(f"Numero di campioni in validation: {n_samples}")

    # 6) Etichette habitat categoriche (fattorizzate) - join per spygen_code
    if args.habitat_labels_file is not None:
        hab_df = pd.read_csv(args.habitat_labels_file)

        if "spygen_code" not in hab_df.columns:
            raise ValueError(
                f"habitat_labels_file deve contenere una colonna 'spygen_code'. "
                f"Colonne disponibili: {list(hab_df.columns)}"
            )
        if "habitat" not in hab_df.columns:
            raise ValueError(
                f"habitat_labels_file deve contenere una colonna 'habitat'. "
                f"Colonne disponibili: {list(hab_df.columns)}"
            )

        # mapping code -> habitat
        hab_df = hab_df.copy()
        hab_df["spygen_code"] = hab_df["spygen_code"].astype(str)

        if hab_df["spygen_code"].duplicated().any():
            hab_df = hab_df.drop_duplicates(subset=["spygen_code"], keep="first")

        code_to_hab = dict(zip(hab_df["spygen_code"].values, hab_df["habitat"].astype(str).values))

        missing = [c for c in val_codes if c not in code_to_hab]
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} codici di validation non trovati nel habitat_labels_file (spygen_code). "
                f"Esempio mancanti: {missing[:10]}"
            )

        hab_str = np.array([code_to_hab[c] for c in val_codes], dtype=str)
        hab_codes, hab_uniques = pd.factorize(hab_str)

        # categorie nominali (per legenda)
        hab_categories = [str(lbl) for lbl in hab_uniques]

        # Logghiamo la mappatura label -> codice su WandB
        mapping_table = wandb.Table(columns=["habitat", "code"])
        for code, lbl in enumerate(hab_uniques):
            mapping_table.add_data(str(lbl), int(code))
        wandb.log({"habitat_label_mapping": mapping_table})

    else:
        # Fallback: se NON viene passato habitat_labels_file,
        # usiamo l'argmax delle feature habitat numeriche del dataset.
        habitats = full_ds.habitats   # numpy array [N, H] o tensor
        if torch.is_tensor(habitats):
            habitats_np = habitats.numpy()
        else:
            habitats_np = habitats

        habitats_np = habitats_np[val_idx]
        hab_codes = habitats_np.argmax(axis=1)
        hab_uniques = np.unique(hab_codes)

        # categorie nominali anche nel fallback (usiamo i codici come stringhe)
        hab_categories = [str(code) for code in hab_uniques]

        mapping_table = wandb.Table(columns=["habitat_code"])
        for code in hab_uniques:
            mapping_table.add_data(int(code))
        wandb.log({"habitat_code_mapping": mapping_table})

    # 7) Subsample per metodi costosi (t-SNE, PCoA)
    latent_sub, prot_sub, hab_sub = maybe_subsample(
        latent, prot_labels, hab_codes, args.max_points
    )
    print(f"Useremo {latent_sub.shape[0]} punti (validation) per t-SNE e PCoA.")

    # 8) PCA 2D
    latent_pca_2d, pca_var = pca_2d(latent_sub)
    wandb.log({
        "pca_var_explained_PC1": float(pca_var[0]),
        "pca_var_explained_PC2": float(pca_var[1]),
    })

    fig_pca_prot = plot_scatter(
        latent_pca_2d, prot_sub,
        title="PCA (2D) - VALIDATION ONLY - colored by protection"
    )
    fig_pca_hab = plot_scatter(
        latent_pca_2d, hab_sub,
        title="PCA (2D) - VALIDATION ONLY - colored by habitat",
        categories=hab_categories
    )
    wandb.log({
        "PCA_protection_val_only": wandb.Image(fig_pca_prot),
        "PCA_habitat_val_only": wandb.Image(fig_pca_hab),
    })
    plt.close(fig_pca_prot)
    plt.close(fig_pca_hab)

    # 9) t-SNE 2D
    latent_tsne_2d = tsne_2d(
        latent_sub,
        perplexity=min(30, max(5, latent_sub.shape[0] // 10))
    )
    fig_tsne_prot = plot_scatter(
        latent_tsne_2d, prot_sub,
        title="t-SNE (2D) - VALIDATION ONLY - colored by protection"
    )
    fig_tsne_hab = plot_scatter(
        latent_tsne_2d, hab_sub,
        title="t-SNE (2D) - VALIDATION ONLY - colored by habitat",
        categories=hab_categories
    )
    wandb.log({
        "tSNE_protection_val_only": wandb.Image(fig_tsne_prot),
        "tSNE_habitat_val_only": wandb.Image(fig_tsne_hab),
    })
    plt.close(fig_tsne_prot)
    plt.close(fig_tsne_hab)

    # 10) PCoA 2D
    latent_pcoa_2d, pcoa_var = pcoa_2d(latent_sub)
    wandb.log({
        "pcoa_var_explained_PC1": float(pcoa_var[0]),
        "pcoa_var_explained_PC2": float(pcoa_var[1]),
    })

    fig_pcoa_prot = plot_scatter(
        latent_pcoa_2d, prot_sub,
        title="PCoA (2D) - VALIDATION ONLY - colored by protection"
    )
    fig_pcoa_hab = plot_scatter(
        latent_pcoa_2d, hab_sub,
        title="PCoA (2D) - VALIDATION ONLY - colored by habitat",
        categories=hab_categories
    )
    wandb.log({
        "PCoA_protection_val_only": wandb.Image(fig_pcoa_prot),
        "PCoA_habitat_val_only": wandb.Image(fig_pcoa_hab),
    })
    plt.close(fig_pcoa_prot)
    plt.close(fig_pcoa_hab)

    wandb.finish()


if __name__ == "__main__":
    main()
