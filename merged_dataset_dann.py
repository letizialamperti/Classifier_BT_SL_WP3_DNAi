import pandas as pd
import torch
from torch.utils.data import Dataset


class MergedDatasetDANN(Dataset):
    """
    Dataset per il modello domain-adversarial (DANN).

    Ritorna:
        emb    : tensor float (embedding features)
        label  : tensor long  (protection level)
        domain : tensor long  (indice intero dell'habitat, per il domain classifier)
    """
    def __init__(self, embeddings_file: str, protection_file: str, habitat_file: str):
        # 1) Caricamento dei CSV
        emb_df  = pd.read_csv(embeddings_file)
        prot_df = pd.read_csv(protection_file)
        hab_df  = pd.read_csv(habitat_file)

        # 2) Chiavi come stringhe
        emb_df["Sample"]       = emb_df["Sample"].astype(str)
        prot_df["spygen_code"] = prot_df["spygen_code"].astype(str)
        hab_df["spygen_code"]  = hab_df["spygen_code"].astype(str)

        # 3) Merge su Sample / spygen_code
        data = pd.merge(
            emb_df,
            prot_df,
            left_on="Sample", right_on="spygen_code"
        )
        data = pd.merge(
            data,
            hab_df,
            on="spygen_code"
        )

        # 4) DOMAIN LABELS (per DANN) — etichette di habitat
        self.domain_labels = data["habitat"].astype(str).values
        unique_domains = sorted(pd.unique(self.domain_labels))
        self.domain_to_idx = {d: i for i, d in enumerate(unique_domains)}
        self.idx_to_domain = {i: d for d, i in self.domain_to_idx.items()}
        self.domain_idx = [self.domain_to_idx[h] for h in self.domain_labels]
        self.num_domains = len(unique_domains)

        # 5) Salva i codici per mapping futuro
        self.codes = data["spygen_code"].values.astype(str)

        # 6) Estrai le colonne di embedding in modo robusto:
        #    tutte le colonne di emb_df tranne "Sample"
        emb_cols = [c for c in emb_df.columns if c != "Sample"]
        self.embeddings = data[emb_cols].values

        # 7) Labels di protezione
        self.labels = data["protection"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb   = torch.tensor(self.embeddings[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx],     dtype=torch.long)
        dom   = torch.tensor(self.domain_idx[idx], dtype=torch.long)
        return emb, label, dom
