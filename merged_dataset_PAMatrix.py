import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MergedDataset(Dataset):
    def __init__(self, embeddings_file: str, protection_file: str, habitat_file: str):
        # 1) load
        emb_df = pd.read_csv(embeddings_file)
        prot_df = pd.read_csv(protection_file)
        hab_df  = pd.read_csv(habitat_file)

        # 2) chiavi stringhe pulite
        emb_df["Sample"] = emb_df["Sample"].astype(str).str.strip()
        prot_df["spygen_code"] = prot_df["spygen_code"].astype(str).str.strip()
        hab_df["spygen_code"]  = hab_df["spygen_code"].astype(str).str.strip()

        # --- colonne specie (tutto tranne Sample) ---
        species_cols = [c for c in emb_df.columns if c != "Sample"]

        # 3) merge
        data = emb_df.merge(prot_df, left_on="Sample", right_on="spygen_code", how="inner")
        data = data.merge(hab_df, on="spygen_code", how="inner")

        # 4) one-hot habitat
        habitat_oh = pd.get_dummies(data["habitat"], prefix="habitat", dummy_na=False)

        # 5) salva codici
        self.codes = data["spygen_code"].astype(str).to_numpy()

        # 6) X specie: forza numerico (0/1) -> float32
        #    Se qualche colonna è stata letta come stringa, to_numeric la converte.
        X = data[species_cols].apply(pd.to_numeric, errors="coerce")

        # sanity: se ci sono NaN, vuol dire che avevi valori non numerici in qualche colonna specie
        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"Found non-numeric values in species columns (NaN after conversion). "
                f"Example bad columns: {bad_cols[:10]}"
            )

        self.embeddings = X.to_numpy(dtype=np.float32)

        # 7) habitat features
        self.habitats = habitat_oh.to_numpy(dtype=np.float32)

        # 8) label protezione (assicurati int)
        self.labels = pd.to_numeric(data["protection"], errors="raise").to_numpy(dtype=np.int64)

        # sanity shapes
        assert self.embeddings.shape[0] == self.habitats.shape[0] == self.labels.shape[0]

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.embeddings[idx])  # float32
        hab = torch.from_numpy(self.habitats[idx])    # float32
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, hab, label
