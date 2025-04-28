import pandas as pd
import torch
from torch.utils.data import Dataset

class MergedDataset(Dataset):
    def __init__(self, embeddings_file: str, protection_file: str, habitat_file: str):
        # 1) Caricamento di file CSV
        self.embeddings = pd.read_csv(embeddings_file)
        self.protection = pd.read_csv(protection_file)
        self.habitat    = pd.read_csv(habitat_file)

        # 2) Assicurati che le chiavi siano stringhe
        self.embeddings['Sample']         = self.embeddings['Sample'].astype(str)
        self.protection['spygen_code']    = self.protection['spygen_code'].astype(str)
        self.habitat['spygen_code']       = self.habitat['spygen_code'].astype(str)

        # 3) Merge dei tre data‐frame su spygen_code / Sample
        self.data = pd.merge(
            self.embeddings,
            self.protection,
            left_on='Sample', right_on='spygen_code'
        )
        self.data = pd.merge(
            self.data,
            self.habitat,
            on='spygen_code'
        )

        # 4) One‐hot encode della colonna 'habitat'
        habitat_one_hot = pd.get_dummies(self.data['habitat'], prefix='', prefix_sep='')
        self.data = pd.concat([self.data, habitat_one_hot], axis=1)

        # 5) Conserva i spygen_code originali per mappature future
        self.codes = self.data['spygen_code'].values.astype(str)

        # 6) Rimuovi colonne non più necessarie
        self.data = self.data.drop(columns=['habitat', 'spygen_code'])

        # 7) Estrai array NumPy per embeddings e habitat one‐hot
        n_habitats = habitat_one_hot.shape[1]
        # - embeddings: tutte le colonne tranne la prima (Sample), l'ultima (protection) e le n_habitats
        self.embeddings = self.data.iloc[:, 1 : -n_habitats-1].values
        # - habitats: le ultime n_habitats colonne
        self.habitats   = self.data.iloc[:, -n_habitats :    ].values
        # - labels di protezione
        self.labels     = self.data['protection'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb     = torch.tensor(self.embeddings[idx], dtype=torch.float)
        hab     = torch.tensor(self.habitats[idx],   dtype=torch.float)
        label   = torch.tensor(self.labels[idx],     dtype=torch.long)
        return emb, hab, label
