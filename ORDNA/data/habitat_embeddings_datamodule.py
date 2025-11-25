import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from typing import Optional

class CombinedDataset(Dataset):
    def __init__(self, embeddings_file: Path, habitat_labels_file: Path):
        self.embeddings_df = pd.read_csv(embeddings_file)
        self.habitat_labels_df = pd.read_csv(habitat_labels_file)
        self.habitat_labels_df.set_index('sample', inplace=True)
        
        self.samples = self.embeddings_df['Sample'].tolist()
        self.embeddings = self.embeddings_df.drop(columns=['Sample']).values

        self.habitat_labels = []
        for sample in self.samples:
            self.habitat_labels.append(self.habitat_labels_df.loc[sample, 'habitat'])
        
        self.habitat_labels = pd.Categorical(self.habitat_labels)
        self.habitat_one_hot = torch.nn.functional.one_hot(
            torch.tensor(self.habitat_labels.codes), 
            num_classes=len(self.habitat_labels.categories)
        ).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float), self.habitat_one_hot[idx]

class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, embeddings_file: Path, habitat_labels_file: Path, batch_size: int = 32):
        super().__init__()
        self.embeddings_file = embeddings_file
        self.habitat_labels_file = habitat_labels_file
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.dataset = CombinedDataset(self.embeddings_file, self.habitat_labels_file)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
