import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset

class BarlowTwinsDataModule(pl.LightningDataModule):
    def __init__(self, embeddings_file: Path, labels_file: Path, habitats_file: Path, batch_size: int = 8) -> None:
        super().__init__()
        self.embeddings_file = embeddings_file
        self.labels_file = labels_file
        self.habitats_file = habitats_file
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = BarlowTwinsDataset(
                embeddings_file=self.embeddings_file,
                labels_file=self.labels_file,
                habitats_file=self.habitats_file
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=12, 
            pin_memory=torch.cuda.is_available(), 
            drop_last=False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=12, 
            pin_memory=torch.cuda.is_available(), 
            drop_last=False
        )
