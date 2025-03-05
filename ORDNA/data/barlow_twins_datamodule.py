import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset

class BarlowTwinsDataModule(pl.LightningDataModule):
    def __init__(self, samples_dir: Path, labels_file: Path, habitats_file: Path, sequence_length: int, sample_subset_size: int, batch_size: int = 8) -> None:
        super().__init__()

        self.train_samples_dir = samples_dir / "train"
        self.val_samples_dir = samples_dir / "valid"
        self.labels_file = labels_file
        self.habitats_file = habitats_file
        self.sequence_length = sequence_length
        self.sample_subset_size = sample_subset_size
        self.batch_size = batch_size

        assert(batch_size is not None) 

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = BarlowTwinsDataset(
                samples_dir=self.train_samples_dir,
                labels_file=self.labels_file,
                habitats_file=self.habitats_file,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length
            )

            self.val_dataset = BarlowTwinsDataset(
                samples_dir=self.val_samples_dir,
                labels_file=self.labels_file,
                habitats_file=self.habitats_file,
                sample_subset_size=self.sample_subset_size,
                sequence_length=self.sequence_length
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
            dataset=self.val_dataset, 
            # dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=12, 
            pin_memory=torch.cuda.is_available(), 
            drop_last=False
        )
