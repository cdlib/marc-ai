import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from .attribute_selector import AttributeSelector
from .similarity_vector_dataset import SimilarityVectorDataset
from typing import List


class MARCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_processed_path: str,
        val_processed_path: str,
        test_processed_path: str,
        attrs: List[str],
        batch_size: int,
    ):
        super().__init__()

        self.train_processed_path = train_processed_path
        self.val_processed_path = val_processed_path
        self.test_processed_path = test_processed_path

        self.batch_size = batch_size
        self.transform = torch.nn.Sequential(AttributeSelector(attrs))

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        self.train_set = SimilarityVectorDataset(
            self.train_processed_path, transform=self.transform
        )
        self.val_set = SimilarityVectorDataset(
            self.val_processed_path, transform=self.transform
        )
        self.test_set = SimilarityVectorDataset(
            self.test_processed_path, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=0)
