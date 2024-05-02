import pytorch_lightning as pl
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torchmetrics import Accuracy


class SimilarityVectorModel(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, lr=0.1, weight_decay=0.0, optimizer="Adam", batch_size=512):
        super().__init__()

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.layers = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        return self.sigmoid(self(x))

    def common_step(self, batch, batch_idx):
        sim, label = batch
        pred = self(sim.float())
        label = label.unsqueeze(1)

        loss = self.criterion(pred, label)
        acc = self.accuracy(pred, label.long())

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizers = {
            "Adadelta": torch.optim.Adadelta,
            "Adagrad": torch.optim.Adagrad,
            "Adam": torch.optim.Adam,
            "RMSprop": torch.optim.RMSprop,
            "SGD": torch.optim.SGD,
        }
        return optimizers[self.optimizer](
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
