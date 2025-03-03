import pytorch_lightning as pl
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torchmetrics import Accuracy


class HardNegativeBCELoss(nn.Module):
    """
    Custom loss function that applies higher weights to hard negative examples.
    Hard negatives are non-matching pairs (label=0) that receive high similarity scores.
    """
    def __init__(self, hard_negative_weight=3.0, mining_threshold=0.7):
        super().__init__()
        self.base_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.hard_negative_weight = hard_negative_weight
        self.mining_threshold = mining_threshold
        
    def forward(self, pred, label):
        # Calculate base loss for all examples
        base_loss = self.base_criterion(pred, label)
        
        # Apply sigmoid to get prediction probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Identify hard negatives (false positives with high confidence)
        # These are samples where label=0 but pred_prob > threshold
        hard_negative_mask = (label < 0.5) & (pred_probs > self.mining_threshold)
        
        # Create weight tensor (default weight 1.0 for all samples)
        weights = torch.ones_like(base_loss)
        
        # Apply higher weight to hard negatives
        weights[hard_negative_mask] = self.hard_negative_weight
        
        # Apply weights to the base loss
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()


class SimilarityVectorModel(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, *, lr, weight_decay, optimizer, batch_size, features, 
                hard_negative_weight=3.0, mining_threshold=0.7, use_hard_negative_mining=True):
        super().__init__()

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.features = features
        self.hard_negative_weight = hard_negative_weight
        self.mining_threshold = mining_threshold
        self.use_hard_negative_mining = use_hard_negative_mining
        self.save_hyperparameters()

        self.layers = nn.Sequential(
            nn.Linear(len(features), 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.sigmoid = nn.Sigmoid()
        self.criterion = HardNegativeBCELoss(hard_negative_weight, mining_threshold) if use_hard_negative_mining else nn.BCEWithLogitsLoss()
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

        # Track hard negatives for logging
        if self.use_hard_negative_mining:
            pred_probs = torch.sigmoid(pred)
            hard_negative_mask = (label < 0.5) & (pred_probs > self.mining_threshold)
            hard_negative_count = hard_negative_mask.sum().item()
            self.log(f"hard_negatives", hard_negative_count, on_step=False, on_epoch=True)

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
