import argparse
import warnings
import torch
import numpy as np
from tqdm import tqdm

import pytorch_lightning as lightning
from pytorch_lightning.callbacks import EarlyStopping, Callback

from marcai.pl import MARCDataModule, SimilarityVectorModel
from marcai.utils import load_config


class HardNegativeMiningCallback(Callback):
    """
    PyTorch Lightning callback for hard negative mining.
    
    After each training epoch, identifies hard negatives and updates the dataset.
    """
    def __init__(self, mining_threshold=0.7, mining_frequency=1):
        super().__init__()
        self.mining_threshold = mining_threshold
        self.mining_frequency = mining_frequency
        self.current_epoch = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        """After each epoch, find hard negatives in the training set"""
        self.current_epoch += 1
        
        # Only mine hard negatives every mining_frequency epochs
        if self.current_epoch % self.mining_frequency != 0:
            return
            
        # Get training dataloader and dataset
        train_dataloader = trainer.train_dataloader
        if hasattr(train_dataloader, 'dataset'):
            dataset = train_dataloader.dataset
        else:
            # Handle CombinedLoader case in PyTorch Lightning
            dataset = train_dataloader.loaders.dataset

        # Skip if dataset doesn't support hard negative mining
        if not hasattr(dataset, 'update_hard_negatives'):
            return
            
        # Disable resampling during prediction
        if hasattr(dataset, 'enable_resampling'):
            dataset.enable_resampling(False)
        
        # Collect predictions for all training samples
        device = pl_module.device
        indices = []
        predictions = []
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        num_samples = len(dataset)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Mining hard negatives"):
                batch_indices = list(range(i, min(i + batch_size, num_samples)))
                batch = [dataset[idx][0] for idx in batch_indices]
                batch_tensor = torch.tensor(np.stack(batch), device=device).float()
                
                # Get predictions
                preds = pl_module.predict(batch_tensor).cpu().numpy().flatten()
                
                indices.extend(batch_indices)
                predictions.extend(preds)
        
        # Update dataset with hard negatives
        dataset.update_hard_negatives(indices, predictions, self.mining_threshold)
        
        # Enable resampling for next training epoch
        if hasattr(dataset, 'enable_resampling'):
            dataset.enable_resampling(True)


def train(name=None):
    config_path = "config.yaml"
    model_config = load_config(config_path)["model"]

    # Create data module from processed data
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    data = MARCDataModule(
        model_config["train_processed_path"],
        model_config["val_processed_path"],
        model_config["test_processed_path"],
        model_config["features"],
        model_config["batch_size"],
    )

    # Create model
    model = SimilarityVectorModel(
        lr=model_config["lr"],
        weight_decay=model_config["weight_decay"],
        optimizer=model_config["optimizer"],
        batch_size=model_config["batch_size"],
        features=model_config["features"],
        hard_negative_weight=model_config.get("hard_negative_weight", 3.0),
        mining_threshold=model_config.get("mining_threshold", 0.7),
        use_hard_negative_mining=model_config.get("use_hard_negative_mining", True)
    )

    callbacks = []
    if model_config["patience"] != -1:
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=model_config["patience"],
            verbose=False,
            mode="max",
        )
        callbacks.append(early_stop_callback)
    
    # Add hard negative mining callback if enabled
    use_hard_negative_mining = model_config.get("use_hard_negative_mining", True)
    if use_hard_negative_mining:
        mining_callback = HardNegativeMiningCallback(
            mining_threshold=model_config.get("mining_threshold", 0.7),
            mining_frequency=model_config.get("mining_frequency", 2)
        )
        callbacks.append(mining_callback)

    trainer = lightning.Trainer(
        max_epochs=model_config["max_epochs"], callbacks=callbacks, accelerator="cpu"
    )
    
    # Setup data module
    data.setup()
    
    # Enable hard negative resampling if supported
    if use_hard_negative_mining and hasattr(data.train_set, 'enable_resampling'):
        data.train_set.enable_resampling(True)
    
    # Train the model
    trainer.fit(model, data)

    # Save model
    model.save_pretrained(name)


def args_parser():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-n", "--run-name", help="Name for training run", required=True)
    return parser


def main(args):
    train(args.run_name)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
