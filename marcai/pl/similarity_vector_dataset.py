from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class SimilarityVectorDataset(Dataset):

    def __init__(self, processed_path: str, transform=None, cache_hard_negatives=True):
        """
        Dataset for similarity vectors with support for hard negative mining.
        
        Args:
            processed_path: Path to the CSV file with processed data
            transform: Transform to apply to the data
            cache_hard_negatives: Whether to cache predictions for hard negative mining
        """
        self.transform = transform
        self.data = pd.read_csv(processed_path)
        self.cache_hard_negatives = cache_hard_negatives
        
        # Dictionary to store model predictions for hard negative mining
        # Maps indices to predicted probabilities
        self.prediction_cache = {}
        
        # Track hard negative indices
        self.hard_negative_indices = []
        
        # Create a balanced dataset with extra hard negatives
        self.resampling_enabled = False

    def __len__(self):
        return self.data.shape[0]
    
    def update_hard_negatives(self, indices, predictions, threshold=0.7):
        """
        Update the hard negative cache based on model predictions.
        
        Args:
            indices: List of indices in the dataset
            predictions: Corresponding model predictions
            threshold: Threshold for identifying hard negatives
        """
        # Add predictions to cache
        for idx, pred in zip(indices, predictions):
            self.prediction_cache[idx] = pred
            
        # Identify hard negatives (non-matches with high confidence)
        self.hard_negative_indices = []
        for idx, pred in self.prediction_cache.items():
            row = self.data.iloc[idx]
            label = float(float(row['cid']) == 1.0)
            if label < 0.5 and pred > threshold:
                self.hard_negative_indices.append(idx)
        
        print(f"Found {len(self.hard_negative_indices)} hard negatives out of {len(self.prediction_cache)} samples")
        
    def enable_resampling(self, enabled=True):
        """Enable or disable hard negative resampling during training"""
        self.resampling_enabled = enabled
        
    def __getitem__(self, idx):
        """Get item with support for hard negative resampling"""
        # If resampling is enabled and we have a 50% chance, sample a hard negative
        if self.resampling_enabled and len(self.hard_negative_indices) > 0 and np.random.rand() < 0.5:
            idx = np.random.choice(self.hard_negative_indices)
            
        row = self.data.iloc[idx].to_dict()
        label = float(float(row['cid']) == 1.0)
                
        if self.transform:
            row = self.transform(row)

        row = np.array(list(row.values())).astype(float)

        return row, label
