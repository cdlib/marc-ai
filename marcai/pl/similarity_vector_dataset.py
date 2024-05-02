from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class SimilarityVectorDataset(Dataset):

    def __init__(self, processed_path: str, transform=None):

        self.transform = transform
        self.data = pd.read_csv(processed_path)

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_dict()
        
        label = float(float(row['cid']) == 1.0)
                
        if self.transform:
            row = self.transform(row)

        row = np.array(list(row.values())).astype(float)

        return row, label
