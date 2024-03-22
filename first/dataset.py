import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_dataloader(dist='line'):

    
    if dist == 'line':
        dataset = LineDataset()

    return dataset


class LineDataset(Dataset):
    def __init__(self, size=1000, dimensions=16):
        """
        Initialize the dataset.
        :param size: Number of samples in the dataset.
        """
        self.size = size
        self.dimensions = dimensions

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Generate and return a single sample from the dataset at the given index.
        Here, x is sampled from a unit Gaussian distribution.
        """
        x = np.random.normal()  # Sample x from a unit Gaussian distribution
        data = x * np.ones(self.dimensions)  # Multiply x by an array of ones
        data = np.expand_dims(data, axis=-1)
        return torch.tensor(data, dtype=torch.float)  # Convert to a PyTorch tensor and return