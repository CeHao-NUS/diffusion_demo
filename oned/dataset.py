import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from scipy.interpolate import CubicSpline


SCALE = 0.25
BIAS = 0.1
FILE_NAME  = 'scale'+str(SCALE)+'_bias'+str(BIAS)

def get_dataloader(dist='line'):

    
    if dist == 'line':
        dataset = LineDataset()
    elif dist == 'cubic_spline':
        dataset = CubicSplineDataset()
    elif dist == 'double_cubic_spline':
        dataset = DoubleCubicSplineDataset()

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
        x = np.random.normal() * SCALE + BIAS # Sample x from a unit Gaussian distribution
        data = x * np.ones(self.dimensions)  # Multiply x by an array of ones
        data = np.expand_dims(data, axis=-1)
        return torch.tensor(data, dtype=torch.float)  # Convert to a PyTorch tensor and return
    


class CubicSplineDataset(Dataset):
    def __init__(self, size=1000, dimensions=16):
        """
        Initialize the dataset.
        :param size: Number of samples in the dataset.
        :param dimensions: Number of dimensions for each sample.
        """
        self.size = size
        self.dimensions = dimensions
        self.x = np.linspace(0, 1, dimensions)  # Define the x-axis for the spline
        self.base_points_x = np.linspace(0, 1, 5)  # 5 points to define the spline

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Generate and return a single sample from the dataset at the given index.
        For each sample, generate a unique set of base_points_y to define the cubic spline,
        then evaluate the spline, and normalize the data by a weight a.
        """
        # Generate a unique set of y values for the cubic spline for each sample
        base_points_y = np.random.rand(5)  # Random y values for the spline

        # Create a cubic spline based on the unique base points for this sample
        spline = CubicSpline(self.base_points_x, base_points_y)

        # Evaluate the spline to generate data
        y = spline(self.x)

        # Sample weight a from a unit Gaussian distribution
        # a = np.random.normal()

        # Normalize the data by the weight a
        normalized_data = SCALE * y + BIAS

        batch_data = np.expand_dims(normalized_data, axis=-1)

        return torch.tensor(batch_data, dtype=torch.float)  # Convert to a PyTorch tensor and return
    

class DoubleCubicSplineDataset(Dataset):
    def __init__(self, size=1000, dimensions=16):
        """
        Initialize the dataset.
        :param size: Number of samples in the dataset.
        :param dimensions: Number of dimensions for each sample.
        """
        self.size = size
        self.dimensions = dimensions
        self.x = np.linspace(0, 1, dimensions)  # Define the x-axis for the spline

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Generate and return a single sample from the dataset at the given index.
        Each sample has its unique cubic spline based on random y values and is biased towards one of the two modes.
        """
        mode = np.random.choice([-1, 1])  # Uniformly select a mode
        base_points_x = np.linspace(0, 1, 5)  # x positions for base points remain constant
        base_points_y = np.random.rand(5)  # Random y values for the spline, unique for each sample
        
        # Create a cubic spline for this sample
        spline = CubicSpline(base_points_x, base_points_y)
        y = spline(self.x)  # Evaluate the spline
        
        normalized_data = SCALE * y + BIAS * mode  # Normalize and bias the data
        batch_data = np.expand_dims(normalized_data, axis=-1)

        return torch.tensor(batch_data, dtype=torch.float)  # Convert to a PyTorch tensor and return
