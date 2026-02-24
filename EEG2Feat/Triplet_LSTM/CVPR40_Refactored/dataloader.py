"""
Data Loading Module for EEG-Image Pairs from CVPR40 Dataset

This module provides PyTorch Dataset and DataLoader utilities for loading
EEG signals paired with their corresponding images from the CVPR40 dataset.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional


class EEGImageDataset(Dataset):
    """
    Dataset class for loading EEG-image pairs.
    
    The dataset loads pre-computed EEG signals and corresponding images,
    normalizes EEG data, and returns them as tuples for training.
    
    Args:
        eeg_signals (torch.Tensor): EEG data of shape (n_samples, seq_len, n_channels)
        images (torch.Tensor): Image data of shape (n_samples, channels, height, width)
        labels (torch.Tensor): Class labels of shape (n_samples,)
    """
    
    def __init__(
        self,
        eeg_signals: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Initialize the dataset.
        
        Args:
            eeg_signals: Tensor of EEG signals
            images: Tensor of images
            labels: Tensor of class labels
            
        Raises:
            ValueError: If tensors have mismatched first dimension
        """
        if not (len(eeg_signals) == len(images) == len(labels)):
            raise ValueError(
                f"Mismatched dataset sizes: EEG={len(eeg_signals)}, "
                f"Images={len(images)}, Labels={len(labels)}"
            )
        
        self.eeg_signals = eeg_signals
        self.images = images
        self.labels = labels
        
        # Pre-compute normalization statistics for EEG
        self.eeg_mean = self.eeg_signals.mean(dim=(0, 1), keepdim=True)
        self.eeg_std = self.eeg_signals.std(dim=(0, 1), keepdim=True, unbiased=False)
        self.eeg_std = self.eeg_std.clamp(min=1e-6)  # Prevent division by zero
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.eeg_signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - eeg (torch.Tensor): Normalized EEG signal
                - image (torch.Tensor): Corresponding image
                - label (torch.Tensor): Class label
        """
        # Get raw data
        eeg = self.eeg_signals[idx]
        image = self.images[idx]
        label = self.labels[idx]
        
        # Normalize EEG: zero-mean, unit-variance per channel
        eeg = (eeg - eeg.mean()) / eeg.std(unbiased=False).clamp_min(1e-6)
        
        return eeg, image, label


def create_eeg_image_dataloader(
    eeg_signals: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for EEG-image pairs.
    
    Args:
        eeg_signals: Tensor of EEG signals
        images: Tensor of images
        labels: Tensor of class labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data at each epoch
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        torch.utils.data.DataLoader: Configured data loader
    """
    dataset = EEGImageDataset(eeg_signals, images, labels)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader


if __name__ == '__main__':
    # Example usage
    print("Example: Creating a dummy EEG-Image dataset")
    
    # Create dummy data
    n_samples = 100
    seq_len = 440
    n_channels = 128
    img_height, img_width = 224, 224
    
    dummy_eeg = torch.randn(n_samples, seq_len, n_channels)
    dummy_images = torch.randn(n_samples, 3, img_height, img_width)
    dummy_labels = torch.randint(0, 40, (n_samples,))
    
    # Create dataset and dataloader
    dataset = EEGImageDataset(dummy_eeg, dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Iterate through one batch
    eeg_batch, img_batch, label_batch = next(iter(dataloader))
    
    print(f"EEG batch shape: {eeg_batch.shape}")
    print(f"Image batch shape: {img_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
