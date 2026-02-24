"""
Configuration Module for EEG Feature Learning with Triplet LSTM

This module manages all hyperparameters and settings for training EEG feature extractors
using Triplet LSTM networks on the CVPR40 dataset.
"""

import torch
import numpy as np

# Set random seeds for reproducibility
np.random.seed(45)
torch.manual_seed(45)


# ============================================================================
# DATASET PATHS
# ============================================================================

# Update these paths to match your system
BASE_PATH = r'D:\DATASETS\EEGCVPR40\\'
DATASET_PATHS = {
    'train': BASE_PATH + r'\data\eeg_imagenet40_cvpr_2017_raw\train\\',
    'validation': BASE_PATH + r'\data\eeg_imagenet40_cvpr_2017_raw\val\\',
    'test': BASE_PATH + r'\data\eeg_imagenet40_cvpr_2017_raw\test\\',
}

# Note: For Linux/macOS systems, use forward slashes and adjust paths accordingly
# DATASET_PATHS = {
#     'train': '/path/to/eeg_imagenet40_cvpr_2017_raw/train/',
#     'validation': '/path/to/eeg_imagenet40_cvpr_2017_raw/val/',
#     'test': '/path/to/eeg_imagenet40_cvpr_2017_raw/test/',
# }


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Feature extraction
FEAT_DIM = 128              # Dimension of EEG feature representation
PROJECTION_DIM = 128        # Dimension of projection space for loss computation

# Dataset properties
NUM_CLASSES = 40            # Number of object categories in CVPR40 dataset
N_CHANNELS = 128            # Number of EEG electrodes/channels
TIMESTEPS = 440             # Temporal length of EEG signal

# LSTM architecture
N_LSTM_LAYERS = 4           # Number of stacked LSTM layers


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 48             # Batch size for training
LEARNING_RATE = 3e-4        # Initial learning rate for Adam optimizer
N_EPOCHS = 8192             # Total number of training epochs


# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

# Time shift augmentation
MAX_TIME_SHIFT = 10         # Maximum number of samples to shift in time domain

# Random crop augmentation
CROP_SIZE = (TIMESTEPS, 110)  # (temporal_length, n_channels) for cropping

# Noise addition
NOISE_FACTOR = 0.05         # Standard deviation of Gaussian noise to add


# ============================================================================
# MONITORING AND VISUALIZATION
# ============================================================================

VISUALIZATION_FREQ = 1      # Frequency (in epochs) for computing visualizations and metrics
K_MEANS_CLUSTERS = 40       # Number of clusters for K-means evaluation


# ============================================================================
# ADAM OPTIMIZER SETTINGS
# ============================================================================

ADAM_BETAS = (0.9, 0.999)   # Beta parameters for Adam optimizer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device():
    """Get the configured PyTorch device."""
    return DEVICE


def get_dataset_path(split):
    """
    Get the path for a specific dataset split.
    
    Args:
        split (str): One of 'train', 'validation', or 'test'
        
    Returns:
        str: Full path to the dataset split
        
    Raises:
        ValueError: If split is not recognized
    """
    if split not in DATASET_PATHS:
        raise ValueError(f"Unknown split: {split}. Choose from {list(DATASET_PATHS.keys())}")
    return DATASET_PATHS[split]
