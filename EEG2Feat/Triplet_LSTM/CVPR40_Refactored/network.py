"""
LSTM-based EEG Feature Extraction Network

This module contains the LSTM network architecture for extracting meaningful features
from EEG signals for the triplet loss training paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class EEGFeatureLSTM(nn.Module):
    """
    LSTM-based feature extractor for EEG signals.
    
    Architecture:
    - Input: Raw EEG time series (batch_size, seq_len, n_channels)
    - LSTM encoder: Extracts temporal patterns
    - Feature layer: Projects to feature space
    - Output: Normalized projection for triplet loss
    
    Args:
        n_classes (int): Number of target classes (default: 40)
        n_channels (int): Number of EEG channels (default: 128)
        feat_dim (int): Dimension of feature representation (default: 128)
        projection_dim (int): Dimension of projection space (default: 128)
        n_lstm_layers (int): Number of stacked LSTM layers (default: 1)
    """
    
    def __init__(
        self,
        n_classes=40,
        n_channels=128,
        feat_dim=128,
        projection_dim=128,
        n_lstm_layers=1
    ):
        super(EEGFeatureLSTM, self).__init__()
        
        self.n_lstm_layers = n_lstm_layers
        self.hidden_size = feat_dim
        
        # LSTM encoder to process temporal EEG signals
        self.lstm_encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=self.hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
            dropout=0.0 if n_lstm_layers == 1 else 0.2
        )
        
        # Projection layer for triplet loss space
        self.projection_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=projection_dim,
            bias=False
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input EEG signal of shape (batch_size, seq_len, n_channels)
            
        Returns:
            torch.Tensor: Normalized projection vector of shape (batch_size, projection_dim)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden and cell states for LSTM
        h_0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size).to(device)
        
        # Process through LSTM
        _, (h_n, c_n) = self.lstm_encoder(x, (h_0, c_0))
        
        # Extract features from the last hidden state of final layer
        features = h_n[-1]
        
        # Project to projection space
        projection = self.projection_layer(features)
        
        # L2 normalize for triplet loss
        projection = F.normalize(projection, dim=-1)
        
        return projection


def create_eeg_feature_extractor(config_dict=None):
    """
    Factory function to create an EEG feature extraction model.
    
    Args:
        config_dict (dict, optional): Configuration dictionary. If None, uses config.py settings.
        
    Returns:
        EEGFeatureLSTM: Initialized model
    """
    if config_dict is None:
        config_dict = {
            'n_classes': config.NUM_CLASSES,
            'n_channels': config.N_CHANNELS,
            'feat_dim': config.FEAT_DIM,
            'projection_dim': config.PROJECTION_DIM,
            'n_lstm_layers': config.N_LSTM_LAYERS,
        }
    
    model = EEGFeatureLSTM(**config_dict)
    return model


if __name__ == '__main__':
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = EEGFeatureLSTM(
        n_channels=config.N_CHANNELS,
        feat_dim=config.FEAT_DIM,
        projection_dim=config.PROJECTION_DIM,
        n_lstm_layers=config.N_LSTM_LAYERS
    ).to(device)
    
    # Create dummy input
    dummy_eeg = torch.randn(8, 440, 128).to(device)
    
    # Forward pass
    output = model(dummy_eeg)
    
    print(f"Input shape: {dummy_eeg.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm (should be ~1.0): {output.norm(dim=-1).mean().item():.4f}")
