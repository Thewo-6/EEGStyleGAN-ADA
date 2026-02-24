"""
Data Augmentation Module for EEG Signals

This module provides various augmentation techniques for EEG time-series data
to increase training data diversity and improve model robustness.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Callable, Dict, Any


class EEGAugmentor:
    """
    Class for applying various augmentations to EEG signals.
    
    Args:
        random_seed (int, optional): Random seed for reproducibility
    """
    
    def __init__(self, random_seed: int = 42):
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def time_shift(
        self,
        eeg: np.ndarray,
        max_shift: int = 10
    ) -> np.ndarray:
        """
        Randomly shift the EEG signal in time domain.
        
        Padding is added to maintain signal length.
        
        Args:
            eeg (np.ndarray): Input EEG of shape (n_channels, n_samples)
            max_shift (int): Maximum shift amount in samples
            
        Returns:
            np.ndarray: Time-shifted EEG signal of same shape
        """
        if max_shift == 0:
            return eeg
        
        # Sample non-zero shift
        shift = np.random.randint(-max_shift, max_shift)
        while shift == 0:
            shift = np.random.randint(-max_shift, max_shift)
        
        n_channels, n_samples = eeg.shape
        
        if shift > 0:
            shifted_eeg = np.pad(eeg[:, shift:], ((0, 0), (0, shift)), mode='constant')
        else:
            shifted_eeg = np.pad(eeg[:, :shift], ((0, 0), (abs(shift), 0)), mode='constant')
        
        return shifted_eeg
    
    def random_crop(
        self,
        eeg: np.ndarray,
        crop_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Randomly crop and pad the EEG signal.
        
        Args:
            eeg (np.ndarray): Input EEG of shape (n_channels, n_samples)
            crop_size (tuple): (n_samples_to_crop, n_channels_to_crop)
            
        Returns:
            np.ndarray: Cropped and padded EEG of original shape
        """
        n_channels, n_samples = eeg.shape
        crop_samples, crop_channels = crop_size
        
        if crop_channels > n_channels or crop_samples > n_samples:
            raise ValueError(
                f"Crop size {crop_size} exceeds input size {eeg.shape}"
            )
        
        # Random crop indices
        channel_idx = np.random.randint(0, n_channels - crop_channels + 1)
        sample_idx = np.random.randint(0, n_samples - crop_samples + 1)
        
        # Crop region
        cropped = eeg[
            channel_idx:channel_idx + crop_channels,
            sample_idx:sample_idx + crop_samples
        ]
        
        # Pad to original size
        pad_samples = n_samples - cropped.shape[1]
        cropped = np.pad(cropped, ((0, 0), (pad_samples, 0)), mode='constant')
        
        # Pad channels if needed
        pad_channels = n_channels - cropped.shape[0]
        if pad_channels > 0:
            cropped = np.pad(cropped, ((pad_channels, 0), (0, 0)), mode='constant')
        
        return cropped
    
    def channel_shuffle(self, eeg: np.ndarray) -> np.ndarray:
        """
        Randomly shuffle channel order.
        
        Args:
            eeg (np.ndarray): Input EEG of shape (n_channels, n_samples)
            
        Returns:
            np.ndarray: EEG with shuffled channels
        """
        n_channels = eeg.shape[0]
        shuffled_indices = np.random.permutation(n_channels)
        return eeg[shuffled_indices, :]
    
    def gaussian_noise(
        self,
        eeg: np.ndarray,
        noise_factor: float = 0.05
    ) -> np.ndarray:
        """
        Add Gaussian white noise to EEG signal.
        
        Args:
            eeg (np.ndarray): Input EEG of shape (n_channels, n_samples)
            noise_factor (float): Standard deviation of noise relative to signal
            
        Returns:
            np.ndarray: Noisy EEG signal
        """
        noise = np.random.normal(0, noise_factor, eeg.shape)
        return eeg + noise
    
    def bandpass_filter(
        self,
        eeg: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        fs: float = 1000.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply bandpass filter to EEG signal.
        
        Args:
            eeg (np.ndarray): Input EEG of shape (n_channels, n_samples)
            lowcut (float): Low cutoff frequency in Hz
            highcut (float): High cutoff frequency in Hz
            fs (float): Sampling frequency in Hz
            order (int): Filter order
            
        Returns:
            np.ndarray: Filtered EEG signal
        """
        nyquist = fs / 2
        low_norm = lowcut / nyquist
        high_norm = highcut / nyquist
        
        # Ensure normalized frequencies are valid
        low_norm = np.clip(low_norm, 0, 1)
        high_norm = np.clip(high_norm, 0, 1)
        
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        filtered_eeg = np.zeros_like(eeg)
        for ch in range(eeg.shape[0]):
            filtered_eeg[ch, :] = signal.filtfilt(b, a, eeg[ch, :])
        
        return filtered_eeg
    
    def apply_augmentation(
        self,
        eeg: np.ndarray,
        augmentation_name: str,
        **kwargs
    ) -> np.ndarray:
        """
        Apply a specific augmentation to EEG signal.
        
        Args:
            eeg (np.ndarray): Input EEG signal
            augmentation_name (str): Name of augmentation ('time_shift', 'crop', 'noise', etc.)
            **kwargs: Additional arguments for the augmentation
            
        Returns:
            np.ndarray: Augmented EEG signal
        """
        augmentations = {
            'time_shift': self.time_shift,
            'crop': self.random_crop,
            'shuffle': self.channel_shuffle,
            'noise': self.gaussian_noise,
            'filter': self.bandpass_filter,
        }
        
        if augmentation_name not in augmentations:
            raise ValueError(
                f"Unknown augmentation: {augmentation_name}. "
                f"Available: {list(augmentations.keys())}"
            )
        
        return augmentations[augmentation_name](eeg, **kwargs)


# Frequency bands for EEG analysis
EEG_FREQUENCY_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 100],
}


def extract_frequency_band(
    eeg: np.ndarray,
    band_name: str = 'alpha',
    fs: float = 1000.0
) -> np.ndarray:
    """
    Extract a specific frequency band from EEG signal.
    
    Args:
        eeg (np.ndarray): Input EEG signal
        band_name (str): Name of frequency band
        fs (float): Sampling frequency in Hz
        
    Returns:
        np.ndarray: EEG signal filtered to the specified band
    """
    if band_name not in EEG_FREQUENCY_BANDS:
        raise ValueError(f"Unknown band: {band_name}")
    
    lowcut, highcut = EEG_FREQUENCY_BANDS[band_name]
    
    augmentor = EEGAugmentor(random_seed=None)
    return augmentor.bandpass_filter(
        eeg,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs
    )


if __name__ == '__main__':
    print("Testing EEG Augmentation Module")
    
    # Create dummy EEG signal
    eeg = np.random.randn(128, 440)  # 128 channels, 440 samples
    
    augmentor = EEGAugmentor(random_seed=42)
    
    # Test augmentations
    print(f"\nOriginal EEG shape: {eeg.shape}")
    
    print("\n1. Time shift augmentation:")
    aug_eeg = augmentor.time_shift(eeg, max_shift=10)
    print(f"   Output shape: {aug_eeg.shape}")
    
    print("\n2. Random crop augmentation:")
    aug_eeg = augmentor.random_crop(eeg, crop_size=(440, 110))
    print(f"   Output shape: {aug_eeg.shape}")
    
    print("\n3. Channel shuffle:")
    aug_eeg = augmentor.channel_shuffle(eeg)
    print(f"   Output shape: {aug_eeg.shape}")
    
    print("\n4. Gaussian noise:")
    aug_eeg = augmentor.gaussian_noise(eeg, noise_factor=0.05)
    print(f"   Output shape: {aug_eeg.shape}")
    
    print("\n5. Bandpass filter (alpha band):")
    aug_eeg = augmentor.bandpass_filter(eeg, lowcut=8, highcut=13)
    print(f"   Output shape: {aug_eeg.shape}")
    
    print("\n6. Extract alpha band:")
    alpha_band = extract_frequency_band(eeg, band_name='alpha')
    print(f"   Alpha band shape: {alpha_band.shape}")
