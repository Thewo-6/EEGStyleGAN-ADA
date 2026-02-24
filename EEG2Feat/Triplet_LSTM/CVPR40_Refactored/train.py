"""
Training Script for EEG Feature Learning with Triplet LSTM

This module handles the complete training pipeline including:
- Data loading from CVPR40 dataset
- Model training with triplet loss
- Validation and evaluation
- Checkpoint management
- Logging and visualization
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from natsort import natsorted
import cv2

from pytorch_metric_learning import miners, losses

# Import local modules
import config
from network import create_eeg_feature_extractor
from dataloader import create_eeg_image_dataloader, EEGImageDataset
from dataaugmentation import EEGAugmentor


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir: str, log_name: str = 'training.log') -> logging.Logger:
    """
    Configure logging for training process.
    
    Args:
        log_dir: Directory to save log files
        log_name: Name of log file
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset_split(
    directory: str,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load EEG signals and images from dataset split directory.
    
    Args:
        directory: Path to dataset split directory
        device: PyTorch device to move data to
        
    Returns:
        Tuple of (eeg_signals, images, labels) as tensors
    """
    eeg_list = []
    image_list = []
    label_list = []
    
    print(f"Loading data from {directory}...")
    
    for filename in tqdm(natsorted(os.listdir(directory))):
        if not filename.endswith('.npy'):
            continue
        
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        
        # Expected format: (image, eeg, label, ...)
        image = data[0]
        eeg = data[1].T  # Transpose to (n_channels, n_samples)
        label = data[2]
        
        # Pre-process image
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))  # CHW format
        
        eeg_list.append(eeg)
        image_list.append(image)
        label_list.append(label)
    
    # Convert to tensors
    eeg_tensor = torch.from_numpy(np.array(eeg_list, dtype=np.float32)).to(device)
    image_tensor = torch.from_numpy(np.array(image_list, dtype=np.float32)).to(device)
    label_tensor = torch.from_numpy(np.array(label_list, dtype=np.long)).to(device)
    
    print(f"Loaded shapes - EEG: {eeg_tensor.shape}, Images: {image_tensor.shape}, Labels: {label_tensor.shape}")
    
    return eeg_tensor, image_tensor, label_tensor


# ============================================================================
# TRAINING & VALIDATION FUNCTIONS
# ============================================================================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    miner,
    loss_fn,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[float, np.ndarray]:
    """
    Train model for one epoch.
    
    Args:
        epoch: Current epoch number
        model: EEG feature extraction model
        dataloader: Training data loader
        optimizer: Optimizer for model parameters
        miner: Hard triplet miner
        loss_fn: Loss function
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, feature_projections)
    """
    model.train()
    running_loss = []
    all_projections = np.array([])
    all_labels = np.array([])
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Training]")
    
    for eeg, images, labels in pbar:
        eeg = eeg.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        projections = model(eeg)
        
        # Mine hard triplets
        hard_pairs = miner(projections, labels)
        
        # Compute loss
        loss = loss_fn(projections, labels, hard_pairs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.detach().cpu().item())
        pbar.set_postfix({'loss': f'{np.mean(running_loss):.4f}'})
        
        # Collect projections for visualization
        all_projections = np.concatenate(
            (all_projections, projections.detach().cpu().numpy()),
            axis=0
        ) if all_projections.size else projections.detach().cpu().numpy()
        all_labels = np.concatenate(
            (all_labels, labels.cpu().numpy()),
            axis=0
        ) if all_labels.size else labels.cpu().numpy()
    
    avg_loss = np.mean(running_loss)
    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    
    return avg_loss, all_projections, all_labels


def validate(
    epoch: int,
    model: nn.Module,
    dataloader,
    miner,
    loss_fn,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Validate model on validation set.
    
    Args:
        epoch: Current epoch number
        model: EEG feature extraction model
        dataloader: Validation data loader
        miner: Hard triplet miner
        loss_fn: Loss function
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, projections, labels)
    """
    model.eval()
    running_loss = []
    all_projections = np.array([])
    all_labels = np.array([])
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Validation]")
    
    with torch.no_grad():
        for eeg, images, labels in pbar:
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # Forward pass
            projections = model(eeg)
            
            # Mine hard triplets
            hard_pairs = miner(projections, labels)
            
            # Compute loss
            loss = loss_fn(projections, labels, hard_pairs)
            
            running_loss.append(loss.detach().cpu().item())
            pbar.set_postfix({'loss': f'{np.mean(running_loss):.4f}'})
            
            # Collect projections
            all_projections = np.concatenate(
                (all_projections, projections.cpu().numpy()),
                axis=0
            ) if all_projections.size else projections.cpu().numpy()
            all_labels = np.concatenate(
                (all_labels, labels.cpu().numpy()),
                axis=0
            ) if all_labels.size else labels.cpu().numpy()
    
    avg_loss = np.mean(running_loss)
    logger.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}")
    
    return avg_loss, all_projections, all_labels


def compute_kmeans_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 40
) -> float:
    """
    Compute K-means clustering accuracy.
    
    Args:
        embeddings: Feature embeddings (n_samples, feat_dim)
        labels: True class labels (n_samples,)
        n_clusters: Number of clusters
        
    Returns:
        Clustering accuracy
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import homogeneity_completeness_v_measure
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        labels, cluster_labels
    )
    
    return v_measure


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> int:
    """
    Load model and optimizer state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load into
        optimizer: Optimizer to load into
        device: PyTorch device
        
    Returns:
        Starting epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch + 1


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer
) -> None:
    """
    Save model and optimizer state to checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer to save
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        filepath
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """
    Main training loop.
    """
    # Setup
    logger = setup_logger(log_dir='./logs')
    device = config.get_device()
    
    logger.info("="*60)
    logger.info("Starting EEG Feature Learning with Triplet LSTM")
    logger.info("="*60)
    
    # Create experiment directory
    existing_experiments = natsorted([
        d for d in os.listdir('.')
        if os.path.isdir(d) and d.startswith('EXPERIMENT_')
    ])
    experiment_num = int(existing_experiments[-1].split('_')[1]) + 1 if existing_experiments else 1
    
    exp_dir = f"EXPERIMENT_{experiment_num}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/best_checkpoints", exist_ok=True)
    
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    train_eeg, train_images, train_labels = load_dataset_split(
        config.get_dataset_path('train'),
        device
    )
    val_eeg, val_images, val_labels = load_dataset_split(
        config.get_dataset_path('validation'),
        device
    )
    
    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader = create_eeg_image_dataloader(
        train_eeg, train_images, train_labels,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = create_eeg_image_dataloader(
        val_eeg, val_images, val_labels,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Create model
    logger.info("\nInitializing model...")
    model = create_eeg_feature_extractor().to(device)
    model = nn.DataParallel(model).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS
    )
    
    # Loss and miner
    miner = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()
    
    # Load checkpoint if exists
    start_epoch = 0
    ckpt_files = natsorted([
        f for f in os.listdir(f"{exp_dir}/checkpoints")
        if f.endswith('.pth')
    ])
    if ckpt_files:
        latest_ckpt = os.path.join(f"{exp_dir}/checkpoints", ckpt_files[-1])
        start_epoch = load_checkpoint(latest_ckpt, model, optimizer, device)
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(start_epoch, config.N_EPOCHS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.N_EPOCHS}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_emb, train_lab = train_one_epoch(
            epoch, model, train_loader, optimizer, miner, loss_fn, device, logger
        )
        
        # Validate every VISUALIZATION_FREQ epochs
        if (epoch + 1) % config.VISUALIZATION_FREQ == 0:
            val_loss, val_emb, val_lab = validate(
                epoch, model, val_loader, miner, loss_fn, device, logger
            )
            
            # Compute clustering metrics
            train_kmeans = compute_kmeans_accuracy(train_emb, train_lab, config.K_MEANS_CLUSTERS)
            val_kmeans = compute_kmeans_accuracy(val_emb, val_lab, config.K_MEANS_CLUSTERS)
            
            logger.info(f"Train K-Means Score: {train_kmeans:.4f}")
            logger.info(f"Val K-Means Score: {val_kmeans:.4f}")
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                save_checkpoint(
                    f"{exp_dir}/best_checkpoints",
                    f"best_model_epoch_{epoch}.pth",
                    epoch, model, optimizer
                )
                logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        # Save regular checkpoint
        save_checkpoint(
            f"{exp_dir}/checkpoints",
            f"checkpoint_epoch_{epoch}.pth",
            epoch, model, optimizer
        )
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
