"""
Example script to analyze CSV training logs from EEGStyleGAN-ADA project.
This script demonstrates how to load and visualize training metrics from CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import argparse


def plot_training_metrics(csv_path, output_dir='analysis_plots'):
    """
    Plot training metrics from a CSV log file.
    
    Args:
        csv_path: Path to the CSV log file
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Get experiment name from path
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
    
    print(f"\nAnalyzing: {exp_name}")
    print(f"CSV file: {csv_path}")
    print(f"Total epochs/ticks: {len(df)}")
    
    # Determine the type of CSV based on columns
    columns = df.columns.tolist()
    
    if 'kimg' in columns:
        # StyleGAN training log
        plot_stylegan_metrics(df, exp_name, output_dir)
    elif 'train_acc' in columns:
        # Finetuning log
        plot_finetuning_metrics(df, exp_name, output_dir)
    elif 'train_kmeans_acc' in columns:
        # Regular training log with k-means
        plot_training_with_kmeans(df, exp_name, output_dir)
    elif 'checkpoint_saved' in columns:
        # EEGClip log
        plot_eegclip_metrics(df, exp_name, output_dir)
    else:
        print("Unknown CSV format. Available columns:", columns)
        return
    
    print(f"\nPlots saved to: {output_dir}/")


def plot_training_with_kmeans(df, exp_name, output_dir):
    """Plot metrics for training scripts with K-means accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{exp_name} - Training Metrics', fontsize=16)
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], marker='o', markersize=3, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    val_data = df[df['val_loss'].notna()]
    if not val_data.empty:
        ax.plot(val_data['epoch'], val_data['val_loss'], marker='s', markersize=5, label='Val Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No validation loss data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 3: K-means Accuracy
    ax = axes[1, 0]
    train_kmeans = df[df['train_kmeans_acc'].notna()]
    val_kmeans = df[df['val_kmeans_acc'].notna()]
    if not train_kmeans.empty:
        ax.plot(train_kmeans['epoch'], train_kmeans['train_kmeans_acc'], marker='o', markersize=5, label='Train K-means', color='green')
    if not val_kmeans.empty:
        ax.plot(val_kmeans['epoch'], val_kmeans['val_kmeans_acc'], marker='s', markersize=5, label='Val K-means', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('K-means Accuracy')
    ax.set_title('K-means Clustering Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Best Validation Accuracy
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['best_val_acc'], marker='o', markersize=3, color='purple')
    ax.axhline(y=df['best_val_acc'].max(), color='r', linestyle='--', label=f'Max: {df["best_val_acc"].max():.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Val Accuracy')
    ax.set_title('Best Validation Accuracy Progress')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_training_metrics.png', dpi=150)
    print(f"  - Saved: {exp_name}_training_metrics.png")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    if not val_data.empty:
        print(f"  Final val loss: {val_data['val_loss'].iloc[-1]:.4f}")
    if not val_kmeans.empty:
        print(f"  Best val K-means accuracy: {df['best_val_acc'].max():.4f} at epoch {df['best_val_epoch'].iloc[-1]}")


def plot_finetuning_metrics(df, exp_name, output_dir):
    """Plot metrics for finetuning scripts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{exp_name} - Finetuning Metrics', fontsize=16)
    
    # Plot 1: Training and Validation Loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], marker='o', markersize=3, label='Train Loss')
    val_data = df[df['val_loss'].notna()]
    if not val_data.empty:
        ax.plot(val_data['epoch'], val_data['val_loss'], marker='s', markersize=5, label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Training and Validation Accuracy
    ax = axes[0, 1]
    train_acc = df[df['train_acc'].notna()]
    val_acc = df[df['val_acc'].notna()]
    if not train_acc.empty:
        ax.plot(train_acc['epoch'], train_acc['train_acc'], marker='o', markersize=5, label='Train Acc', color='green')
    if not val_acc.empty:
        ax.plot(val_acc['epoch'], val_acc['val_acc'], marker='s', markersize=5, label='Val Acc', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Loss Comparison (Last 50 epochs)
    ax = axes[1, 0]
    recent_data = df.tail(50)
    ax.plot(recent_data['epoch'], recent_data['train_loss'], marker='o', markersize=3, label='Train Loss')
    recent_val = recent_data[recent_data['val_loss'].notna()]
    if not recent_val.empty:
        ax.plot(recent_val['epoch'], recent_val['val_loss'], marker='s', markersize=5, label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Recent Loss (Last 50 Epochs)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Best Validation Accuracy
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['best_val_acc'], marker='o', markersize=3, color='purple')
    ax.axhline(y=df['best_val_acc'].max(), color='r', linestyle='--', label=f'Max: {df["best_val_acc"].max():.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Val Accuracy')
    ax.set_title('Best Validation Accuracy Progress')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_finetuning_metrics.png', dpi=150)
    print(f"  - Saved: {exp_name}_finetuning_metrics.png")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    if not train_acc.empty:
        print(f"  Final train accuracy: {train_acc['train_acc'].iloc[-1]:.4f}")
    if not val_acc.empty:
        print(f"  Best val accuracy: {df['best_val_acc'].max():.4f} at epoch {df['best_val_epoch'].iloc[-1]}")


def plot_stylegan_metrics(df, exp_name, output_dir):
    """Plot metrics for StyleGAN training."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{exp_name} - StyleGAN Training Metrics', fontsize=16)
    
    # Plot 1: Generator and Discriminator Loss
    ax = axes[0, 0]
    ax.plot(df['kimg'], df['Loss_G'], label='Generator Loss', marker='o', markersize=2)
    ax.plot(df['kimg'], df['Loss_D'], label='Discriminator Loss', marker='s', markersize=2)
    ax.set_xlabel('kimg')
    ax.set_ylabel('Loss')
    ax.set_title('Generator and Discriminator Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Training Time Metrics
    ax = axes[0, 1]
    ax.plot(df['kimg'], df['sec_per_kimg'], marker='o', markersize=2, color='green')
    ax.set_xlabel('kimg')
    ax.set_ylabel('Seconds per kimg')
    ax.set_title('Training Speed')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Augmentation Probability
    ax = axes[0, 2]
    ax.plot(df['kimg'], df['augment_p'], marker='o', markersize=2, color='purple')
    ax.set_xlabel('kimg')
    ax.set_ylabel('Augmentation p')
    ax.set_title('ADA Augmentation Probability')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory Usage
    ax = axes[1, 0]
    ax.plot(df['kimg'], df['cpu_mem_gb'], marker='o', markersize=2, label='CPU Memory', color='blue')
    ax.plot(df['kimg'], df['peak_gpu_mem_gb'], marker='s', markersize=2, label='GPU Memory', color='red')
    ax.set_xlabel('kimg')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Usage')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Loss Signs Real
    ax = axes[1, 1]
    ax.plot(df['kimg'], df['Loss_signs_real'], marker='o', markersize=2, color='orange')
    ax.set_xlabel('kimg')
    ax.set_ylabel('Loss Signs Real')
    ax.set_title('Real Image Loss Signs')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Total Training Time
    ax = axes[1, 2]
    ax.plot(df['kimg'], df['total_sec'] / 3600, marker='o', markersize=2, color='brown')
    ax.set_xlabel('kimg')
    ax.set_ylabel('Hours')
    ax.set_title('Total Training Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_stylegan_metrics.png', dpi=150)
    print(f"  - Saved: {exp_name}_stylegan_metrics.png")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total kimg: {df['kimg'].iloc[-1]:.1f}")
    print(f"  Total training time: {df['total_sec'].iloc[-1] / 3600:.2f} hours")
    print(f"  Final G loss: {df['Loss_G'].iloc[-1]:.4f}")
    print(f"  Final D loss: {df['Loss_D'].iloc[-1]:.4f}")
    print(f"  Average sec/kimg: {df['sec_per_kimg'].mean():.2f}")


def plot_eegclip_metrics(df, exp_name, output_dir):
    """Plot metrics for EEGClip training."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'{exp_name} - EEGClip Training Metrics', fontsize=16)
    
    ax.plot(df['epoch'], df['train_loss'], marker='o', markersize=3, label='Train Loss')
    
    # Mark epochs where checkpoints were saved
    checkpoints = df[df['checkpoint_saved'] == 'Yes']
    if not checkpoints.empty:
        ax.scatter(checkpoints['epoch'], checkpoints['train_loss'], 
                  color='red', s=100, marker='*', zorder=5, label='Checkpoint Saved')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_eegclip_metrics.png', dpi=150)
    print(f"  - Saved: {exp_name}_eegclip_metrics.png")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total epochs: {len(df)}")
    print(f"  Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Checkpoints saved: {len(checkpoints)}")


def compare_experiments(csv_paths, output_dir='analysis_plots'):
    """
    Compare multiple experiments by plotting their metrics together.
    
    Args:
        csv_paths: List of CSV file paths
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment Comparison', fontsize=16)
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        
        # Plot training loss
        axes[0].plot(df['epoch'], df['train_loss'], marker='o', markersize=2, label=exp_name)
        
        # Plot validation metrics if available
        if 'val_loss' in df.columns:
            val_data = df[df['val_loss'].notna()]
            if not val_data.empty:
                axes[1].plot(val_data['epoch'], val_data['val_loss'], marker='s', markersize=3, label=exp_name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/experiments_comparison.png', dpi=150)
    print(f"\nComparison plot saved: {output_dir}/experiments_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze CSV training logs from EEGStyleGAN-ADA project')
    parser.add_argument('csv_path', nargs='?', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments (provide multiple CSV paths)')
    parser.add_argument('--output-dir', default='analysis_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        compare_experiments(args.compare, args.output_dir)
    elif args.csv_path:
        # Analyze single experiment
        if os.path.isfile(args.csv_path):
            plot_training_metrics(args.csv_path, args.output_dir)
        elif os.path.isdir(args.csv_path):
            # Find all CSV files in directory
            csv_files = glob(os.path.join(args.csv_path, '**/logs/*.csv'), recursive=True)
            if not csv_files:
                print(f"No CSV files found in {args.csv_path}")
                return
            print(f"Found {len(csv_files)} CSV file(s)")
            for csv_file in csv_files:
                plot_training_metrics(csv_file, args.output_dir)
        else:
            print(f"Invalid path: {args.csv_path}")
    else:
        # Find all CSV files in current directory and subdirectories
        csv_files = glob('**/logs/*.csv', recursive=True)
        if not csv_files:
            print("No CSV log files found in current directory")
            print("\nUsage:")
            print("  python analyze_csv_logs.py <csv_file_or_directory>")
            print("  python analyze_csv_logs.py --compare exp1.csv exp2.csv exp3.csv")
            return
        
        print(f"Found {len(csv_files)} CSV file(s) in current directory")
        for csv_file in csv_files:
            plot_training_metrics(csv_file, args.output_dir)


if __name__ == '__main__':
    main()
