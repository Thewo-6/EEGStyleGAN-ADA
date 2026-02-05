# CSV Logging for Training Scripts

This document describes the CSV logging functionality that has been added to all training scripts in the EEGStyleGAN-ADA project.

## Overview

All training and finetuning scripts now automatically save training metrics to CSV files for easy analysis and visualization. The CSV files are saved in the `logs` subdirectory of each experiment folder.

## CSV Logger Utility

### Location
`csv_logger.py` - Root directory of the project

### Usage
```python
from csv_logger import CSVLogger

# Initialize logger
logger = CSVLogger(
    log_dir='path/to/logs',
    filename='training_log.csv',
    fieldnames=['epoch', 'loss', 'accuracy']
)

# Log metrics
logger.log({'epoch': 1, 'loss': 0.5, 'accuracy': 0.85})

# Close logger when done
logger.close()
```

### Features
- **Automatic file creation**: Creates log directory and CSV file automatically
- **Header management**: Writes CSV header automatically on first log
- **Immediate flush**: Data is written immediately to disk (no buffering)
- **Context manager support**: Can be used with `with` statement
- **Safe cleanup**: Automatically closes file on deletion

## Training Scripts

### EEG2Feat Scripts
**Location**: `EEG2Feat/Triplet_LSTM/` and `EEG2Feat/Triplet_CNN/` subdirectories

**CSV Location**: `EXPERIMENT_{num}/logs/training_log.csv`

**Logged Metrics**:
- `epoch`: Current epoch number
- `train_loss`: Average training loss for the epoch
- `train_kmeans_acc`: K-means clustering accuracy on training set (logged at vis_freq intervals)
- `val_loss`: Average validation loss
- `val_kmeans_acc`: K-means clustering accuracy on validation set
- `best_val_acc`: Best validation accuracy achieved so far
- `best_val_epoch`: Epoch at which best validation accuracy was achieved

### EEG2Feat_Unseen Scripts
**Location**: `EEG2Feat_Unseen/train.py`

**CSV Location**: `EXPERIMENT_{num}/logs/training_log.csv`

**Logged Metrics**: Same as EEG2Feat scripts

### EEGClip Scripts
**Location**: `EEGClip/main.py`

**CSV Location**: `EEGClip_ckpt/EXPERIMENT_{num}/logs/training_log.csv`

**Logged Metrics**:
- `epoch`: Current epoch number
- `train_loss`: Average training loss for the epoch
- `checkpoint_saved`: Indicates if a checkpoint was saved ('Yes' or empty)

### Image2EEG Scripts
**Location**: `Image2EEG/train.py`

**CSV Location**: `EXPERIMENT_{num}/logs/training_log.csv`

**Logged Metrics**:
- `epoch`: Current epoch number
- `train_loss`: Average training loss for the epoch
- `train_kmeans_acc`: K-means clustering accuracy on training set
- `val_loss`: Average validation loss
- `val_kmeans_acc`: K-means clustering accuracy on validation set
- `best_val_acc`: Best validation accuracy achieved so far
- `best_val_epoch`: Epoch at which best validation accuracy was achieved

### EEGStyleGAN-ADA Scripts
**Location**: `EEGStyleGAN-ADA_CVPR40/training/training_loop.py` and `EEGStyleGAN-ADA_ThoughtViz/training/training_loop.py`

**CSV Location**: `{run_dir}/logs/training_log.csv` (within the output directory specified when launching training)

**Logged Metrics**:
- `tick`: Progress tick number
- `kimg`: Thousands of images processed
- `total_sec`: Total seconds elapsed since training start
- `sec_per_tick`: Seconds per tick
- `sec_per_kimg`: Seconds per thousand images
- `maintenance_sec`: Maintenance time in seconds
- `cpu_mem_gb`: CPU memory usage in GB
- `peak_gpu_mem_gb`: Peak GPU memory usage in GB
- `augment_p`: Augmentation probability (ADA parameter)
- `Loss_G`: Generator loss
- `Loss_D`: Discriminator loss
- `Loss_signs_real`: Loss signs for real images

## Finetuning Scripts

### All Finetuning Scripts
**Locations**: Multiple subdirectories in `EEG2Feat/`, `EEG2Feat_Unseen/`, and `Image2EEG/`

**CSV Location**: `EXPERIMENT_{num}/logs/finetuning_log.csv`

**Logged Metrics**:
- `epoch`: Current epoch number
- `train_loss`: Average training loss for the epoch
- `train_acc`: Training accuracy
- `val_loss`: Average validation loss
- `val_acc`: Validation accuracy
- `best_val_acc`: Best validation accuracy achieved so far
- `best_val_epoch`: Epoch at which best validation accuracy was achieved

## Analyzing CSV Logs

### Using Pandas
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('EXPERIMENT_1/logs/training_log.csv')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.savefig('training_progress.png')
plt.show()

# Print best validation accuracy
best_idx = df['val_kmeans_acc'].idxmax()
print(f"Best validation accuracy: {df.loc[best_idx, 'val_kmeans_acc']:.4f} at epoch {df.loc[best_idx, 'epoch']}")
```

### Using Excel/Google Sheets
CSV files can be directly opened in Excel or Google Sheets for analysis and visualization.

### Command Line
```bash
# View first few lines
head EXPERIMENT_1/logs/training_log.csv

# View last few lines
tail EXPERIMENT_1/logs/training_log.csv

# Search for specific patterns
grep "epoch,100" EXPERIMENT_1/logs/training_log.csv
```

## Benefits

1. **Easy Analysis**: CSV format is universally supported and easy to analyze
2. **No Data Loss**: Immediate flushing ensures data is saved even if training is interrupted
3. **Visualization**: Can be easily imported into plotting libraries or spreadsheet software
4. **Comparison**: Multiple experiment logs can be loaded and compared side-by-side
5. **Reproducibility**: Complete training history is preserved for each experiment

## Notes

- CSV files are created in the same directory structure as existing experiment logs
- The logger is automatically closed when training completes or if the script is interrupted
- Empty values in CSV indicate metrics that weren't logged at that particular epoch (e.g., validation metrics that are only computed at certain intervals)
- All CSV files use UTF-8 encoding with proper newline handling for cross-platform compatibility

## Example CSV Output

### Training Script CSV
```csv
epoch,train_loss,train_kmeans_acc,val_loss,val_kmeans_acc,best_val_acc,best_val_epoch
0,1.2345,,0.9876,0.7500,0.7500,0
1,1.1234,,,,,0.7500,0
2,1.0123,,,,,0.7500,0
3,0.9876,,,,,0.7500,0
4,0.9234,,,,,0.7500,0
5,0.8765,0.8200,0.8234,0.7850,0.7850,5
```

### Finetuning Script CSV
```csv
epoch,train_loss,train_acc,val_loss,val_acc,best_val_acc,best_val_epoch
0,2.3456,0.3500,2.1234,0.4200,0.4200,0
1,1.8765,0.5200,1.7654,0.5800,0.5800,1
2,1.4567,0.6500,1.5432,0.6400,0.6400,2
3,1.2345,0.7200,1.3456,0.7000,0.7000,3
```

### StyleGAN Training CSV
```csv
tick,kimg,total_sec,sec_per_tick,sec_per_kimg,maintenance_sec,cpu_mem_gb,peak_gpu_mem_gb,augment_p,Loss_G,Loss_D,Loss_signs_real
0,4.0,45.23,45.23,11.31,0.12,8.45,10.23,0.000,1.234,0.876,0.512
1,8.0,89.67,44.44,11.11,0.15,8.47,10.25,0.050,1.123,0.845,0.523
2,12.0,133.45,43.78,10.95,0.13,8.49,10.28,0.100,1.034,0.812,0.534
```
