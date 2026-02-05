# EEGStyleGAN-ADA: CVPR40 Dataset Reproduction Guide

This comprehensive guide will help you reproduce the work presented in the paper "Learning Robust Deep Visual Representations from EEG Brain Recordings" (WACV 2024) using the CVPR40 dataset.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Stage 1: EEG Feature Extraction](#stage-1-eeg-feature-extraction)
6. [Stage 2: Image-to-EEG Mapping](#stage-2-image-to-eeg-mapping)
7. [Stage 3: StyleGAN-ADA Training](#stage-3-stylegan-ada-training)
8. [Stage 4: Image Generation](#stage-4-image-generation)
9. [Stage 5: Evaluation](#stage-5-evaluation)
10. [Pretrained Checkpoints](#pretrained-checkpoints)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Repository Does

This project implements a two-stage deep learning pipeline that:
1. **Extracts meaningful features from EEG brain signals** using LSTM/CNN networks with triplet loss
2. **Maps images to the learned EEG space** to enable reconstruction
3. **Generates images from EEG signals** using StyleGAN2-ADA conditioned on EEG features
4. **Reconstructs unseen images** by transforming them to EEG space and back to images

### Pipeline Architecture

```
Raw EEG Signals (128 channels × 440 timepoints)
    ↓
[Stage 1: Triplet LSTM Network]
    ↓
EEG Feature Space (128-dim)
    ↓
[Stage 2: Image→EEG Projection Network]
    ↓
Learned EEG Representations
    ↓
[Stage 3: StyleGAN2-ADA with EEG Conditioning]
    ↓
Generated/Reconstructed Images (128×128 RGB)
```

### CVPR40 Dataset

- **Source**: EEG signals recorded while subjects viewed ImageNet images
- **Classes**: 40 object categories
- **Subjects**: 6 participants
- **EEG Channels**: 128 electrodes
- **Temporal Resolution**: 440 timepoints per trial
- **Splits**: Train / Validation / Test

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **CUDA**: Version 11.0 or compatible
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for dataset + experiments

### Software Requirements
- **OS**: Linux (Ubuntu 18.04/20.04 recommended) or Windows 10/11
- **Anaconda/Miniconda**: Python 3.8
- **CUDA Toolkit**: 11.0

---

## Environment Setup

### Step 1: Create Conda Environment

You can either use the provided environment file or create manually:

#### Option A: Using Environment File
```bash
cd anaconda
conda env create -f to1.7.yml
conda activate to1.7
```

#### Option B: Manual Setup
```bash
# Create environment
conda create -n to1.7 anaconda python=3.8
conda activate to1.7

# Install PyTorch with CUDA 11.0
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install required packages
pip install opencv-python==4.5.4.58 opencv-contrib-python==4.5.4.58
pip install natsort
pip install pytorch-metric-learning
pip install lpips
pip install click
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install umap-learn
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 1.7.0+cu110
CUDA Available: True
```

---

## Dataset Preparation

### Dataset Structure

The CVPR40 dataset should be organized as follows:
```
dataset/
└── eeg_imagenet40_cvpr_2017_raw/
    ├── train/
    │   ├── class_01/
    │   │   ├── sample_001.npy
    │   │   ├── sample_002.npy
    │   │   └── ...
    │   ├── class_02/
    │   └── ...
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)
```

### Data Format

Each `.npy` file contains:
```python
# Structure: [image, eeg_data, label, class_name]
loaded_array = np.load('sample_001.npy', allow_pickle=True)
image = loaded_array[0]        # Shape: (H, W, 3) - RGB image
eeg_data = loaded_array[1]     # Shape: (128, 440) - EEG signals
label = loaded_array[2]        # Integer class label (0-39)
class_name = loaded_array[3]   # String class name
```

### Download Dataset

**Note**: You need to obtain the CVPR40 EEG-ImageNet dataset from the original authors. Contact the authors of the paper:
- Paper: "Learning to Generate Images from EEG Brain Recordings"
- Dataset request: Check the original paper for access instructions

### Data Preprocessing (if needed)

If you have raw EEG data, ensure:
1. EEG signals are normalized
2. Images are resized to 224×224 (will be resized to 128×128 during training)
3. Data is saved in the numpy format described above

---

## Stage 1: EEG Feature Extraction

### Objective
Train a Triplet LSTM network to learn a robust, class-discriminative EEG feature space.

### Navigate to Directory
```bash
cd EEG2Feat/Triplet_LSTM/CVPR40/
```

### Step 1: Configure Training

Edit `config.py`:
```python
base_path       = '/path/to/your/dataset/'  # Update this!
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'
test_path       = 'eeg_imagenet40_cvpr_2017_raw/test/'

# Network hyperparameters
feat_dim       = 128    # Feature dimension
projection_dim = 128    # Projection dimension
num_classes    = 40     # Number of classes
input_size     = 128    # EEG channels
timestep       = 440    # Temporal samples
num_layers     = 4      # LSTM layers
batch_size     = 256    # Batch size
epoch          = 8192   # Training epochs
lr             = 3e-4   # Learning rate
```

### Step 2: Train EEG Feature Extractor

```bash
# Train the Triplet LSTM model
python train.py
```

**Training Process**:
- Uses Triplet Semi-Hard Loss with hard negative mining
- Monitors K-means clustering accuracy on validation set
- Saves best checkpoint based on validation clustering accuracy
- Training takes ~12-24 hours on a single GPU

**Expected Output**:
```
Train:[1, 0.245]
[Epoch: 1, Train KMeans score Proj: 0.3245]
Val:[1, 0.198]
[Epoch: 1, Val KMeans score Proj: 0.3512]
...
```

### Step 3: Evaluate Feature Quality

```bash
# Evaluate on test set
python evaluate.py
```

This will generate:
- K-means clustering accuracy
- t-SNE visualizations
- Class separability metrics

**Target Metrics**:
- Validation K-means accuracy: >90%
- Clear class separation in t-SNE plots

### Step 4: Save EEG Embeddings

```bash
# Extract and save embeddings for all data
python linearprobing.py
```

This saves embeddings that will be used in later stages.

---

## Stage 2: Image-to-EEG Mapping

### Objective
Learn to project images into the learned EEG feature space for reconstruction purposes.

### Navigate to Directory
```bash
cd ../../../Image2EEG/
```

### Step 1: Configure Training

Edit `config.py`:
```python
base_path       = '/path/to/your/dataset/'  # Update this!
train_path      = 'eeg_imagenet40_cvpr_2017_raw/train/'
validation_path = 'eeg_imagenet40_cvpr_2017_raw/val/'

# Point to the trained EEG feature extractor
eeg_model_path  = '../EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_XX/bestckpt/eegfeat_lstm.pth'

# Network hyperparameters
feat_dim       = 128
projection_dim = 128
num_classes    = 40
batch_size     = 32
epoch          = 5000
lr             = 3e-4
```

### Step 2: Train Image-to-EEG Network

```bash
python train.py
```

**Training Process**:
- Loads frozen EEG feature extractor
- Trains GoogleNet-based image encoder to match EEG embeddings
- Uses MSE loss between image projections and EEG features
- Training takes ~8-12 hours

**Expected Output**:
```
Train:[1, 0.0234]
[Epoch: 1, Train KMeans score Proj: 0.5123]
Val:[1, 0.0189]
[Epoch: 1, Val KMeans score Proj: 0.5467]
```

### Step 3: Evaluate Mapping

```bash
python evaluate.py
```

---

## Stage 3: StyleGAN-ADA Training

### Objective
Train StyleGAN2-ADA conditioned on EEG features to generate images.

### Navigate to Directory
```bash
cd ../EEGStyleGAN-ADA_CVPR40/
```

### Step 1: Configure Training

Edit `config.py`:
```python
# EEG Data paths
train_data_path = '/path/to/dataset/eeg_imagenet40_cvpr_2017_raw/train/*'
val_data_path   = '/path/to/dataset/eeg_imagenet40_cvpr_2017_raw/val/*'
test_data_path  = '/path/to/dataset/eeg_imagenet40_cvpr_2017_raw/test/*'

# Model hyperparameters
image_height   = 128
image_width    = 128
batch_size     = 32
latent_dim     = 128
n_classes      = 40
n_subjects     = 6
c_dim          = 128  # Conditioning dimension
feat_dim       = 128
input_size     = 128  # EEG channels
input_shape    = (1, 440, 128)
EPOCH          = 5001
lr             = 3e-4
```

### Step 2: Prepare EEG Feature Network

Copy the trained EEG feature extractor checkpoint:
```bash
mkdir -p eegbestckpt/
cp ../EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_XX/bestckpt/eegfeat_lstm.pth \
   eegbestckpt/eegfeat_lstm_all_0.9665178571428571.pth
```

### Step 3: Train StyleGAN-ADA

```bash
# Basic training command
python train.py --outdir=./out/ \
                --data=../dataset/eeg_imagenet40_cvpr_2017_raw/train/* \
                --cond=1 \
                --gpus=1 \
                --cfg=cifar \
                --mirror=1 \
                --augpipe=bgcfnc

# For multi-GPU training (e.g., 4 GPUs)
python train.py --outdir=./out/ \
                --data=../dataset/eeg_imagenet40_cvpr_2017_raw/train/* \
                --cond=1 \
                --gpus=4 \
                --cfg=cifar \
                --mirror=1 \
                --augpipe=bgcfnc \
                --batch=64
```

**Training Parameters**:
- `--outdir`: Output directory for checkpoints and samples
- `--data`: Path to training data
- `--cond=1`: Enable conditional generation
- `--gpus`: Number of GPUs to use
- `--cfg=cifar`: Base configuration (suitable for 128×128 images)
- `--mirror=1`: Enable horizontal flipping augmentation
- `--augpipe=bgcfnc`: Augmentation pipeline (background, color, filter, noise, cutout)
- `--batch`: Total batch size across all GPUs

**Training Process**:
- Adaptive Discriminator Augmentation (ADA) automatically adjusts augmentation
- Checkpoints saved every 10 ticks (default)
- Sample images generated periodically
- Training takes ~3-7 days depending on GPU

**Expected Output**:
```
tick 1     kimg 32.0    time 2m 34s      min/tick 2m 34s      G_loss 0.8234  D_loss 0.6123  ...
tick 2     kimg 64.0    time 5m 08s      min/tick 2m 34s      G_loss 0.7821  D_loss 0.5934  ...
...
```

### Step 4: Monitor Training

Check generated samples:
```bash
# Sample images are saved in
out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/fakes*.png
```

Monitor training progress:
```bash
# Check logs
tail -f out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/log.txt
```

---

## Stage 4: Image Generation

### Generate Images from EEG Signals

```bash
# Navigate to StyleGAN directory
cd EEGStyleGAN-ADA_CVPR40/

# Generate images from test set EEG signals
python generate.py \
    --network=out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/network-snapshot-005443.pkl \
    --outdir=generated_images/ \
    --seeds=0-100 \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/*
```

**Parameters**:
- `--network`: Path to trained GAN checkpoint
- `--outdir`: Output directory for generated images
- `--seeds`: Range of random seeds / EEG samples to use
- `--data`: Path to test data

### Reconstruct Unseen Images

```bash
# Reconstruct images using Image→EEG→Image pipeline
python image2eeg2image.py \
    --network=out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/network-snapshot-005443.pkl \
    --outdir=reconstructed_images/ \
    --seeds=0-100 \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/*
```

This uses the Image2EEG network to project images to EEG space, then generates images from those projections.

---

## Stage 5: Evaluation

### Calculate Metrics

Navigate to StyleGAN directory:
```bash
cd EEGStyleGAN-ADA_CVPR40/
```

### Inception Score (IS)
```bash
python calc_metrics.py \
    --metrics=is50k \
    --network=out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/network-snapshot-005443.pkl \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/* \
    --mirror=1
```

### Fréchet Inception Distance (FID)
```bash
python calc_metrics.py \
    --metrics=fid50k_full \
    --network=out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/network-snapshot-005443.pkl \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/* \
    --mirror=1
```

### Kernel Inception Distance (KID)
```bash
python calc_metrics.py \
    --metrics=kid50k_full \
    --network=out/00000-EEGImageCVPR40-cond-mirror-cifar-bgcfnc/network-snapshot-005443.pkl \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/* \
    --mirror=1
```

### Batch Evaluation Script

```bash
# Use the provided script for all metrics
bash metriccompute.sh
```

### Expected Results

According to the paper, you should achieve approximately:

| Metric | Expected Value |
|--------|---------------|
| Inception Score | 62.9% improvement over baseline |
| FID | Competitive with state-of-the-art |
| KID | Competitive with state-of-the-art |
| K-means Accuracy (EEG Features) | >90% |

---

## Pretrained Checkpoints

If you want to skip training and use pretrained models:

### Download Checkpoints

1. **EEG Feature Extractor (CVPR40)**:
   - [Download Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EXn-8R80rxtHjlMCzPfhL9UBj80opHXyq3MnBBXXE6IsQw?e=Xbt2zO)
   - Extract to: `EEGStyleGAN-ADA_CVPR40/eegbestckpt/`

2. **Image2EEG Network**:
   - [Download Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EQuJKUdXz8lGn04O2KSwsAUBUdsL-rjj0FdDwNH1Z8V9jw?e=716Xuq)
   - Extract to: `EEGStyleGAN-ADA_CVPR40/imageckpt/`

3. **StyleGAN-ADA Generator**:
   - [Download Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EXn-8R80rxtHjlMCzPfhL9UBj80opHXyq3MnBBXXE6IsQw?e=Xbt2zO)

### Using Pretrained Models

```bash
# Generate images with pretrained model
python generate.py \
    --network=path/to/downloaded/network-snapshot.pkl \
    --outdir=generated_images/ \
    --seeds=0-100 \
    --data=../dataset/eeg_imagenet40_cvpr_2017_raw/test/*
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in config files
```python
batch_size = 16  # or 8
```

#### 2. Missing Dependencies
```
ModuleNotFoundError: No module named 'pytorch_metric_learning'
```
**Solution**:
```bash
pip install pytorch-metric-learning
```

#### 3. Data Loading Errors
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: Update paths in config files to absolute paths
```python
base_path = '/absolute/path/to/dataset/'
```

#### 4. StyleGAN Training Crashes
```
RuntimeError: NVRTC error
```
**Solution**: Update CUDA and PyTorch, or reduce resolution:
```bash
--cfg=cifar  # Uses 128x128 instead of 256x256
```

#### 5. Slow Training
**Solution**: Enable cuDNN benchmarking and mixed precision:
- Edit train.py to set `torch.backends.cudnn.benchmark = True`
- Use multiple GPUs with `--gpus=4`

#### 6. Poor Generation Quality
**Causes**:
- Insufficient training epochs
- EEG feature extractor not well trained
- Augmentation too aggressive

**Solutions**:
- Train longer (check sample quality over time)
- Retrain EEG feature extractor with higher K-means accuracy
- Adjust `--augpipe` parameter

### Verification Steps

After each stage, verify:

**Stage 1 (EEG Features)**:
```bash
# Should see clear class clusters
ls EXPERIMENT_XX/figures/*tsne*.png
```

**Stage 2 (Image2EEG)**:
```bash
# Check if checkpoints exist
ls eegckpt/*.pth
```

**Stage 3 (StyleGAN)**:
```bash
# Check generated samples improve over time
ls out/*/fakes*.png
```

### Getting Help

1. Check [GitHub Issues](https://github.com/prajwalsingh/EEGStyleGAN-ADA/issues)
2. Refer to original paper: [arXiv:2310.16532](https://arxiv.org/abs/2310.16532)
3. Contact authors (see paper for details)

---

## Additional Notes

### Dataset-Specific Considerations

- **Subject Variability**: EEG signals vary significantly between subjects. The model learns subject-invariant representations.
- **Trial Averaging**: Each class has multiple trials. Better results with more trials.
- **Preprocessing**: EEG data should already be preprocessed (filtered, epoched).

### Hyperparameter Tuning

Key hyperparameters to tune:
1. **Learning rate**: 3e-4 works well, but try 1e-4 to 1e-3
2. **Batch size**: Larger is better for GAN stability (32-128)
3. **Feature dimension**: 128 is standard, 256 may improve quality
4. **LSTM layers**: 4 layers works well, 2-6 range
5. **Augmentation**: Adjust based on dataset size

### Performance Optimization

1. **Use mixed precision training** (requires PyTorch >= 1.6):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

2. **Enable persistent workers**:
   ```python
   DataLoader(..., persistent_workers=True)
   ```

3. **Pin memory**:
   ```python
   DataLoader(..., pin_memory=True)
   ```

### Customization for Your Data

If using different EEG data:
1. Update `input_size` for different channel counts
2. Update `timestep` for different temporal resolution
3. Update `n_classes` for different number of categories
4. Adjust network architecture in `network.py`

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{singh2024learning,
  title={Learning Robust Deep Visual Representations from EEG Brain Recordings},
  author={Singh, Prajwal and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

---

## License

This project uses code from NVIDIA's StyleGAN2-ADA-PyTorch, which is licensed under the NVIDIA Source Code License. See `LICENSE.txt` for details.

---

## Acknowledgments

- NVIDIA for StyleGAN2-ADA implementation
- Original CVPR40 dataset creators
- PyTorch team for the framework

---

**Last Updated**: January 2026
**Repository**: https://github.com/prajwalsingh/EEGStyleGAN-ADA
**Paper**: https://arxiv.org/abs/2310.16532
