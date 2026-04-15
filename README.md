# Brain Tumor Segmentation with U-Net

A PyTorch implementation of the U-Net architecture for pixel-wise binary segmentation of brain tumors from MRI scans, trained on the [BraTS 2020 dataset].

Given a 2D axial slice of a FLAIR MRI brain scan, the model predicts a pixel-wise binary mask highlighting the tumor region.

## Overview

This project implements U-Net from scratch in PyTorch and applies it to brain tumor segmentation. U-Net's symmetric encoder–decoder design with skip connections makes it well-suited for biomedical image segmentation, where spatial precision matters and labeled data is scarce.

---

## Architecture

The network consists of four encoder blocks, a bottleneck, four symmetric decoder blocks, and a final 1×1 convolution.

| Stage | Channels | Spatial Size |
|---|---|---|
| Input | 1 | 256 × 256 |
| Encoder 1 | 64 | 256 → 128 |
| Encoder 2 | 128 | 128 → 64 |
| Encoder 3 | 256 | 64 → 32 |
| Encoder 4 | 512 | 32 → 16 |
| Bottleneck | 1024 | 16 × 16 |
| Decoder 4 | 512 | 16 → 32 |
| Decoder 3 | 256 | 32 → 64 |
| Decoder 2 | 128 | 64 → 128 |
| Decoder 1 | 64 | 128 → 256 |
| Output | 1 | 256 × 256 |

Key design choices:
- **Double convolution blocks** (`Conv2d → BatchNorm → ReLU`) at every stage
- **Skip connections** route fine-grained spatial detail from encoder to decoder
- **Bottleneck dropout** (`p=0.3`) regularises the deepest representation
- **Transposed convolutions** for learned upsampling in the decoder

---

## Dataset

**BraTS 2020** — 369 patients, each with a 3D FLAIR MRI volume (155 axial slices) and a corresponding expert-annotated segmentation mask.

### Slice Selection

Rather than using all 155 slices per patient (most of which are tumor-free), only the **top-3 most tumor-rich slices** per patient are kept. This yields 1,107 slice pairs and avoids severe class imbalance.


## Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Initial learning rate | 1 × 10⁻⁴ |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Max epochs | 50 |
| Batch size | 4 |
| Early stopping patience | 10 epochs |
| Loss function | Dice loss |

Training stopped at epoch 34 (early stopping). The best validation loss of **0.1201** was achieved at epoch 24.

### Why Dice Loss?

Cross-entropy is a poor choice when tumor pixels can be as few as 1–2% of all pixels. Dice loss directly optimises the overlap between prediction and ground truth, making it robust to severe class imbalance.

---

## Results

| Metric | Value |
|---|---|
| Test Dice Score | **0.9025** |
| Best Val Dice Loss | 0.1201 (epoch 24) |
| Training stopped | Epoch 34 / 50 |

The qualitative output shows three panels per test sample: the raw FLAIR scan, the ground-truth expert mask, and the U-Net prediction. A Dice score of ~0.90 confirms strong generalisation despite training on a small slice subset.

