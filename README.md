# SRPPF: Seismic Reflectivity Estimation by Pre-training and Physics-guided Fine-tuning

This repository contains the official PyTorch implementation of the paper:

> **Deep Learning-based Seismic Reflectivity Estimation by Pre-training on Labeled Synthetic Data and Physics-guided Fine-tuning in Field Data**
> Yuting Wang, Jintao Li, Xiaoming Sun, Xinming Wu
> *Computers & Geosciences*

---

## Overview

**SRPPF** is a two-stage deep learning framework for 3D post-stack seismic reflectivity estimation that requires no field-data labels.

**Stage 1 — Supervised Pre-training on Synthetic Data**
A 3D U-Net is trained on 400 pairs of synthetic seismic data and reflectivity labels. Synthetic data are generated with realistic fold/fault structures, Ricker wavelets (25–75 Hz), and field noise. This stage produces a stable, geologically reasonable initial reflectivity estimate.

**Stage 2 — Physics-guided Self-supervised Fine-tuning on Field Data**
Starting from the pre-trained model, the network is fine-tuned on the target field survey without any reflectivity labels. Three physics-based constraints guide the process:
- **Data reconstruction loss** — predicted reflectivity convolved with the wavelet must reproduce the observed seismic
- **Sparsity loss** — L1 penalty encourages sparse reflectivity
- **Structure-oriented smoothness loss** — reflectivity gradients are aligned with local structural tensors (u/v/w eigenvectors)

An early-stopping criterion based on well-log cross-correlation automatically selects the best checkpoint.

---

## Repository Structure

```
SRPPF-reflectivity-estimation/
│
├── quick_test.py           # Quick installation test (no data needed)
│
├── train.py                # Fine-tuning script (physics-guided self-supervised)
├── testset.py              # Inference / prediction script
│
├── unet.py                 # 3D U-Net + LNORMALLoss_3D + COSINELoss_3D
├── unets2wyt.py            # Attention U-Net variant (ablation study)
├── common.py               # ResBlock, BasicBlock shared modules
│
├── ssim.py                 # 2D/3D SSIM and MS-SSIM loss implementations
├── utils.py                # GeoConv (physics convolution), normalization, I/O
│
├── ndatagenerator.py       # DataLoader for single-volume field data (fine-tuning)
├── newndatagenerator.py    # DataLoader with patch cropping + augmentation (pre-training)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

```
torch>=1.12.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
tensorboard>=2.9.0
```

```bash
pip install -r requirements.txt
```

Tested on: Ubuntu 20.04, Python 3.9, CUDA 11.3, NVIDIA V100 GPU.

---

## Quick Test (No Data Required)

To verify that the installation is correct and the model runs as expected:

```bash
python quick_test.py
```

This script automatically generates a small synthetic seismic volume (64×32×32),
passes it through the U-Net with random weights, and checks output shape correctness.

**Expected output:**
```
==================================================
SRPPF Quick Test
==================================================

Generating synthetic seismic volume (64×32×32)...
Input shape : torch.Size([1, 1, 64, 32, 32])
Output shape: torch.Size([1, 1, 64, 32, 32])

Quick test PASSED.
==================================================
```

> This test uses random network weights. To use the pre-trained checkpoint,
> download it as described below and pass it via `--resume`.

---

## Pre-trained Model

The pre-trained checkpoint (trained on 400 synthetic 3D volumes of 256×256×256)
is available on the **GitHub Releases** page:

**Download page:** https://github.com/Wytclean/SRPPF-reflectivity-estimation/releases/tag/v1.0

**Direct download:**
```bash
wget https://github.com/Wytclean/SRPPF-reflectivity-estimation/releases/download/v1.0/pretrain_model.pt
```

**SHA-256 checksum** (to verify file integrity after download):
```
43d1c83a3a40f189f38cf033842027ad0e624abe2a65301bc0def4e72639fcba
```

Verify with:
```bash
# Linux / macOS
sha256sum pretrain_model.pt

# Windows (PowerShell)
Get-FileHash pretrain_model.pt -Algorithm SHA256
```

Place the downloaded file in the `./model/` directory:
```bash
mkdir -p model
mv pretrain_model.pt ./model/
```

---

## Input Data Format

All files are raw binary **float32** (`.dat`) arrays, read with `numpy.fromfile`.

### Fine-tuning inputs (field data)

| File | Shape | Description |
|------|-------|-------------|
| Seismic volume | `(n1, n2, n3)` | Observed field seismic data `d_obs` |
| Wavelet | `(nw,)` | Extracted seismic wavelet |
| Mask | `(n1, n2, n3)` | Valid-trace mask |
| u1, u2, u3 | `(n1, n2, n3)` each | Dip-direction unit eigenvectors |
| v1, v2, v3 | `(n1, n2, n3)` each | Normal-plane eigenvectors |
| w1, w2, w3 | `(n1, n2, n3)` each | Strike-direction eigenvectors |

> Field datasets in this paper: **128 × 128 × 512** (crossline × inline × time).

The u/v/w eigenvector volumes are computed from the seismic data using the
structure-tensor method in [Wu (2017), GJI], available in the
[Mines JTK](https://github.com/dhale/jtk) library.

---

## Usage

### Stage 2: Physics-guided fine-tuning on field data

```bash
python train.py \
  --droot  /path/to/seismic.dat \
  --wroot  /path/to/wavelet.dat \
  --rroot  /path/to/mask.dat \
  --u1root /path/to/u1.dat \
  --u2root /path/to/u2.dat \
  --u3root /path/to/u3.dat \
  --v1root /path/to/v1.dat \
  --v2root /path/to/v2.dat \
  --v3root /path/to/v3.dat \
  --w1root /path/to/w1.dat \
  --w2root /path/to/w2.dat \
  --w3root /path/to/w3.dat \
  --n1 128 --n2 128 --n3 512 \
  --epochs 1000 --lr 1e-4 \
  --resume ./model/pretrain_model.pt \
  --pred_dir ./predict_reflection \
  --recov_dir ./recon_seismic
```

Monitor training:
```bash
tensorboard --logdir=.
```

Fine-tuning hyperparameters (from the paper):

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr = 1×10⁻⁴ |
| α1 (MSE weight) | 1 |
| α2 (MS-SSIM weight) | 5 |
| α3 (Sparsity weight) | 0.005 |
| α4 (Structure weight) | 1 |
| Early stopping | Well-log cross-correlation peak |

### Inference

```bash
python testset.py \
  --data_path  /path/to/seismic.dat \
  --model_path ./model/pretrain_model.pt \
  --out_path   ./output/prediction.dat \
  --n1 128 --n2 128 --n3 512
```

---

## Loss Functions

### Pre-training loss
```
L_pretrain = λ1·MSE(r̂, r) + λ2·(1 − MS-SSIM(r̂, r)) + λ3·mean(|r̂|)
```
λ1=1, λ2=5, λ3=1. Inputs are predicted vs. ground-truth reflectivity labels.

### Fine-tuning loss
```
L_finetune = α1·MSE(d_recons, d_obs) + α2·(1 − MS-SSIM(d_recons, d_obs))
           + α3·mean(|r|) + α4·L_str

L_str(u)   = cos(∇r, u)
L_str(v,w) = mean(|∇r·v| + |∇r·w|)
```
where `d_recons = w * r` (convolution of predicted reflectivity with wavelet).

---

## Results

| Method | Corr. (clean) | MSE (clean) | Corr. (11 dB) | Corr. (5 dB) |
|--------|:---:|:---:|:---:|:---:|
| Only pre-trained | 0.8453 | 0.0052 | 0.8303 | 0.7959 |
| Only self-supervised | 0.8012 | 0.0098 | 0.5292 | 0.4504 |
| **SRPPF (ours)** | **0.9832** | **0.0006** | **0.9302** | **0.9010** |

*Evaluated on 2D Marmousi synthetic data.*

---

## Citation

```bibtex
@article{wang2025srppf,
  title   = {Deep Learning-based Seismic Reflectivity Estimation by Pre-training on
             Labeled Synthetic Data and Physics-guided Fine-tuning in Field Data},
  author  = {Wang, Yuting and Li, Jintao and Sun, Xiaoming and Wu, Xinming},
  journal = {Computers \& Geosciences},
  year    = {2025},
  doi     = {xxx}
}
```

---

## License

Released under the MIT License. See `LICENSE` for details.

---

## Contact

- Yuting Wang: wangyuting051@mail.ustc.edu.cn
- Xinming Wu: xinmwu@ustc.edu.cn
