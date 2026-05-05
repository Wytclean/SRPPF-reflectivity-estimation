# SRPPF: Seismic Reflectivity Estimation by Pre-training and Physics-guided Fine-tuning

This repository contains the official PyTorch implementation of the paper:

> **Deep Learning-based Seismic Reflectivity Estimation by Pre-training on Labeled Synthetic Data and Physics-guided Fine-tuning in Field Data**
> Yuting Wang, Jintao Li, Xiaoming Sun, Xinming Wu
> *Computers & Geosciences*, [Year], DOI: [xxx]

---

## Overview

**SRPPF** is a two-stage deep learning framework for 3D post-stack seismic reflectivity estimation that requires no field-data labels.

**Stage 1 — Supervised Pre-training on Synthetic Data**
A 3D U-Net is trained on 400 pairs of synthetic seismic data and reflectivity labels. The synthetic data are generated with realistic fold/fault structures, Ricker wavelets (25–75 Hz), and field noise extracted from real surveys. This stage produces a stable, geologically reasonable initial reflectivity estimate.

**Stage 2 — Physics-guided Self-supervised Fine-tuning on Field Data**
Starting from the pre-trained model, the network is fine-tuned on the target field survey in a self-supervised manner. No reflectivity labels are needed. The fine-tuning is guided by three physics-based constraints:
- **Data reconstruction loss** — the predicted reflectivity convolved with the wavelet must reproduce the observed seismic data
- **Sparsity loss** — reflectivity is encouraged to be sparse (L1 penalty)
- **Structure-oriented smoothness loss** — gradients of the predicted reflectivity are aligned with the 3D local structural tensor (u/v/w eigenvectors), enforcing dip-consistent lateral continuity

An early-stopping criterion based on well-log cross-correlation is used to automatically select the best fine-tuning checkpoint.



---

## Repository Structure

```
SRPPF-reflectivity-estimation/
│
├── train.py                # Fine-tuning script (physics-guided self-supervised)
├── testset.py              # Inference / prediction script
│
├── unet.py                 # 3D U-Net + LNORMALLoss_3D + COSINELoss_3D
├── unets2wyt.py            # Attention U-Net variant (used in ablation study)
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

> **Note on pre-training code:** The supervised pre-training pipeline uses `newndatagenerator.py` for loading synthetic training pairs and the same U-Net architecture in `unet.py`. The fine-tuning pipeline uses `ndatagenerator.py` for single-volume field data loading. The `train.py` script implements the fine-tuning stage.

---

## Pre-trained Model

The pre-trained model checkpoint (trained on 400 synthetic 3D volumes of size 256×256×256) is available for download:

- **Download:** [Google Drive / Zenodo link — add your link here]
- **File:** `pretrained_unet.pt`
- Place the downloaded file in the `./model/` directory before running fine-tuning.

---

## Requirements

```
torch>=1.12.0
numpy>=1.21.0
matplotlib>=3.4.0
tensorboard>=2.9.0
```

```bash
pip install -r requirements.txt
```

Tested on: Ubuntu 20.04, Python 3.9, CUDA 11.3, NVIDIA A100 GPU.

---

## Input Data Format

All files are raw binary **float32** (`.dat`) arrays, read with `numpy.fromfile`.

### Fine-tuning inputs (field data)

| File | Shape | Description |
|------|-------|-------------|
| Seismic volume | `(n1, n2, n3)` | Observed field seismic data `d_obs` |
| Wavelet | `(nw,)` | Extracted seismic wavelet (e.g., from Hampson-Russell) |
| Mask | `(n1, n2, n3)` | Valid-trace mask for correlation evaluation |
| u1, u2, u3 | `(n1, n2, n3)` each | Dip-direction unit eigenvectors (from structure tensor) |
| v1, v2, v3 | `(n1, n2, n3)` each | Normal-plane eigenvectors |
| w1, w2, w3 | `(n1, n2, n3)` each | Strike-direction eigenvectors |

> The field datasets used in this paper have dimensions **128 × 128 × 512** (crossline × inline × time samples).

### Structure tensor computation
The u/v/w eigenvector volumes can be computed from the input seismic data using the structure-tensor method described in [Wu (2017), Geophysical Journal International]. We used the implementation available in the [Mines JTK](https://github.com/dhale/jtk) library.

### Pre-training inputs (synthetic data)
400 pairs of 3D synthetic seismic volumes and reflectivity labels at **256 × 256 × 256** resolution.
Synthetic data generation follows the procedure in [Wu et al. (2020), Geophysics]:
1. Generate heterogeneous flat-layer impedance models via stochastic simulation
2. Add fold structures by vertical shearing
3. Add fault structures using random fault parameters
4. Compute reflectivity from impedance
5. Convolve with Ricker wavelets (25–75 Hz, randomly selected)
6. Add field noise extracted from real surveys

---

## Usage

### Stage 1: Pre-training (supervised, on synthetic data)

Prepare 400 pairs of synthetic seismic and reflectivity volumes as `.dat` files, then run fine-tuning using the pre-training loss (MSE + MS-SSIM + Sparsity on predicted vs. label reflectivity directly). Adjust the loss inputs accordingly in `train.py`.

Pre-training hyperparameters (from the paper):

| Parameter | Value |
|-----------|-------|
| Training samples | 400 pairs (256³) |
| Train / Val / Test split | 80% / 10% / 10% |
| Optimizer | Adam, lr = 1×10⁻⁴ |
| LR schedule | ReduceLROnPlateau (factor=0.8, patience=5) |
| λ1 (MSE weight) | 1 |
| λ2 (MS-SSIM weight) | 5 |
| λ3 (Sparsity weight) | 1 |

### Stage 2: Physics-guided fine-tuning (self-supervised, on field data)

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
  --resume ./model/pretrained_unet.pt \
  --pred_dir ./predict_reflection \
  --recov_dir ./recon_seismic
```

Monitor with TensorBoard:
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
| α4 (Structure loss weight) | 1 |
| Early stopping | Well-log cross-correlation peak |

### Inference

```bash
python testset.py \
  --data_path  /path/to/seismic.dat \
  --model_path ./model/model_best.pt \
  --out_path   ./output/prediction.dat \
  --n1 128 --n2 128 --n3 512
```

---

## Loss Functions

### Pre-training loss
```
L_pretrain = λ1·MSE(r̂, r) + λ2·(1 − MS-SSIM(r̂, r)) + λ3·mean(|r̂|)
```
where `r̂` is the predicted reflectivity and `r` is the ground-truth reflectivity label.

### Fine-tuning loss
```
L_finetune = α1·MSE(d_recons, d_obs) + α2·(1 − MS-SSIM(d_recons, d_obs))
           + α3·mean(|r|) + α4·L_str
```
where `d_recons = w * r` is the reconstructed seismic (convolution of predicted reflectivity with wavelet), and:
```
L_str(u)   = cos(∇r, u)          # dip alignment loss
L_str(v,w) = mean(|∇r·v| + |∇r·w|)  # normal-plane smoothness loss
```

---

## Results

| Method | Correlation (clean) | MSE (clean) | Correlation (SNR=11dB) | Correlation (SNR=5dB) |
|--------|--------------------:|------------:|-----------------------:|----------------------:|
| Only pre-trained | 0.8453 | 0.0052 | 0.8303 | 0.7959 |
| Only self-supervised | 0.8012 | 0.0098 | 0.5292 | 0.4504 |
| **SRPPF (ours)** | **0.9832** | **0.0006** | **0.9302** | **0.9010** |

*Evaluated on 2D Marmousi synthetic data.*

---

## Citation

If you use this code in your research, please cite:

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

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions, please open a GitHub Issue or contact:
- Yuting Wang: [your email]
- Xinming Wu: xinmwu@ustc.edu.cn
