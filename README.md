# SRPPF — Seismic Reflectivity Estimation via Pre-training & Physics-guided Fine-tuning

> **Deep Learning-based Seismic Reflectivity Estimation by Pre-training on Labeled Synthetic Data and Physics-guided Fine-tuning in Field Data**
>
> *Yuting Wang, Jintao Li, Xiaoming Sun, Xinming Wu*
> *University of Science and Technology of China (USTC)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C.svg)](https://pytorch.org/)

---

## Overview

**SRPPF** is a two-stage deep learning framework for post-stack seismic reflectivity estimation.

| Stage | Strategy | Data | Supervision |
|---|---|---|---|
| **Pre-training** | Supervised learning | Labeled 3-D synthetic seismic volumes | Paired (seismic, reflectivity) |
| **Fine-tuning** | Physics-guided self-supervised | Target field seismic data | Physics constraints only |

### Key ideas

- **Pre-training on synthetic data** – A U-Net with residual blocks is trained on 400 synthetic 3-D volumes (256 × 256 × 256), providing a geologically consistent initial reflectivity estimate that avoids the instability of random initialisation in self-supervised inversion.
- **Physics-guided fine-tuning** – The pre-trained model is adapted to the specific field survey using three geophysical constraints:
  - *Sparsity* – promotes sharp, sparse reflection coefficients.
  - *Structure-oriented smoothness* – enforces lateral continuity along seismic dip directions derived from the 3-D structure tensor.
  - *Data reconstruction* – minimises the residual between the convolved reflectivity and the observed seismic data.
- **Well-log early stopping** – Cross-correlation between predicted reflectivity and well-log impedance is monitored epoch-by-epoch; training stops when the metric peaks, preventing noise overfitting.

### Combined loss functions

**Pre-training:**
$$L_{\text{pre-train}} = \lambda_1 L_{\text{MSE}} + \lambda_2 L_{\text{MS-SSIM}} + \lambda_3 L_{\text{sparsity}}$$

**Fine-tuning:**
$$L_{\text{fine-tune}} = \alpha_1 L_{\text{MSE}} + \alpha_2 L_{\text{MS-SSIM}} + \alpha_3 L_{\text{sparsity}} + \alpha_4 L_{\text{str}}$$

---

## Repository structure

```
SRPPF/
├── README.md
├── environment.yml          # Conda environment specification
├── requirements.txt         # pip-only alternative
│
├── data/
│   └── demo/
│       ├── demo_seismic.npy        # Small demo seismic cube (64×64×256)
│       ├── demo_reflectivity.npy   # Corresponding reflectivity labels
│       ├── demo_wavelet.npy        # 25 Hz Ricker wavelet
│       └── demo_welllog.npy        # Synthetic well-log impedance (optional)
│
├── configs/
│   ├── pretrain.yaml        # Pre-training hyperparameters
│   └── finetune.yaml        # Fine-tuning hyperparameters
│
├── srppf/
│   ├── __init__.py
│   ├── network.py           # U-Net + residual block architecture
│   ├── losses.py            # MSE, MS-SSIM, sparsity, structure-oriented smoothness
│   ├── structure_tensor.py  # 3-D structure tensor & eigenvector decomposition
│   ├── data_utils.py        # Synthetic data generation & DataLoader helpers
│   └── metrics.py           # Cross-correlation early-stopping metric
│
├── pretrain.py              # Pre-training entry point
├── finetune.py              # Fine-tuning entry point
└── predict.py               # Inference on a new seismic volume
```

---

## Environment & dependencies

The code was developed and tested on **2 × NVIDIA Tesla V100 (32 GB)** with CUDA 11.3.

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate srppf
```

<details>
<summary>environment.yml</summary>

```yaml
name: srppf
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - cudatoolkit=11.3
  - pytorch=1.13.1
  - torchvision=0.14.1
  - numpy=1.24
  - scipy=1.10
  - matplotlib=3.7
  - scikit-image=0.20
  - tqdm
  - pyyaml
  - h5py
  - pip:
      - pytorch-msssim==0.2.1
      - segyio==1.9.12
```

</details>

### Option B — pip

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
torch>=1.13.1
torchvision>=0.14.1
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
scikit-image>=0.20
tqdm
pyyaml
h5py
pytorch-msssim==0.2.1
segyio>=1.9.12
```

</details>

> **CUDA note:** The code uses `torch.nn.DataParallel` to distribute training across the two V100s automatically; no additional configuration is needed as long as both GPUs are visible (`nvidia-smi`).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SRPPF.git
cd SRPPF

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate srppf

# 3. Verify GPU availability
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s) available')"
# Expected output: 2 GPU(s) available
```

---

## Quick start

### 1. Run the demo (CPU or single GPU, ~2 min)

This uses the small demo data in `data/demo/` to verify your installation end-to-end:

```bash
python predict.py \
    --seismic   data/demo/demo_seismic.npy \
    --wavelet   data/demo/demo_wavelet.npy \
    --checkpoint pretrained_weights/srppf_pretrained.pth \
    --output    results/demo_reflectivity.npy
```

### 2. Pre-training on synthetic data (2 × V100)

```bash
python pretrain.py --config configs/pretrain.yaml
```

Key settings in `configs/pretrain.yaml`:

```yaml
# configs/pretrain.yaml
data:
  train_dir: /path/to/synthetic/train      # directory of .npy pairs
  val_split: 0.1
  test_split: 0.1
  patch_size: [256, 256, 256]

model:
  in_channels: 1
  base_channels: 64
  depth: 4
  num_res_blocks: 3

training:
  epochs: 100
  batch_size: 2                            # 1 per GPU × 2 GPUs
  lr: 1.0e-4
  lr_patience: 5
  lr_factor: 0.8
  loss_weights:                            # λ1, λ2, λ3
    mse: 1.0
    ms_ssim: 5.0
    sparsity: 1.0

hardware:
  gpus: [0, 1]                             # 2 × V100
  num_workers: 8
```

### 3. Physics-guided fine-tuning on field data

```bash
python finetune.py --config configs/finetune.yaml
```

Key settings in `configs/finetune.yaml`:

```yaml
# configs/finetune.yaml
data:
  field_seismic: /path/to/field_seismic.npy   # 3-D array [nz, nx, ny]
  wavelet: /path/to/wavelet.npy
  welllog: /path/to/welllog.npy               # optional; used for early stopping
  welllog_location: [64, 64]                  # [inline, crossline] index

model:
  checkpoint: pretrained_weights/srppf_pretrained.pth

training:
  max_epochs: 200
  lr: 5.0e-5
  loss_weights:                               # α1, α2, α3, α4
    mse: 1.0
    ms_ssim: 5.0
    sparsity: 1.0
    structure: 2.0
  early_stopping:
    enabled: true
    metric: welllog_correlation               # stops when corr peaks
    patience: 20

hardware:
  gpus: [0, 1]
  num_workers: 4
```

### 4. Inference only

```bash
python predict.py \
    --seismic     /path/to/field_seismic.npy \
    --wavelet     /path/to/wavelet.npy \
    --checkpoint  pretrained_weights/srppf_finetuned.pth \
    --output      results/reflectivity.npy
```

---

## Demo data

The `data/demo/` directory contains a small synthetic example generated from a 2-D Marmousi-derived impedance profile (cropped to 64 × 64 × 256 for fast testing):

| File | Shape | Description |
|---|---|---|
| `demo_seismic.npy` | `(64, 64, 256)` | Synthetic seismic with added field noise, SNR ≈ 11 dB |
| `demo_reflectivity.npy` | `(64, 64, 256)` | Ground-truth reflectivity labels |
| `demo_wavelet.npy` | `(64,)` | 25 Hz Ricker wavelet, dt = 1 ms |
| `demo_welllog.npy` | `(256,)` | Synthetic impedance log at inline 32, crossline 32 |

> **Field data:** The proprietary 3-D field datasets (C and D) used in the paper are provided under confidentiality agreement and cannot be redistributed. Contact the authors if you have a relevant dataset and wish to test the method.

---

## Reproducing paper results

| Experiment | Script | Config |
|---|---|---|
| Table 1 — clean synthetic | `finetune.py` | `configs/marmousi_clean.yaml` |
| Table 2 — noisy synthetic (11/5 dB) | `finetune.py` | `configs/marmousi_noisy.yaml` |
| Field data C & D | `finetune.py` | `configs/field_C.yaml` / `configs/field_D.yaml` |
| Ablation — loss functions | `finetune.py` | `configs/ablation_*.yaml` |

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{wang2025srppf,
  title   = {Deep Learning-based Seismic Reflectivity Estimation by Pre-training
             on Labeled Synthetic Data and Physics-guided Fine-tuning in Field Data},
  author  = {Wang, Yuting and Li, Jintao and Sun, Xiaoming and Wu, Xinming},
  journal = {Computers \& Geosciences},
  year    = {2025},
  doi     = {}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

This research is supported by the University of Science and Technology of China (USTC).
The Marmousi model is courtesy of the Institut Français du Pétrole (IFP).
