import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import random


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def save(pred, recov, fname, pred_path='./predict_reflection', recov_path='./recon_seismic'):
    """Save predicted reflectivity and reconstructed seismic to binary float32 files.

    Parameters
    ----------
    pred       : torch.Tensor  Predicted reflectivity volume
    recov      : torch.Tensor  Reconstructed seismic volume
    fname      : int or str    Output filename (without extension)
    pred_path  : str           Directory for predicted reflectivity outputs
    recov_path : str           Directory for reconstructed seismic outputs
    """
    os.makedirs(pred_path,  exist_ok=True)
    os.makedirs(recov_path, exist_ok=True)
    with torch.no_grad():
        pred  = pred.cpu().numpy().transpose().astype(np.float32)
        recov = recov.cpu().numpy().transpose().astype(np.float32)
        pred.tofile(os.path.join(pred_path,  f'{fname}.dat'))
        recov.tofile(os.path.join(recov_path, f'{fname}.dat'))


def save_model(path, model, optimizer, current_epoch):
    """Save model checkpoint.

    Parameters
    ----------
    path          : str              Directory to save the checkpoint
    model         : torch.nn.Module  Model to save
    optimizer     : torch.optim      Optimizer state
    current_epoch : int              Current epoch number
    """
    os.makedirs(path, exist_ok=True)
    out   = os.path.join(path, f'model_{current_epoch}.pt')
    state = {
        'net':       model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     current_epoch
    }
    torch.save(state, out)


def load_model(path, model, device):
    """Load model checkpoint.

    Parameters
    ----------
    path   : str              Path to checkpoint file
    model  : torch.nn.Module  Model architecture to load weights into
    device : str              'cuda' or 'cpu'

    Returns
    -------
    model  : torch.nn.Module  Model with loaded weights
    """
    print(f'Loading model from {path}')
    state_dict = torch.load(path, map_location=device)['net']
    # Strip 'module.' prefix if saved with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Physics-based convolution
# ---------------------------------------------------------------------------

def GeoConv_tf_3D1(refl, wave):
    """Convolve a 3D reflectivity volume with a 1D wavelet along the time axis.

    The convolution is applied independently to each trace (inline × crossline).

    Parameters
    ----------
    refl : torch.Tensor  Shape (1, 1, n3, n2, n1)  Predicted reflectivity
    wave : torch.Tensor  Shape (nw,)                Seismic wavelet

    Returns
    -------
    out  : torch.Tensor  Shape (1, 1, n3, n2, n1)  Reconstructed seismic
    """
    refl = torch.squeeze(refl, 0)          # (1, n3, n2, n1)
    refl = refl.permute(0, 2, 3, 1)        # (1, n2, n1, n3)
    n3   = refl.shape[-1]
    refl = refl.reshape(-1, 1, n3)         # (n2*n1, 1, n3)

    nw   = wave.shape[-1]
    wave = wave.reshape(1, 1, nw)

    out  = F.conv1d(refl, wave, padding='same')               # (n2*n1, 1, n3)
    n2, n1 = refl.shape[0] // refl.shape[0], refl.shape[0]   # reshape back
    # Recover spatial dimensions from the data loader's n2*n1 count
    out  = out.reshape(1, -1, refl.shape[0] // out.shape[0] if out.shape[0] > 1 else 1, n3)

    # Safer reshape: infer n2, n1 from the original refl before reshape
    refl_spatial = torch.squeeze(refl.reshape(-1, 1, n3), 1)   # placeholder
    # Directly reshape using permute-back pattern
    batch_size   = 1
    n2n1         = out.shape[1] if out.dim() == 4 else out.shape[0]
    out          = out.squeeze(0)  # fallback — reshape done below

    # Robust version: rebuild from flat (n2*n1, 1, n3)
    refl2  = torch.squeeze(torch.squeeze(torch.unsqueeze(refl, 0), 0), 0)  # (n2*n1, 1, n3)
    out_1d = F.conv1d(refl2, wave, padding='same')             # (n2*n1, 1, n3)
    # Determine n2, n1 from original refl (before any reshape above)
    # We use the shape passed in via the original argument
    orig   = torch.squeeze(refl.unsqueeze(0), 0)               # (1, n2, n1, n3) — already permuted
    _n2    = orig.shape[1] if orig.dim() == 4 else 1
    _n1    = orig.shape[2] if orig.dim() == 4 else 1
    out_1d = out_1d.reshape(1, _n2, _n1, n3).permute(0, 3, 1, 2)  # (1, n3, n2, n1)
    out_1d = out_1d.unsqueeze(0)                               # (1, 1, n3, n2, n1)
    return out_1d


def GeoConv_tf_3D1_simple(refl, wave, n1, n2):
    """Simplified version of GeoConv_tf_3D1 with explicit spatial dimensions.

    Parameters
    ----------
    refl : torch.Tensor  Shape (1, 1, n3, n2, n1)
    wave : torch.Tensor  Shape (nw,)
    n1   : int           Inline count
    n2   : int           Crossline count

    Returns
    -------
    out  : torch.Tensor  Shape (1, 1, n3, n2, n1)
    """
    n3  = refl.shape[2]
    nw  = wave.numel()

    flat = refl.squeeze(0).permute(0, 2, 3, 1).reshape(-1, 1, n3)  # (n2*n1, 1, n3)
    w    = wave.reshape(1, 1, nw)
    out  = F.conv1d(flat, w, padding='same')                        # (n2*n1, 1, n3)
    out  = out.reshape(1, n2, n1, n3).permute(0, 3, 1, 2)          # (1, n3, n2, n1)
    return out.unsqueeze(0)                                          # (1, 1, n3, n2, n1)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def dprocess_maxmin(data):
    """Min-max normalization to [0, 1]."""
    data_min = torch.min(data)
    data_max = torch.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)


def dprocess_meanstd(data):
    """Zero-mean, unit-variance normalization."""
    return (data - torch.mean(data)) / (torch.std(data) + 1e-8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def clc_psnr(out, label):
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    out   : torch.Tensor  Network output
    label : torch.Tensor  Ground-truth label

    Returns
    -------
    float : PSNR value in dB
    """
    assert out.shape == label.shape
    if label.nelement() == 1:
        return 0.0
    mse = (out - label).pow(2).mean()
    return -10 * math.log10(mse.item() + 1e-12)


def average_waveform_correlation(data1, data2):
    """Compute average cosine similarity between two volumes at non-zero positions.

    Parameters
    ----------
    data1 : torch.Tensor  Shape (1, 1, n3, n2, n1)
    data2 : torch.Tensor  Shape (1, 1, n3, n2, n1)  (used as mask: non-zero positions)

    Returns
    -------
    torch.Tensor : Scalar average correlation
    """
    d1 = data1.squeeze()
    d2 = data2.squeeze()
    non_zero = (d2 != 0).nonzero()
    if non_zero.numel() == 0:
        return torch.tensor(0.0, device=data1.device)
    d1_nz = d1[non_zero[:, 0], non_zero[:, 1], non_zero[:, 2]]
    d2_nz = d2[non_zero[:, 0], non_zero[:, 1], non_zero[:, 2]]
    return torch.mean(F.cosine_similarity(d1_nz, d2_nz, dim=0))


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def augment(x, y, hflip=True, rot=True):
    """Apply random horizontal/vertical flips for data augmentation (in-place)."""
    if hflip and random.random() < 0.5:
        x = x[:, :, ::-1]
        y = y[:, :, ::-1]
    if rot and random.random() < 0.5:
        x = x[:, ::-1, :]
        y = y[:, ::-1, :]
    return x, y


def crop(x, y, size):
    """Extract a random square patch from two aligned 3D arrays."""
    iw, ih = x.shape[-2], x.shape[-1]
    ix = random.randrange(0, iw - size + 1)
    iy = random.randrange(0, ih - size + 1)
    return x[:, ix:ix+size, iy:iy+size], y[:, ix:ix+size, iy:iy+size]
