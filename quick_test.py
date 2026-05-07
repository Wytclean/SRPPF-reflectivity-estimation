"""
quick_test.py — SRPPF Quick Test with Synthetic Data

This script generates a small synthetic seismic volume and runs a
single forward pass through the U-Net to verify the installation.

No real seismic data is required. The test uses random network weights.

Usage:
    python quick_test.py

Expected output:
    Input shape : torch.Size([1, 1, 64, 32, 32])
    Output shape: torch.Size([1, 1, 64, 32, 32])
    Quick test PASSED.

To run with the pre-trained checkpoint, download pretrain_model.pt from:
    https://github.com/Wytclean/SRPPF-reflectivity-estimation/releases/tag/v1.0

SHA-256: 43d1c83a3a40f189f38cf033842027ad0e624abe2a65301bc0def4e72639fcba

Verify the download:
    sha256sum pretrain_model.pt                          # Linux/macOS
    Get-FileHash pretrain_model.pt -Algorithm SHA256     # Windows PowerShell
"""

import torch
import numpy as np
from unet import make_model


def generate_synthetic_seismic(n3=64, n2=32, n1=32, freq=30.0, dt=0.002):
    """Generate a small synthetic 3D seismic volume.

    A sparse reflectivity series is convolved with a Ricker wavelet
    along the time axis to produce band-limited synthetic seismic data.

    Parameters
    ----------
    n3   : int    Number of time samples
    n2   : int    Number of crosslines
    n1   : int    Number of inlines
    freq : float  Dominant frequency of the Ricker wavelet (Hz)
    dt   : float  Sampling interval (s)

    Returns
    -------
    seismic : np.ndarray  Shape (1, n3, n2, n1), float32, zero-mean unit-variance
    """
    # Ricker wavelet
    nw = 51
    t  = (np.arange(nw) - nw // 2) * dt
    wavelet = (1.0 - 2.0 * (np.pi * freq * t) ** 2) * np.exp(-(np.pi * freq * t) ** 2)

    # Sparse reflectivity — four horizontal layers
    reflectivity = np.zeros((n3, n2, n1), dtype=np.float32)
    for layer in [10, 25, 40, 55]:
        if layer < n3:
            reflectivity[layer, :, :] = np.random.uniform(0.05, 0.15, (n2, n1)).astype(np.float32)

    # Convolve each trace along the time axis
    try:
        from scipy.signal import fftconvolve
        seismic = np.zeros_like(reflectivity)
        for i2 in range(n2):
            for i1 in range(n1):
                seismic[:, i2, i1] = fftconvolve(
                    reflectivity[:, i2, i1], wavelet, mode='same'
                ).astype(np.float32)
    except ImportError:
        # scipy not available — use random noise as a fallback
        print("  [Warning] scipy not found. Using random noise instead of synthetic seismic.")
        seismic = np.random.randn(n3, n2, n1).astype(np.float32)

    # Zero-mean, unit-variance normalisation
    seismic = (seismic - seismic.mean()) / (seismic.std() + 1e-8)

    return seismic[np.newaxis]   # shape: (1, n3, n2, n1)


def run_quick_test():
    print("=" * 50)
    print("SRPPF Quick Test")
    print("=" * 50)

    # 1. Generate synthetic input
    n3, n2, n1 = 64, 32, 32
    print(f"\nGenerating synthetic seismic volume ({n3}×{n2}×{n1})...")
    seismic = generate_synthetic_seismic(n3=n3, n2=n2, n1=n1)

    # shape: (1, 1, n3, n2, n1)
    x = torch.from_numpy(seismic).unsqueeze(0)
    print(f"Input shape : {x.shape}")

    # 2. Build model (CPU, random weights)
    device = "cpu"
    model  = make_model().to(device)
    model.eval()

    # 3. Forward pass
    with torch.no_grad():
        out = model(x.to(device))

    print(f"Output shape: {out.shape}")

    # 4. Sanity check
    assert out.shape == x.shape, \
        f"Shape mismatch — input {x.shape}, output {out.shape}"

    print("\nQuick test PASSED.")
    print("=" * 50)
    print(
        "\nNote: This test uses random network weights.\n"
        "To use the pre-trained model, download pretrain_model.pt from:\n"
        "  https://github.com/Wytclean/SRPPF-reflectivity-estimation/releases/tag/v1.0\n"
        "\n"
        "SHA-256: 43d1c83a3a40f189f38cf033842027ad0e624abe2a65301bc0def4e72639fcba\n"
        "\n"
        "Then run fine-tuning with:\n"
        "  python train.py --resume ./model/pretrain_model.pt [other args]\n"
    )


if __name__ == "__main__":
    run_quick_test()
