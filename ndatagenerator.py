import numpy as np
import torch
from torch.utils.data import Dataset


class Seismicloader(Dataset):
    """DataLoader for a single 3D seismic volume with structural tensor inputs.

    Reads one large binary float32 volume and all auxiliary direction-vector
    files. Returns them as tensors ready for self-supervised training.

    Parameters
    ----------
    droot  : str   Path to seismic data file (.dat, float32)
    wroot  : str   Path to wavelet file (.dat, float32)
    rroot  : str   Path to mask/validity file (.dat, float32)
    u1/u2/u3root : str  Dip-direction unit vector files
    v1/v2/v3root : str  Normal-plane vector files
    w1/w2/w3root : str  Strike-direction vector files
    n1     : int   Inline dimension  (default 128)
    n2     : int   Crossline dimension (default 128)
    n3     : int   Time-sample dimension (default 512)
    """

    def __init__(self, droot, wroot, rroot,
                 u1root, u2root, u3root,
                 v1root, v2root, v3root,
                 w1root, w2root, w3root,
                 n1=128, n2=128, n3=512):
        super().__init__()
        self.n1, self.n2, self.n3 = n1, n2, n3
        shape = (n1, n2, n3)

        self.datan = np.fromfile(droot,  dtype=np.float32).reshape(shape)
        self.rn    = np.fromfile(rroot,  dtype=np.float32).reshape(shape)
        self.wn    = np.fromfile(wroot,  dtype=np.float32)[::-1].copy()  # time-reversed wavelet
        self.u1n   = np.fromfile(u1root, dtype=np.float32).reshape(shape)
        self.u2n   = np.fromfile(u2root, dtype=np.float32).reshape(shape)
        self.u3n   = np.fromfile(u3root, dtype=np.float32).reshape(shape)
        self.v1n   = np.fromfile(v1root, dtype=np.float32).reshape(shape)
        self.v2n   = np.fromfile(v2root, dtype=np.float32).reshape(shape)
        self.v3n   = np.fromfile(v3root, dtype=np.float32).reshape(shape)
        self.w1n   = np.fromfile(w1root, dtype=np.float32).reshape(shape)
        self.w2n   = np.fromfile(w2root, dtype=np.float32).reshape(shape)
        self.w3n   = np.fromfile(w3root, dtype=np.float32).reshape(shape)

    def __len__(self):
        # Single-volume dataset: treated as one sample per epoch
        return 1

    def __getitem__(self, index):
        n1, n2, n3 = self.n1, self.n2, self.n3

        def _prep(arr):
            """Transpose (n1,n2,n3) → (n3,n2,n1) and add channel dim."""
            t = arr.transpose()                       # (n3, n2, n1)
            out = np.zeros((1, n3, n2, n1), dtype=np.float32)
            out[0] = t
            return out

        X   = _prep(self.datan)
        rz  = _prep(self.rn)
        u1z = _prep(self.u1n)
        u2z = _prep(self.u2n)
        u3z = _prep(self.u3n)
        v1z = _prep(self.v1n)
        v2z = _prep(self.v2n)
        v3z = _prep(self.v3n)
        w1z = _prep(self.w1n)
        w2z = _prep(self.w2n)
        w3z = _prep(self.w3n)

        return (X, self.wn, rz,
                u1z, u2z, u3z,
                v1z, v2z, v3z,
                w1z, w2z, w3z)
