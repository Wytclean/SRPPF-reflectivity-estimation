import os
import argparse
import numpy as np
import torch
import torch.utils.data as data

from utils import load_model
from unet import make_model


class TestLoader(data.Dataset):
    """DataLoader for inference on 3D seismic volumes.

    Parameters
    ----------
    data_path : str   Path to a single seismic volume (.dat, float32)
    n1        : int   Inline dimension
    n2        : int   Crossline dimension
    n3        : int   Time-sample dimension
    """

    def __init__(self, data_path, n1=128, n2=128, n3=256):
        super().__init__()
        self.data_path = data_path
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raw  = np.fromfile(self.data_path, dtype=np.float32)
        vol  = raw.reshape(1, self.n3, self.n2, self.n1)   # (1, n3, n2, n1)
        mean = np.mean(vol)
        std  = np.std(vol)
        vol  = (vol - mean) / (std + 1e-8)
        return vol.astype(np.float32), vol.astype(np.float32)


def inference(model, test_loader, out_path, device='cuda'):
    """Run inference and save the predicted reflectivity volume.

    Parameters
    ----------
    model       : torch.nn.Module
    test_loader : DataLoader
    out_path    : str   Output file path (e.g., './output/result.dat')
    device      : str   'cuda' or 'cpu'
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (vol, _) in enumerate(test_loader):
            vol = vol.to(device)
            out = model(vol)
            out = out.squeeze().cpu().numpy().astype(np.float32)
            out.tofile(out_path)
            print(f'Saved prediction to {out_path}  shape={out.shape}')


def parse_args():
    parser = argparse.ArgumentParser(description='3D Seismic Reflectivity Estimation — Inference')
    parser.add_argument('--data_path',  required=True, help='Input seismic volume (.dat)')
    parser.add_argument('--model_path', required=True, help='Trained model checkpoint (.pt)')
    parser.add_argument('--out_path',   default='./output/prediction.dat',
                        help='Output file path for predicted reflectivity')
    parser.add_argument('--n1', type=int, default=128, help='Inline dimension')
    parser.add_argument('--n2', type=int, default=128, help='Crossline dimension')
    parser.add_argument('--n3', type=int, default=256, help='Time-sample dimension')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = make_model()
    model = load_model(args.model_path, model, device)

    test_loader = data.DataLoader(
        TestLoader(args.data_path, n1=args.n1, n2=args.n2, n3=args.n3),
        batch_size=1, shuffle=False
    )

    inference(model, test_loader, out_path=args.out_path, device=device)
