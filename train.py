import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ssim import MultiScaleSSIMLoss3d
from utils import GeoConv_tf_3D1, dprocess_meanstd, average_waveform_correlation, save, save_model, load_model
from unet import make_model, LNORMALLoss_3D, COSINELoss_3D
from ndatagenerator import Seismicloader

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Loss weights
LAMBDA1 = 1       # MSE loss
LAMBDA2 = 5       # MS-SSIM loss
LAMBDA3 = 0.005   # L1 sparsity loss
LAMBDA4 = 1       # L_DVW structural loss
LAMBDA5 = 1       # L_DU structural loss


def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised 3D Seismic Reflectivity Estimation')

    # Data paths
    parser.add_argument('--droot',  required=True, help='Path to seismic data (.dat)')
    parser.add_argument('--wroot',  required=True, help='Path to wavelet file (.dat)')
    parser.add_argument('--rroot',  required=True, help='Path to mask file (.dat)')
    parser.add_argument('--u1root', required=True, help='Path to u1 direction vector (.dat)')
    parser.add_argument('--u2root', required=True, help='Path to u2 direction vector (.dat)')
    parser.add_argument('--u3root', required=True, help='Path to u3 direction vector (.dat)')
    parser.add_argument('--v1root', required=True, help='Path to v1 direction vector (.dat)')
    parser.add_argument('--v2root', required=True, help='Path to v2 direction vector (.dat)')
    parser.add_argument('--v3root', required=True, help='Path to v3 direction vector (.dat)')
    parser.add_argument('--w1root', required=True, help='Path to w1 direction vector (.dat)')
    parser.add_argument('--w2root', required=True, help='Path to w2 direction vector (.dat)')
    parser.add_argument('--w3root', required=True, help='Path to w3 direction vector (.dat)')

    # Volume dimensions
    parser.add_argument('--n1', type=int, default=128, help='Dimension n1 (inline)')
    parser.add_argument('--n2', type=int, default=128, help='Dimension n2 (crossline)')
    parser.add_argument('--n3', type=int, default=512, help='Dimension n3 (time samples)')

    # Training settings
    parser.add_argument('--lr',         type=float, default=1e-4,  help='Learning rate')
    parser.add_argument('--epochs',     type=int,   default=1000,  help='Total training epochs')
    parser.add_argument('--batch_size', type=int,   default=1,     help='Batch size')
    parser.add_argument('--gpu',        type=str,   default='0',   help='GPU device id(s)')

    # Output
    parser.add_argument('--model_dir',  default='./model',    help='Directory to save model checkpoints')
    parser.add_argument('--pred_dir',   default='./predict_reflection', help='Directory to save predicted reflectivity')
    parser.add_argument('--recov_dir',  default='./recon_seismic',      help='Directory to save reconstructed seismic')
    parser.add_argument('--log_dir',    default='.',           help='TensorBoard log directory')
    parser.add_argument('--resume',     default=None,          help='Path to checkpoint to resume from')

    return parser.parse_args()


def train_one_epoch(model, data_loader, optimizer, scheduler, zeros,
                    MSELoss, MSSSIMLoss, SMAELoss, writer, epoch, epoch_max, is_gpu):
    """Run one training epoch and return total loss."""
    loss_epoch = mseloss_e = msssim_e = smae_e = dvw_e = du_e = diff_e = 0.0
    model.train()

    for step, (seismic, w, r, u1, u2, u3, v1, v2, v3, w1, w2, w3) in enumerate(data_loader):
        if is_gpu:
            seismic = seismic.cuda()
            w  = w.cuda();  r  = r.cuda()
            u1 = u1.cuda(); u2 = u2.cuda(); u3 = u3.cuda()
            v1 = v1.cuda(); v2 = v2.cuda(); v3 = v3.cuda()
            w1 = w1.cuda(); w2 = w2.cuda(); w3 = w3.cuda()

        indvw = LNORMALLoss_3D(v1, v2, v3, w1, w2, w3)
        indu  = COSINELoss_3D(u1, u2, u3)

        out   = model(seismic)
        sxpre = GeoConv_tf_3D1(out, w)
        sx_norm    = dprocess_meanstd(seismic)
        sxpre_norm = dprocess_meanstd(sxpre)
        diff = average_waveform_correlation(out, r)

        loss1 = LAMBDA1 * MSELoss(sx_norm, sxpre_norm)
        loss2 = LAMBDA2 * MSSSIMLoss(sx_norm, sxpre_norm)
        loss3 = LAMBDA3 * SMAELoss(out, zeros)
        loss4 = LAMBDA4 * indvw(out.squeeze())
        loss5 = LAMBDA5 * indu(out)
        loss  = loss1 + loss2 + loss3 + loss4 + loss5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch[{epoch}/{epoch_max}] Step[{step}/{len(data_loader)}] '
              f'MSE:{loss1.item():.4f} MSSSIM:{loss2.item():.4f} '
              f'L1:{loss3.item():.4f} DVW:{loss4.item():.4f} DU:{loss5.item():.4f}')

        loss_epoch += loss.item()
        mseloss_e  += loss1.item()
        msssim_e   += loss2.item()
        smae_e     += loss3.item()
        dvw_e      += loss4.item()
        du_e       += loss5.item()
        diff_e     += diff.item()

    scheduler.step(loss_epoch)

    writer.add_scalars('Losses', {
        'MSE': mseloss_e, 'L1': smae_e,
        'MSSSIM': msssim_e, 'DVW': dvw_e, 'DU': du_e
    }, epoch)
    writer.add_scalar('Total_Loss',    loss_epoch, epoch)
    writer.add_scalar('Diff_Loss',     diff_e,     epoch)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    return loss_epoch, out, sxpre


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    num_GPU  = torch.cuda.device_count() if use_cuda else 0
    is_gpu   = use_cuda

    os.makedirs(args.model_dir, exist_ok=True)

    zeros = torch.zeros(args.batch_size, 1, args.n3, args.n2, args.n1)
    if is_gpu:
        zeros = zeros.cuda()

    dataset = Seismicloader(
        args.droot, args.wroot, args.rroot,
        args.u1root, args.u2root, args.u3root,
        args.v1root, args.v2root, args.v3root,
        args.w1root, args.w2root, args.w3root,
        n1=args.n1, n2=args.n2, n3=args.n3
    )
    data_loader = data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, drop_last=True
    )

    model = make_model()
    if args.resume:
        model = load_model(args.resume, model, 'cuda' if is_gpu else 'cpu')
    model = model.to(device)
    if num_GPU > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    MSELoss   = nn.MSELoss()
    MSSSIMLoss = MultiScaleSSIMLoss3d()
    SMAELoss  = nn.L1Loss()

    writer = SummaryWriter(log_dir=args.log_dir)
    c = 0

    for epoch in range(args.epochs):
        loss_epoch, out, sxpre = train_one_epoch(
            model, data_loader, optimizer, scheduler, zeros,
            MSELoss, MSSSIMLoss, SMAELoss, writer, epoch, args.epochs, is_gpu
        )
        save(out, sxpre, c, pred_path=args.pred_dir, recov_path=args.recov_dir)
        save_model(args.model_dir, model, optimizer, epoch)
        c += 1
        print(f'Epoch[{epoch}/{args.epochs}] Total Loss: {loss_epoch:.4f}')

    writer.close()


if __name__ == '__main__':
    main()
