
import torch
from skimage.metrics import structural_similarity



def psnr_per_image(pred, gt):
    """NCHW image-wise PSNR, [0,1]; shape [B]."""
    mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse)  # exact match -> +inf


def ssim_rgb(pred_np, gt_np):
    return float(structural_similarity(
        gt_np, pred_np, data_range=1.0, channel_axis=2,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))