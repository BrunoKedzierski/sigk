import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn

import torch.nn.functional as F
import cv2
import numpy as np
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

from SIGK.losses_inpainting import combined_inpainting_loss
from SIGK.figures_inpainting import show_examples

def _bbox_from_mask(m):
    # m: tensor [H, W] 
    # returns bounding box coordinates (y0, y1, x0, x1)
    ys = torch.nonzero(torch.any(m > 0.5, dim=1)).flatten()
    xs = torch.nonzero(torch.any(m > 0.5, dim=0)).flatten()
    y0, y1 = ys[0].item(), ys[-1].item() + 1
    x0, x1 = xs[0].item(), xs[-1].item() + 1
    return y0, y1, x0, x1


def _hole_metrics_per_sample(comp, gt, mask, psnr_metric, ssim_metric, lpips_model, margin=24):
    # metrics are computed locally around hole (per image), because global ones are dominated by identical background (GT).
    # PSNR and SNE are on hole (pixel mettrcis)
    # SSIM and LPIPS are on hole + margin (perceptiual metrics)
    
    H, W = comp.shape[-2], comp.shape[-1]
    psnr_vals, ssim_vals, lpips_vals, sne_vals, l1_vals = [], [], [], [], []

    for i in range(comp.size(0)):
        y0, y1, x0, x1 = _bbox_from_mask(mask[i, 0])

        # tight hole slice, pixel metrics
        c = comp[i:i + 1, :, y0:y1, x0:x1]
        g = gt[i:i + 1, :, y0:y1, x0:x1]

        # clip psnr to 100 dB so that one perfect sample does not ruin the average
        # because perfect sample will give inf psnr value and break the average via log10
        p = psnr_metric(c, g).item()
        psnr_vals.append(min(p, 100.0))

        # l2 / l1 loss on hole vs gt (hole slice only)
        sne_vals.append(((c - g) ** 2).sum().item())
        l1_vals.append((c - g).abs().mean().item())
        psnr_metric.reset()

        # slice + margin  for perceptual metrics
        ym0, ym1 = max(0, y0 - margin), min(H, y1 + margin)
        xm0, xm1 = max(0, x0 - margin), min(W, x1 + margin)
        cm = comp[i:i + 1, :, ym0:ym1, xm0:xm1]
        gm = gt[i:i + 1, :, ym0:ym1, xm0:xm1]

        # perceptual metrics
        ssim_vals.append(ssim_metric(cm, gm).item())
        lpips_vals.append(lpips_model(cm * 2 - 1, gm * 2 - 1).mean().item())
        ssim_metric.reset()
        
    return psnr_vals, ssim_vals, lpips_vals, sne_vals, l1_vals


def validate_and_compare(model, val_loader, device, psnr_metric, ssim_metric, lpips_model, tag="WALIDACJA"):
    # validation mode
    model.eval()

    # train metrics holders
    unet_psnr, unet_ssim, unet_lpips, unet_sne, unet_l1 = [], [], [], [], []
    cv2_psnr, cv2_ssim, cv2_lpips, cv2_sne, cv2_l1 = [], [], [], [], []
    
    # metrics for full image as a reference
    
    unet_psnr_full, unet_ssim_full, unet_lpips_full = [], [], []
    cv2_psnr_full, cv2_ssim_full, cv2_lpips_full = [], [], []
    wins = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'sne': 0}  # win-rate per metryka
    total_samples = 0
    val_loss_sum = 0.0
    n_batches = 0

    # validation loop
    with torch.no_grad():
        for masked_img, mask, gt_img in val_loader:
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            gt_img = gt_img.to(device)

            # U-NET
            raw_pred = model(masked_img, mask)
            val_loss_sum += combined_inpainting_loss(raw_pred.float(), gt_img, mask, lpips_model).item()
            n_batches += 1

            unet_comp = gt_img * (1.0 - mask) + raw_pred * mask
            u_psnr, u_ssim, u_lpips, u_sne, u_l1 = _hole_metrics_per_sample(
                unet_comp, gt_img, mask, psnr_metric, ssim_metric, lpips_model
            )

            # OPENCV (Telea)
            batch_size = masked_img.size(0)
            cv2_preds = []
            for i in range(batch_size):
                m = mask[i, 0].cpu().numpy()
                m_uint8 = (m * 255).astype(np.uint8)
                img_np = masked_img[i].cpu().permute(1, 2, 0).numpy()
                img_bgr = cv2.cvtColor(np.clip(np.round(img_np * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                res_bgr = cv2.inpaint(img_bgr, m_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
                cv2_preds.append(torch.from_numpy(res_rgb).permute(2, 0, 1).float() / 255.0)

            cv2_comp = torch.stack(cv2_preds).to(device)

            # composition as for UNet: outside hole exact GT (float), inside hole Telea's inpainting.
            # this way margin and full image are fair (no uint8 quantization difference compared to UNet).
            cv2_comp = gt_img * (1.0 - mask) + cv2_comp * mask
            c_psnr, c_ssim, c_lpips, c_sne, c_l1 = _hole_metrics_per_sample(
                cv2_comp, gt_img, mask, psnr_metric, ssim_metric, lpips_model
            )

            unet_psnr += u_psnr; unet_ssim += u_ssim; unet_lpips += u_lpips; unet_sne += u_sne; unet_l1 += u_l1
            cv2_psnr += c_psnr; cv2_ssim += c_ssim; cv2_lpips += c_lpips; cv2_sne += c_sne; cv2_l1 += c_l1

            # metrics for full image as a reference
            unet_psnr_full.append(psnr_metric(unet_comp, gt_img).item()); psnr_metric.reset()
            unet_ssim_full.append(ssim_metric(unet_comp, gt_img).item()); ssim_metric.reset()
            unet_lpips_full.append(lpips_model(unet_comp * 2 - 1, gt_img * 2 - 1).mean().item())
            cv2_psnr_full.append(psnr_metric(cv2_comp, gt_img).item()); psnr_metric.reset()
            cv2_ssim_full.append(ssim_metric(cv2_comp, gt_img).item()); ssim_metric.reset()
            cv2_lpips_full.append(lpips_model(cv2_comp * 2 - 1, gt_img * 2 - 1).mean().item())

            # Win-rate per metric (PSNR/SSIM: more=better; LPIPS/SNE: less=better)
            for a, b in zip(u_psnr, c_psnr): wins['psnr'] += int(a > b)
            for a, b in zip(u_ssim, c_ssim): wins['ssim'] += int(a > b)
            for a, b in zip(u_lpips, c_lpips): wins['lpips'] += int(a < b)
            for a, b in zip(u_sne, c_sne): wins['sne'] += int(a < b)
            total_samples += batch_size

    import statistics as _st
    def _mean(xs):
        return sum(xs) / max(len(xs), 1)
    def _std(xs):
        return _st.pstdev(xs) if len(xs) > 1 else 0.0
    def _wr(k):
        return 100.0 * wins[k] / max(total_samples, 1)

    avg_val_loss = val_loss_sum / max(n_batches, 1)
    print(f"--- Wyniki Ewaluacji [{tag}] ---")
    print(f"Val Loss: {avg_val_loss:.4f}  (n={total_samples})")
    print(f"[LOKALNIE | dziura (PSNR,SNE,L1) / dziura+margines (SSIM,LPIPS)]  (srednia +/- std)")
    print(f"  U-Net  | PSNR: {_mean(unet_psnr):.2f}+/-{_std(unet_psnr):.2f} | SSIM: {_mean(unet_ssim):.4f}+/-{_std(unet_ssim):.4f} | LPIPS: {_mean(unet_lpips):.4f}+/-{_std(unet_lpips):.4f} | SNE: {_mean(unet_sne):.2f} | L1: {_mean(unet_l1):.4f}")
    print(f"  Telea  | PSNR: {_mean(cv2_psnr):.2f}+/-{_std(cv2_psnr):.2f} | SSIM: {_mean(cv2_ssim):.4f}+/-{_std(cv2_ssim):.4f} | LPIPS: {_mean(cv2_lpips):.4f}+/-{_std(cv2_lpips):.4f} | SNE: {_mean(cv2_sne):.2f} | L1: {_mean(cv2_l1):.4f}")
    print(f"  U-Net wygrywa: PSNR {wins['psnr']}/{total_samples} ({_wr('psnr'):.1f}%) | SSIM {wins['ssim']}/{total_samples} ({_wr('ssim'):.1f}%) | LPIPS {wins['lpips']}/{total_samples} ({_wr('lpips'):.1f}%) | SNE {wins['sne']}/{total_samples} ({_wr('sne'):.1f}%)")
    print(f"[PEŁNY OBRAZ | referencyjnie]")
    print(f"  U-Net  | PSNR: {_mean(unet_psnr_full):.2f} | SSIM: {_mean(unet_ssim_full):.4f} | LPIPS: {_mean(unet_lpips_full):.4f}")
    print(f"  Telea  | PSNR: {_mean(cv2_psnr_full):.2f} | SSIM: {_mean(cv2_ssim_full):.4f} | LPIPS: {_mean(cv2_lpips_full):.4f}")
    print("-" * 30)

    return {
        'loss': avg_val_loss,
        'n_samples': total_samples,
        'unet': {
            'psnr': _mean(unet_psnr), 'psnr_std': _std(unet_psnr),
            'ssim': _mean(unet_ssim), 'ssim_std': _std(unet_ssim),
            'lpips': _mean(unet_lpips), 'lpips_std': _std(unet_lpips),
            'sne': _mean(unet_sne), 'sne_std': _std(unet_sne),
            'l1': _mean(unet_l1), 'l1_std': _std(unet_l1),
            'psnr_full': _mean(unet_psnr_full), 'ssim_full': _mean(unet_ssim_full), 'lpips_full': _mean(unet_lpips_full),
        },
        'telea': {
            'psnr': _mean(cv2_psnr), 'psnr_std': _std(cv2_psnr),
            'ssim': _mean(cv2_ssim), 'ssim_std': _std(cv2_ssim),
            'lpips': _mean(cv2_lpips), 'lpips_std': _std(cv2_lpips),
            'sne': _mean(cv2_sne), 'sne_std': _std(cv2_sne),
            'l1': _mean(cv2_l1), 'l1_std': _std(cv2_l1),
            'psnr_full': _mean(cv2_psnr_full), 'ssim_full': _mean(cv2_ssim_full), 'lpips_full': _mean(cv2_lpips_full),
        },
        'wins': dict(wins),
        'win_rate_pct': {k: _wr(k) for k in wins},
        # short-cuts for training loop compatibility (scheduler/saving best):
        'psnr': _mean(unet_psnr), 'ssim': _mean(unet_ssim), 'lpips': _mean(unet_lpips), 'sne': _mean(unet_sne),
    }
