import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from SIGK.directories import INPAINTING_EXPERIMENTS_DIR


def _hole_psnr(pred, gt, mask):
    # PSNR computed ONLY on hole pixels (mask==1), averaged over 3 channels
    mse = ((pred - gt) ** 2 * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    return (10 * torch.log10(1.0 / (mse + 1e-12))).item()


def _hole_sne(pred, gt, mask):
    # SNE (Squared Norm Error) = sum of squared differences on hole pixels
    return ((pred - gt) ** 2 * mask).sum().item()


@torch.no_grad()
def show_examples(model, dataset, device, indices=(0, 1, 2), save_path=None):
    """Shows n examples: GT | input with hole | mask | OpenCV (Telea) | model."""
    if save_path is None:
        save_path = INPAINTING_EXPERIMENTS_DIR / "examples.png"

    model.eval()
    indices = list(indices)
    n = len(indices)
    col_titles = ['Ground Truth', 'Wejscie (z dziura)', 'Maska', 'OpenCV (Telea)', 'Model (U-Net)']

    fig, axes = plt.subplots(n, 5, figsize=(5 * 2.6, n * 2.6))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        masked_img, mask, gt_img = dataset[idx]
        masked_b = masked_img.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)
        gt_b = gt_img.unsqueeze(0).to(device)

        # model reproduction (composition: GT outside hole + prediction in hole)
        raw_pred = model(masked_b, mask_b)
        unet_comp = gt_b * (1.0 - mask_b) + raw_pred * mask_b

        # OpenCV (Telea) reproduction
        m_uint8 = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        img_np = masked_img.permute(1, 2, 0).numpy()
        img_bgr = cv2.cvtColor(np.clip(np.round(img_np * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        res_bgr = cv2.inpaint(img_bgr, m_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        cv2_comp = torch.from_numpy(res_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        telea_psnr = _hole_psnr(cv2_comp, gt_b, mask_b)
        unet_psnr_v = _hole_psnr(unet_comp, gt_b, mask_b)
        telea_sne = _hole_sne(cv2_comp, gt_b, mask_b)
        unet_sne_v = _hole_sne(unet_comp, gt_b, mask_b)

        imgs = [
            gt_img.permute(1, 2, 0).numpy(),
            masked_img.permute(1, 2, 0).numpy(),
            mask[0].cpu().numpy(),
            cv2_comp[0].cpu().permute(1, 2, 0).numpy(),
            unet_comp[0].cpu().permute(1, 2, 0).numpy(),
        ]

        for col in range(5):
            ax = axes[row, col]
            if col == 2:
                ax.imshow(imgs[col], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(imgs[col], 0, 1))
            ax.axis('off')

            # titles: column names in the first row; PSNR in hole for each row
            if col == 3:
                ax.set_title(f"{col_titles[col]}\nPSNR: {telea_psnr:.2f} | SNE: {telea_sne:.1f}", fontsize=9)
            elif col == 4:
                ax.set_title(f"{col_titles[col]}\nPSNR: {unet_psnr_v:.2f} | SNE: {unet_sne_v:.1f}", fontsize=9)
            elif row == 0:
                ax.set_title(col_titles[col], fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Zapisano przykłady do: {save_path}")


def show_test_examples(model, test_dataset, device, weights_path=None, save_path=None):
    # load best model (based on val loss) for visualization
    if weights_path is None:
        weights_path = INPAINTING_EXPERIMENTS_DIR / "unet_best.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # examples of reproduction: 8 different images from the TEST set (evenly spread)
    _n = len(test_dataset)
    _idx = tuple(int(i) for i in np.linspace(0, _n - 1, 8).astype(int))
    show_examples(model, test_dataset, device, indices=_idx, save_path=save_path)


def plot_training_history(history=None, save_path=None):
    if save_path is None:
        save_path = INPAINTING_EXPERIMENTS_DIR / "training_history.png"
    # if run separately (after session restart) -> load history from file
    if history is None:
        history = json.load(open(INPAINTING_EXPERIMENTS_DIR / "history.json"))

    epochs = history['epoch']
    # (title, train_key, val_key, higher_better)
    panels = [
        ('Loss',                     'train_loss',  'val_loss',  False),
        ('PSNR (dziura)',            'train_psnr',  'val_psnr',  True),
        ('SSIM (dziura+margines)',   'train_ssim',  'val_ssim',  True),
        ('LPIPS (dziura+margines)',  'train_lpips', 'val_lpips', False),
        ('SNE (dziura)',             'train_sne',   'val_sne',   False),
    ]

    n_ax = len(panels) + 1  # +1 on learning rate panel
    fig, axes = plt.subplots(1, n_ax, figsize=(4 * n_ax, 4))
    for ax, (title, k_tr, k_val, higher_better) in zip(axes, panels):
        ax.plot(epochs, history[k_tr],  label='train', marker='o', ms=3)
        ax.plot(epochs, history[k_val], label='val',   marker='s', ms=3)
        arrow = ' (wyzej lepiej)' if higher_better else ' (nizej lepiej)'
        ax.set_title(title + arrow, fontsize=10)
        ax.set_xlabel('Epoka')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # learning rate panel (separate, single curve)
    ax = axes[-1]
    ax.plot(epochs, history.get('lr', [float('nan')] * len(epochs)), marker='o', ms=3, color='tab:green')
    ax.set_title('Learning rate', fontsize=10)
    ax.set_xlabel('Epoka')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Zapisano wykres do: {save_path}")
