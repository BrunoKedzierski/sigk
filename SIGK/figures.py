import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


def plot_history(history, out_dir):
    """Saved train/validation learning curves and learning-rate trajectory."""
    ep = [r['epoch'] for r in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(ep, [r['train_l1'] for r in history], label='Train L1')
    val = [(r['epoch'], r['val_l1']) for r in history if r['val_l1'] != '']
    ax1.plot([p[0] for p in val], [p[1] for p in val],
             marker='o', markersize=3, label='Validation L1')
    ax1.set_ylabel('L1 loss (lower better)')
    ax1.legend()
    ax1.grid(alpha=0.25)
    ax2.step(ep, [r['lr_used'] for r in history], where='post')
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning rate')
    ax2.set_xlabel('Epoch')
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'training_curves.png', dpi=160)
    plt.close(fig)


def plot_overall_comparison(summary_rows, out_root):
    """Side-by-side model versus bicubic summary for each input resolution."""
    if not summary_rows:
        return
    for scale in (32, 64):
        subset = [r for r in summary_rows if r['input_size'] == scale]
        if not subset:
            continue
        labels = [r['architecture'] for r in subset]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, metric, title in zip(
            axes, ('psnr', 'ssim', 'lpips'),
            ('PSNR ↑', 'SSIM ↑', 'LPIPS ↓'),
        ):
            w = 0.35
            ax.bar(x - w / 2, [r[f'model_{metric}'] for r in subset],
                   width=w, label='Model')
            ax.bar(x + w / 2, [r[f'bicubic_{metric}'] for r in subset],
                   width=w, label='Bicubic')
            ax.set_xticks(x, labels)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.2)
        axes[0].legend()
        fig.suptitle(f'Input {scale}×{scale} → 256×256')
        fig.tight_layout()
        fig.savefig(out_root / f'comparison_x{scale}.png',
                    dpi=160, bbox_inches='tight')
        plt.close(fig)


def save_rgb_png(rgb, path):
    arr = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path)


def save_gallery(sample_data, out_dir):
    """Creates comparison grid and individual PNGs for selected test indices."""
    if not sample_data:
        return
    n = len(sample_data)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n), squeeze=False)
    image_dir = out_dir / 'example_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for row, item in enumerate(sample_data):
        name = item['name']
        stem = Path(name).stem
        comparisons = [
            ('LR', item['lr']),
            (f"Model (PSNR {item['model_psnr']:.2f})", item['model']),
            (f"Bicubic (PSNR {item['bicubic_psnr']:.2f})", item['bicubic']),
            ('Ground truth', item['gt']),
        ]
        for col, (title, img) in enumerate(comparisons):
            axes[row, col].imshow(img, interpolation='nearest', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(stem, rotation=90, fontsize=9)
        for short, arr in [('lr', item['lr']), ('model', item['model']),
                           ('bicubic', item['bicubic']), ('gt', item['gt'])]:
            save_rgb_png(arr, image_dir / f'{stem}_{short}.png')
    fig.tight_layout()
    fig.savefig(out_dir / 'examples_grid.png', dpi=160, bbox_inches='tight')
    plt.close(fig)