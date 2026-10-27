from tqdm import tqdm
import torch
from torch.nn import functional as F
import time
from pathlib import Path
import cv2
import numpy as np
from SIGK.model_utils import save_json, save_csv,rgb_numpy, check_dataset, make_loader, seed_everything
import random
import lpips
import math
from SIGK.metrics import psnr_per_image, ssim_rgb
from SIGK.figures import save_gallery, plot_history, plot_overall_comparison

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total = 0.0
    n = 0

    for lr, gt in tqdm(loader, desc="Training", leave=False):
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(lr)
        loss = F.l1_loss(pred, gt)

        loss.backward()
        optimizer.step()

        batch_size = len(lr)
        total += loss.item() * batch_size
        n += batch_size

    return total / n

@torch.inference_mode()
def validation_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for lr, gt in loader:
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        loss = F.l1_loss(model(lr), gt)
        batch = len(lr)
        total += loss.item() * batch
        n += batch
    return total / n


@torch.inference_mode()
def evaluate_on_test(model, test_loader, dataset, lpips_model,
                     device, out_dir, example_indices):
    model.eval()
    lpips_model.eval()
    rows = []
    samples = []
    seen = 0

    for lr_cpu, gt_cpu in test_loader:
        lr = lr_cpu.to(device, non_blocking=True)
        gt = gt_cpu.to(device, non_blocking=True)
        pred = model(lr).clamp(0, 1)

        # OpenCV resize on RGB float arrays; RGB remains RGB.
        bic_np = [np.clip(cv2.resize(
            rgb_numpy(x), (gt.shape[-1], gt.shape[-2]),
            interpolation=cv2.INTER_CUBIC,
        ), 0, 1) for x in lr_cpu]
        bic = torch.from_numpy(np.stack(bic_np)).permute(0, 3, 1, 2)
        bic = bic.to(device=device, dtype=gt.dtype)

        psnr_pred = psnr_per_image(pred, gt).cpu().tolist()
        psnr_bic = psnr_per_image(bic, gt).cpu().tolist()
        mae_pred = (pred - gt).abs().mean(dim=(1, 2, 3)).cpu().tolist()
        mae_bic = (bic - gt).abs().mean(dim=(1, 2, 3)).cpu().tolist()

        # LPIPS expects [-1,1]; outputs one score for every image.
        lp_pred = lpips_model(pred * 2 - 1, gt * 2 - 1).reshape(-1).cpu().tolist()
        lp_bic = lpips_model(bic * 2 - 1, gt * 2 - 1).reshape(-1).cpu().tolist()

        for j in range(len(lr_cpu)):
            global_idx = seen + j
            file_name = Path(dataset.tr_files[global_idx]).name
            pred_np = rgb_numpy(pred[j])
            gt_np = rgb_numpy(gt[j])
            bic_j = bic_np[j]
            model_ssim = ssim_rgb(pred_np, gt_np)
            bicubic_ssim = ssim_rgb(bic_j, gt_np)
            row = {
                'file': file_name,
                'original_id': Path(file_name).stem.split('_crop_x256_')[0],
                'model_l1': mae_pred[j],
                'model_psnr': psnr_pred[j],
                'model_ssim': model_ssim,
                'model_lpips': lp_pred[j],
                'bicubic_l1': mae_bic[j],
                'bicubic_psnr': psnr_bic[j],
                'bicubic_ssim': bicubic_ssim,
                'bicubic_lpips': lp_bic[j],
            }
            rows.append(row)
            if global_idx in example_indices:
                samples.append({
                    'name': file_name,
                    'lr': rgb_numpy(lr_cpu[j]),
                    'model': pred_np,
                    'bicubic': bic_j,
                    'gt': gt_np,
                    'model_psnr': psnr_pred[j],
                    'bicubic_psnr': psnr_bic[j],
                })
        seen += len(lr_cpu)

    save_csv(out_dir / 'test_per_image.csv', rows)
    samples.sort(key=lambda d: d['name'])
    save_gallery(samples, out_dir)

    report = {'n_test_images': len(rows)}
    for method in ('model', 'bicubic'):
        for metric in ('l1', 'psnr', 'ssim', 'lpips'):
            key = f'{method}_{metric}'
            report[key] = float(np.mean([r[key] for r in rows]))
            report[f'{key}_std'] = float(np.std([r[key] for r in rows]))
    for metric in ('psnr', 'ssim', 'lpips', 'l1'):
        report[f'delta_{metric}'] = (
            report[f'model_{metric}'] - report[f'bicubic_{metric}']
        )
    save_json(out_dir / 'test_summary.json', report)
    return report



def run_all_experiments(datasets, config, models):
    """Run six independent train/val/test experiments in Colab."""
    cfg = config 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = Path(cfg['output_dir'])
    out_root.mkdir(parents=True, exist_ok=True)

    for input_size in (32, 64):
        for split in ('train', 'val', 'test'):
            check_dataset(datasets[input_size][split], input_size, split)

    # One common metric network, not trained along with your SR models.
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    summary_rows = []
    for input_size in (32, 64):
        ds = datasets[input_size]
        # Deterministic per-scale examples shared by all 3 architectures.
        example_indices = set(random.Random(cfg['seed']).sample(
            range(len(ds['test'])),
            k=min(cfg['examples_per_experiment'], len(ds['test'])),
        ))

        for model_name, model_cls in models.items():
            run_name = f'{model_name}_x{input_size}'
            out_dir = out_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(cfg['seed'])

            model = model_cls(input_size=input_size).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg['scheduler_factor'],
                patience=cfg['scheduler_patience'], min_lr=cfg['min_lr'], threshold = cfg['scheduler_threshold'],
                threshold_mode = cfg['scheduler_thresh_mode'],
            )
            loaders = {
                split: make_loader(
                    ds[split], cfg['batch_size'], shuffle=(split == 'train'),
                    num_workers=cfg['num_workers'], device=device,
                    seed=cfg['seed'],
                ) for split in ('train', 'val', 'test')
            }
            save_json(out_dir / 'config.json', {
                **cfg,
                'run_name': run_name,
                'architecture': model_name,
                'input_size': input_size,
                'upscaling_factor': 256 // input_size,
                'trainable_parameters': n_params,
                'train_samples': len(ds['train']),
                'val_samples': len(ds['val']),
                'test_samples': len(ds['test']),
                'optimizer': 'Adam',
                'loss_function': 'L1Loss',
                'scheduler': 'ReduceLROnPlateau',
                'device': str(device),
                'torch_version': torch.__version__,
                'metric_details': 'RGB [0,1] after clamp; LPIPS alex; '
                                  'SSIM Gaussian sigma=1.5; average per image',
            })
            print(f'\n=== {run_name} | trainable params: {n_params:,} ===', flush=True)

            best_val = math.inf
            best_epoch = None
            history = []
            checkpoint_path = out_dir / 'best.pt'
            train_seconds = 0.0
            bad_checks = 0

            for epoch in range(1, cfg['epochs'] + 1):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                lr_used = optimizer.param_groups[0]['lr']
                train_l1 = train_one_epoch(model, loaders['train'], optimizer, device)
                val_l1 = None
                if epoch % cfg['val_every'] == 0 or epoch == cfg['epochs']:
                    val_l1 = validation_loss(model, loaders['val'], device)
                    scheduler.step(val_l1)
                    if val_l1 < best_val:
                        bad_checks = 0
                        best_val = val_l1
                        best_epoch = epoch
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'val_l1': val_l1,
                            'model_name': model_name,
                            'input_size': input_size,
                        }, checkpoint_path)
                    else:
                      bad_checks += 1
                    if bad_checks >= 5:
                      break
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                secs = time.perf_counter() - start
                train_seconds += secs
                history.append({
                    'epoch': epoch,
                    'train_l1': train_l1,
                    'val_l1': val_l1 if val_l1 is not None else '',
                    'lr_used': lr_used,
                    'lr_next': optimizer.param_groups[0]['lr'],
                    'epoch_seconds': secs,
                })
                save_csv(out_dir / 'history.csv', history)
                print(f'Epoch {epoch:3d} train={train_l1:.5f} '
                      f'val={val_l1 if val_l1 is not None else "-"} '
                      f'lr={lr_used:.6g} time={secs:.1f}s', flush=True)

            plot_history(history, out_dir)

            # Load *best validation* weights, not those from the final epoch.
            ckpt = torch.load(checkpoint_path, map_location=device,
                              weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            test_report = evaluate_on_test(
                model, loaders['test'], ds['test'], lpips_model, device,
                out_dir, example_indices,
            )
            run_summary = {
                'run_name': run_name,
                'architecture': model_name,
                'input_size': input_size,
                'scale': 256 // input_size,
                'parameters': n_params,
                'best_epoch': best_epoch,
                'best_val_l1': best_val,
                'total_train_seconds': train_seconds,
                'avg_epoch_seconds': train_seconds / cfg['epochs'],
                **test_report,
            }
            save_json(out_dir / 'run_summary.json', run_summary)
            summary_rows.append(run_summary)
            save_csv(out_root / 'summary_all.csv', summary_rows)
            plot_overall_comparison(summary_rows, out_root)
            print(f'{run_name}: PSNR {test_report["model_psnr"]:.2f} vs '
                  f'bicubic {test_report["bicubic_psnr"]:.2f}; '
                  f'SSIM {test_report["model_ssim"]:.4f}; '
                  f'LPIPS {test_report["model_lpips"]:.4f}', flush=True)

            del model, optimizer, scheduler, loaders
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return summary_rows