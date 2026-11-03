import os
import random
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

from SIGK.dataset_inpainting import InpaintingDataset
from SIGK.unet_inpainting import UNet
from SIGK.losses_inpainting import combined_inpainting_loss
from SIGK.metrics_inpainting import validate_and_compare, _hole_metrics_per_sample


from SIGK.directories import (
    TRAIN_CROPPED_x256_DIR,
    VALID_CROPPED_x256_DIR,
    TEST_CROPPED_x256_DIR,
    INPAINTING_EXPERIMENTS_DIR
)

OUT = INPAINTING_EXPERIMENTS_DIR
OUT.mkdir(parents=True, exist_ok=True)

# train / val / test config
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def seed_worker(worker_id):
    ws = (SEED + worker_id) % (2 ** 32)
    np.random.seed(ws)
    random.seed(ws)
def main(epochs: int = 75, hole_size: int = 3, batch_size: int = 16, num_workers: int = 4):
        
    _g = torch.Generator()
    _g.manual_seed(SEED)

    TRAIN_DIR = TRAIN_CROPPED_x256_DIR
    VAL_DIR = VALID_CROPPED_x256_DIR
    TEST_DIR = TEST_CROPPED_x256_DIR
    BATCH_SIZE = batch_size
    NUM_WORKERS = num_workers
    EPOCHS = epochs

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    print(f"epochs={EPOCHS} | hole_size={hole_size} | batch_size={BATCH_SIZE} | num_workers={NUM_WORKERS}")
    use_amp = device.type in ("cuda", "mps")
    pin_memory = device.type == "cuda"

    # Dataset & Dataloader
    train_dataset = InpaintingDataset(images_dir=TRAIN_DIR, is_train=True, hole_size=hole_size)
    val_dataset = InpaintingDataset(images_dir=VAL_DIR, is_train=False, hole_size=hole_size)
    test_dataset = InpaintingDataset(images_dir=TEST_DIR, is_train=False, hole_size=hole_size)
    print(f"Train: {len(train_dataset)} | Valid: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=_g,
        )
    val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )
    test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # Model & Optimizer
    model = UNet().to(device)  
    NUM_PARAMS = sum(p.numel() for p in model.parameters())
    print(f"Liczba parametrów modelu: {NUM_PARAMS:,} ({NUM_PARAMS / 1e6:.2f}M)")

    optimizer = optim.Adam(model.parameters(), lr=2e-4) 

    # scheduler tracks validation LPIPS (perceptual quality in hole+margin)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )

    # metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)


    # GradScaler is stable on CUDA; MPS/CPU use autocast (MPS) or plain fp32 (CPU)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float('inf')   # best (lowest) validation loss -> save model based on it
    best_val_lpips = float('inf')  # best (lowest) LPIPS in hole (tracked by scheduler)

    TRACK_TRAIN_METRICS = True  # whether to compute metrics on training set (adds 1 LPIPS/batch)

    # training history for later plotting
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_psnr': [], 'val_psnr': [],
        'train_ssim': [], 'val_ssim': [],
        'train_lpips': [], 'val_lpips': [],
        'train_sne': [], 'val_sne': [],
        'lr': [],
    }

    import time
    _t_start = time.time()  # training time measurement (wall-clock)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        tr_psnr_vals, tr_ssim_vals, tr_lpips_vals, tr_sne_vals = [], [], [], []

        for batch_idx, (masked_img, mask, gt_img) in enumerate(train_loader):
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            gt_img = gt_img.to(device)

            optimizer.zero_grad()

            
            # fp16 autocast on CUDA/MPS; CPU stays fp32
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_img = model(masked_img, mask)

            # go back to fp32 for loss computation
            loss = combined_inpainting_loss(pred_img.float(), gt_img, mask, lpips_vgg)

            if scaler is not None:
                # scalers the loss to thousands of times to prevent zeroing of fp16
                scaler.scale(loss).backward()
                # scaler prevents gradient explosion
                scaler.unscale_(optimizer)
                # clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()

            # train metrics
            # for each photo in a batch
            # compute metrics on hole (pixel metrics) and hole+margin (perceptual metrics)
            # reported per epoch
            if TRACK_TRAIN_METRICS:
                with torch.no_grad():
                    tr_comp = gt_img * (1.0 - mask) + pred_img.float() * mask
                    bp, bs, bl, bsne, _ = _hole_metrics_per_sample(tr_comp, gt_img, mask, psnr, ssim, lpips_vgg)
                    tr_psnr_vals += bp
                    tr_ssim_vals += bs
                    tr_lpips_vals += bl
                    tr_sne_vals += bsne

        avg_train_loss = train_loss / len(train_loader)

        def _mean(xs):
            return sum(xs) / max(len(xs), 1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

        # validation
        val = validate_and_compare(model, val_loader, device, psnr, ssim, lpips_vgg)
        val_loss, val_lpips = val['loss'], val['lpips']
        scheduler.step(val_lpips)  # scheduler działa na LPIPS (mode="min")

        # save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_psnr'].append(_mean(tr_psnr_vals) if TRACK_TRAIN_METRICS else float('nan'))
        history['val_psnr'].append(val['psnr'])
        history['train_ssim'].append(_mean(tr_ssim_vals) if TRACK_TRAIN_METRICS else float('nan'))
        history['val_ssim'].append(val['ssim'])
        history['train_lpips'].append(_mean(tr_lpips_vals) if TRACK_TRAIN_METRICS else float('nan'))
        history['val_lpips'].append(val['lpips'])
        history['train_sne'].append(_mean(tr_sne_vals) if TRACK_TRAIN_METRICS else float('nan'))
        history['val_sne'].append(val['sne'])
        history['lr'].append(current_lr)  # LR użyty w tej epoce (do krzywej uczenia)

        # track best model based on validation LPIPS
        if val_lpips < best_val_lpips:
            best_val_lpips = val_lpips

        # save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUT / 'unet_best.pth')
            print(f"  -> Nowy najlepszy model (Val Loss: {val_loss:.4f}) zapisany.")

        # display best results in each epoch
        print(f"  Best Val Loss: {best_val_loss:.4f} | Best LPIPS: {best_val_lpips:.4f}")

    # save last model state after loop
    torch.save(model.state_dict(), OUT / 'unet_last.pth')

    # training time (wall-clock)
    train_time_min = (time.time() - _t_start) / 60.0
    history['train_time_min'] = train_time_min  # scalar next to per-epoch lists

    # save history to file (JSON + CSV) for later use
    import json, csv
    with open(OUT / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(OUT / 'history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # only per-epoch keys (lists); skip scalar train_time_min
        keys = [k for k, v in history.items() if isinstance(v, list)]
        writer.writerow(keys)
        for row in zip(*[history[k] for k in keys]):
            writer.writerow(row)

    print(f"Trening zakończony w {train_time_min:.2f} min. Najlepsza Val Loss: {best_val_loss:.4f} | Najlepszy LPIPS: {best_val_lpips:.4f}")
    print("Historia zapisana do '" + str(OUT / 'history.json') + "' i '" + str(OUT / 'history.csv') + "'")


    # TEST
    # load the best model (based on val loss) and evaluate on the UNSEEN TEST set.
    # validate_and_compare calculates metrics in the hole (PSNR/SSIM/LPIPS/SNE), win-rate and full-image view.
    model.load_state_dict(torch.load(OUT / 'unet_best.pth', map_location=device))

    print('=========== FINALNY TEST (zbior TEST) ===========')
    test_metrics = validate_and_compare(model, test_loader, device, psnr, ssim, lpips_vgg, tag="TEST")

    # save full test results to file (for report)
    import json
    test_results = {
        'seed': SEED,
        'num_params': int(NUM_PARAMS),
        'epochs': EPOCHS,
        'hole_size': hole_size,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'lr_start': optimizer.defaults.get('lr'),
        'train_time_min': history.get('train_time_min') if 'history' in dir() else None,
        'test': test_metrics,
    }
    with open(OUT / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print('Wyniki testu zapisane do:' + str(OUT / 'test_results.json'))

    # =========================  test artficats =========================
    # save the whole set, to ensure fully reproducible results without running the code.
    # for each test image we save: GT, input with hole, mask and both models (UNet and Telea).
    # test masks are deterministic, but we save the whole set to ensure fully reproducible results without running the code.
    import os

    ART_DIR = OUT / 'test_artifacts'
    for sub in ['gt', 'masked', 'mask', 'unet', 'unet_raw', 'telea']:
        os.makedirs(os.path.join(ART_DIR, sub), exist_ok=True)


    def _save_png(tensor_chw, path):
        arr = np.clip(np.round(tensor_chw.cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        Image.fromarray(arr).save(path)


    model.eval()
    sample_idx = 0
    with torch.no_grad():
        for masked_img, mask, gt_img in test_loader:
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            gt_img = gt_img.to(device)

            raw_pred = model(masked_img, mask)
            unet_comp = gt_img * (1.0 - mask) + raw_pred * mask

            for i in range(masked_img.size(0)):
                # Telea (per obraz)
                m_uint8 = (mask[i, 0].cpu().numpy() * 255).astype(np.uint8)
                img_np = masked_img[i].cpu().permute(1, 2, 0).numpy()
                img_bgr = cv2.cvtColor(np.clip(np.round(img_np * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                res_bgr = cv2.inpaint(img_bgr, m_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                telea_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
                telea_t = torch.from_numpy(telea_rgb).permute(2, 0, 1).float() / 255.0

                name = f"{sample_idx:05d}.png"
                _save_png(gt_img[i], os.path.join(ART_DIR, 'gt', name))
                _save_png(masked_img[i], os.path.join(ART_DIR, 'masked', name))
                _save_png(mask[i], os.path.join(ART_DIR, 'mask', name))
                _save_png(unet_comp[i], os.path.join(ART_DIR, 'unet', name))
                _save_png(raw_pred[i], os.path.join(ART_DIR, 'unet_raw', name))  # surowe wyjscie sieci (bez kompozycji)
                _save_png(telea_t, os.path.join(ART_DIR, 'telea', name))
                sample_idx += 1

    print(f"Zapisano artefakty dla {sample_idx} obrazów do: {ART_DIR}")
    print("  podfoldery: gt / masked / mask / unet / unet_raw / telea")


    from SIGK.figures_inpainting import plot_training_history, show_test_examples
    plot_training_history(history)
    show_test_examples(model, test_dataset, device)

if __name__ == "__main__":
    main()