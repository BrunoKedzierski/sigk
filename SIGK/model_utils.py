import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save_json(path: Path, obj: object) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding='utf-8',
    )


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset, batch_size, shuffle, num_workers, device, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=(num_workers > 0),
    )

def check_dataset(dataset, input_size, split):
    if len(dataset) == 0:
        raise ValueError(f'Empty {split} dataset x{input_size}. Check paths!')
    lr, gt = dataset[0]
    if tuple(lr.shape) != (3, input_size, input_size):
        raise ValueError(f'{split} x{input_size}: unexpected LR shape {lr.shape}')
    if tuple(gt.shape) != (3, 256, 256):
        raise ValueError(f'{split} x{input_size}: unexpected GT shape {gt.shape}')
    if len(dataset.tr_files) != len(dataset):
        raise ValueError('Dataset .tr_files must match dataset indices.')

def rgb_numpy(x):
    return x.detach().float().cpu().permute(1, 2, 0).numpy()



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
