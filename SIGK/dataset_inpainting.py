import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class InpaintingDataset(Dataset):
    def __init__(self, images_dir, image_size=256, hole_size=3, is_train=True):
        self.image_paths = [
            os.path.join(images_dir, file_name)
            for file_name in os.listdir(images_dir)
            if file_name.lower().endswith((".png", ".jpeg", ".jpg"))
        ]
        self.hole_size = hole_size
        self.is_train = is_train
        
        if self.is_train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor()
            ])
        else:
            # val/test, deterministic
            self.transform = T.Compose([
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def _generate_mask(self, h, w, idx):
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        max_x = w - self.hole_size
        max_y = h - self.hole_size

        if self.is_train:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        else:
            # val/test, deterministic via idx seed
            rng = random.Random(idx)
            x = rng.randint(0, max_x)
            y = rng.randint(0, max_y)

        mask[0, y : y + self.hole_size, x : x + self.hole_size] = 1.0
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        gt_img = self.transform(img)

        _, h, w = gt_img.shape
        mask = self._generate_mask(h, w, idx)
        masked_img = gt_img * (1.0 - mask)

        return masked_img, mask, gt_img