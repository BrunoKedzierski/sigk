

import torchvision.io as io
from pathlib import Path
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    def __init__(self, gt_dir, tr_dir, input_img_size):
        self.gt_dir = Path(gt_dir)
        self.tr_dir = Path(tr_dir)
        self.input_img_size = input_img_size
        self.gt_files = sorted(self.gt_dir.glob("*.png"))
        self.tr_files = sorted(self.tr_dir.glob("*.png"))

    def __len__(self):
        return len(self.tr_files)

    def __getitem__(self, idx):

          # Przykład: 0001_crop_x256_0_resized_x32.png
          tr_path = self.tr_files[idx]

          # 0001_crop_x256_0_resized_x32.png -> 0001_crop_x256.png
          gt_name = tr_path.stem.removesuffix(f"_resized_x{self.input_img_size}") + ".png"

          # Ścieżka do obrazu referencyjnego
          gt_path = self.gt_dir / gt_name

          # Wczytanie obrazów
          tr_img_tensor = io.read_image(tr_path) / 255
          gt_img_tensor = io.read_image(gt_path) / 255


          # Zamiana na tensory
          return tr_img_tensor, gt_img_tensor