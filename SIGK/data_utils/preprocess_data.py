from collections import defaultdict
from pathlib import Path
import random
import shutil
from zipfile import ZipFile

from torchvision import io
from torchvision.transforms import v2
from tqdm import tqdm

from SIGK.directories import DATA_DIR

FILES = {
    "train": {
        "zip_path": DATA_DIR / "raw" / "DIV2K" / "DIV2K_train_HR.zip",
        "extract_dir": DATA_DIR / "processed" / "DIV2K" / "train",
        "cropped_dir": DATA_DIR / "processed" / "DIV2K" / "train_cropped_x256",
        "cropped_resized_x32_dir": DATA_DIR / "processed" / "DIV2K" / "train_cropped_x256_resized_x32",
        "cropped_resized_x64_dir": DATA_DIR / "processed" / "DIV2K" / "train_cropped_x256_resized_x64",
    },
    "valid": {
        "zip_path": DATA_DIR / "raw" / "DIV2K" / "DIV2K_valid_HR.zip",
        "zip_extract_dir": DATA_DIR / "processed" / "DIV2K" / "valid",
        "cropped_x256_dir": DATA_DIR / "processed" / "DIV2K" / "valid_cropped_x256",
        "cropped_resized_x32_dir": DATA_DIR / "processed" / "DIV2K" / "valid_cropped_x256_resized_x32",
        "cropped_resized_x64_dir": DATA_DIR / "processed" / "DIV2K" / "valid_cropped_x256_resized_x64",
    
    },
    "test": {
        "test_dir_32": DATA_DIR / "processed" / "DIV2K" / "test_cropped_x256_resized_x32",
        "test_dir_64": DATA_DIR / "processed" / "DIV2K" / "test_cropped_x256_resized_x64",
        "test_dir_hr": DATA_DIR / "processed" / "DIV2K" / "test_cropped_x256",
    }
}





def unzip_files(zip_path, extract_dir, split):
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        extract_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"Extracted {split} data to {extract_dir}")

def crop_data(original_dir,output_dir, crop_size, crop_num):

  RandCrop = v2.RandomCrop(size=(crop_size, crop_size))
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  image_paths = [p for p in Path(original_dir).rglob("*") if p.is_file()]
  for p in tqdm(image_paths, desc="Cropping images", unit="image"):
    img_tensor = io.read_image(str(p))
    for i in range(crop_num):
      out_image = RandCrop(img_tensor)
      io.write_png(input=out_image, filename=f"{output_dir}/{p.stem}_crop_x{crop_size}_{i}.png")


def resize_data(original_dir,output_dir, target_size):

  resize = v2.Resize(size=(target_size, target_size))
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  image_paths = [p for p in Path(original_dir).rglob("*") if p.is_file()]
  for p in tqdm(image_paths, desc="Resizing images", unit="image"):
    img_tensor = io.read_image(str(p))
    out_image = resize(img_tensor)
    io.write_png(
        input=out_image,
        filename=f"{output_dir}/{p.stem!s}_resized_x{target_size}.png",
    )


def build_test_from_val(gt_dir, lr32_dir, lr64_dir, test_dir_32, test_dir_64, test_dir_hr):
    groups = defaultdict(list)

    for hr_path in sorted(gt_dir.glob("*.png")):
        # 0801_crop_x256_2.png -> 0801
        original_id = hr_path.stem.split("_crop_x256_")[0]
        groups[original_id].append(hr_path)

    print("Original_images", len(groups))
    print("Crop number:", sum(len(crops) for crops in groups.values()))

    pairs = []

    for original_id, crops in groups.items():
        for hr_path in crops:
            stem = hr_path.stem

            lr32_path = lr32_dir / f"{stem}_resized_x32.png"
            lr64_path = lr64_dir / f"{stem}_resized_x64.png"

            if not lr32_path.exists():
                raise FileNotFoundError(f"Missing: {lr32_path}")

            if not lr64_path.exists():
                raise FileNotFoundError(f"Missing: {lr64_path}")

            pairs.append((original_id, hr_path, lr32_path, lr64_path))

    assert len(pairs) == len(list(lr32_dir.glob("*.png")))
    assert len(pairs) == len(list(lr64_dir.glob("*.png")))

    for folder in [test_dir_32, test_dir_64, test_dir_hr]:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(
                f"Folder {folder} is already populated. "
                "Abort ."
            )

    original_ids = sorted(groups.keys())

    rng = random.Random(42)
    rng.shuffle(original_ids)

    n_val = round(len(original_ids) * 0.20)

    val_ids = set(original_ids[:n_val])
    test_ids = set(original_ids[n_val:])

    print(f"Validation: {len(val_ids)} original imgs")
    print(f"Test: {len(test_ids)} original imgs")

    for folder in [test_dir_32, test_dir_64, test_dir_hr]:
        folder.mkdir(parents=True, exist_ok=True)

    moved = 0

    for original_id, hr_path, lr32_path, lr64_path in tqdm(
        pairs, desc="Building test split", unit="image"
    ):
        if original_id not in test_ids:
            continue

        shutil.move(str(hr_path), str(test_dir_hr / hr_path.name))
        shutil.move(str(lr32_path), str(test_dir_32 / lr32_path.name))
        shutil.move(str(lr64_path), str(test_dir_64 / lr64_path.name))

        moved += 1

    
    print("\nMoved files to test set:")
    print(f"Moved to test: {moved} crops in each folder")








def main():
    train = FILES["train"]
    unzip_files(train["zip_path"], train["extract_dir"], "train")
    crop_data(train["extract_dir"], train["cropped_dir"], crop_size=256, crop_num=5)
    resize_data(train["cropped_dir"], train["cropped_resized_x32_dir"], target_size=32)
    resize_data(train["cropped_dir"], train["cropped_resized_x64_dir"], target_size=64)

    valid = FILES["valid"]
    unzip_files(valid["zip_path"], valid["zip_extract_dir"], "valid")
    crop_data(valid["zip_extract_dir"], valid["cropped_x256_dir"], crop_size=256, crop_num=5)
    resize_data(valid["cropped_x256_dir"], valid["cropped_resized_x32_dir"], target_size=32)
    resize_data(valid["cropped_x256_dir"], valid["cropped_resized_x64_dir"], target_size=64)

    test = FILES["test"]
    build_test_from_val(
        gt_dir=valid["cropped_x256_dir"],
        lr32_dir=valid["cropped_resized_x32_dir"],
        lr64_dir=valid["cropped_resized_x64_dir"],
        test_dir_32=test["test_dir_32"],
        test_dir_64=test["test_dir_64"],
        test_dir_hr=test["test_dir_hr"],
    )


if __name__ == "__main__":
    main()