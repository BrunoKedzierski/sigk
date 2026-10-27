from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] 

DATA_DIR = ROOT / "data"

SUPERRESOLUTION_EXPERIMENTS_DIR = ROOT / "experiments" / "superresolution"
PROCESSED_DIV2K_DIR = DATA_DIR / "processed" / "DIV2K"

TEST_CROPPED_x256_DIR = PROCESSED_DIV2K_DIR  / "test_cropped_x256"
TRAIN_CROPPED_x256_DIR = PROCESSED_DIV2K_DIR  / "train_cropped_x256"
VALID_CROPPED_x256_DIR = PROCESSED_DIV2K_DIR  / "valid_cropped_x256"

TEST_CROPPED_x256_RESIZED_x32_DIR = PROCESSED_DIV2K_DIR  / "test_cropped_x256_resized_x32"
TRAIN_CROPPED_x256_RESIZED_x32_DIR = PROCESSED_DIV2K_DIR  / "train_cropped_x256_resized_x32"
VALID_CROPPED_x256_RESIZED_x32_DIR = PROCESSED_DIV2K_DIR  / "valid_cropped_x256_resized_x32"

TEST_CROPPED_x256_RESIZED_x64_DIR = PROCESSED_DIV2K_DIR  / "test_cropped_x256_resized_x64"
TRAIN_CROPPED_x256_RESIZED_x64_DIR = PROCESSED_DIV2K_DIR  / "train_cropped_x256_resized_x64"
VALID_CROPPED_x256_RESIZED_x64_DIR = PROCESSED_DIV2K_DIR  / "valid_cropped_x256_resized_x64"

