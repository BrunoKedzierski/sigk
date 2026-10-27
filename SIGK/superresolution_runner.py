from SIGK.directories import (
    SUPERRESOLUTION_EXPERIMENTS_DIR,
    TEST_CROPPED_x256_DIR,
    TEST_CROPPED_x256_RESIZED_x32_DIR,
    TEST_CROPPED_x256_RESIZED_x64_DIR,
    TRAIN_CROPPED_x256_DIR,
    TRAIN_CROPPED_x256_RESIZED_x32_DIR,
    TRAIN_CROPPED_x256_RESIZED_x64_DIR,
    VALID_CROPPED_x256_DIR,
    VALID_CROPPED_x256_RESIZED_x32_DIR,
    VALID_CROPPED_x256_RESIZED_x64_DIR,
)
from SIGK.engine import run_all_experiments
from SIGK.superresolution_dataset import SuperResolutionDataset
from SIGK.supperresolution_models import BaseNet, Netv2, Netv3

train_dataset_x32 = SuperResolutionDataset(
    gt_dir=TRAIN_CROPPED_x256_DIR,
    tr_dir=TRAIN_CROPPED_x256_RESIZED_x32_DIR,
    input_img_size = 32
)


train_dataset_x64 = SuperResolutionDataset(
    gt_dir=TRAIN_CROPPED_x256_DIR,
    tr_dir=TRAIN_CROPPED_x256_RESIZED_x64_DIR,
    input_img_size = 64
)

val_dataset_x32 = SuperResolutionDataset(
    gt_dir=VALID_CROPPED_x256_DIR,
    tr_dir=VALID_CROPPED_x256_RESIZED_x32_DIR,
    input_img_size = 32
)


val_dataset_x64 = SuperResolutionDataset(
    gt_dir=VALID_CROPPED_x256_DIR,
    tr_dir=VALID_CROPPED_x256_RESIZED_x64_DIR,
    input_img_size = 64
)



test_dataset_x32 = SuperResolutionDataset(
    gt_dir=TEST_CROPPED_x256_DIR,
    tr_dir=TEST_CROPPED_x256_RESIZED_x32_DIR,
    input_img_size = 32
)


test_dataset_x64 = SuperResolutionDataset(
    gt_dir=TEST_CROPPED_x256_DIR,
    tr_dir=TEST_CROPPED_x256_RESIZED_x64_DIR,
    input_img_size = 64
)






MODELS = {'BaseNet': BaseNet, 'Netv2': Netv2, 'Netv3': Netv3}




DATASETS = {
    32: {
        "train": train_dataset_x32,
        "val": val_dataset_x32,
        "test": test_dataset_x32,
    },
    64: {
        "train": train_dataset_x64,
        "val": val_dataset_x64,
        "test": test_dataset_x64,
    },
}

CONFIG = {
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "val_every": 2,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_threshold": 0.003,
    "scheduler_thresh_mode": "rel",
    "seed": 42,
    "min_lr": 1e-5,
    "num_workers": 2,
    "examples_per_experiment": 6,
    "output_dir": SUPERRESOLUTION_EXPERIMENTS_DIR / "ep30_b32_lr1e3",
}


def main():
    print("Super-resolution runner script executed.")
    run_all_experiments(DATASETS, CONFIG, MODELS)


if __name__ == "__main__":
    main()