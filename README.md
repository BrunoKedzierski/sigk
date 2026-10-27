# SIGK

SIGK is a PyTorch project for training and evaluating image super-resolution
models on DIV2K. It downloads the DIV2K high-resolution images, prepares
256x256 crops and 32x32/64x64 low-resolution inputs, and runs experiments with
three model architectures.

## Project structure

```text
SIGK/
├── data/
│   ├── external/                  # Data from external sources
│   ├── interim/                   # Intermediate data
│   ├── processed/
│   │   └── DIV2K/                  # Extracted images, crops and resized splits
│   └── raw/
│       └── DIV2K/                  # Downloaded DIV2K archives
├── experiments/
│   └── superresolution/            # Training run outputs
├── reports/
│   └── figures/
│       └── super_resolution/       # Generated experiment figures
├── SIGK/
│   ├── data_utils/
│   │   ├── download_div2k.py       # Download DIV2K train and validation archives
│   │   ├── preprocess_data.py      # Extract, crop, resize, and create test split
│   │   └── summarize_images.py     # Count images by split and resolution
│   ├── directories.py              # Project data and output paths
│   ├── engine.py                   # Training, validation, and experiment orchestration
│   ├── figures.py                  # Plot and comparison figure generation
│   ├── metrics.py                  # Image quality metrics
│   ├── model_utils.py              # Serialization, data loaders, and helpers
│   ├── superresolution_dataset.py  # PyTorch dataset for paired LR/HR images
│   ├── superresolution_runner.py  # Dataset, model, and training configuration
│   └── supperresolution_models.py  # BaseNet, Netv2, and Netv3 architectures
├── LICENSE
├── Makefile
├── pyproject.toml                  # Package metadata, dependencies, and commands
├── README.md
└── uv.lock                         # Locked dependency versions
```

The `data/`, `experiments/`, and `reports/` directories contain generated or
downloaded files. The DIV2K preprocessing script writes the train, validation,
and test images beneath `data/processed/DIV2K/`.

## Setup

From the project root, sync the environment and install the project commands:

```powershell
uv sync
```

## Commands

Run these commands from the project root:

```powershell
uv run download_div2k
uv run preprocess_data
uv run summarize_images
uv run superresolution_train
```

Run the commands in order: download the archives, preprocess the dataset, then
train. `summarize_images` reports image counts for each split at 256px, 32px,
and 64px. Training outputs are written to `experiments/superresolution/`, with
training curves, comparisons, and sample grids saved alongside each experiment.
