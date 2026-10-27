from SIGK.data_utils.preprocess_data import FILES

IMAGE_DIRECTORIES = {
    "train": {
        "256": FILES["train"]["cropped_dir"],
        "32": FILES["train"]["cropped_resized_x32_dir"],
        "64": FILES["train"]["cropped_resized_x64_dir"],
    },
    "valid": {
        "256": FILES["valid"]["cropped_x256_dir"],
        "32": FILES["valid"]["cropped_resized_x32_dir"],
        "64": FILES["valid"]["cropped_resized_x64_dir"],
    },
    "test": {
        "256": FILES["test"]["test_dir_hr"],
        "32": FILES["test"]["test_dir_32"],
        "64": FILES["test"]["test_dir_64"],
    },
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def count_images(directory):
    if not directory.is_dir():
        return None

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    print(f"{'Split':<8} {'256px':>10} {'32px':>10} {'64px':>10}")
    print("-" * 42)

    for split, sizes in IMAGE_DIRECTORIES.items():
        counts = []
        for directory in sizes.values():
            count = count_images(directory)
            counts.append(str(count) if count is not None else "missing")

        print(f"{split:<8} {counts[0]:>10} {counts[1]:>10} {counts[2]:>10}")


if __name__ == "__main__":
    main()
