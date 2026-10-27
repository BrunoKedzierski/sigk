from pathlib import Path

import httpx
from tqdm import tqdm

from SIGK.directories import DATA_DIR

TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VAL_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

OUTPUT_DIR = Path(DATA_DIR / "raw" / "DIV2K")
ARCHIVES = (
    ("DIV2K_train_HR.zip", TRAIN_URL),
    ("DIV2K_valid_HR.zip", VAL_URL),
)


def download_file():
    existing_archives = [
        OUTPUT_DIR / filename
        for filename, _ in ARCHIVES
        if (OUTPUT_DIR / filename).exists()
    ]
    if existing_archives:
        paths = "\n".join(str(path) for path in existing_archives)
        raise SystemExit(
            "DIV2K archive destination already contains files:\n"
            f"{paths}\n"
            f"Clear {OUTPUT_DIR} before running the download command."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for zip_filename, url in ARCHIVES:
        zip_path = OUTPUT_DIR / zip_filename

        partial_path = zip_path.with_suffix(zip_path.suffix + ".part")
        print(f"Downloading {zip_filename}...")
        try:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))

                with partial_path.open("wb") as file, tqdm(
                    total=total, unit_scale=True, unit_divisor=1024, unit="B"
                ) as progress:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
                        progress.update(len(chunk))

            partial_path.replace(zip_path)
        finally:
            partial_path.unlink(missing_ok=True)

        print(f"Download completed: {zip_path}")

def main():
    download_file()


if __name__ == "__main__":
    main()
    # print(f"Data directory: {DATA_DIR}")
    # print(f"Output directory: {OUTPUT_DIR}")
    # print(f"Zip file path: {ZIP_PATH}")