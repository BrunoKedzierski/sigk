from pathlib import Path

import httpx
from tqdm import tqdm

from SIGK.directories import DATA_DIR

TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VAL_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

OUTPUT_DIR = Path(DATA_DIR / "raw" / "DIV2K")


def download_file():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for zip_filename, url in [("DIV2K_train_HR.zip", TRAIN_URL), ("DIV2K_valid_HR.zip", VAL_URL)]:
        zip_path = OUTPUT_DIR / zip_filename

   
    
        print("Starting download...")
        with httpx.stream("GET", url, follow_redirects=True) as response:
            total = int(response.headers.get("Content-Length", 0))
        
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                    progress.update(len(chunk))
                    
    print(f" Download completed: {zip_path}")

def main():
    download_file()


if __name__ == "__main__":
    main()
    # print(f"Data directory: {DATA_DIR}")
    # print(f"Output directory: {OUTPUT_DIR}")
    # print(f"Zip file path: {ZIP_PATH}")