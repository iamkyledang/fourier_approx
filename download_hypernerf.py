"""
Download and extract the HyperNeRF interp subset dataset.
All scenes are downloaded from the official GitHub release and extracted
into a single 'hypernerf_interp/' directory.

Usage:
    python download_hypernerf.py [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import urllib.request
import zipfile

BASE_URL = "https://github.com/google/hypernerf/releases/download/v0.1"

# Each entry: (zip_suffix, final_scene_dir)
# zip URL: {BASE_URL}/interp_{zip_suffix}.zip
# final_scene_dir: name used by training/render scripts
INTERP_SCENES = [
    ("aleks-teapot", "aleks-teapot"),
    ("chickchicken",  "chickchicken"),
    ("cut-lemon",     "cut-lemon1"),      # interp_cut-lemon.zip → cut-lemon1
    ("hand",          "hand1-dense-v2"),  # interp_hand.zip      → hand1-dense-v2
    ("slice-banana",  "slice-banana"),
    ("torchocolate",  "torchocolate"),
]


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with a simple progress indicator."""
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            mb_done = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def download_scene(zip_suffix: str, scene_dir: str, output_dir: str) -> None:
    final_path = os.path.join(output_dir, scene_dir)
    if os.path.isdir(final_path):
        print(f"[skip] '{scene_dir}' already exists at {final_path}")
        return

    zip_name = f"interp_{zip_suffix}.zip"
    url = f"{BASE_URL}/{zip_name}"
    zip_path = os.path.join(output_dir, zip_name)

    print(f"[download] {scene_dir}  ({zip_name})")
    print(f"  URL: {url}")
    try:
        download_with_progress(url, zip_path)
    except Exception as e:
        print(f"  ERROR downloading {zip_name}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return

    print(f"[extract] {zip_name} -> {output_dir}/")
    before = set(os.listdir(output_dir))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
    except zipfile.BadZipFile as e:
        print(f"  ERROR extracting {zip_name}: {e}")
        os.remove(zip_path)
        return
    os.remove(zip_path)

    # Detect any newly created directory and rename to the expected scene name
    after = set(os.listdir(output_dir))
    new_dirs = [d for d in (after - before) if os.path.isdir(os.path.join(output_dir, d))]
    if new_dirs and new_dirs[0] != scene_dir:
        extracted = os.path.join(output_dir, new_dirs[0])
        os.rename(extracted, final_path)
        print(f"[rename]  {new_dirs[0]} -> {scene_dir}")

    print(f"[done]    {scene_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Download HyperNeRF interp subset dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hypernerf_interp",
        help="Directory to download and extract datasets into (default: hypernerf_interp/)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")

    for zip_suffix, scene_dir in INTERP_SCENES:
        download_scene(zip_suffix, scene_dir, output_dir)

    print("All scenes processed.")


if __name__ == "__main__":
    main()
