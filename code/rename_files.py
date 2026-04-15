import os
import re
from pathlib import Path


def rename_files_with_offset(input_folder: str, output_folder: str, offset: int) -> None:
    """
    Rename files from 'Number_suffix.ext' format by adding `offset` to the number.

    Args:
        input_folder:  Path to the folder containing the original files.
        output_folder: Path to the folder where renamed files will be saved.
        offset:        Integer to add to each file's number prefix.
    """
    os.makedirs(output_folder, exist_ok=True)

    pattern = re.compile(r'^(\d+)_(.+)$')

    for filename in os.listdir(input_folder):
        src_path = os.path.join(input_folder, filename)

        if not os.path.isfile(src_path):
            continue  # skip subdirectories

        name, ext = os.path.splitext(filename)   # e.g. "1_zed", ".png"
        match = pattern.match(name)

        if not match:
            print(f"[SKIP] Does not match expected pattern: {filename}")
            continue

        number  = int(match.group(1))            # 1
        suffix  = match.group(2)                 # "zed"
        new_name = f"{number + offset}_{suffix}{ext}"   # "41_zed.png"

        dst_path = os.path.join(output_folder, new_name)
        shutil.copy2(src_path, dst_path)         # copy; use os.rename to move
        print(f"[OK] {filename} → {new_name}")


# ── Example usage ────────────────────────────────────────────────────────────
import shutil

parent_dir = Path(__file__).resolve().parents[2]
date = "13042026"
dataset_dir_source = parent_dir / "datasets" / f"dataset_{date}_2"
source_dir = dataset_dir_source / "stereo_4k_depth" / "depth"
out_dir = parent_dir / "datasets" / f"dataset_{date}_2_shifted" / "depth"
debug = 0

rename_files_with_offset(
    input_folder  = str(source_dir),
    output_folder = str(out_dir),
    offset        = 100
)