#!/usr/bin/env python3
"""
NPY → 16-bit Depth PNG for FoundationPose
"""

import numpy as np
import cv2
from pathlib import Path
import argparse


def npy_to_depth_png(npy_path, output_path, max_depth_m=65.535):
    depth_m = np.load(npy_path).astype(np.float32)

    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=max_depth_m, neginf=0.0)

    depth_png = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

    print(depth_png.shape)
    success = cv2.imwrite(str(output_path), depth_png)
    if success:
        print(f"✓ {npy_path} → {output_path} (METERS)")
        print(f"  Shape: {depth_png.shape}, Min/Max: {depth_png.min() / 1000:.3f}-{depth_png.max() / 1000:.3f} m")
        print(f"  NaN count before: {(np.isnan(np.load(npy_path))).sum()}")
    else:
        raise RuntimeError(f"Failed to save {output_path}")
    return depth_png

def batch_convert(input_dir, pattern="*.npy", max_depth_m=10.000):
    input_dir = Path(input_dir)

    npy_files = sorted(input_dir.glob(pattern))
    print(f"Found {len(npy_files)} file(s)")

    if not npy_files:
        print("No .npy files found.")
        return

    output_path = input_dir.parent / "depth_pngs"
    output_path.mkdir(parents=True, exist_ok=True)

    for npy_file in npy_files:
        name = npy_file.stem.replace("_depth", "")
        png_out = output_path / f"{name}.png"

        npy_to_depth_png(npy_file, png_out, max_depth_m=max_depth_m)

    print("Ready for FoundationPose!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPY → 16-bit Depth PNG")
    parser.add_argument('input', nargs='?', help="Input .npy file or directory")
    parser.add_argument('output', nargs='?', help="Output PNG file or directory")
    parser.add_argument('--scale', type=float, default=1000.0, help="Depth scale")
    parser.add_argument('--batch', action='store_true', help="Batch mode")

    args = parser.parse_args()

    # Default paths
    parent_dir = Path(__file__).resolve().parents[3]
    date = "07042026"

    if not args.input:
        args.input = str(
            parent_dir / "datasets" / f'dataset_{date}' / "cameras_downstream_task" / "scene_009" / "depth" )
    if not args.output:
        args.output = str(parent_dir / "out" / f"out_{date}" / "depth_visualized" )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_convert(input_path)
