from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import MIN_COMMON_CORNERS, estimate_relative_pose
from prepare_paths import prepare_relative_pose_paths


# keep your existing imports
# import cv2
# import numpy as np
# ...
# from prepare_paths import prepare_relative_pose_paths


@dataclass
class RelativePoseRunConfig:
    parent_dir: Path
    date: str
    cam2_suffix: str = "left"
    debug: int = 0
    squares_horizontally: int = 6
    squares_vertically: int = 8
    squares_length: float = 45.0
    marker_length: float = 31.0
    min_common_corners: int = MIN_COMMON_CORNERS
    use_pair_weights: bool = True
    max_imgs: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate relative pose between RGBD cameras and a reference camera."
    )
    parser.add_argument(
        "--date",
        type=str,
        default="28032026",
        help="Dataset date suffix.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug visualization level.",
    )
    parser.add_argument(
        "--max-imgs",
        type=int,
        default=None,
        help="Maximum number of synchronized image pairs to use.",
    )
    parser.add_argument(
        "--squares-horizontally",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--squares-vertically",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--squares-length",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=31.0,
    )
    return parser.parse_args()


def build_default_run_config(args: argparse.Namespace) -> RelativePoseRunConfig:
    parent_dir = Path(__file__).resolve().parents[3]

    return RelativePoseRunConfig(
        parent_dir=parent_dir,
        date=args.date,
        cam2_suffix="left",
        debug=args.debug,
        squares_horizontally=args.squares_horizontally,
        squares_vertically=args.squares_vertically,
        squares_length=args.squares_length,
        marker_length=args.marker_length,
        min_common_corners=MIN_COMMON_CORNERS,
        use_pair_weights=True,
        max_imgs=args.max_imgs,
    )


def run_relative_pose_for_suffix(
    rgbd_cam_suffix: str,
    config: RelativePoseRunConfig,
) -> Path:
    dataset_dir, relative_pose_dir, out_dir, out_dir_save, calib_dict_stereo, calib_dict_rgbd = (
        prepare_relative_pose_paths(
            config.parent_dir,
            config.date,
            rgbd_cam_suffix,
        )
    )

    cam1_calib = calib_dict_rgbd
    cam2_calib = calib_dict_stereo

    print(f"\n=== Running relative pose for rgbd_cam_suffix='{rgbd_cam_suffix}' ===")
    print(f"Image dir: {relative_pose_dir}")
    print(f"cam1 calib: {cam1_calib}")
    print(f"cam2 calib: {cam2_calib}")
    print(f"output dir: {out_dir_save}")

    estimate_relative_pose(
        image_dir=relative_pose_dir,
        cam1_calib=cam1_calib,
        cam2_calib=cam2_calib,
        output_path=out_dir_save,
        cam1_suffix=rgbd_cam_suffix,
        cam2_suffix=config.cam2_suffix,
        debug=config.debug,
        squares_horizontally=config.squares_horizontally,
        squares_vertically=config.squares_vertically,
        squares_length=config.squares_length,
        marker_length=config.marker_length,
        min_common_corners=config.min_common_corners,
        use_pair_weights=config.use_pair_weights,
        max_imgs=config.max_imgs,
    )

    return out_dir_save


def run_relative_pose_for_suffixes(
    rgbd_cam_suffixes: list[str],
    config: RelativePoseRunConfig,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    for rgbd_cam_suffix in rgbd_cam_suffixes:
        try:
            out_path = run_relative_pose_for_suffix(
                rgbd_cam_suffix=rgbd_cam_suffix,
                config=config,
            )
            outputs[rgbd_cam_suffix] = out_path
        except Exception as exc:
            print(f"[ERROR] Failed for suffix '{rgbd_cam_suffix}': {exc}")

    return outputs


def main() -> None:
    args = parse_args()
    config = build_default_run_config(args)

    outputs = run_relative_pose_for_suffixes(
        rgbd_cam_suffixes=["realsense", "zed"],
        config=config,
    )

    print("\n=== Finished ===")
    for suffix, out_path in outputs.items():
        print(f"{suffix}: {out_path}")


if __name__ == "__main__":
    main()