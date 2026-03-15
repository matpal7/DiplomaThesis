from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.image import get_undistort_functions
from utils import load_dict


def _read_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    try:
        K = fs.getNode("K").mat()
        D = fs.getNode("D").mat()
    finally:
        fs.release()

    if K is None or D is None:
        raise ValueError(f"Calibration file {path} must contain nodes 'K' and 'D'.")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    return K, D


def _validate_transform(T: np.ndarray, path: Path) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Transform matrix in {path} must have shape (4, 4), got {T.shape}.")
    return T


def _read_relative_pose(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        matrix_keys = (
            "T_cam2_cam1",
            "T_right_left",
            "T_right_from_left",
            "T_cam1_cam2",
        )
        key = next((k for k in matrix_keys if k in data), None)
        if key is None:
            raise ValueError(f"Could not find a 4x4 transform matrix in JSON file: {path}")

        T = np.asarray(data[key], dtype=np.float64)
        if key == "T_cam1_cam2":
            T = np.linalg.inv(T)

        return _validate_transform(T, path)

    if suffix in {".yml", ".yaml"}:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open relative pose file: {path}")

        try:
            T = fs.getNode("T_cam2_cam1").mat()
            if T is None:
                T_12 = fs.getNode("T_cam1_cam2").mat()
                if T_12 is None:
                    raise ValueError(f"Could not find 'T_cam2_cam1' or 'T_cam1_cam2' in: {path}")
                T = np.linalg.inv(T_12)
        finally:
            fs.release()

        return _validate_transform(T, path)

    raise ValueError(f"Unsupported relative pose file format: {path}")


def _find_lr_pairs(image_dir: Path, left_suffix: str, right_suffix: str) -> list[tuple[Path, Path]]:
    left_paths = sorted(image_dir.glob(f"*{left_suffix}"))
    right_paths = sorted(image_dir.glob(f"*{right_suffix}"))

    by_stem_left = {p.name[: -len(left_suffix)]: p for p in left_paths}
    by_stem_right = {p.name[: -len(right_suffix)]: p for p in right_paths}

    common_keys = sorted(set(by_stem_left.keys()) & set(by_stem_right.keys()), key=lambda x: int(x) if x.isdigit() else x)
    pairs = [(by_stem_left[k], by_stem_right[k]) for k in common_keys]

    if not pairs:
        raise FileNotFoundError(
            f"No left/right pairs found in {image_dir} with suffixes {left_suffix!r}, {right_suffix!r}."
        )

    return pairs


def _draw_epi_lines(img: np.ndarray, step: int = 50) -> np.ndarray:
    vis = img.copy()
    h = vis.shape[0]
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rectify stereo pairs using intrinsics + relative pose (T_cam2_cam1).")
    p.add_argument("--image-dir", type=Path, required=True, help="Directory with original stereo images.")
    p.add_argument("--left-suffix", type=str, default="_left.png", help="Suffix for left camera images.")
    p.add_argument("--right-suffix", type=str, default="_right.png", help="Suffix for right camera images.")

    p.add_argument("--left-calib", type=Path, required=True, help="Left camera calibration YAML containing K,D.")
    p.add_argument("--right-calib", type=Path, required=True, help="Right camera calibration YAML containing K,D.")
    p.add_argument("--relative-pose", type=Path, required=True, help="Relative pose file with T_cam2_cam1.")
    p.add_argument(
        "--invert-relative-pose",
        action="store_true",
        help="Invert loaded transform before use (when your file stores opposite direction).",
    )

    p.add_argument("--alpha", type=float, default=0.0, help="StereoRectify alpha (0=crop, 1=keep all).")
    p.add_argument(
        "--new-width",
        type=int,
        default=None,
        help="Optional output width. If omitted, input width is used.",
    )
    p.add_argument(
        "--new-height",
        type=int,
        default=None,
        help="Optional output height. If omitted, input height is used.",
    )

    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for rectified pairs.")
    p.add_argument(
        "--save-preview",
        action="store_true",
        help="Save side-by-side preview images with horizontal epipolar lines.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parent_dir = Path(__file__).resolve().parent.parent
    dataset_dir = parent_dir / "dataset_11032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir / "out" / "cameras_parameters"
    debug = 2

    calib_dict = load_dict(out_dir / "calib_data.npy")

    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=False)


    pairs = _find_lr_pairs(args.image_dir, args.left_suffix, args.right_suffix)
    left0 = cv2.imread(str(pairs[0][0]), cv2.IMREAD_COLOR)
    right0 = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if left0 is None or right0 is None:
        raise FileNotFoundError("Cannot read first pair of images.")
    if left0.shape[:2] != right0.shape[:2]:
        raise ValueError(f"Left/right image sizes differ: {left0.shape[:2]} vs {right0.shape[:2]}")

    in_h, in_w = left0.shape[:2]
    out_w = args.new_width if args.new_width is not None else in_w
    out_h = args.new_height if args.new_height is not None else in_h

    K1, D1 = _read_camera_calibration(args.left_calib)
    K2, D2 = _read_camera_calibration(args.right_calib)
    T_cam2_cam1 = _read_relative_pose(args.relative_pose)
    if args.invert_relative_pose:
        T_cam2_cam1 = np.linalg.inv(T_cam2_cam1)

    R = T_cam2_cam1[:3, :3]
    t = T_cam2_cam1[:3, 3].reshape(3, 1)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=(in_w, in_h),
        R=R,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=args.alpha,
        newImageSize=(out_w, out_h),
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (out_w, out_h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (out_w, out_h), cv2.CV_32FC1)

    out_left = args.out_dir / "left"
    out_right = args.out_dir / "right"
    out_preview = args.out_dir / "preview"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        out_preview.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Pairs found: {len(pairs)}")
    print(f"Input size: {in_w}x{in_h}")
    print(f"Output size: {out_w}x{out_h}")
    print(f"ROI left: {roi1}, ROI right: {roi2}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 72)

    for left_path, right_path in pairs:
        stem = left_path.name[: -len(args.left_suffix)]

        img_l = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        img_r = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            print(f"[WARN] Skipping pair {left_path.name}, {right_path.name} (cannot read).")
            continue

        rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

        out_l_path = out_left / f"{stem}_left_rect.png"
        out_r_path = out_right / f"{stem}_right_rect.png"
        cv2.imwrite(str(out_l_path), rect_l)
        cv2.imwrite(str(out_r_path), rect_r)

        if args.save_preview:
            cat = np.hstack([rect_l, rect_r])
            cat = _draw_epi_lines(cat, step=max(30, out_h // 15))
            cv2.imwrite(str(out_preview / f"{stem}_preview.png"), cat)

    # np.savez(
    #     str(args.out_dir / "stereo_rectify_data.npz"),
    #     R1=R1,
    #     R2=R2,
    #     P1=P1,
    #     P2=P2,
    #     Q=Q,
    #     roi1=np.array(roi1),
    #     roi2=np.array(roi2),
    #     map1x=map1x,
    #     map1y=map1y,
    #     map2x=map2x,
    #     map2y=map2y,
    # )
    print(f"Saved rectification data: {args.out_dir / 'stereo_rectify_data.npz'}")


if __name__ == "__main__":
    main()
