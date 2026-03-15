from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration.image import get_undistort_functions


def _load_calib_dict(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        calib_dict = data.item()
    elif hasattr(data, "item"):
        try:
            calib_dict = data.item()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Cannot parse calibration dict from {path}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported calibration container in {path}")

    required = ["K_l", "D_l", "xi_l", "K_r", "D_r", "xi_r", "rvec", "tvec", "img_dim_l", "img_dim_r"]
    missing = [k for k in required if k not in calib_dict]
    if missing:
        raise KeyError(f"Calibration dict {path} missing keys: {missing}")

    return calib_dict


def _find_lr_pairs(image_dir: Path, left_suffix: str, right_suffix: str) -> list[tuple[Path, Path]]:
    left_paths = sorted(image_dir.glob(f"*{left_suffix}"))
    right_paths = sorted(image_dir.glob(f"*{right_suffix}"))

    by_stem_left = {p.name[: -len(left_suffix)]: p for p in left_paths}
    by_stem_right = {p.name[: -len(right_suffix)]: p for p in right_paths}
    common = sorted(set(by_stem_left) & set(by_stem_right), key=lambda x: int(x) if x.isdigit() else x)

    pairs = [(by_stem_left[k], by_stem_right[k]) for k in common]
    if not pairs:
        raise FileNotFoundError(
            f"No stereo pairs found in {image_dir} for suffixes {left_suffix!r} and {right_suffix!r}."
        )
    return pairs


def _draw_epi_lines(img: np.ndarray, step: int = 50) -> np.ndarray:
    vis = img.copy()
    for y in range(0, vis.shape[0], step):
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
    return vis





def _pick_new_k(calib: dict, left: bool, use_wide: bool, balance: float) -> np.ndarray:
    base_key = "K_l" if left else "K_r"
    new_key = "new_K_l" if left else "new_K_r"
    wide_key = "new_K_l_wide" if left else "new_K_r_wide"

    if use_wide and wide_key in calib:
        K = np.asarray(calib[wide_key], dtype=np.float64)
    elif new_key in calib:
        K = np.asarray(calib[new_key], dtype=np.float64)
    else:
        K = np.asarray(calib[base_key], dtype=np.float64).copy()

    K[0, 1] = 0.0
    if use_wide and wide_key not in calib and balance > 0:
        # fallback "wider" view by reducing focal length
        K[0, 0] /= (1.0 + balance)
        K[1, 1] /= (1.0 + balance)

    return K

def _as_omnidir_xi(xi_value: np.ndarray | float) -> np.ndarray:
    """OpenCV omnidir expects xi as a 1-element float matrix (total()==1)."""
    xi = np.asarray(xi_value, dtype=np.float64).reshape(-1)
    if xi.size != 1:
        raise ValueError(f"xi must contain exactly one value, got shape={np.asarray(xi_value).shape}")
    return xi.reshape(1, 1)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stereo rectification for omnidirectional cameras using calib_dict from "
            "cv2.omnidir.stereoCalibrate."
        )
    )
    p.add_argument("--calib-dict", type=Path, required=False, help="Path to calib_data.npy (dict with K_l, D_l, xi_l, ...).")
    p.add_argument("--image-dir", type=Path, required=False, help="Directory with stereo images.")
    p.add_argument("--left-suffix", default="_left.png", help="Left image suffix.")
    p.add_argument("--right-suffix", default="_right.png", help="Right image suffix.")
    p.add_argument("--out-dir", type=Path, required=False, default="", help="Output directory for rectified pairs.")
    p.add_argument("--use-wide", action="store_true", help="Use new_K_l_wide/new_K_r_wide if available.")
    p.add_argument("--balance", type=float, default=0.0, help="Scale factor for focal length when wide intrinsics missing.")
    p.add_argument("--save-preview", action="store_true", help="Save side-by-side preview with epipolar lines.")
    return p.parse_args()

def _make_preview_pair(img_l, img_r, preview_size=(680, 480), draw_lines=True, step=30):
    l = cv2.resize(img_l, preview_size)
    r = cv2.resize(img_r, preview_size)
    cat = np.hstack([l, r])
    if draw_lines:
        cat = _draw_epi_lines(cat, step=step)
    return cat

def main() -> None:
    args = parse_args()

    parent_dir = Path(__file__).resolve().parent
    dataset_dir = parent_dir / "dataset_11032026"
    calib_imgs_dir = dataset_dir / "stereo_4k_calibration" / "rgb"
    relative_pose_dir = dataset_dir / "stereo_4k_relative_pose" / "rgb"
    out_dir = parent_dir / "out" / "cameras_parameters"
    debug = 2

    args.calib_dict = out_dir / "calib_data.npy"
    args.image_dir = calib_imgs_dir

    calib = _load_calib_dict(args.calib_dict)
    pairs = _find_lr_pairs(args.image_dir, args.left_suffix, args.right_suffix)

    undistort_l, undistort_r = get_undistort_functions(calib, correct_horizon=False)


    K_l = np.asarray(calib["K_l"], dtype=np.float64)
    D_l = np.asarray(calib["D_l"], dtype=np.float64).reshape(-1)
    xi_l = _as_omnidir_xi(calib["xi_l"])

    K_r = np.asarray(calib["K_r"], dtype=np.float64)
    D_r = np.asarray(calib["D_r"], dtype=np.float64).reshape(-1)
    xi_r = _as_omnidir_xi(calib["xi_r"])

    rvec = np.asarray(calib["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calib["tvec"], dtype=np.float64).reshape(3, 1)
    R_lr, _ = cv2.Rodrigues(rvec)

    img_dim_l = tuple(int(x) for x in np.asarray(calib["img_dim_l"]).reshape(-1)[:2])
    img_dim_r = tuple(int(x) for x in np.asarray(calib["img_dim_r"]).reshape(-1)[:2])
    if img_dim_l != img_dim_r:
        raise ValueError(f"Left/right calibration image dimensions differ: {img_dim_l} vs {img_dim_r}")

    # OpenCV expects imageSize as (width, height)
    img_size = img_dim_l

    new_K_l = _pick_new_k(calib, left=True, use_wide=args.use_wide, balance=args.balance)
    new_K_r = _pick_new_k(calib, left=False, use_wide=args.use_wide, balance=args.balance)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=new_K_l,
        distCoeffs1=np.zeros((4, 1), dtype=np.float64),
        cameraMatrix2=new_K_r,
        distCoeffs2=np.zeros((4, 1), dtype=np.float64),
        imageSize=img_size,
        R=R_lr,
        T=tvec,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map1_l, map2_l = cv2.omnidir.initUndistortRectifyMap(
        K_l,
        D_l,
        xi_l,
        R1,
        P1[:3, :3],
        img_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
    )
    map1_r, map2_r = cv2.omnidir.initUndistortRectifyMap(
        K_r,
        D_r,
        xi_r,
        R2,
        P2[:3, :3],
        img_size,
        cv2.CV_16SC2,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
    )

    out_left = args.out_dir / "left"
    out_right = args.out_dir / "right"
    out_preview = args.out_dir / "preview"

    # out_left.mkdir(parents=True, exist_ok=True)
    # out_right.mkdir(parents=True, exist_ok=True)
    # out_preview.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Pairs: {len(pairs)}")
    print(f"Image size: {img_size}")
    print(f"ROI L={roi1}, ROI R={roi2}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 72)


    for l_path, r_path in pairs:
        stem = l_path.name[: -len(args.left_suffix)]

        img_l = cv2.imread(str(l_path), cv2.IMREAD_COLOR)
        img_r = cv2.imread(str(r_path), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            print(f"[WARN] Skipping unreadable pair {l_path.name}, {r_path.name}")
            continue

        # BEFORE rectification
        before_preview = _make_preview_pair(
            undistort_l(img_l),
            undistort_r(img_r),
            preview_size=(680, 480),
            draw_lines=True,
            step=30,
        )

        # RECTIFICATION
        rect_l = cv2.remap(
            img_l, map1_l, map2_l,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        rect_r = cv2.remap(
            img_r, map1_r, map2_r,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        # AFTER rectification
        after_preview = _make_preview_pair(
            rect_l,
            rect_r,
            preview_size=(680, 480),
            draw_lines=True,
            step=30,
        )

        # Optional 2x2 comparison grid
        before_l_small = cv2.resize(undistort_l(img_l), (680, 480))
        before_r_small = cv2.resize(undistort_r(img_r), (680, 480))
        after_l_small = cv2.resize(rect_l, (680, 480))
        after_r_small = cv2.resize(rect_r, (680, 480))

        cv2.putText(before_l_small, "Before Left", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(before_r_small, "Before Right", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(after_l_small, "Rectified Left", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(after_r_small, "Rectified Right", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        top = np.hstack([before_l_small, before_r_small])
        bottom = np.hstack([after_l_small, after_r_small])

        top = _draw_epi_lines(top, step=30)
        bottom = _draw_epi_lines(bottom, step=30)

        comparison = np.vstack([top, bottom])

        # Show all three views
        # cv2.imshow("Before rectification", before_preview)
        # cv2.imshow("After rectification", after_preview)
        cv2.imshow("Before vs After", comparison)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
