from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.utils import scale_intrinsics
from code.visualize_depth import colorize_depth


def _read_transform(path: Path) -> np.ndarray:
    """Read 4x4 transform from OpenCV YAML/XML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open transform file: {path}")

    try:
        t_cam2_cam1 = fs.getNode("T_cam2_cam1").mat()
        t_cam1_cam2 = fs.getNode("T_cam1_cam2").mat()
    finally:
        fs.release()

    if t_cam2_cam1 is None and t_cam1_cam2 is None:
        raise ValueError(f"Could not find T_cam2_cam1 or T_cam1_cam2 in {path}")

    if t_cam2_cam1 is not None:
        t_cam2_cam1 = np.asarray(t_cam2_cam1, dtype=np.float64)
    if t_cam1_cam2 is not None:
        t_cam1_cam2 = np.asarray(t_cam1_cam2, dtype=np.float64)

    if t_cam2_cam1 is not None and t_cam2_cam1.shape != (4, 4):
        raise ValueError(f"Expected 4x4 T_cam2_cam1 in {path}, got {t_cam2_cam1.shape}.")
    if t_cam1_cam2 is not None and t_cam1_cam2.shape != (4, 4):
        raise ValueError(f"Expected 4x4 T_cam1_cam2 in {path}, got {t_cam1_cam2.shape}.")

    if t_cam2_cam1 is not None:
        return t_cam2_cam1
    return np.linalg.inv(t_cam1_cam2)


def _back_project_depth(depth: np.ndarray, k: np.ndarray, d: np.ndarray, depth_scale: float) -> np.ndarray:
    """Back-project valid depth pixels to 3D camera points."""
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    uv = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, k, d).reshape(-1, 2)

    z = depth[ys, xs].astype(np.float64) * depth_scale
    return np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])


def _splat_depth(
    uv: np.ndarray,
    z: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer splat with 2x2 neighborhood to reduce holes."""
    h, w = target_hw
    projected = np.full((h, w), np.inf, dtype=np.float64)

    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)

    offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    for du, dv in offsets:
        uu = u0 + du
        vv = v0 + dv
        inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        if not np.any(inside):
            continue
        np.minimum.at(projected, (vv[inside], uu[inside]), z[inside])

    valid = np.isfinite(projected)
    projected[~valid] = 0.0
    return projected, valid


def _fill_small_holes(depth_m: np.ndarray, valid: np.ndarray, max_kernel: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Fill sparse holes for visualization/metrics overlap using nearest valid depth."""
    if not np.any(valid):
        return depth_m, valid

    depth32 = depth_m.astype(np.float32)
    holes = (~valid).astype(np.uint8)
    if not np.any(holes):
        return depth_m, valid

    dilated = depth32.copy()
    for k in (3, max_kernel):
        kernel = np.ones((k, k), dtype=np.uint8)
        candidate = cv2.dilate(dilated, kernel)
        take = (dilated <= 0) & (candidate > 0)
        dilated[take] = candidate[take]

    valid_new = dilated > 0
    return dilated.astype(np.float64), valid_new


def warp_depth_to_target(
    source_depth: np.ndarray,
    k_source: np.ndarray,
    d_source: np.ndarray,
    k_target: np.ndarray,
    d_target: np.ndarray,
    t_target_source: np.ndarray,
    source_depth_scale: float,
    target_hw: tuple[int, int],
    fill_holes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    points_source = _back_project_depth(source_depth, k_source, d_source, source_depth_scale)

    if points_source.shape[0] == 0:
        h, w = target_hw
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)

    points_source_h = np.hstack([points_source, np.ones((points_source.shape[0], 1), dtype=np.float64)])
    points_target = (t_target_source @ points_source_h.T).T[:, :3]

    front = points_target[:, 2] > 0
    points_target = points_target[front]

    print("points_source count:", points_source.shape[0])
    print("points_target count before front filter:", points_target.shape[0])
    print("front count:", np.count_nonzero(front))

    if points_target.shape[0] == 0:
        h, w = target_hw
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)

    projected_pixels, _ = cv2.projectPoints(
        points_target.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        k_target,
        d_target,
    )
    uv = projected_pixels.reshape(-1, 2)
    z = points_target[:, 2]
    print("uv min:", uv.min(axis=0))
    print("uv max:", uv.max(axis=0))

    h, w = target_hw
    inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    print("target_hw:", target_hw)
    print("inside count:", int(np.count_nonzero(inside)))
    print("inside ratio:", float(np.mean(inside)))

    projected, valid = _splat_depth(uv, z, target_hw)

    print("valid after splat:", int(np.count_nonzero(valid)))
    print("projected min/max valid z:",
          float(projected[valid].min()) if np.any(valid) else None,
          float(projected[valid].max()) if np.any(valid) else None)

    if fill_holes:
        projected, valid = _fill_small_holes(projected, valid)



    return projected, valid


def estimate_depth_scale(source_depth, gt_depth_m):
    valid = (
        np.isfinite(source_depth) & (source_depth > 0) &
        np.isfinite(gt_depth_m) & (gt_depth_m > 0)
    )
    if np.count_nonzero(valid) == 0:
        raise ValueError("No valid overlapping pixels.")

    s = np.median(gt_depth_m[valid] / source_depth[valid])
    return float(s)

def evaluate_scaled_depth(source_depth, gt_depth_m):
    valid = (
        np.isfinite(source_depth) & (source_depth > 0) &
        np.isfinite(gt_depth_m) & (gt_depth_m > 0)
    )
    if np.count_nonzero(valid) == 0:
        raise ValueError("No valid overlapping pixels.")

    s = np.median(gt_depth_m[valid] / source_depth[valid])
    source_scaled = source_depth * s

    err = np.abs(source_scaled[valid] - gt_depth_m[valid])
    mae = float(np.mean(err))
    med = float(np.median(err))

    return {
        "scale": float(s),
        "mae_m": mae,
        "median_abs_error_m": med,
    }

def _compute_metrics(pred_target_m: np.ndarray, gt_target: np.ndarray, gt_depth_scale: float, valid_projected: np.ndarray) -> dict:
    # gt_target_m = gt_target.astype(np.float64) * gt_depth_scale
    result = evaluate_scaled_depth(pred_target_m, gt_target)
    print(result)

    valid = (
        valid_projected
        & np.isfinite(pred_target_m)
        & np.isfinite(gt_target)
        & (pred_target_m > 0)
        & (gt_target > 0)
    )

    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"num_valid_pixels": 0}

    pred = pred_target_m[valid]
    gt = gt_target[valid]
    diff = pred - gt
    abs_diff = np.abs(diff)

    return {
        "num_valid_pixels": n,
        "mae_m": float(np.mean(abs_diff)),
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "median_abs_error_m": float(np.median(abs_diff)),
        "abs_rel": float(np.mean(abs_diff / np.maximum(gt, 1e-8))),
        "delta_1_25": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25)),
        "delta_1_25_sq": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25 ** 2)),
        "delta_1_25_cu": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25 ** 3)),
    }


def _colorize_depth(depth_m: np.ndarray, valid: np.ndarray, max_depth_m: float) -> np.ndarray:
    d = depth_m.astype(np.float32).copy()
    d[~valid] = 0.0

    scale = max(max_depth_m, 1e-6)
    norm = np.clip((d / scale) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def _save_visualization(pred_m: np.ndarray, gt_raw: np.ndarray, gt_scale: float, valid_pred: np.ndarray, out_png: Path) -> None:
    gt_m = gt_raw.astype(np.float64) * gt_scale
    valid_gt = np.isfinite(gt_m) & (gt_m > 0)
    valid_both = valid_gt & valid_pred

    if np.any(valid_both):
        depth_max = float(np.percentile(gt_m[valid_both], 99))
    elif np.any(valid_gt):
        depth_max = float(np.percentile(gt_m[valid_gt], 99))
    else:
        depth_max = 5.0

    pred_vis = _colorize_depth(pred_m, valid_pred, depth_max)
    gt_vis = _colorize_depth(gt_m, valid_gt, depth_max)

    error = np.zeros_like(gt_m, dtype=np.float64)
    error[valid_both] = np.abs(pred_m[valid_both] - gt_m[valid_both])
    err_max = max(float(np.percentile(error[valid_both], 99)) if np.any(valid_both) else 0.5, 1e-6)
    err_norm = np.clip((error / err_max) * 255.0, 0, 255).astype(np.uint8)
    err_vis = cv2.applyColorMap(err_norm, cv2.COLORMAP_INFERNO)
    err_vis[~valid_both] = (0, 0, 0)

    panel = cv2.hconcat([pred_vis, gt_vis, err_vis])
    cv2.putText(panel, "Warped source depth", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "Target GT depth", (pred_vis.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "|pred-gt|", (pred_vis.shape[1] * 2 + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), panel)

def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize depth map while preserving depth values.
    target_hw = (H, W)
    """
    return cv2.resize(depth, target_hw, interpolation=cv2.INTER_NEAREST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp depth from source camera to target camera and compare with target GT depth.")
    parser.add_argument("--source-depth", type=Path, required=False, help="Source depth map (.npy)")
    parser.add_argument("--target-depth", type=Path, required=False, help="Target/GT depth map (.npy)")
    parser.add_argument("--source-calib", type=Path, required=False, help="Source camera calibration (.yaml/.yml/.npy)")
    parser.add_argument("--target-calib", type=Path, required=False, help="Target camera calibration (.yaml/.yml/.npy)")
    parser.add_argument("--source-suffix", type=str, default="realsense", help="Suffix when loading .npy stereo calib: left/right")
    parser.add_argument("--target-suffix", type=str, default="left", help="Suffix when loading .npy stereo calib: left/right")
    parser.add_argument("--relative-pose", type=Path, required=False, help="YAML with T_cam2_cam1 or T_cam1_cam2")
    parser.add_argument(
        "--pose-convention",
        choices=["target_from_source", "source_from_target"],
        default="source_from_target",
        help="How to interpret T_cam2_cam1 from --relative-pose.",
    )
    parser.add_argument("--source-depth-scale", type=float, default=1.0, help="Meters per unit in source depth map")
    parser.add_argument("--target-depth-scale", type=float, default=1.0, help="Meters per unit in target depth map")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable small hole filling after projection")
    parser.add_argument("--out-json", type=Path, default=Path("out/depth_compare_metrics.json"))
    parser.add_argument("--out-vis", type=Path, default=Path("out/depth_compare_visualization.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # source Left
    # target Realsense

    parent_dir = Path(__file__).resolve().parents[2]

    date = "27032026"
    NNname = "FoundationStereo"

    dataset_dir = parent_dir / 'datasets' / f"dataset_{date}"
    depth_dir = dataset_dir / "stereo_4k_depth" / "rgb"
    depth_estimation_dir = parent_dir / "out_estimation" / "stereo" / NNname / f"dataset_{date}" / "depth"
    print(depth_estimation_dir)
    print(depth_dir)
    img_number = 4
    args.image_cam1 = depth_dir / f"{img_number}_realsense.png"
    args.image_cam2 = depth_dir / f"{img_number}_left.png"

    out_dir = parent_dir / 'out' / f"out_{date}" / "cameras_parameters"

    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_stereo = out_dir / "calib_data.npy"

    args.relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_left_v3.yaml"
    args.cam1_calib = calib_dict_realsense
    args.cam2_calib = calib_dict_stereo
    source_depth = depth_estimation_dir / f"{img_number}_depth.npy"
    target_depth =dataset_dir / "stereo_4k_depth" / "depth" / f"{img_number}_realsense_depth.npy"
    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_stereo = out_dir / "calib_data.npy"
    relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_left_v3.yaml"

    source_depth = np.load(source_depth)
    target_depth = np.load(target_depth)

    frame_size = (960, 540)

    vis1 = colorize_depth(source_depth)
    vis2 = colorize_depth(target_depth)

    vis1 = cv2.resize(vis1, frame_size)
    vis2 = cv2.resize(vis2, frame_size)
    vis = cv2.hconcat([vis1, vis2])
    # cv2.imshow("Depth map Realsense " + NNname, vis)
    # cv2.waitKey(0)

    source_depth = resize_depth(source_depth, frame_size)
    target_depth = resize_depth(target_depth, frame_size)

    print("source_depth shape:", source_depth.shape)
    print("target_depth shape:", target_depth.shape)

    metrics = evaluate_scaled_depth(source_depth, target_depth)
    scale = metrics["scale"]
    print(metrics["scale"])


    if source_depth.ndim != 2 or target_depth.ndim != 2:
        raise ValueError(f"Depth maps must be 2D, got source={source_depth.shape}, target={target_depth.shape}")

    k_source, d_source = load_camera_calibration(calib_dict_stereo, suffix="left")
    k_source = scale_intrinsics(k_source, (3840, 2160), (960, 540))

    k_target, d_target = load_camera_calibration(calib_dict_realsense)
    k_target = scale_intrinsics(k_target, (1280, 720), (960, 540))

    # k_source[0,0] *= scale
    # k_source[1,1] *= scale
    t_cam2_cam1 = _read_transform(relative_pose)
    if args.pose_convention == "target_from_source":
        t_target_source = t_cam2_cam1
    else:
        t_target_source = np.linalg.inv(t_cam2_cam1)

    t_target_source[:3, 3] /= 1000.0
    print("t_target_source:\n", t_target_source)
    print("translation norm:", np.linalg.norm(t_target_source[:3, 3]))



    pred_warped_m, valid_pred = warp_depth_to_target(
        source_depth=source_depth,
        k_source=k_source,
        d_source=d_source,
        k_target=k_target,
        d_target=d_target,
        t_target_source=t_target_source,
        source_depth_scale=1.0,
        target_hw=(target_depth.shape[0], target_depth.shape[1]),
        fill_holes=not args.no_fill_holes,
    )

    vis3 = colorize_depth(pred_warped_m)
    vis = cv2.hconcat([vis1, vis2, vis3])
    cv2.imshow("Depth map Realsense " + NNname, vis)
    cv2.waitKey(0)

    metrics = _compute_metrics(pred_warped_m, target_depth, args.target_depth_scale, valid_pred)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _save_visualization(pred_warped_m, target_depth, args.target_depth_scale, valid_pred, args.out_vis)

    print("=== Depth comparison (source -> target) ===")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {args.out_json}")
    print(f"Visualization saved to: {args.out_vis}")


if __name__ == "__main__":
    main()
