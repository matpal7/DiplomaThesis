from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.visualize_depth import colorize_depth


def _read_camera_calibration(path: Path, use_undistorted: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Read K and D (or K_new and D_new) from OpenCV YAML/XML calibration file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    key_k = "new_K_l" if use_undistorted else "K"
    key_d = "D_new" if use_undistorted else "D"

    try:
        k = fs.getNode(key_k).mat()
        d = fs.getNode(key_d).mat()
    finally:
        fs.release()

    if k is None or d is None:
        raise ValueError(f"Calibration {path} must contain '{key_k}' and '{key_d}'.")

    return np.asarray(k, dtype=np.float64).reshape(3, 3), np.asarray(d, dtype=np.float64).reshape(-1, 1)


def _read_transform(path: Path) -> np.ndarray:
    """Read 4x4 target<-source transform from OpenCV YAML/XML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open transform file: {path}")

    try:
        t_target_source = fs.getNode("T_cam2_cam1").mat()
        if t_target_source is None:
            t_source_target = fs.getNode("T_cam1_cam2").mat()
            if t_source_target is None:
                raise ValueError(f"Could not find T_cam2_cam1 or T_cam1_cam2 in {path}")
            t_target_source = np.linalg.inv(np.asarray(t_source_target, dtype=np.float64))
    finally:
        fs.release()

    t_target_source = np.asarray(t_target_source, dtype=np.float64)
    if t_target_source.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform in {path}, got {t_target_source.shape}.")
    return t_target_source


def _back_project_depth(depth: np.ndarray, k: np.ndarray, d: np.ndarray, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid depth pixels to camera 3D points."""
    h, w = depth.shape
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.int32)

    uv = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, k, d).reshape(-1, 2)  # normalized pinhole rays

    z = depth[ys, xs].astype(np.float64) * depth_scale
    xyz = np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])
    pix = np.column_stack([xs, ys]).astype(np.int32)
    return xyz, pix


def _forward_warp_depth(
    source_depth: np.ndarray,
    k_source: np.ndarray,
    d_source: np.ndarray,
    k_target: np.ndarray,
    d_target: np.ndarray,
    t_target_source: np.ndarray,
    depth_scale: float,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project source depth map into target camera frame and image.

    Returns:
      projected_depth_target: depth map in target image coordinates (meters)
      valid_mask: pixels where projected depth is available
    """
    target_h, target_w = target_hw
    projected = np.full((target_h, target_w), np.inf, dtype=np.float64)

    points_source, _ = _back_project_depth(source_depth, k_source, d_source, depth_scale)
    if len(points_source) == 0:
        return np.zeros((target_h, target_w), dtype=np.float64), np.zeros((target_h, target_w), dtype=np.uint8)

    points_source_h = np.hstack([points_source, np.ones((points_source.shape[0], 1), dtype=np.float64)])
    points_target = (t_target_source @ points_source_h.T).T[:, :3]

    front = points_target[:, 2] > 0
    points_target = points_target[front]
    if len(points_target) == 0:
        return np.zeros((target_h, target_w), dtype=np.float64), np.zeros((target_h, target_w), dtype=np.uint8)

    projected_pixels, _ = cv2.projectPoints(
        points_target.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        k_target,
        d_target,
    )
    uv = np.round(projected_pixels.reshape(-1, 2)).astype(np.int32)

    for i, (u, v) in enumerate(uv):
        if 0 <= u < target_w and 0 <= v < target_h:
            z = points_target[i, 2]
            if z < projected[v, u]:
                projected[v, u] = z

    valid = np.isfinite(projected)
    projected[~valid] = 0.0
    return projected, valid.astype(np.uint8)


def _compute_metrics(pred_target_m: np.ndarray, gt_target: np.ndarray, gt_depth_scale: float, valid_projected: np.ndarray) -> dict:
    """Compute depth error metrics against RGBD ground truth."""
    gt_target_m = gt_target.astype(np.float64) * gt_depth_scale

    valid = (
        (valid_projected > 0)
        & np.isfinite(pred_target_m)
        & np.isfinite(gt_target_m)
        & (pred_target_m > 0)
        & (gt_target_m > 0)
    )

    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"num_valid_pixels": 0}

    pred = pred_target_m[valid]
    gt = gt_target_m[valid]
    diff = pred - gt
    abs_diff = np.abs(diff)

    metrics = {
        "num_valid_pixels": n,
        "mae_m": float(np.mean(abs_diff)),
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "median_abs_error_m": float(np.median(abs_diff)),
        "abs_rel": float(np.mean(abs_diff / np.maximum(gt, 1e-8))),
        "delta_1_25": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25)),
        "delta_1_25_sq": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25 ** 2)),
        "delta_1_25_cu": float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25 ** 3)),
    }
    return metrics


def _colorize_depth(depth_m: np.ndarray, valid: np.ndarray, max_depth_m: float) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)
    d[~valid] = 0.0

    scale = max(max_depth_m, 1e-6)
    norm = np.clip((d / scale) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def _save_visualization(pred_m: np.ndarray, gt_raw: np.ndarray, gt_scale: float, valid_pred: np.ndarray, out_png: Path) -> None:
    gt_m = gt_raw.astype(np.float64) * gt_scale
    valid_gt = np.isfinite(gt_m) & (gt_m > 0)
    valid_both = valid_gt & (valid_pred > 0)

    if np.any(valid_both):
        depth_max = float(np.percentile(gt_m[valid_both], 99))
    elif np.any(valid_gt):
        depth_max = float(np.percentile(gt_m[valid_gt], 99))
    else:
        depth_max = 5.0

    pred_vis = _colorize_depth(pred_m, valid_pred > 0, depth_max)
    gt_vis = _colorize_depth(gt_m, valid_gt, depth_max)

    error = np.zeros_like(gt_m, dtype=np.float64)
    error[valid_both] = np.abs(pred_m[valid_both] - gt_m[valid_both])
    err_max = max(float(np.percentile(error[valid_both], 99)) if np.any(valid_both) else 0.5, 1e-6)
    err_norm = np.clip((error / err_max) * 255.0, 0, 255).astype(np.uint8)
    err_vis = cv2.applyColorMap(err_norm, cv2.COLORMAP_INFERNO)
    err_vis[~valid_both] = (0, 0, 0)

    panel_top = cv2.hconcat([pred_vis, gt_vis, err_vis])

    overlay = np.zeros_like(pred_vis)
    overlay[..., 1] = (valid_pred > 0).astype(np.uint8) * 255
    overlay[..., 2] = valid_gt.astype(np.uint8) * 255

    cv2.putText(panel_top, "Pred warped", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel_top, "GT RGBD", (pred_vis.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel_top, "|pred-gt|", (pred_vis.shape[1] * 2 + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    canvas = cv2.vconcat([panel_top, overlay])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp predicted depth from source camera to RGBD camera and compute depth metrics."
    )
    # parser.add_argument("--pred-depth", type=Path, required=True, help="Predicted/source depth .npy")
    # parser.add_argument("--gt-depth", type=Path, required=True, help="Ground-truth RGBD depth .npy")
    # parser.add_argument("--source-calib", type=Path, required=True, help="Source camera calibration YAML")
    # parser.add_argument("--target-calib", type=Path, required=True, help="Target (RGBD) camera calibration YAML")
    # parser.add_argument("--relative-pose", type=Path, required=True, help="YAML with T_cam2_cam1 (target<-source)")
    parser.add_argument("--pred-depth-scale", type=float, default=1.0, help="Meters per unit in predicted depth")
    # parser.add_argument("--gt-depth-scale", type=float, default=0.001, help="Meters per unit in GT depth")
    # parser.add_argument("--source-use-undistorted", action="store_true", help="Use K_new/D_new for source")
    # parser.add_argument("--target-use-undistorted", action="store_true", help="Use K_new/D_new for target")
    # parser.add_argument("--out-json", type=Path, default=Path("out/depth_compare_metrics.json"))
    # parser.add_argument("--out-vis", type=Path, default=Path("out/depth_compare_visualization.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parent_dir = Path(__file__).resolve().parents[2]

    date = "28032026"
    NNname = "FoundationStereo"

    dataset_dir = parent_dir / 'datasets' / f"dataset_{date}"
    depth_dir = dataset_dir / "stereo_4k_depth" / "rgb"
    depth_estimation_dir = parent_dir / "out_estimation" / "stereo" / NNname / f"dataset_{date}" / "depth"
    print(depth_estimation_dir)
    print(depth_dir)
    img_number = 0
    args.image_cam1 = depth_dir / f"{img_number}_realsense.png"
    args.image_cam2 = depth_dir / f"{img_number}_left.png"

    out_dir = parent_dir / 'out' / f"out_{date}" / "cameras_parameters"

    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_stereo = out_dir / "calib_data.npy"

    args.relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_left_v3.yaml"
    args.cam1_calib = calib_dict_realsense
    args.cam2_calib = calib_dict_stereo
    depth_map_cam1 = dataset_dir / "stereo_4k_depth" / "depth" / f"{img_number}_realsense_depth.npy"
    depth_map_cam2 = depth_estimation_dir / f"{img_number}_depth.npy"

    frame_size = (960, 540)
    depth_cam1 = np.load(depth_map_cam1)
    depth_cam2 = np.load(depth_map_cam2)

    vis1 = colorize_depth(depth_cam1)
    vis2 = colorize_depth(depth_cam2)

    vis1 = cv2.resize(vis1, frame_size)
    vis2 = cv2.resize(vis2, frame_size)




    source_depth = depth_cam2
    gt_depth = depth_cam1

    if source_depth.ndim != 2 or gt_depth.ndim != 2:
        raise ValueError(f"Depth maps must be 2D, got source={source_depth.shape}, gt={gt_depth.shape}")

    calib_dict_realsense = out_dir / "realsense_calibration_1280x720.yaml"
    calib_dict_stereo = out_dir / "calib_data.npy"
    relative_pose = out_dir / "relative_pose" / "relative_pose_realsense_to_left_v3.yaml"

    k_source, d_source = load_camera_calibration(calib_dict_realsense)
    k_target, d_target = load_camera_calibration(calib_dict_stereo, suffix="left")
    from code.utils import scale_intrinsics
    print((gt_depth.shape[1], gt_depth.shape[0]), (source_depth.shape[1], source_depth.shape[0]))
    k_source = scale_intrinsics(k_source, (gt_depth.shape[1], gt_depth.shape[0]), (source_depth.shape[1], source_depth.shape[0]))
    t_target_source = _read_transform(relative_pose)

    pred_warped_m, valid_pred = _forward_warp_depth(
        source_depth=source_depth,
        k_source=k_source,
        d_source=d_source,
        k_target=k_target,
        d_target=d_target,
        t_target_source=t_target_source,
        depth_scale=args.pred_depth_scale,
        target_hw=(gt_depth.shape[0], gt_depth.shape[1]),
    )

    vis_projected =  colorize_depth(pred_warped_m)
    vis = cv2.hconcat([vis1, vis2])

    cv2.imshow("Depth map Realsense " + NNname, vis_projected)
    cv2.waitKey(0)

    metrics = _compute_metrics(pred_warped_m, gt_depth, args.gt_depth_scale, valid_pred)
    #
    # args.out_json.parent.mkdir(parents=True, exist_ok=True)
    # with args.out_json.open("w", encoding="utf-8") as f:
    #     json.dump(metrics, f, indent=2)
    #
    # _save_visualization(pred_warped_m, gt_depth, args.gt_depth_scale, valid_pred, args.out_vis)
    #
    # print("=== Depth comparison (predicted source -> target RGBD) ===")
    # print(json.dumps(metrics, indent=2))
    # print(f"Metrics saved to: {args.out_json}")
    # print(f"Visualization saved to: {args.out_vis}")


if __name__ == "__main__":
    main()
