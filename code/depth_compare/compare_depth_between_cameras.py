from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.save_diff_data import save_per_image_results, save_summary_results
from code.image import load_yaml_calibration, get_undistort_function_mono, load_rgbd_images
from code.utils import scale_intrinsics, load_estimated_depth_map
from code.visualize_depth import colorize_depth
from prepare_paths import prepare_depth_comparison_paths

logger = logging.getLogger(__name__)

def _read_transform(path: Path, direction: str = "cam2_from_cam1") -> np.ndarray:
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
        if t_cam2_cam1.shape != (4, 4):
            raise ValueError(f"Expected 4x4 T_cam2_cam1 in {path}, got {t_cam2_cam1.shape}.")

    if t_cam1_cam2 is not None:
        t_cam1_cam2 = np.asarray(t_cam1_cam2, dtype=np.float64)
        if t_cam1_cam2.shape != (4, 4):
            raise ValueError(f"Expected 4x4 T_cam1_cam2 in {path}, got {t_cam1_cam2.shape}.")

    if direction == "cam2_from_cam1":
        if t_cam2_cam1 is not None:
            return t_cam2_cam1
        return np.linalg.inv(t_cam1_cam2)

    if direction == "cam1_from_cam2":
        if t_cam1_cam2 is not None:
            return t_cam1_cam2
        return np.linalg.inv(t_cam2_cam1)

    raise ValueError(f"Unsupported direction: {direction}")


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
    fill_holes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    points_source = _back_project_depth(source_depth, k_source, d_source, source_depth_scale)

    if points_source.shape[0] == 0:
        h, w = target_hw
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)

    points_source_h = np.hstack([points_source, np.ones((points_source.shape[0], 1), dtype=np.float64)])
    points_target = (t_target_source @ points_source_h.T).T[:, :3]

    front = points_target[:, 2] > 0
    points_target = points_target[front]

    logger.debug("points_source count: %d", points_source.shape[0])
    logger.debug("points_target count before front filter: %d", points_target.shape[0])
    logger.debug("front count: %d", np.count_nonzero(front))

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

    h, w = target_hw
    inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    projected, valid = _splat_depth(uv, z, target_hw)

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

def _compute_metrics(pred_target_m: np.ndarray, gt_target: np.ndarray, valid_projected: np.ndarray) -> dict:
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
    gt   = gt_target[valid]
    diff = pred - gt
    abs_diff = np.abs(diff)

    return {
        "num_valid_pixels":      n,                                                              # kept for diagnostics only
        "abs_rel":               float(np.mean(abs_diff / np.maximum(gt, 1e-8))),               # primary error ↓
        "rmse_m":                float(np.sqrt(np.mean(diff ** 2))),                             # absolute error ↓
        "median_abs_error_m":    float(np.median(abs_diff)),                                     # robust typical error ↓
        "delta_1_25":            float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25)),        # primary accuracy ↑
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

def undistort_depth_map(depth: np.ndarray, calib_path: Path) -> np.ndarray:
    """Undistort depth map using mono-camera undistortion maps."""
    calib = load_yaml_calibration(calib_path)
    undistort = get_undistort_function_mono(calib)

    depth32 = depth.astype(np.float32)
    depth_undist = undistort(depth32)

    # Keep invalid regions as zeros after remap.
    invalid = ~np.isfinite(depth_undist)
    depth_undist[invalid] = 0.0
    return depth_undist.astype(depth.dtype, copy=False)

def create_comparison_visualization(
    depth_gt: np.ndarray,
    depth_est: np.ndarray,
    pred_warped_m: np.ndarray,
    valid_pred: np.ndarray,
    metrics: dict = None,
) -> np.ndarray:
    # ── shared depth range ────────────────────────────────────────────────────
    valid_gt  = np.isfinite(depth_gt)      & (depth_gt      > 0)
    valid_est = np.isfinite(depth_est)     & (depth_est     > 0)
    valid_w   = np.isfinite(pred_warped_m) & (pred_warped_m > 0)

    all_vals = np.concatenate([
        depth_gt[valid_gt], depth_est[valid_est], pred_warped_m[valid_w],
    ])
    shared_vmin = float(np.percentile(all_vals, 2))  if len(all_vals) else 0.0
    shared_vmax = float(np.percentile(all_vals, 98)) if len(all_vals) else 5.0

    vis_gt     = colorize_depth(depth_gt,     vmin=shared_vmin, vmax=shared_vmax)
    vis_est    = colorize_depth(depth_est,     vmin=shared_vmin, vmax=shared_vmax)
    vis_warped = colorize_depth(pred_warped_m, vmin=shared_vmin, vmax=shared_vmax)
    vis_error  = create_error_heatmap(pred_warped_m, depth_gt, valid_pred)

    # ── scale all panels to same height ──────────────────────────────────────
    def resize_h(img, h):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)

    panel_h = vis_gt.shape[0]
    vis_gt     = resize_h(vis_gt,     panel_h)
    vis_est    = resize_h(vis_est,    panel_h)
    vis_warped = resize_h(vis_warped, panel_h)
    vis_error  = resize_h(vis_error,  panel_h)

    # ── label helper: white text with black outline ───────────────────────────
    def _label(img, text, y=28):
        out = img.copy()
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0),       2, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top_row = cv2.hconcat([
        _label(vis_gt,     "GT depth"),
        _label(vis_est,    "Estimated depth"),
        _label(vis_warped, "Warped to GT frame"),
    ])

    # metrics text on its own line at top of error panel (y=28)
    # "worst regions" label at a different y to avoid overlap
    err_panel = vis_error.copy()
    if metrics:
        txt = (f"AbsRel={metrics.get('abs_rel', 0):.3f}  "
               f"MAE={metrics.get('mae_m', 0):.3f}m  "
               f"d1={metrics.get('delta_1_25', 0)*100:.1f}%")
        cv2.putText(err_panel, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0),     2, cv2.LINE_AA)
        cv2.putText(err_panel, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 230, 0),   1, cv2.LINE_AA)

    cv2.putText(err_panel, "Error |warped - GT|  (circles = worst regions)",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(err_panel, "Error |warped - GT|  (circles = worst regions)",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── scale top_row and bottom_row to fit DISPLAY_MAX_W ─────────────────────
    max_w = max(top_row.shape[1], err_panel.shape[1])
    if max_w > 1920:
        scale = 1920 / max_w
        def scale_img(img):
            return cv2.resize(img,
                              (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        top_row   = scale_img(top_row)
        err_panel = scale_img(err_panel)

    # match widths for vconcat
    w = min(top_row.shape[1], err_panel.shape[1])
    return cv2.vconcat([top_row[:, :w], err_panel[:, :w]])


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
        choices=["cam1_from_cam2", "cam2_from_cam1"],
        default="cam1_from_cam2",
        help="How to interpret T_cam2_cam1 from --relative-pose.",
    )
    parser.add_argument("--source-depth-scale", type=float, default=1.0, help="Meters per unit in source depth map")
    parser.add_argument("--target-depth-scale", type=float, default=1.0, help="Meters per unit in target depth map")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable small hole filling after projection")
    parser.add_argument("--out-json", type=Path, default=Path("out/depth_compare_metrics.json"))
    parser.add_argument("--out-vis", type=Path, default=Path("out/depth_compare_visualization.png"))
    return parser.parse_args()


def run_depth_comparison_experiment(
    parent_dir: Path,
    date: str,
    nn_name: str,
    rgbd_camera_suffix: str = "realsense",
    max_imgs: int = None,
    pose_convention: str = "cam1_from_cam2",  #T_rgbd_from_left
    debug: int = 0,
) -> None:

    gt_data_dir, relative_pose_path, calib_rgbd_path, calib_stereo_path, depth_estimation_dir, depth_comparison_dir=prepare_depth_comparison_paths(parent_dir, date, rgbd_camera_suffix)
    paths = {
        "gt_data_dir": gt_data_dir,
        "relative_pose_path": relative_pose_path,
        "calib_rgbd_path": calib_rgbd_path,
        "calib_stereo_path": calib_stereo_path,
        "depth_estimation_dir": depth_estimation_dir,
        "depth_comparison_dir": depth_comparison_dir,
    }

    for name, path in paths.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗ NOT FOUND"
        print(f"  {status}  {name:25s}  {path}")

    imgs_rgb = load_rgbd_images(gt_data_dir, suffix=rgbd_camera_suffix, max_imgs=max_imgs)

    estimated_depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=max_imgs)
    frame_size = estimated_depth_maps[0].shape[::-1]
    logger.debug(f"NN frame size: {frame_size}")

    k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
    k_source = scale_intrinsics(k_source, (3840, 2160), frame_size)

    k_target, d_target = load_camera_calibration(calib_rgbd_path)
    k_target = scale_intrinsics(k_target, (1280, 720), frame_size)

    transform_target_from_source = _read_transform(relative_pose_path, pose_convention)

    transform_target_from_source[:3, 3] /= 1000.0
    norm = np.linalg.norm(transform_target_from_source[:3, 3])
    logger.debug(f"translation norm: {norm}")

    all_metrics = []

    for img_gt, depth_est in tqdm(zip(imgs_rgb, estimated_depth_maps), desc="evaluating depth estimation"):
        image_number = img_gt.get_image_number()
        depth_est = depth_est
        depth_gt = img_gt.get_depth()
        # target_depth = undistort_depth_map(target_depth, calib_dict_realsense)

        vis1 = colorize_depth(depth_gt)
        vis2 = colorize_depth(depth_est)

        vis1 = cv2.resize(vis1, frame_size)
        vis2 = cv2.resize(vis2, frame_size)

        depth_est = resize_depth(depth_est, frame_size)
        depth_gt = resize_depth(depth_gt, frame_size)

        if depth_est.ndim != 2 or depth_est.ndim != 2:
            raise ValueError(f"Depth maps must be 2D, got source={depth_est.shape}, target={depth_est.shape}")

        pred_warped_m, valid_pred = warp_depth_to_target(
            source_depth=depth_est,
            k_source=k_source,
            d_source=d_source,
            k_target=k_target,
            d_target=d_target,
            t_target_source=transform_target_from_source,
            source_depth_scale=1.0,
            target_hw=(depth_gt.shape[0], depth_gt.shape[1]),
        )

        vis3 = colorize_depth(pred_warped_m)
        vis = cv2.hconcat([vis1, vis2, vis3])
        if debug > 0:
            cv2.imshow(f"Depth map {rgbd_camera_suffix} | {nn_name}", vis)
            cv2.waitKey(0)

        metrics = _compute_metrics(pred_warped_m, depth_gt, valid_pred)
        metrics_with_id = {
            "image_id": image_number,
            **metrics,
        }

        comparison_vis = create_comparison_visualization(
            depth_gt=depth_gt,
            depth_est=depth_est,
            pred_warped_m=pred_warped_m,
            valid_pred=valid_pred,  # ← add
            metrics=metrics,  # ← add
        )

        save_per_image_results(
            out_root=depth_comparison_dir,
            image_id=image_number,
            pred_warped_m=pred_warped_m,
            gt_depth_m=depth_gt,
            depth_est_m=depth_est,
            valid_mask=valid_pred,
            metrics=metrics_with_id,
            comparison_vis=comparison_vis,
        )

        all_metrics.append(metrics_with_id)
        if debug > 0:
            cv2.imshow(f"Comparison {rgbd_camera_suffix} | {nn_name}", comparison_vis)
            cv2.waitKey(0)

    save_summary_results(
        out_root=depth_comparison_dir,
        all_metrics=all_metrics,
    )

def create_error_heatmap(
    pred_warped_m: np.ndarray,
    depth_gt: np.ndarray,
    valid_pred: np.ndarray,
    percentile_clip: float = 95.0,
) -> np.ndarray:
    valid = (
        valid_pred
        & np.isfinite(pred_warped_m) & (pred_warped_m > 0)
        & np.isfinite(depth_gt)      & (depth_gt > 0)
    )

    h, w = depth_gt.shape
    abs_err = np.zeros((h, w), dtype=np.float32)
    abs_err[valid] = np.abs(pred_warped_m[valid] - depth_gt[valid])

    if np.any(valid):
        err_max = float(np.percentile(abs_err[valid], percentile_clip))
    else:
        err_max = 1.0

    err_norm = np.clip(abs_err / max(err_max, 1e-6), 0, 1)
    err_u8   = (err_norm * 255).astype(np.uint8)
    heatmap  = cv2.applyColorMap(err_u8, cv2.COLORMAP_INFERNO)
    heatmap[~valid] = (30, 30, 30)

    # ── worst regions — use larger patch to spread circles out ────────────────
    if np.any(valid):
        patch = max(h // 10, 32)   # adaptive patch: ~10% of image height
        worst = []
        for py in range(0, h - patch, patch):
            for px in range(0, w - patch, patch):
                region_valid = valid[py:py+patch, px:px+patch]
                if region_valid.sum() < patch * patch * 0.3:
                    continue
                region_err = abs_err[py:py+patch, px:px+patch]
                mean_err = float(region_err[region_valid].mean())
                worst.append((mean_err, px + patch // 2, py + patch // 2))

        worst.sort(reverse=True)
        radius = max(patch // 2, 15)
        for rank, (err_val, cx, cy) in enumerate(worst[:5]):
            cv2.circle(heatmap, (cx, cy), radius, (0, 255, 0), 2)
            # keep label inside image bounds
            lx = min(cx + radius + 4, w - 120)
            ly = max(cy + 6, 15)
            cv2.putText(heatmap, f"#{rank+1} {err_val:.2f}m",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0),     2, cv2.LINE_AA)
            cv2.putText(heatmap, f"#{rank+1} {err_val:.2f}m",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0),   1, cv2.LINE_AA)

    # ── colorbar ──────────────────────────────────────────────────────────────
    bar_h = h
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    colorbar = cv2.applyColorMap(np.repeat(gradient, 20, axis=1), cv2.COLORMAP_INFERNO)

    label_panel = np.full((bar_h, 90, 3), 245, dtype=np.uint8)
    cv2.putText(label_panel, f"{err_max:.2f}m", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(label_panel, "0.00m",           (4, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(label_panel, f"p{percentile_clip:.0f}", (4, bar_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    return cv2.hconcat([heatmap, colorbar, label_panel])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parent_dir = Path(__file__).resolve().parents[3]
    date = "27032026"
    nn_name = "FoundationStereo"
    # NNname = "FoundationStereo"
    debug = 1
    rgbd_camera_suffix = "zed"
    max_imgs = 2
    run_depth_comparison_experiment(
        parent_dir,
        date=date,
        nn_name=nn_name,
        rgbd_camera_suffix = rgbd_camera_suffix,
        max_imgs=max_imgs,
        debug=debug
    )
