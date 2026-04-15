from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.save_diff_data import save_per_image_results, save_summary_results
from code.image import load_yaml_calibration, get_undistort_function_mono, load_rgbd_images
from code.prepare_paths import prepare_depth_comparison_paths
from code.utils import scale_intrinsics, load_estimated_depth_map
from code.visualize_depth import colorize_depth

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ── IGNORE RECTS ── Global state for interactive rectangle drawing
# ─────────────────────────────────────────────────────────────────────────────

# Each rect: (x, y, w, h) in pixel coordinates of the target frame
# Can be pre-populated for batch runs, or drawn interactively in debug mode.
IGNORE_RECTS: list[tuple[int, int, int, int]] = []

_rect_drawing   = False   # mouse button held
_rect_start     = (0, 0)
_rect_current   = (0, 0)


def build_ignore_mask(shape_hw: tuple[int, int],
                      rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Returns a bool mask (H, W) that is False inside every ignore rectangle.
    Pixel is True  → include in metrics.
    Pixel is False → exclude from metrics.
    """
    mask = np.ones(shape_hw, dtype=bool)
    for (x, y, w, h) in rects:
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, shape_hw[1]), min(y + h, shape_hw[0])
        mask[y1:y2, x1:x2] = False
    return mask


def draw_ignore_rects_on(img: np.ndarray,
                         rects: list[tuple[int, int, int, int]],
                         current_rect: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Overlay confirmed ignore rectangles (red) and the one being drawn (yellow).
    """
    out = img.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 220), 2)
        # semi-transparent fill
        roi = out[y:y+h, x:x+w]
        fill = np.full_like(roi, (0, 0, 80))
        cv2.addWeighted(fill, 0.35, roi, 0.65, 0, roi)
        out[y:y+h, x:x+w] = roi
        cv2.putText(out, "IGNORED", (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    if current_rect:
        cx, cy, cw, ch = current_rect
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), (0, 220, 220), 2)
    return out


def _make_rect_mouse_cb(window_name: str, display_scale: float):
    """
    Returns an OpenCV mouse callback that lets the user draw ignore rectangles
    directly on the comparison window.

    Controls (while hovering the window):
      • Left-drag  → draw a new ignore rectangle
      • Right-click → remove the last rectangle
    """
    global _rect_drawing, _rect_start, _rect_current

    def cb(event, x, y, flags, param):
        global _rect_drawing, _rect_start, _rect_current, IGNORE_RECTS
        # Map display coords back to depth-map coords
        rx = int(x / display_scale)
        ry = int(y / display_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            _rect_drawing = True
            _rect_start   = (rx, ry)
            _rect_current = (rx, ry)

        elif event == cv2.EVENT_MOUSEMOVE and _rect_drawing:
            _rect_current = (rx, ry)

        elif event == cv2.EVENT_LBUTTONUP and _rect_drawing:
            _rect_drawing = False
            x0, y0 = _rect_start
            x1, y1 = rx, ry
            rect_w = abs(x1 - x0)
            rect_h = abs(y1 - y0)
            if rect_w > 4 and rect_h > 4:   # ignore accidental tiny clicks
                IGNORE_RECTS.append((min(x0, x1), min(y0, y1), rect_w, rect_h))
                print(f"  [ignore rect added]  total={len(IGNORE_RECTS)}  "
                      f"last={IGNORE_RECTS[-1]}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if IGNORE_RECTS:
                removed = IGNORE_RECTS.pop()
                print(f"  [ignore rect removed]  removed={removed}  "
                      f"remaining={len(IGNORE_RECTS)}")
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
        raise ValueError(f"No T_cam2_cam1 or T_cam1_cam2 in {path}")
    if t_cam2_cam1 is not None:
        t_cam2_cam1 = np.asarray(t_cam2_cam1, dtype=np.float64)
    if t_cam1_cam2 is not None:
        t_cam1_cam2 = np.asarray(t_cam1_cam2, dtype=np.float64)
    if direction == "cam2_from_cam1":
        return t_cam2_cam1 if t_cam2_cam1 is not None else np.linalg.inv(t_cam1_cam2)
    if direction == "cam1_from_cam2":
        return t_cam1_cam2 if t_cam1_cam2 is not None else np.linalg.inv(t_cam2_cam1)
    raise ValueError(f"Unsupported direction: {direction}")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _back_project_depth(depth, k, d, depth_scale):
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    uv   = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, k, d).reshape(-1, 2)
    z    = depth[ys, xs].astype(np.float64) * depth_scale
    return np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])


def _splat_depth(uv, z, target_hw, exclude_left_px=0, exclude_top_px=0,
                 exclude_right_px=0, exclude_bottom_px=0):
    h, w = target_hw
    projected = np.full((h, w), np.inf, dtype=np.float64)
    keep = (
        (uv[:, 0] >= exclude_left_px)  & (uv[:, 0] <  w - exclude_right_px)
        & (uv[:, 1] >= exclude_top_px) & (uv[:, 1] <  h - exclude_bottom_px)
    )
    uv, z = uv[keep], z[keep]
    u0 = np.floor(uv[:, 0]).astype(np.int32)
    v0 = np.floor(uv[:, 1]).astype(np.int32)
    for du, dv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        uu, vv = u0 + du, v0 + dv
        inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        if np.any(inside):
            np.minimum.at(projected, (vv[inside], uu[inside]), z[inside])
    valid = np.isfinite(projected)
    projected[~valid] = 0.0
    return projected, valid


def _fill_small_holes(depth_m, valid, max_kernel=5):
    if not np.any(valid):
        return depth_m, valid
    dilated = depth_m.astype(np.float32).copy()
    for k in (3, max_kernel):
        kernel    = np.ones((k, k), dtype=np.uint8)
        candidate = cv2.dilate(dilated, kernel)
        take      = (dilated <= 0) & (candidate > 0)
        dilated[take] = candidate[take]
    return dilated.astype(np.float64), dilated > 0


def warp_depth_to_target(source_depth, k_source, d_source, k_target, d_target,
                         t_target_source, source_depth_scale, target_hw, fill_holes=False):
    points_source = _back_project_depth(source_depth, k_source, d_source, source_depth_scale)
    h, w = target_hw
    if points_source.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)
    pts_h         = np.hstack([points_source, np.ones((points_source.shape[0], 1))])
    points_target = (t_target_source @ pts_h.T).T[:, :3]
    front         = points_target[:, 2] > 0
    points_target = points_target[front]
    if points_target.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=bool)
    projected_pixels, _ = cv2.projectPoints(
        points_target.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), k_target, d_target,
    )
    uv = projected_pixels.reshape(-1, 2)
    z  = points_target[:, 2]
    projected, valid = _splat_depth(uv, z, target_hw)
    if fill_holes:
        projected, valid = _fill_small_holes(projected, valid)
    return projected, valid


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  — now takes ignore_mask
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(pred_target_m: np.ndarray,
                     gt_target: np.ndarray,
                     valid_projected: np.ndarray,
                     ignore_mask: Optional[np.ndarray] = None,   # ── IGNORE RECTS ──
                     ) -> dict:
    """
    ignore_mask: bool array same shape as pred/gt.
                 True = keep pixel, False = exclude from metrics.
    """
    valid = (
        valid_projected
        & np.isfinite(pred_target_m) & (pred_target_m > 0)
        & np.isfinite(gt_target)     & (gt_target     > 0)
    )
    # ── IGNORE RECTS ── apply exclusion mask
    if ignore_mask is not None:
        valid = valid & ignore_mask

    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"num_valid_pixels": 0}
    pred, gt = pred_target_m[valid], gt_target[valid]
    diff     = pred - gt
    abs_diff = np.abs(diff)
    return {
        "num_valid_pixels":   n,
        "abs_rel":            float(np.mean(abs_diff / np.maximum(gt, 1e-8))),
        "rmse_m":             float(np.sqrt(np.mean(diff ** 2))),
        "median_abs_error_m": float(np.median(abs_diff)),
        "delta_1_25":         float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Depth resize / undistort  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def resize_depth(depth, target_hw):
    return cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def undistort_depth_map(depth, calib_path):
    calib     = load_yaml_calibration(calib_path)
    undistort = get_undistort_function_mono(calib)
    out       = undistort(depth.astype(np.float32))
    out[~np.isfinite(out)] = 0.0
    return out.astype(depth.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colorize_depth(depth_m, valid, max_depth_m):
    d = depth_m.astype(np.float32).copy()
    d[~valid] = 0.0
    norm  = np.clip((d / max(max_depth_m, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def _label(img, text, y=28):
    out = img.copy()
    for color, thickness in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
    return out


def create_error_heatmap(pred_warped_m, depth_gt, valid_pred,
                         percentile_clip=95.0,
                         ignore_mask: Optional[np.ndarray] = None):  # ── IGNORE RECTS ──
    valid = (
        valid_pred
        & np.isfinite(pred_warped_m) & (pred_warped_m > 0)
        & np.isfinite(depth_gt)      & (depth_gt > 0)
    )
    if ignore_mask is not None:
        valid = valid & ignore_mask

    h, w    = depth_gt.shape
    abs_err = np.zeros((h, w), dtype=np.float32)
    abs_err[valid] = np.abs(pred_warped_m[valid] - depth_gt[valid]).astype(np.float32)
    err_max  = float(np.percentile(abs_err[valid], percentile_clip)) if np.any(valid) else 1.0
    err_norm = np.clip(abs_err / max(err_max, 1e-6), 0, 1)
    err_u8   = (err_norm * 255).astype(np.uint8)
    heatmap  = cv2.applyColorMap(err_u8, cv2.COLORMAP_INFERNO)
    heatmap[~valid] = (30, 30, 30)

    if np.any(valid):
        patch = max(h // 10, 32)
        worst = []
        for py in range(0, h - patch, patch):
            for px in range(0, w - patch, patch):
                reg_v = valid[py:py+patch, px:px+patch]
                if reg_v.sum() < patch * patch * 0.3:
                    continue
                worst.append((float(abs_err[py:py+patch, px:px+patch][reg_v].mean()),
                               px + patch // 2, py + patch // 2))
        worst.sort(reverse=True)
        radius = max(patch // 2, 15)
        for rank, (ev, cx, cy) in enumerate(worst[:5]):
            cv2.circle(heatmap, (cx, cy), radius, (0, 255, 0), 2)
            lx = min(cx + radius + 4, w - 120)
            ly = max(cy + 6, 15)
            for color, th in [((0, 0, 0), 2), ((0, 255, 0), 1)]:
                cv2.putText(heatmap, f"#{rank+1} {ev:.2f}m", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, th, cv2.LINE_AA)

    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    colorbar  = cv2.applyColorMap(np.repeat(gradient, 20, axis=1), cv2.COLORMAP_INFERNO)
    labels    = np.full((h, 90, 3), 245, dtype=np.uint8)
    cv2.putText(labels, f"{err_max:.2f}m", (4, 15),    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(labels, "0.00m",           (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(labels, f"p{percentile_clip:.0f}", (4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)
    return cv2.hconcat([heatmap, colorbar, labels])


def create_comparison_visualization(
    depth_gt: np.ndarray,
    depth_est: np.ndarray,
    pred_warped_m: np.ndarray,
    valid_pred: np.ndarray,
    metrics: dict | None = None,
    ignore_mask: Optional[np.ndarray] = None,   # ── IGNORE RECTS ──
    ignore_rects: list[tuple[int, int, int, int]] | None = None,  # ── IGNORE RECTS ──
) -> np.ndarray:
    valid_gt  = np.isfinite(depth_gt)      & (depth_gt      > 0)
    valid_est = np.isfinite(depth_est)     & (depth_est     > 0)
    valid_w   = np.isfinite(pred_warped_m) & (pred_warped_m > 0)

    all_vals    = np.concatenate([depth_gt[valid_gt], depth_est[valid_est], pred_warped_m[valid_w]])
    shared_vmin = float(np.percentile(all_vals, 2))  if len(all_vals) else 0.0
    shared_vmax = float(np.percentile(all_vals, 98)) if len(all_vals) else 5.0

    vis_gt     = colorize_depth(depth_gt,     vmin=shared_vmin, vmax=shared_vmax)
    vis_est    = colorize_depth(depth_est,     vmin=shared_vmin, vmax=shared_vmax)
    vis_warped = colorize_depth(pred_warped_m, vmin=shared_vmin, vmax=shared_vmax)

    # ── IGNORE RECTS ── overlay rectangles on all three top panels
    if ignore_rects:
        vis_gt     = draw_ignore_rects_on(vis_gt,     ignore_rects)
        vis_est    = draw_ignore_rects_on(vis_est,     ignore_rects)
        vis_warped = draw_ignore_rects_on(vis_warped, ignore_rects)

    panel_h = vis_gt.shape[0]
    def _resize_h(img, h):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)

    vis_gt     = _resize_h(_label(vis_gt,     "GT depth"),          panel_h)
    vis_est    = _resize_h(_label(vis_est,     "Estimated depth"),   panel_h)
    vis_warped = _resize_h(_label(vis_warped, "Warped to GT frame"), panel_h)
    top_row    = cv2.hconcat([vis_gt, vis_est, vis_warped])

    # error heatmap — also masks out ignore rects
    err_raw   = create_error_heatmap(pred_warped_m, depth_gt, valid_pred,
                                     ignore_mask=ignore_mask)  # ── IGNORE RECTS ──
    err_panel = cv2.resize(err_raw, (top_row.shape[1], panel_h), interpolation=cv2.INTER_AREA)

    if ignore_rects:  # ── IGNORE RECTS ── draw on error panel too
        err_panel = draw_ignore_rects_on(err_panel, ignore_rects)

    if metrics:
        txt = (f"AbsRel={metrics.get('abs_rel', 0):.3f}  "
               f"RMSE={metrics.get('rmse_m', 0):.3f}m  "
               f"MedAE={metrics.get('median_abs_error_m', 0)*100:.1f}cm  "
               f"d1={metrics.get('delta_1_25', 0)*100:.1f}%")
        for color, th in [((0, 0, 0), 2), ((0, 230, 0), 1)]:
            cv2.putText(err_panel, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, th, cv2.LINE_AA)

    err_panel = _label(err_panel, "Error |warped - GT|  (circles = worst regions)", y=54)
    composite = cv2.vconcat([top_row, err_panel])

    max_display_w = 1920
    if composite.shape[1] > max_display_w:
        s         = max_display_w / composite.shape[1]
        composite = cv2.resize(composite,
                               (int(composite.shape[1] * s), int(composite.shape[0] * s)),
                               interpolation=cv2.INTER_AREA)
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_depth_comparison_experiment(
    parent_dir: Path,
    date: str,
    nn_name: str,
    rgbd_camera_suffix: str = "realsense",
    max_imgs: int | None = None,
    pose_convention: str = "cam1_from_cam2",
    debug: int = 0,
    ignore_rects: list[tuple[int, int, int, int]] | None = None,  # ── IGNORE RECTS ──
) -> None:
    global IGNORE_RECTS
    if ignore_rects:
        IGNORE_RECTS = list(ignore_rects)

    (
        gt_data_dir, relative_pose_path, calib_rgbd_path,
        calib_stereo_path, depth_estimation_dir, depth_comparison_dir,
    ) = prepare_depth_comparison_paths(parent_dir, date, rgbd_camera_suffix)

    imgs_rgb             = load_rgbd_images(gt_data_dir, suffix=rgbd_camera_suffix, max_imgs=max_imgs)
    estimated_depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=max_imgs)
    frame_size           = estimated_depth_maps[0].shape[::-1]

    k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
    k_source = scale_intrinsics(k_source, (3840, 2160), frame_size)
    k_target, d_target = load_camera_calibration(calib_rgbd_path)
    k_target = scale_intrinsics(k_target, (1280, 720), frame_size)

    transform_target_from_source = _read_transform(relative_pose_path, pose_convention)
    transform_target_from_source[:3, 3] /= 1000.0

    target_hw   = (frame_size[1], frame_size[0])
    window_name = f"Depth comparison  [{rgbd_camera_suffix} | {nn_name}]"

    # ── IGNORE RECTS ── register mouse callback once if in debug mode
    if debug > 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_scale = min(1.0, 1920 / max(target_hw[1] * 3, 1))
        cv2.setMouseCallback(window_name,
                             _make_rect_mouse_cb(window_name, display_scale))
        print("\n[Ignore rects]  Left-drag to add • Right-click to remove last")
        print(f"  Pre-loaded: {IGNORE_RECTS}\n")

    all_metrics = []

    for img_gt, depth_est in tqdm(zip(imgs_rgb, estimated_depth_maps), desc="evaluating"):
        image_number = img_gt.get_image_number()
        depth_gt_raw = img_gt.get_depth()

        depth_est_r = resize_depth(depth_est,    target_hw)
        depth_gt_r  = resize_depth(depth_gt_raw, target_hw)

        pred_warped_m, valid_pred = warp_depth_to_target(
            source_depth=depth_est_r,
            k_source=k_source, d_source=d_source,
            k_target=k_target, d_target=d_target,
            t_target_source=transform_target_from_source,
            source_depth_scale=1.0,
            target_hw=target_hw,
        )

        # ── IGNORE RECTS ── build mask from current rectangles (may change frame-to-frame)
        ignore_mask = build_ignore_mask(target_hw, IGNORE_RECTS) if IGNORE_RECTS else None

        metrics = _compute_metrics(pred_warped_m, depth_gt_r, valid_pred,
                                   ignore_mask=ignore_mask)   # ── IGNORE RECTS ──
        metrics_with_id = {"image_id": image_number, **metrics}

        print(f"  [{image_number}]  RMSE={metrics.get('rmse_m', 0):.3f}m  "
              f"AbsRel={metrics.get('abs_rel', 0):.4f}  "
              f"d1={metrics.get('delta_1_25', 0)*100:.1f}%  "
              f"ignored_rects={len(IGNORE_RECTS)}")   # ── IGNORE RECTS ──

        comparison_vis = create_comparison_visualization(
            depth_gt=depth_gt_r,
            depth_est=depth_est_r,
            pred_warped_m=pred_warped_m,
            valid_pred=valid_pred,
            metrics=metrics,
            ignore_mask=ignore_mask,        # ── IGNORE RECTS ──
            ignore_rects=IGNORE_RECTS,      # ── IGNORE RECTS ──
        )

        save_per_image_results(
            out_root=depth_comparison_dir,
            image_id=image_number,
            pred_warped_m=pred_warped_m,
            gt_depth_m=depth_gt_r,
            depth_est_m=depth_est_r,
            valid_mask=valid_pred,
            metrics=metrics_with_id,
            comparison_vis=comparison_vis,
        )
        all_metrics.append(metrics_with_id)

        if debug > 0:
            # ── IGNORE RECTS ── rebuild vis with any newly drawn rectangles
            current_vis = create_comparison_visualization(
                depth_gt=depth_gt_r, depth_est=depth_est_r,
                pred_warped_m=pred_warped_m, valid_pred=valid_pred,
                metrics=metrics,
                ignore_mask=build_ignore_mask(target_hw, IGNORE_RECTS) if IGNORE_RECTS else None,
                ignore_rects=IGNORE_RECTS,
            )
            # ── IGNORE RECTS ── live-redraw while holding on a frame (debug > 1 = waitKey(0))
            while True:
                cv2.imshow(window_name, current_vis)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('n'):
                    break
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                if key == ord('r'):   # ── IGNORE RECTS ── 'r' recomputes wnith current rects
                    ignore_mask = build_ignore_mask(target_hw, IGNORE_RECTS) if IGNORE_RECTS else None
                    metrics     = _compute_metrics(pred_warped_m, depth_gt_r, valid_pred,
                                                   ignore_mask=ignore_mask)
                    current_vis = create_comparison_visualization(
                        depth_gt=depth_gt_r, depth_est=depth_est_r,
                        pred_warped_m=pred_warped_m, valid_pred=valid_pred,
                        metrics=metrics,
                        ignore_mask=ignore_mask,
                        ignore_rects=IGNORE_RECTS,
                    )
                    print(f"  [recomputed]  RMSE={metrics.get('rmse_m', 0):.3f}m  "
                          f"rects={IGNORE_RECTS}")
                elif debug > 1:
                    continue          # stay on this frame
                else:
                    break             # advance automatically

    if debug > 0:
        cv2.destroyAllWindows()

    save_summary_results(out_root=depth_comparison_dir, all_metrics=all_metrics)
    # ── IGNORE RECTS ── print final rect config so you can hard-code it next run
    if IGNORE_RECTS:
        print(f"\n[Ignore rects used]  ignore_rects={IGNORE_RECTS}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parent_dir = Path(__file__).resolve().parents[3]

    run_depth_comparison_experiment(
        parent_dir,
        date="13042026",
        nn_name="S2M2_stereo",
        rgbd_camera_suffix="zed",
        max_imgs=50,
        debug=2,
        # ── IGNORE RECTS ── optionally hard-code known bad regions, e.g.:
        # ignore_rects=[(0, 0, 80, 720), (1200, 600, 80, 120)],
        ignore_rects=None,
    )