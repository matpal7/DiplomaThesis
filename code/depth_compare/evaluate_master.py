from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.ZED_tolerance_depth import within_sensor_tolerance, sensor_filtered_errors
from code.depth_compare.compare_depth_between_cameras import warp_depth_to_target, _read_transform, _compute_metrics
from code.depth_compare.evaluation import eval_depth_single
# import torch

# from code.depth_compare.evaluation import eval_depth
from code.image import load_rgbd_images
from code.utils import load_estimated_depth_map, scale_intrinsics
from code.visualize_depth import colorize_depth
from prepare_paths import get_depth_estimation_network_names, \
    prepare_depth_comparison_paths

parent_dir = Path(__file__).resolve().parent.parent.parent.parent
date       = "27032026"
rgbd_suffix = "zed"
max_imgs    = None

(
    gt_data_dir,
    relative_pose_path,
    calib_rgbd_path,
    calib_stereo_path,
    depth_estimation_dir,
    depth_comparison_dir,
) = prepare_depth_comparison_paths(parent_dir, date, rgbd_suffix)

imgs_rgb = load_rgbd_images(gt_data_dir, suffix=rgbd_suffix, max_imgs=max_imgs)

k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
k_target, d_target = load_camera_calibration(calib_rgbd_path)
pose_convention = "cam1_from_cam2"
transform_target_from_source = _read_transform(relative_pose_path, pose_convention)
transform_target_from_source[:3, 3] /= 1000.0

def resize_depth_safe(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    valid = np.isfinite(depth) & (depth > 0)
    depth_filled = depth.copy()
    depth_filled[~valid] = 0.0
    depth_sum = cv2.resize(depth_filled.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid_sum = cv2.resize(valid.astype(np.float32),        (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w), np.nan, dtype=np.float32)
    valid_out = valid_sum > 0.1
    out[valid_out] = depth_sum[valid_out] / valid_sum[valid_out]
    return out

# ── load estimated depth maps ─────────────────────────────────────────────────
estimated_depth_maps = {}
for nn_name in get_depth_estimation_network_names():
    estimated_depth_maps[nn_name] = load_estimated_depth_map(
        depth_estimation_dir, nn_name, max_imgs=max_imgs
    )

metrics_out_dir = depth_comparison_dir / "metrics"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

REPORT_COLS    = ["abs_rel", "rmse_m", "median_abs_error_m", "delta_1_25"]
EVAL_DEPTH_AGG = [
    "d1",           # threshold accuracy (raw)
    "d1_ssi",       # threshold accuracy (scale+shift aligned)
    "d1_si",        # threshold accuracy (scale aligned)
    "arel",         # relative error (raw)
    "arel_ssi",     # relative error (scale+shift aligned)
    "rmse",         # absolute error sensitivity
    "median_abs_error",  # robust typical error
    "silog",        # scale-invariant log error
]

# ZED sensor tolerance — one per concept
TOL_AGG = [
    "within_1.0sigma",          # strict: % within ZED 1σ noise floor
    "within_3.0sigma",          # lenient: % within ZED 3σ band
    "median_error_in_sigmas",   # robust: typical error in sensor units
    "sensor_sigma_mean_m",      # reference: ZED's own noise at scene depth
    "abs_error_mean_m",         # absolute error (all pixels)
    "pct_outside",              # % pixels with genuine network error
    "filtered_median_abs_error_m",  # robust error on failure pixels only
    "filtered_arel",            # relative error on failure pixels only
]
all_networks_summary = []


for nn_name in estimated_depth_maps:
    print(f"\n{'━'*60}")
    print(f"  Network: {nn_name}")
    print(f"{'━'*60}")

    frame_size     = estimated_depth_maps[nn_name][0].shape[::-1]
    k_source_scaled = scale_intrinsics(k_source.copy(), (3840, 2160), frame_size)
    k_target_scaled = scale_intrinsics(k_target.copy(), (1280, 720),  frame_size)

    all_gt, all_est = [], []
    per_image_rows  = []

    for img_gt, depth_est in tqdm(
        zip(imgs_rgb, estimated_depth_maps[nn_name]),
        desc=f"[{nn_name}]",
    ):
        image_number = img_gt.get_image_number()
        depth_gt     = img_gt.get_depth()
        depth_gt     = resize_depth_safe(depth_gt, depth_est.shape)

        valid = (
            np.isfinite(depth_est) & (depth_est > 0) &
            np.isfinite(depth_gt)  & (depth_gt  > 0)
        )

        # ── warp + geometric metrics ──────────────────────────────────────────
        pred_warped_m, valid_pred = warp_depth_to_target(
            source_depth=depth_est,
            k_source=k_source_scaled,
            d_source=d_source,
            k_target=k_target_scaled,
            d_target=d_target,
            t_target_source=transform_target_from_source,
            source_depth_scale=1.0,
            target_hw=(depth_gt.shape[0], depth_gt.shape[1]),
        )
        # warp_metrics = _compute_metrics(pred_warped_m, depth_gt, valid_pred)

        # ── eval_depth metrics (no warp, direct comparison) ───────────────────
        mask_eval  = (
            np.isfinite(pred_warped_m) & (pred_warped_m > 0) &
            np.isfinite(depth_gt)  & (depth_gt  > 0)
        )
        eval_metrics = eval_depth_single(
            gt=depth_gt,
            pred=pred_warped_m,
            mask=mask_eval,
            max_depth=None,
        )

        tol_1sigma = within_sensor_tolerance(
            gt=depth_gt, pred=pred_warped_m, mask=mask_eval, n_sigma=1.0
        )
        tol_3sigma = within_sensor_tolerance(
            gt=depth_gt, pred=pred_warped_m, mask=mask_eval, n_sigma=3.0
        )
        # filtered metrics use 1σ as the exclusion threshold
        filtered = sensor_filtered_errors(
            gt=depth_gt, pred=pred_warped_m, mask=mask_eval, n_sigma=1.0
        )

        within_tolerance_combined = {**tol_1sigma, **tol_3sigma, **filtered}

        row = {
            "image_id": image_number,
            "nn_name": nn_name,
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
            **{f"tol_{k}": v for k, v in within_tolerance_combined.items()},
        }
        per_image_rows.append(row)

        if np.any(valid):
            all_gt.append(depth_gt[valid])
            all_est.append(depth_est[valid])

        # ── console per-image summary ─────────────────────────────────────────
        print(f"  [{image_number}]  "
              f"eval d1={eval_metrics.get('d1', float('nan'))*100:.1f}%  "
              f"arel={eval_metrics.get('arel', float('nan')):.3f}  "
              f"silog={eval_metrics.get('silog', float('nan')):.2f}"
              f"medAE={eval_metrics.get('median_abs_error', float('nan')):.3f}m"
        )

    # ── save per-image CSV ────────────────────────────────────────────────────
    df_per_image  = pd.DataFrame(per_image_rows)
    per_image_csv = metrics_out_dir / f"{nn_name}_per_image_metrics.csv"
    df_per_image.to_csv(per_image_csv, index=False)

    if len(all_gt) == 0:
        print("  No valid overlapping pixels — skipping summary.")
        continue

    all_gt_cat  = np.concatenate(all_gt)
    all_est_cat = np.concatenate(all_est)
    global_scale = float(np.median(all_gt_cat) / np.median(all_est_cat))

    # ── aggregate both metric groups ─────────────────────────────────────────
    aggregated = {"nn_name": nn_name, "global_scale": global_scale}


    # eval_depth metrics
    for col in EVAL_DEPTH_AGG:
        eval_col = f"eval_{col}"
        if eval_col in df_per_image.columns:
            aggregated[f"{col}_mean"] = float(df_per_image[eval_col].mean())

    for col in TOL_AGG:
        tol_col = f"tol_{col}"
        if tol_col in df_per_image.columns:
            aggregated[f"{col}_mean"] = float(df_per_image[tol_col].mean())

    all_networks_summary.append(aggregated)

    # ── save per-network summary CSV ──────────────────────────────────────────
    df_net = pd.DataFrame([aggregated])
    df_net.to_csv(metrics_out_dir / f"{nn_name}_summary_metrics.csv", index=False)

    # ── console network summary ───────────────────────────────────────────────
    print(f"\n  ── {nn_name} summary ──")
    print(f"  Global scale          : {global_scale:.4f}")
    print(f"  [Depth accuracy]")
    print(f"    d1 (raw)            : {aggregated.get('d1_mean', float('nan')) * 100:.2f}%")
    print(f"    d1_ssi (aligned)    : {aggregated.get('d1_ssi_mean', float('nan')) * 100:.2f}%")
    print(f"    d1_si  (scale only) : {aggregated.get('d1_si_mean', float('nan')) * 100:.2f}%")
    print(f"    arel (raw)          : {aggregated.get('arel_mean', float('nan')):.4f}")
    print(f"    arel_ssi (aligned)  : {aggregated.get('arel_ssi_mean', float('nan')):.4f}")
    print(f"    rmse                : {aggregated.get('rmse_mean', float('nan')):.4f} m")
    print(f"    median abs error    : {aggregated.get('median_abs_error_mean', float('nan')) * 100:.2f} cm")
    print(f"    silog               : {aggregated.get('silog_mean', float('nan')):.4f}")
    print(f"  [ZED sensor tolerance]")
    print(f"    within 1σ           : {aggregated.get('within_1.0sigma_mean', float('nan')) * 100:.2f}%")
    print(f"    within 3σ           : {aggregated.get('within_3.0sigma_mean', float('nan')) * 100:.2f}%")
    print(f"    median error        : {aggregated.get('median_error_in_sigmas_mean', float('nan')):.2f} σ")
    print(f"    sensor sigma (ref)  : {aggregated.get('sensor_sigma_mean_m_mean', float('nan')) * 100:.2f} cm")
    print(f"  [Failure pixel analysis]")
    print(f"    pixels outside 1σ   : {aggregated.get('pct_outside_mean', float('nan')):.1f}%")
    print(f"    failure median AE   : {aggregated.get('filtered_median_abs_error_m_mean', float('nan')) * 100:.2f} cm")
    print(f"    failure arel        : {aggregated.get('filtered_arel_mean', float('nan')):.4f}")
    print(f"  Saved → {metrics_out_dir / f'{nn_name}_summary_metrics.csv'}")

    # ── global summary across ALL networks ───────────────────────────────────────
if all_networks_summary:
    df_global = pd.DataFrame(all_networks_summary)
    global_csv = metrics_out_dir / "all_networks_global_summary.csv"
    df_global.to_csv(global_csv, index=False)

    print(f"\n{'═'*60}")
    print("  GLOBAL SUMMARY — ALL NETWORKS")
    print(f"{'═'*60}")
    print(df_global.to_string(index=False))
    print(f"\n  Saved → {global_csv}")