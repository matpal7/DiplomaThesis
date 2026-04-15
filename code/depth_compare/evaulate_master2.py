from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.camera_model_parameters import load_sensor_model
from code.depth_compare.compare_depth_between_cameras import warp_depth_to_target, _read_transform
from code.depth_compare.evaluation import eval_depth_single
from code.image import load_rgbd_images
from code.prepare_paths import prepare_depth_comparison_paths, get_depth_estimation_network_names
from code.utils import load_estimated_depth_map, scale_intrinsics


parent_dir  = Path(__file__).resolve().parents[3]
date        = "12042026"
rgbd_suffix = "realsense"
max_imgs    = 50

(
    gt_data_dir,
    relative_pose_path,
    calib_rgbd_path,
    calib_stereo_path,
    depth_estimation_dir,
    depth_comparison_dir,
) = prepare_depth_comparison_paths(parent_dir, date, rgbd_suffix)

imgs_rgb = load_rgbd_images(gt_data_dir, suffix=rgbd_suffix, max_imgs=max_imgs)

print(calib_rgbd_path)

k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
k_target, d_target = load_camera_calibration(calib_rgbd_path)
transform_target_from_source = _read_transform(relative_pose_path, "cam1_from_cam2")
transform_target_from_source[:3, 3] /= 1000.0

camera_stats_dir = parent_dir / "out" / "out_09042026" / "cameras_statistic_model"
sensor_model     = load_sensor_model(camera_stats_dir, rgbd_suffix)

assert sensor_model is not None, (
    f"[✗] Sensor model not found in {camera_stats_dir}. "
    "Run camera statistics fitting before this script."
)
alpha = sensor_model["rel_alpha"]
beta  = sensor_model["rel_beta"]
print(f"[✓] Loaded sensor model: σ_rel(Z) = {alpha:.8f} · Z^{beta:.3f}")


# ── Metric lists ──────────────────────────────────────────────────────────────
EVAL_DEPTH_AGG = [
    "d1", "d1_ssi", "d1_si",
    "d2", "d3",
    "tau", "tau_ssi", "tau_si",
    "d_auc",
    "arel", "arel_ssi", "arel_si",
    "sqrel",
    "rmse", "rmselog",
    "median_abs_error",
    "silog",
]

TOL_AGG = [
    "within_1.960sigma",
    "median_error_in_sigmas",
    "sensor_sigma_mean_m",
    "abs_error_mean_m",
    "pct_outside",
    "filtered_median_abs_error_m",
    "filtered_arel",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_depth_safe(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w   = target_hw
    valid  = np.isfinite(depth) & (depth > 0)
    filled = depth.copy()
    filled[~valid] = 0.0
    depth_sum = cv2.resize(filled.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid_sum = cv2.resize(valid.astype(np.float32),  (w, h), interpolation=cv2.INTER_LINEAR)
    out       = np.full((h, w), np.nan, dtype=np.float32)
    ok        = valid_sum > 0.1
    out[ok]   = depth_sum[ok] / valid_sum[ok]
    return out


def within_power_model_tolerance(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    beta: float,
    n_sigma: float = 1.96,
) -> tuple[dict, np.ndarray]:
    """
    Pixels pass where |pred - gt| <= n_sigma * alpha * gt^(beta+1).

    Returns:
        metrics         — dict of tolerance statistics
        outside_mask    — full-size bool array, True where pixel is outside the band
    """
    gt_m   = gt[mask].astype(np.float64)
    pred_m = pred[mask].astype(np.float64)

    sigma_abs = alpha * np.power(gt_m, beta + 1.0)
    abs_err   = np.abs(pred_m - gt_m)
    err_sigma = abs_err / np.clip(sigma_abs, 1e-9, None)
    within    = abs_err <= n_sigma * sigma_abs
    outside   = ~within

    filt_median_ae = float(np.median(abs_err[outside])) if outside.any() else 0.0
    filt_arel      = float(np.median(abs_err[outside] / gt_m[outside])) if outside.any() else 0.0

    outside_mask_full = np.zeros(gt.shape, dtype=bool)
    outside_mask_full[mask] = outside

    metrics = {
        f"within_{n_sigma:.3f}sigma":      float(within.mean()),
        "median_error_in_sigmas":          float(np.median(err_sigma)),
        "sensor_sigma_mean_m":             float(sigma_abs.mean()),
        "abs_error_mean_m":                float(abs_err.mean()),
        "pct_outside":                     float(outside.mean() * 100),
        "filtered_median_abs_error_m":     filt_median_ae,
        "filtered_arel":                   filt_arel,
    }
    return metrics, outside_mask_full


# ── Load estimated depth maps ─────────────────────────────────────────────────
estimated_depth_maps = {}
for nn_name in get_depth_estimation_network_names():
    estimated_depth_maps[nn_name] = load_estimated_depth_map(
        depth_estimation_dir, nn_name, max_imgs=max_imgs
    )

metrics_out_dir = depth_comparison_dir / "metrics"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

all_networks_summary = []

# ── Per-network loop ──────────────────────────────────────────────────────────
for nn_name in estimated_depth_maps:
    print(f"\n{'━'*60}")
    print(f"  Network : {nn_name}")
    print(f"  Model   : σ(Z) = {alpha:.8f} · Z^{beta:.3f}")
    print(f"{'━'*60}")

    frame_size      = estimated_depth_maps[nn_name][0].shape[::-1]
    k_source_scaled = scale_intrinsics(k_source.copy(), (3840, 2160), frame_size)
    k_target_scaled = scale_intrinsics(k_target.copy(), (1280,  720), frame_size)

    all_gt, all_est = [], []
    per_image_rows  = []

    for img_gt, depth_est in tqdm(
        zip(imgs_rgb, estimated_depth_maps[nn_name]),
        desc=f"[{nn_name}]",
    ):
        image_number = img_gt.get_image_number()
        depth_gt     = resize_depth_safe(img_gt.get_depth(), depth_est.shape)

        valid = (
            np.isfinite(depth_est) & (depth_est > 0) &
            np.isfinite(depth_gt)  & (depth_gt  > 0)
        )

        pred_warped_m, _ = warp_depth_to_target(
            source_depth=depth_est,
            k_source=k_source_scaled,
            d_source=d_source,
            k_target=k_target_scaled,
            d_target=d_target,
            t_target_source=transform_target_from_source,
            source_depth_scale=1.0,
            target_hw=(depth_gt.shape[0], depth_gt.shape[1]),
        )

        mask_eval = (
            np.isfinite(pred_warped_m) & (pred_warped_m > 0) &
            np.isfinite(depth_gt)      & (depth_gt      > 0)
        )

        # ── tolerance metrics + outside mask ─────────────────────────────────
        tol_metrics, outside_mask = within_power_model_tolerance(
            gt=depth_gt, pred=pred_warped_m, mask=mask_eval,
            alpha=alpha, beta=beta, n_sigma=1.96,
        )

        # ── depth accuracy on failure pixels only ─────────────────────────────
        failure_mask = mask_eval & outside_mask
        fail_eval = (
            eval_depth_single(gt=depth_gt, pred=pred_warped_m,
                              mask=failure_mask, max_depth=None)
            if failure_mask.any() else {}
        )

        # ── console per-image summary ─────────────────────────────────────────
        print(
            f"  [{image_number}]  "
            f"band={tol_metrics.get('within_1.960sigma', float('nan'))*100:.1f}%  "
            f"pctOut={tol_metrics.get('pct_outside', float('nan')):.1f}%  "
            f"absErr={tol_metrics.get('abs_error_mean_m', float('nan'))*100:.2f}cm  "
            f"filtMedAE={tol_metrics.get('filtered_median_abs_error_m', float('nan'))*100:.2f}cm  "
            f"fail_d1={fail_eval.get('d1', float('nan'))*100:.1f}%  "
            f"fail_arel={fail_eval.get('arel', float('nan')):.3f}  "
            f"fail_rmse={fail_eval.get('rmse', float('nan'))*100:.2f}cm  "
            f"fail_silog={fail_eval.get('silog', float('nan')):.2f}"
        )

        row = {
            "image_id": image_number,
            "nn_name":  nn_name,
            **{f"tol_{k}":  v for k, v in tol_metrics.items()},
            **{f"fail_{k}": v for k, v in fail_eval.items()},
        }
        per_image_rows.append(row)

        if valid.any():
            all_gt.append(depth_gt[valid])
            all_est.append(depth_est[valid])

    # ── save per-image CSV ────────────────────────────────────────────────────
    df_per_image = pd.DataFrame(per_image_rows)
    df_per_image.to_csv(metrics_out_dir / f"{nn_name}_per_image_metrics.csv", index=False)
    print(f"  Saved per-image → {metrics_out_dir / f'{nn_name}_per_image_metrics.csv'}")

    if not all_gt:
        print("  No valid overlapping pixels — skipping summary.")
        continue

    global_scale = float(
        np.median(np.concatenate(all_gt)) / np.median(np.concatenate(all_est))
    )

    # ── aggregate ─────────────────────────────────────────────────────────────
    aggregated = {
        "nn_name":            nn_name,
        "global_scale":       global_scale,
        "sensor_model_alpha": alpha,
        "sensor_model_beta":  beta,
    }

    for col in TOL_AGG:
        tol_col = f"tol_{col}"
        if tol_col in df_per_image.columns:
            aggregated[f"{col}_mean"] = float(df_per_image[tol_col].mean())

    for col in EVAL_DEPTH_AGG:
        fail_col = f"fail_{col}"
        if fail_col in df_per_image.columns:
            aggregated[f"fail_{col}_mean"] = float(df_per_image[fail_col].mean())

    all_networks_summary.append(aggregated)
    pd.DataFrame([aggregated]).to_csv(
        metrics_out_dir / f"{nn_name}_summary_metrics.csv", index=False
    )

    # ── console network summary ───────────────────────────────────────────────
    print(f"\n  ── {nn_name} summary ──")
    print(f"  Global scale          : {global_scale:.4f}")
    print(f"  [Sensor tolerance — all pixels]")
    print(f"    within 1.96σ (95%)  : {aggregated.get('within_1.960sigma_mean', float('nan'))*100:.2f}%")
    print(f"    abs error mean      : {aggregated.get('abs_error_mean_m_mean', float('nan'))*100:.2f} cm")
    print(f"    sensor σ ref        : {aggregated.get('sensor_sigma_mean_m_mean', float('nan'))*100:.2f} cm")
    print(f"    median error (σ)    : {aggregated.get('median_error_in_sigmas_mean', float('nan')):.3f} σ")
    print(f"  [Failure pixels — outside 1.96σ band]")
    print(f"    pct outside         : {aggregated.get('pct_outside_mean', float('nan')):.1f}%")
    print(f"    failure median AE   : {aggregated.get('filtered_median_abs_error_m_mean', float('nan'))*100:.2f} cm")
    print(f"    failure arel        : {aggregated.get('filtered_arel_mean', float('nan')):.4f}")
    print(f"  [Depth accuracy — failure pixels only]")
    print(f"    fail d1             : {aggregated.get('fail_d1_mean', float('nan'))*100:.2f}%")
    print(f"    fail d1_ssi         : {aggregated.get('fail_d1_ssi_mean', float('nan'))*100:.2f}%")
    print(f"    fail d2             : {aggregated.get('fail_d2_mean', float('nan'))*100:.2f}%")
    print(f"    fail d3             : {aggregated.get('fail_d3_mean', float('nan'))*100:.2f}%")
    print(f"    fail tau            : {aggregated.get('fail_tau_mean', float('nan'))*100:.2f}%")
    print(f"    fail d_auc          : {aggregated.get('fail_d_auc_mean', float('nan')):.4f}")
    print(f"    fail arel           : {aggregated.get('fail_arel_mean', float('nan')):.4f}")
    print(f"    fail arel_ssi       : {aggregated.get('fail_arel_ssi_mean', float('nan')):.4f}")
    print(f"    fail sqrel          : {aggregated.get('fail_sqrel_mean', float('nan')):.6f}")
    print(f"    fail rmse           : {aggregated.get('fail_rmse_mean', float('nan'))*100:.2f} cm")
    print(f"    fail rmselog        : {aggregated.get('fail_rmselog_mean', float('nan')):.4f}")
    print(f"    fail median abs err : {aggregated.get('fail_median_abs_error_mean', float('nan'))*100:.2f} cm")
    print(f"    fail silog          : {aggregated.get('fail_silog_mean', float('nan')):.4f}")
    print(f"  Saved → {metrics_out_dir / f'{nn_name}_summary_metrics.csv'}")

# ── global summary ────────────────────────────────────────────────────────────
if all_networks_summary:
    df_global = pd.DataFrame(all_networks_summary)
    global_csv = metrics_out_dir / "all_networks_global_summary.csv"
    df_global.to_csv(global_csv, index=False)
    print(f"\n{'═'*60}")
    print("  GLOBAL SUMMARY — ALL NETWORKS")
    print(f"{'═'*60}")
    print(df_global.to_string(index=False))
    print(f"\n  Saved → {global_csv}")