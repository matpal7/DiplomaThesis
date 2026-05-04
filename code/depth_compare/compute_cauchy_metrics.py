from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

from CI_calculation import load_zed_gamma
from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.compare_depth_between_cameras import _read_transform, warp_depth_to_target
from code.image import load_rgbd_images
from code.prepare_paths import get_depth_estimation_network, prepare_depth_comparison_paths
from code.utils import load_estimated_depth_map, scale_intrinsics

DATE = "24042026"
RGBD_SUFFIX = "zed"
MAX_IMGS = None
# ── Multiple confidence levels ──────────────────────────────────────────────
CI_LEVELS = [0.90, 0.95, 0.99, 0.999]
CI_LABELS = {0.90: "90", 0.95: "95", 0.99: "99", 0.999: "999"}
# ────────────────────────────────────────────────────────────────────────────
SOURCE_HW = (2160, 3840)
TARGET_HW = (720, 1280)
OUTPUT_SUBDIR = "metrics_cauchy"


def resize_depth_validity_weighted(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    valid = np.isfinite(depth) & (depth > 0)
    weighted_depth = depth.astype(np.float32).copy()
    weighted_depth[~valid] = 0.0
    depth_interp = cv2.resize(weighted_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    valid_interp = cv2.resize(valid.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    output = np.full((target_h, target_w), np.nan, dtype=np.float32)
    valid_out = valid_interp > 1e-2
    output[valid_out] = depth_interp[valid_out] / valid_interp[valid_out]
    return output


def _standard_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / np.clip(gt, 1e-9, None)
    sq_err = (pred - gt) ** 2
    ratio = np.maximum(pred / np.clip(gt, 1e-9, None), gt / np.clip(pred, 1e-9, None))
    log_diff = np.log(np.clip(pred, 1e-9, None)) - np.log(np.clip(gt, 1e-9, None))
    silog_var = np.mean(log_diff ** 2) - np.mean(log_diff) ** 2
    return {
        "AbsRel":   float(np.mean(rel_err)),
        "RMSE":     float(np.sqrt(np.mean(sq_err))),
        "MAE":      float(np.mean(abs_err)),
        "MedianAE": float(np.median(abs_err)),
        "SILog":    float(np.sqrt(max(silog_var, 0.0))),
        "delta1":   float(np.mean(ratio < 1.25)),
        "delta2":   float(np.mean(ratio < (1.25 ** 2))),
        "delta3":   float(np.mean(ratio < (1.25 ** 3))),
    }


def compute_cauchy_metrics(
    depth_gt: np.ndarray,
    pred_warped: np.ndarray,
    valid_mask: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    gt   = depth_gt[valid_mask].astype(np.float64)
    pred = pred_warped[valid_mask].astype(np.float64)

    eps = (pred - gt) / np.clip(gt, 1e-9, None)
    z   = eps / gamma
    std_all = _standard_metrics(gt, pred)

    out: dict[str, float] = {
        "n_pixels":        int(valid_mask.sum()),
        "within_1gamma_pct": float(np.mean(np.abs(z) <= 1.0) * 100.0),
        "within_2gamma_pct": float(np.mean(np.abs(z) <= 2.0) * 100.0),
        "median_abs_z":    float(np.median(np.abs(z))),
        "mean_abs_z":      float(np.mean(np.abs(z))),
        "mean_log_lik":    float(np.mean(scipy_stats.cauchy.logpdf(eps, loc=0.0, scale=gamma))),
        "cauchy_ks_stat":  float(scipy_stats.kstest(z, "cauchy").statistic),
        "cauchy_ks_pvalue":float(scipy_stats.kstest(z, "cauchy").pvalue),
    }

    # ── Per-CI-level metrics ────────────────────────────────────────────────
    for ci in CI_LEVELS:
        label     = CI_LABELS[ci]
        ci_thresh = float(scipy_stats.cauchy.ppf(0.5 + ci / 2.0, loc=0, scale=1))
        ci_bound  = gamma * ci_thresh
        within_ci = np.abs(eps) <= ci_bound

        out[f"within_ci{label}_pct"]       = float(np.mean(within_ci) * 100.0)
        out[f"outside_ci{label}_pct"]      = float(np.mean(~within_ci) * 100.0)
        out[f"n_pixels_outside_ci{label}"] = int((~within_ci).sum())
    # ────────────────────────────────────────────────────────────────────────

    for key, value in std_all.items():
        out[f"all_{key}"] = value

    return out


def main() -> None:
    parent_dir = Path(__file__).resolve().parents[3]
    (
        gt_data_dir, relative_pose_path, calib_rgbd_path,
        calib_stereo_path, depth_estimation_dir, depth_comparison_dir,
    ) = prepare_depth_comparison_paths(parent_dir, DATE, RGBD_SUFFIX)

    output_dir = depth_comparison_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
    k_target, d_target = load_camera_calibration(calib_rgbd_path)
    transform_target_from_source = _read_transform(relative_pose_path, "cam1_from_cam2")
    transform_target_from_source[:3, 3] /= 1000.0

    camera_stats_path = parent_dir / "out" / f"out_{DATE}" / "cameras_statistic_model" / "best_distribution_models.json"
    zed_gamma = load_zed_gamma(camera_stats_path, camera="zed")

    gt_frames = load_rgbd_images(gt_data_dir, suffix=RGBD_SUFFIX, max_imgs=MAX_IMGS)

    summaries = []
    for nn_key, nn_value in get_depth_estimation_network().items():
        pred_depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_key, max_imgs=MAX_IMGS)
        if not pred_depth_maps:
            continue

        frame_size       = pred_depth_maps[0].shape[::-1]
        k_source_scaled  = scale_intrinsics(k_source.copy(), SOURCE_HW[::-1], frame_size)
        k_target_scaled  = scale_intrinsics(k_target.copy(), TARGET_HW[::-1], frame_size)

        rows = []
        for gt_frame, pred_depth in tqdm(zip(gt_frames, pred_depth_maps), desc=f"[{nn_value}]", total=len(pred_depth_maps)):
            depth_gt = resize_depth_validity_weighted(gt_frame.get_depth(), pred_depth.shape)
            pred_warped, _ = warp_depth_to_target(
                source_depth=pred_depth,
                k_source=k_source_scaled,
                d_source=d_source,
                k_target=k_target_scaled,
                d_target=d_target,
                t_target_source=transform_target_from_source,
                source_depth_scale=1.0,
                target_hw=depth_gt.shape,
            )

            valid_mask = (
                np.isfinite(pred_warped) & (pred_warped > 0) &
                np.isfinite(depth_gt)   & (depth_gt > 0)
            )
            if not valid_mask.any():
                continue

            metrics = compute_cauchy_metrics(depth_gt, pred_warped, valid_mask, zed_gamma)
            rows.append({"image_id": gt_frame.get_image_number(), "nn_name": nn_key, **metrics})

        if not rows:
            continue

        per_image_df = pd.DataFrame(rows)
        per_image_df.to_csv(output_dir / f"{nn_key}_per_image.csv", index=False)

        summary = {"nn_name": nn_key, "zed_gamma": zed_gamma, "ci_levels": str(CI_LEVELS)}
        for col in per_image_df.columns:
            if col in {"image_id", "nn_name"}:
                continue
            summary[col] = float(
                per_image_df[col].sum()
                if col in {"n_pixels"} or col.startswith("n_pixels_outside_ci")
                else per_image_df[col].mean()
            )
        summaries.append(summary)

    if not summaries:
        print("[WARN] No metrics were computed.")
        return

    df_summary = pd.DataFrame(summaries).sort_values("median_abs_z")
    global_path = output_dir / "all_networks_summary.csv"
    df_summary.to_csv(global_path, index=False)

    print(f"\n{'═' * 130}")
    print("Global Cauchy + MDE network summary")
    print(f"{'═' * 130}")

    # Build column list dynamically for all CI levels
    ci_cols = []
    for ci in CI_LEVELS:
        label = CI_LABELS[ci]
        ci_cols += [f"within_ci{label}_pct", f"outside_ci{label}_pct"]

    table_cols = [
        "nn_name",
        "median_abs_z", "mean_abs_z",
        *ci_cols,
        "within_1gamma_pct", "within_2gamma_pct",
        "mean_log_lik", "cauchy_ks_stat", "cauchy_ks_pvalue",
        "all_AbsRel", "all_RMSE", "all_MAE", "all_MedianAE",
        "all_SILog", "all_delta1", "all_delta2", "all_delta3",
    ]
    available_cols = [c for c in table_cols if c in df_summary.columns]

    formatters = {
        "median_abs_z":      "{:.4f}".format,
        "mean_abs_z":        "{:.4f}".format,
        "within_1gamma_pct": "{:.2f}".format,
        "within_2gamma_pct": "{:.2f}".format,
        "mean_log_lik":      "{:.5f}".format,
        "cauchy_ks_stat":    "{:.5f}".format,
        "cauchy_ks_pvalue":  "{:.3e}".format,
        "all_AbsRel":        "{:.5f}".format,
        "all_RMSE":          "{:.5f}".format,
        "all_MAE":           "{:.5f}".format,
        "all_MedianAE":      "{:.5f}".format,
        "all_SILog":         "{:.5f}".format,
        "all_delta1":        "{:.4f}".format,
        "all_delta2":        "{:.4f}".format,
        "all_delta3":        "{:.4f}".format,
    }
    for ci in CI_LEVELS:
        label = CI_LABELS[ci]
        formatters[f"within_ci{label}_pct"]  = "{:.2f}".format
        formatters[f"outside_ci{label}_pct"] = "{:.2f}".format

    print(df_summary[available_cols].to_string(index=False, formatters=formatters))

    print(f"\n{'─' * 130}")
    print("Best network by criterion")
    print(f"{'─' * 130}")
    print(f"  • lowest  median_abs_z        : {df_summary.loc[df_summary['median_abs_z'].idxmin(), 'nn_name']}")
    print(f"  • lowest  mean_abs_z          : {df_summary.loc[df_summary['mean_abs_z'].idxmin(), 'nn_name']}")
    print(f"  • highest within_1gamma_pct   : {df_summary.loc[df_summary['within_1gamma_pct'].idxmax(), 'nn_name']}")
    print(f"  • highest within_2gamma_pct   : {df_summary.loc[df_summary['within_2gamma_pct'].idxmax(), 'nn_name']}")
    print(f"  • highest mean_log_lik        : {df_summary.loc[df_summary['mean_log_lik'].idxmax(), 'nn_name']}")
    print(f"  • lowest  cauchy_ks_stat      : {df_summary.loc[df_summary['cauchy_ks_stat'].idxmin(), 'nn_name']}")
    for ci in CI_LEVELS:
        label = CI_LABELS[ci]
        col   = f"within_ci{label}_pct"
        best  = df_summary.loc[df_summary[col].idxmax(), "nn_name"]
        print(f"  • highest within_ci{label}_pct     : {best}")

    print(f"\n[INFO] Saved metrics to: {global_path}")


if __name__ == "__main__":
    main()