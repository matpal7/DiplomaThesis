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
from code.prepare_paths import get_depth_estimation_network_names, prepare_depth_comparison_paths
from code.utils import load_estimated_depth_map, scale_intrinsics

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATE = "24042026"
RGBD_SUFFIX = "zed"
MAX_IMGS = None
CONFIDENCE = 0.95
SOURCE_HW = (2160, 3840)
TARGET_HW = (720, 1280)
DEBUG_VIS = True
DEBUG_SAVE = False
DEBUG_MAX_FRAMES_PER_NETWORK = 5


def resize_depth_validity_weighted(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize depth while avoiding invalid (0/NaN/Inf) values via validity-weighted interpolation."""
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


def depth_to_colormap(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert depth map to a color image for quick visual debugging."""
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 0)

    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid_mask.any():
        return vis

    d = depth.astype(np.float32).copy()
    d[~valid_mask] = np.nan
    d_min = float(np.nanpercentile(d, 2))
    d_max = float(np.nanpercentile(d, 98))
    if d_max <= d_min:
        d_max = d_min + 1e-6

    d_norm = np.clip((d - d_min) / (d_max - d_min), 0, 1)
    d_norm[~valid_mask] = 0
    d_u8 = (d_norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    vis[~valid_mask] = 0
    return vis


def debug_show_depth_triplet(
    image_id: int,
    nn_name: str,
    rgb_img: np.ndarray,
    depth_gt: np.ndarray,
    depth_pred: np.ndarray,
    depth_warped: np.ndarray,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Create and optionally display/save a concatenated debug image:
    [RGB | GT depth | Predicted depth | Warped predicted depth].
    """
    gt_mask = np.isfinite(depth_gt) & (depth_gt > 0)
    pred_mask = np.isfinite(depth_pred) & (depth_pred > 0)
    warped_mask = np.isfinite(depth_warped) & (depth_warped > 0)

    gt_vis = depth_to_colormap(depth_gt, gt_mask)
    pred_vis = depth_to_colormap(depth_pred, pred_mask)
    warped_vis = depth_to_colormap(depth_warped, warped_mask)

    # cv2.putText(gt_vis, "GT depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(pred_vis, "Predicted depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(warped_vis, "Warped depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    rgb_vis = cv2.resize(rgb_img, (depth_gt.shape[1], depth_gt.shape[0]), interpolation=cv2.INTER_AREA)
    # cv2.putText(rgb_vis, "RGB", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    concat = np.concatenate([rgb_vis, gt_vis, pred_vis, warped_vis], axis=1)
    header = np.full((36, concat.shape[1], 3), 20, dtype=np.uint8)
    caption = f"{nn_name} | image_id={image_id}"
    cv2.putText(header, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    concat = np.concatenate([header, concat], axis=0)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{nn_name}_{image_id:06d}_debug.png"), concat)

    if DEBUG_VIS:
        cv2.imshow("Depth debug [RGB | GT | Predicted | Warped]", concat)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            return concat, True

    return concat, False


def _standard_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / np.clip(gt, 1e-9, None)
    sq_err = (pred - gt) ** 2

    ratio = np.maximum(
        pred / np.clip(gt, 1e-9, None),
        gt / np.clip(pred, 1e-9, None),
    )

    log_diff = np.log(np.clip(pred, 1e-9, None)) - np.log(np.clip(gt, 1e-9, None))
    silog_var = np.mean(log_diff ** 2) - np.mean(log_diff) ** 2

    return {
        "AbsRel": float(np.mean(rel_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "MAE": float(np.mean(abs_err)),
        "MedianAE": float(np.median(abs_err)),
        "SILog": float(np.sqrt(max(silog_var, 0.0))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < (1.25 ** 2))),
        "delta3": float(np.mean(ratio < (1.25 ** 3))),
    }


def compute_metrics(depth_gt: np.ndarray, pred_warped: np.ndarray, valid_mask: np.ndarray, gamma: float) -> dict[str, float]:
    gt = depth_gt[valid_mask].astype(np.float64)
    pred = pred_warped[valid_mask].astype(np.float64)

    eps = (pred - gt) / np.clip(gt, 1e-9, None)
    abs_err = np.abs(pred - gt)
    ratio = np.maximum(pred / np.clip(gt, 1e-9, None), gt / np.clip(pred, 1e-9, None))

    std_all = _standard_metrics(gt, pred)

    z = eps / gamma
    median_abs_z = float(np.median(np.abs(z)))
    mean_abs_z = float(np.mean(np.abs(z)))

    ci_thresh = float(scipy_stats.cauchy.ppf(0.5 + CONFIDENCE / 2.0, loc=0, scale=1))
    ci_bound = gamma * ci_thresh

    within_ci = np.abs(eps) <= ci_bound
    outside_ci = ~within_ci

    within_ci_pct = float(np.mean(within_ci) * 100.0)
    within_1gamma_pct = float(np.mean(np.abs(z) <= 1.0) * 100.0)
    within_2gamma_pct = float(np.mean(np.abs(z) <= 2.0) * 100.0)
    mean_log_lik = float(np.mean(scipy_stats.cauchy.logpdf(eps, loc=0.0, scale=gamma)))

    if outside_ci.any():
        std_out = _standard_metrics(gt[outside_ci], pred[outside_ci])
        ci_median_eps_out = float(np.median(np.abs(eps[outside_ci])))
    else:
        std_out = {k: 0.0 for k in std_all}
        ci_median_eps_out = 0.0

    out = {
        "n_pixels": int(valid_mask.sum()),
        "n_pixels_outside_ci": int(outside_ci.sum()),
        "outside_ci_pct": float(np.mean(outside_ci) * 100.0),
        "ci_thresh": ci_thresh,
        "ci_low": -ci_bound,
        "ci_high": ci_bound,
        "within_ci_pct": within_ci_pct,
        "within_1gamma_pct": within_1gamma_pct,
        "within_2gamma_pct": within_2gamma_pct,
        "median_abs_z": median_abs_z,
        "mean_abs_z": mean_abs_z,
        "mean_log_lik": mean_log_lik,
        "ci_median_eps_out": ci_median_eps_out,
    }

    for key, value in std_all.items():
        out[f"all_{key}"] = value
    for key, value in std_out.items():
        out[f"out_{key}"] = value

    return out


def print_network_report(summary_row: dict[str, float], nn_name: str, gamma: float) -> None:
    print(f"\n{'━' * 78}")
    print(f"Network: {nn_name}")
    print(f"{'━' * 78}")
    print(f"Total valid pixels:      {int(summary_row['n_pixels']):>12,}")
    print(
        f"Pixels outside 95% CI:   {int(summary_row['n_pixels_outside_ci']):>12,} "
        f"({summary_row['outside_ci_pct']:.2f}%)"
    )

    metrics = ["AbsRel", "RMSE", "MAE", "MedianAE", "SILog", "delta1", "delta2", "delta3"]
    print("\nStandard MDE metrics")
    print(f"{'Metric':<14} {'All pixels':>14} {'Outside-CI pixels':>20}")
    print("-" * 52)
    for name in metrics:
        a = summary_row[f"all_{name}"]
        o = summary_row[f"out_{name}"]
        if name.startswith("delta"):
            print(f"{name:<14} {a * 100:>13.2f}% {o * 100:>19.2f}%")
        elif name in {"RMSE", "MAE", "MedianAE"}:
            print(f"{name:<14} {a:>14.5f} {o:>20.5f}")
        else:
            print(f"{name:<14} {a:>14.5f} {o:>20.5f}")

    print("\nCauchy camera-model metrics")
    print(f"  gamma:             {gamma:.6f}")
    print(f"  CI bounds (ε):     [{summary_row['ci_low']:.6f}, {summary_row['ci_high']:.6f}]")
    print(f"  within_ci_pct:     {summary_row['within_ci_pct']:.2f}%")
    print(f"  median_abs_z:      {summary_row['median_abs_z']:.4f}")
    print(f"  mean_abs_z:        {summary_row['mean_abs_z']:.4f}")
    print(f"  within_1gamma_pct: {summary_row['within_1gamma_pct']:.2f}%")
    print(f"  within_2gamma_pct: {summary_row['within_2gamma_pct']:.2f}%")
    print(f"  mean_log_lik:      {summary_row['mean_log_lik']:.5f}")


def main() -> None:
    parent_dir = Path(__file__).resolve().parents[3]
    (
        gt_data_dir,
        relative_pose_path,
        calib_rgbd_path,
        calib_stereo_path,
        depth_estimation_dir,
        depth_comparison_dir,
    ) = prepare_depth_comparison_paths(parent_dir, DATE, RGBD_SUFFIX)

    metrics_out_dir = depth_comparison_dir / "metrics_ci_extended"
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
    k_target, d_target = load_camera_calibration(calib_rgbd_path)

    transform_target_from_source = _read_transform(relative_pose_path, "cam1_from_cam2")
    transform_target_from_source[:3, 3] /= 1000.0  # mm -> m

    camera_stats_path = parent_dir / "out" / f"out_{DATE}" / "cameras_statistic_model" / "best_distribution_models.json"
    zed_gamma = load_zed_gamma(camera_stats_path, camera="zed")
    ci_thresh = float(scipy_stats.cauchy.ppf(0.5 + CONFIDENCE / 2.0, loc=0.0, scale=1.0))

    print(f"[INFO] Loaded ZED gamma: {zed_gamma:.6f}")
    print(f"[INFO] Standardized Cauchy 95% threshold: {ci_thresh:.6f}")

    gt_frames = load_rgbd_images(gt_data_dir, suffix=RGBD_SUFFIX, max_imgs=MAX_IMGS)

    estimated_depth_maps: dict[str, list[np.ndarray]] = {}
    for nn_name in get_depth_estimation_network_names():
        depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=MAX_IMGS)
        if depth_maps:
            estimated_depth_maps[nn_name] = depth_maps

    all_networks_summary: list[dict[str, float]] = []
    visualization_enabled = DEBUG_VIS

    for nn_name, depth_maps in estimated_depth_maps.items():
        frame_size = depth_maps[0].shape[::-1]
        k_source_scaled = scale_intrinsics(k_source.copy(), SOURCE_HW[::-1], frame_size)
        k_target_scaled = scale_intrinsics(k_target.copy(), TARGET_HW[::-1], frame_size)

        per_frame_rows = []
        debug_counter = 0
        debug_save_dir = metrics_out_dir / "debug_triplets" / nn_name if DEBUG_SAVE else None
        for gt_frame, pred_depth in tqdm(zip(gt_frames, depth_maps), desc=f"[{nn_name}]", total=len(depth_maps)):
            image_id = gt_frame.get_image_number()
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
                np.isfinite(pred_warped)
                & (pred_warped > 0)
                & np.isfinite(depth_gt)
                & (depth_gt > 0)
            )

            if not valid_mask.any():
                continue

            if visualization_enabled or DEBUG_SAVE:
                if debug_counter < DEBUG_MAX_FRAMES_PER_NETWORK:
                    _, stop_visualization = debug_show_depth_triplet(
                        image_id=image_id,
                        nn_name=nn_name,
                        rgb_img=gt_frame.get_img(),
                        depth_gt=depth_gt,
                        depth_pred=pred_depth,
                        depth_warped=pred_warped,
                        save_dir=debug_save_dir,
                    )
                    debug_counter += 1
                    if stop_visualization:
                        visualization_enabled = False
                        print("[INFO] Visualization stopped by user (pressed 'q' or ESC).")
                        cv2.destroyAllWindows()

            metrics = compute_metrics(depth_gt=depth_gt, pred_warped=pred_warped, valid_mask=valid_mask, gamma=zed_gamma)
            per_frame_rows.append({"image_id": image_id, "nn_name": nn_name, **metrics})

        if not per_frame_rows:
            print(f"[WARN] No valid pixels for {nn_name}; skipping")
            continue

        df_per_image = pd.DataFrame(per_frame_rows)
        df_per_image.to_csv(metrics_out_dir / f"{nn_name}_per_image.csv", index=False)

        summary: dict[str, float] = {"nn_name": nn_name}
        for col in df_per_image.columns:
            if col in {"image_id", "nn_name"}:
                continue
            if col in {"n_pixels", "n_pixels_outside_ci"}:
                summary[col] = float(df_per_image[col].sum())
            else:
                summary[col] = float(df_per_image[col].mean())

        summary["zed_gamma"] = zed_gamma
        summary["confidence"] = CONFIDENCE

        all_networks_summary.append(summary)
        pd.DataFrame([summary]).to_csv(metrics_out_dir / f"{nn_name}_summary.csv", index=False)
        print_network_report(summary, nn_name, zed_gamma)

    if not all_networks_summary:
        print("[WARN] No network summaries generated.")
        return

    df_global = pd.DataFrame(all_networks_summary)
    global_path = metrics_out_dir / "all_networks_summary.csv"
    df_global.to_csv(global_path, index=False)

    print(f"\n{'═' * 90}")
    print("Global network summary")
    print(f"{'═' * 90}")

    table_cols = [
        "nn_name",
        "median_abs_z",
        "within_ci_pct",
        "mean_log_lik",
        "all_AbsRel",
        "all_RMSE",
        "all_MAE",
        "all_MedianAE",
        "all_SILog",
    ]
    print(df_global[table_cols].to_string(index=False))

    best_lowest_median_abs_z = df_global.loc[df_global["median_abs_z"].idxmin(), "nn_name"]
    best_highest_within_ci = df_global.loc[df_global["within_ci_pct"].idxmax(), "nn_name"]
    best_highest_mean_log_lik = df_global.loc[df_global["mean_log_lik"].idxmax(), "nn_name"]

    print("\nBest network by criterion:")
    print(f"  • lowest median_abs_z  : {best_lowest_median_abs_z}")
    print(f"  • highest within_ci_pct: {best_highest_within_ci}")
    print(f"  • highest mean_log_lik : {best_highest_mean_log_lik}")
    print(f"\n[INFO] Saved: {global_path}")
    if visualization_enabled:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
