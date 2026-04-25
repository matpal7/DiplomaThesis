from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.camera_model_parameters import load_sensor_model
from code.depth_compare.compare_depth_between_cameras import warp_depth_to_target, _read_transform, _compute_metrics
from code.depth_compare.evaluation import eval_depth_single
from code.image import load_rgbd_images
from code.prepare_paths import prepare_depth_comparison_paths, get_depth_estimation_network_names
from code.utils import load_estimated_depth_map, scale_intrinsics

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
parent_dir  = Path(__file__).resolve().parent.parent.parent.parent
date        = "13042026"
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

# ─────────────────────────────────────────────────────────────────────────────
# Sensor model — Cauchyho γ
# ─────────────────────────────────────────────────────────────────────────────
camera_stats_dir = parent_dir / "out" / "out_09042026" / "cameras_statistic_model"
sensor_model     = load_sensor_model(camera_stats_dir, rgbd_suffix)

# Cauchyho gamma — načítaj z modelu alebo použi fallback pre ZED
CAUCHY_GAMMA = sensor_model.get("cauchy_gamma", 0.00297) if sensor_model else 0.00297
CONFIDENCE   = 0.95
CI_LO, CI_HI = stats.cauchy.interval(CONFIDENCE, loc=0.0, scale=CAUCHY_GAMMA)

print(f"[✓] Cauchyho model: γ={CAUCHY_GAMMA:.5f}, {CONFIDENCE*100:.0f}% CI=[{CI_LO:.5f}, {CI_HI:.5f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Pomocné funkcie
# ─────────────────────────────────────────────────────────────────────────────
def resize_depth_safe(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    valid       = np.isfinite(depth) & (depth > 0)
    depth_filled = depth.copy()
    depth_filled[~valid] = 0.0
    depth_sum = cv2.resize(depth_filled.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid_sum = cv2.resize(valid.astype(np.float32),        (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w), np.nan, dtype=np.float32)
    valid_out = valid_sum > 0.1
    out[valid_out] = depth_sum[valid_out] / valid_sum[valid_out]
    return out


def within_cauchy_tolerance(
    gt:         np.ndarray,
    pred:       np.ndarray,
    mask:       np.ndarray,
    gamma:      float = CAUCHY_GAMMA,
    confidence: float = CONFIDENCE,
) -> dict:
    """
    Pre každý pixel vypočíta relatívnu chybu ε = (pred - gt) / gt
    a overí, či leží v confidence intervale Cauchyho modelu.

    CI hranice sú konštantné pre všetky pixely — závislosť od hĺbky
    je už implicitne zahrnutá v relatívnej chybe.
    """
    gt_m   = gt[mask].astype(np.float64)
    pred_m = pred[mask].astype(np.float64)

    eps = (pred_m - gt_m) / gt_m

    lo, hi  = stats.cauchy.interval(confidence, loc=0.0, scale=gamma)
    within  = (eps >= lo) & (eps <= hi)
    outside = ~within

    outlier_median_eps = float(np.median(np.abs(eps[outside]))) if outside.any() else 0.0

    return {
        "pct_within_ci":          float(within.mean() * 100),
        "pct_outside_ci":         float(outside.mean() * 100),
        "median_abs_eps":         float(np.median(np.abs(eps))),
        "outlier_median_abs_eps": outlier_median_eps,
        "ci_lo":                  float(lo),
        "ci_hi":                  float(hi),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agregačné stĺpce
# ─────────────────────────────────────────────────────────────────────────────
EVAL_AGG = ["d1", "d1_ssi", "arel", "rmse", "median_abs_error", "silog"]
TOL_AGG  = ["pct_within_ci", "pct_outside_ci", "median_abs_eps", "outlier_median_abs_eps"]

# ─────────────────────────────────────────────────────────────────────────────
# Načítanie odhadnutých hĺbkových máp
# ─────────────────────────────────────────────────────────────────────────────
estimated_depth_maps = {}
for nn_name in get_depth_estimation_network_names():
    depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=max_imgs)
    if len(depth_maps) == 0:
        continue
    estimated_depth_maps[nn_name] = depth_maps

metrics_out_dir = depth_comparison_dir / "metrics"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

all_networks_summary = []

# ─────────────────────────────────────────────────────────────────────────────
# Hlavná slučka — per network
# ─────────────────────────────────────────────────────────────────────────────
for nn_name in estimated_depth_maps:
    print(f"\n{'━'*60}")
    print(f"  Network: {nn_name}")
    print(f"{'━'*60}")

    frame_size      = estimated_depth_maps[nn_name][0].shape[::-1]
    k_source_scaled = scale_intrinsics(k_source.copy(), (3840, 2160), frame_size)
    k_target_scaled = scale_intrinsics(k_target.copy(), (1280, 720),  frame_size)

    per_image_rows = []

    for img_gt, depth_est in tqdm(
        zip(imgs_rgb, estimated_depth_maps[nn_name]),
        desc=f"[{nn_name}]",
    ):
        image_number = img_gt.get_image_number()
        depth_gt     = img_gt.get_depth()
        depth_gt     = resize_depth_safe(depth_gt, depth_est.shape)

        # ── Warp predikcie do súradníc GT kamery ─────────────────────────
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

        # ── Spoločná maska platných pixelov ──────────────────────────────
        mask_eval = (
            np.isfinite(pred_warped_m) & (pred_warped_m > 0) &
            np.isfinite(depth_gt)      & (depth_gt > 0)
        )

        # ── Štandardné eval metriky ───────────────────────────────────────
        eval_metrics = eval_depth_single(
            gt=depth_gt, pred=pred_warped_m,
            mask=mask_eval, max_depth=None,
        )

        # ── Cauchyho CI tolerancia ────────────────────────────────────────
        tol_metrics = within_cauchy_tolerance(
            gt=depth_gt, pred=pred_warped_m,
            mask=mask_eval,
        )

        row = {
            "image_id":  image_number,
            "nn_name":   nn_name,
            # Eval metriky
            "eval_d1":               eval_metrics.get("d1"),
            "eval_d1_ssi":           eval_metrics.get("d1_ssi"),
            "eval_arel":             eval_metrics.get("arel"),
            "eval_rmse":             eval_metrics.get("rmse"),
            "eval_median_abs_error": eval_metrics.get("median_abs_error"),
            "eval_silog":            eval_metrics.get("silog"),
            # Cauchyho CI metriky
            "tol_pct_within_ci":          tol_metrics["pct_within_ci"],
            "tol_pct_outside_ci":         tol_metrics["pct_outside_ci"],
            "tol_median_abs_eps":         tol_metrics["median_abs_eps"],
            "tol_outlier_median_abs_eps": tol_metrics["outlier_median_abs_eps"],
            "tol_ci_lo":                  tol_metrics["ci_lo"],
            "tol_ci_hi":                  tol_metrics["ci_hi"],
        }
        per_image_rows.append(row)

        print(
            f"  [{image_number}]  "
            f"d1={eval_metrics.get('d1', float('nan'))*100:.1f}%  "
            f"arel={eval_metrics.get('arel', float('nan')):.3f}  "
            f"within_CI={tol_metrics['pct_within_ci']:.1f}%  "
            f"outliers={tol_metrics['pct_outside_ci']:.1f}%"
        )

    # ── Uloženie per-image CSV ────────────────────────────────────────────
    df_per_image = pd.DataFrame(per_image_rows)
    df_per_image.to_csv(metrics_out_dir / f"{nn_name}_per_image_metrics.csv", index=False)

    if df_per_image.empty:
        print("  No valid overlapping pixels — skipping summary.")
        continue

    # ── Agregácia ─────────────────────────────────────────────────────────
    aggregated = {
        "nn_name":      nn_name,
        "cauchy_gamma": CAUCHY_GAMMA,
        "confidence":   CONFIDENCE,
        "ci_lo":        CI_LO,
        "ci_hi":        CI_HI,
    }
    for col in EVAL_AGG:
        aggregated[f"{col}_mean"] = float(df_per_image[f"eval_{col}"].mean())
    for col in TOL_AGG:
        aggregated[f"{col}_mean"] = float(df_per_image[f"tol_{col}"].mean())

    all_networks_summary.append(aggregated)
    df_net = pd.DataFrame([aggregated])
    df_net.to_csv(metrics_out_dir / f"{nn_name}_summary_metrics.csv", index=False)

    # ── Výpis súhrnu ──────────────────────────────────────────────────────
    print(f"\n  ── {nn_name} ──")
    print(f"  [Hĺbková presnosť]")
    print(f"    d1 (raw)          : {aggregated['d1_mean']*100:.2f}%")
    print(f"    d1_ssi (aligned)  : {aggregated['d1_ssi_mean']*100:.2f}%")
    print(f"    arel              : {aggregated['arel_mean']:.4f}")
    print(f"    rmse              : {aggregated['rmse_mean']:.4f} m")
    print(f"    median abs error  : {aggregated['median_abs_error_mean']*100:.2f} cm")
    print(f"    silog             : {aggregated['silog_mean']:.4f}")
    print(f"  [Cauchyho CI  γ={CAUCHY_GAMMA:.5f}  {CONFIDENCE*100:.0f}% CI=[{CI_LO:.5f}, {CI_HI:.5f}]]")
    print(f"    pixely v CI       : {aggregated['pct_within_ci_mean']:.1f}%")
    print(f"    pixely mimo CI    : {aggregated['pct_outside_ci_mean']:.1f}%")
    print(f"    medián |ε|        : {aggregated['median_abs_eps_mean']:.5f}")
    print(f"    outlier medián |ε|: {aggregated['outlier_median_abs_eps_mean']:.5f}")
    print(f"  Uložené → {metrics_out_dir / f'{nn_name}_summary_metrics.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# Globálny súhrn — všetky siete
# ─────────────────────────────────────────────────────────────────────────────
if all_networks_summary:
    df_global = pd.DataFrame(all_networks_summary)
    global_csv = metrics_out_dir / "all_networks_global_summary.csv"
    df_global.to_csv(global_csv, index=False)

    # Prehľadná tabuľka pre konzolu
    display_cols = [
        "nn_name",
        "d1_mean", "d1_ssi_mean", "arel_mean", "rmse_mean",
        "pct_within_ci_mean", "pct_outside_ci_mean",
        "median_abs_eps_mean", "outlier_median_abs_eps_mean",
    ]
    df_display = df_global[display_cols].copy()
    df_display["d1_mean"]    *= 100
    df_display["d1_ssi_mean"] *= 100
    df_display["pct_within_ci_mean"]  = df_display["pct_within_ci_mean"].map("{:.1f}%".format)
    df_display["pct_outside_ci_mean"] = df_display["pct_outside_ci_mean"].map("{:.1f}%".format)
    df_display.rename(columns={
        "nn_name":                   "Sieť",
        "d1_mean":                   "d1 (%)",
        "d1_ssi_mean":               "d1_ssi (%)",
        "arel_mean":                 "arel",
        "rmse_mean":                 "rmse (m)",
        "pct_within_ci_mean":        "v CI",
        "pct_outside_ci_mean":       "mimo CI",
        "median_abs_eps_mean":       "med|ε|",
        "outlier_median_abs_eps_mean": "outlier med|ε|",
    }, inplace=True)

    print(f"\n{'═'*60}")
    print(f"  GLOBÁLNY SÚHRN — VŠETKY SIETE")
    print(f"  Cauchyho model: γ={CAUCHY_GAMMA:.5f}, {CONFIDENCE*100:.0f}% CI=[{CI_LO:.5f}, {CI_HI:.5f}]")
    print(f"{'═'*60}")
    print(df_display.to_string(index=False))
    print(f"\n  Uložené → {global_csv}")