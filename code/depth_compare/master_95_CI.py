from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats as scipy_stats

from CI_calculation import load_zed_gamma, is_within_ci
from code.calibration.ChArUco.charuco_relative_pose_pnp_v3 import load_camera_calibration
from code.depth_compare.compare_depth_between_cameras import warp_depth_to_target, _read_transform

from code.image import load_rgbd_images
from code.prepare_paths import prepare_depth_comparison_paths, get_depth_estimation_network_names
from code.utils import load_estimated_depth_map, scale_intrinsics

# ─────────────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────────────
parent_dir  = Path(__file__).resolve().parents[3]
date        = "24042026"
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

metrics_out_dir = depth_comparison_dir / "metrics_ci"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

k_source, d_source = load_camera_calibration(calib_stereo_path, suffix="left")
k_target, d_target = load_camera_calibration(calib_rgbd_path)

pose_convention               = "cam1_from_cam2"
transform_target_from_source  = _read_transform(relative_pose_path, pose_convention)
transform_target_from_source[:3, 3] /= 1000.0


camera_stats_dir = parent_dir / "out" / f"out_{date}" / "cameras_statistic_model"
ZED_GAMMA        = load_zed_gamma(camera_stats_dir / "best_distribution_models.json", camera="zed")
CONFIDENCE       = 0.95
print(f"[✓] ZED Cauchy γ = {ZED_GAMMA:.5f}  →  95% CI = ±{ZED_GAMMA * np.tan(CONFIDENCE * np.pi / 2):.5f}")


def resize_depth_safe(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w    = target_hw
    valid   = np.isfinite(depth) & (depth > 0)
    filled  = depth.copy()
    filled[~valid] = 0.0
    d_sum   = cv2.resize(filled.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    v_sum   = cv2.resize(valid.astype(np.float32),  (w, h), interpolation=cv2.INTER_LINEAR)
    out     = np.full((h, w), np.nan, dtype=np.float32)
    ok      = v_sum > 0.1
    out[ok] = d_sum[ok] / v_sum[ok]
    return out


def compute_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    confidence: float = 0.95,
) -> dict:
    gt_m    = gt[mask].astype(np.float64)
    pred_m  = pred[mask].astype(np.float64)
    eps     = (pred_m - gt_m) / np.clip(gt_m, 1e-9, None)
    abs_err = np.abs(pred_m - gt_m)
    ratio   = np.maximum(pred_m / np.clip(gt_m, 1e-9, None),
                         gt_m   / np.clip(pred_m, 1e-9, None))

    # ── Cauchyho CI ───────────────────────────────────────────────────────
    within_ci, (ci_lo, ci_hi) = is_within_ci(eps, gt_m, gamma=gamma, confidence=confidence)
    outside_ci = ~within_ci
    ci_pct_in  = float(within_ci.mean() * 100)
    ci_pct_out = float(outside_ci.mean() * 100)
    ci_median_eps_outside = float(np.median(np.abs(eps[outside_ci]))) if outside_ci.any() else 0.0

    # ── NOVÉ: Vzdialenosť chýb voči modelu kamery ─────────────────────────
    # z_i = ε_i / γ  →  koľkonásobne je chyba väčšia ako šum kamery
    z = eps / gamma

    median_abs_z      = float(np.median(np.abs(z)))
    mean_abs_z        = float(np.mean(np.abs(z)))
    z_within_1gamma   = float(np.mean(np.abs(z) <= 1.0) * 100)   # % pixelov v ±1γ
    z_within_2gamma   = float(np.mean(np.abs(z) <= 2.0) * 100)   # % pixelov v ±2γ

    # Log-likelihood každého pixelu pod Cauchyho modelom kamery
    # → čím menej záporné, tým bližšie sú chyby siete k šumu kamery
    log_lik_per_pixel = scipy_stats.cauchy.logpdf(eps, loc=0, scale=gamma)
    mean_log_lik      = float(np.mean(log_lik_per_pixel))

    # ── Pomocná funkcia pre výpočet skupiny metrík ────────────────────────
    def _metrics_on(gt_f, pred_f, abs_f, ratio_f):
        log_diff = np.log(np.clip(pred_f, 1e-9, None)) - np.log(np.clip(gt_f, 1e-9, None))
        return dict(
            arel      = float(np.mean(np.abs((pred_f - gt_f) / np.clip(gt_f, 1e-9, None)))),
            rmse      = float(np.sqrt(np.mean(abs_f ** 2))),
            mae       = float(np.mean(abs_f)),
            median_ae = float(np.median(abs_f)),
            silog     = float(np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)),
            d1        = float(np.mean(ratio_f < 1.25)),
            d2        = float(np.mean(ratio_f < 1.25 ** 2)),
            d3        = float(np.mean(ratio_f < 1.25 ** 3)),
        )

    all_m = _metrics_on(gt_m, pred_m, abs_err, ratio)

    if outside_ci.any():
        out_m = _metrics_on(
            gt_m[outside_ci], pred_m[outside_ci],
            abs_err[outside_ci], ratio[outside_ci],
        )
    else:
        out_m = {k: 0.0 for k in all_m}

    return dict(
        n_pixels              = int(mask.sum()),
        n_pixels_outside_ci   = int(outside_ci.sum()),
        ci_pct_within         = ci_pct_in,
        ci_pct_outside        = ci_pct_out,
        ci_median_eps_out     = ci_median_eps_outside,
        ci_lo                 = ci_lo,
        ci_hi                 = ci_hi,
        # ── NOVÉ: vzdialenosť voči modelu kamery ──────────────────────
        median_abs_z          = median_abs_z,     # typická chyba v násobkoch γ
        mean_abs_z            = mean_abs_z,
        z_within_1gamma_pct   = z_within_1gamma,  # % pixelov v ±1γ  (~50% CI kamery)
        z_within_2gamma_pct   = z_within_2gamma,  # % pixelov v ±2γ  (~70% CI kamery)
        mean_log_lik          = mean_log_lik,      # fit pod Cauchyho modelom
        # ── existujúce metriky ────────────────────────────────────────
        **{f"all_{k}": v for k, v in all_m.items()},
        **{f"out_{k}": v for k, v in out_m.items()},
    )


def print_metrics(m: dict, nn_name: str) -> None:
    print(f"\n  ── {nn_name} ──")
    print(f"  Pixely celkom       : {m['n_pixels']:>12,}")
    print(f"  Pixely mimo CI      : {m['n_pixels_outside_ci']:>12,}  ({m['ci_pct_outside']:.1f}%)")
    print(f"  {'─'*55}")
    print(f"  {'Metrika':<20} {'Všetky pixely':>16}  {'Mimo CI (chyba siete)':>20}")
    print(f"  {'─'*55}")
    print(f"  {'d1 (δ<1.25)':<20} {m['all_d1']*100:>15.2f}%  {m['out_d1']*100:>19.2f}%")
    print(f"  {'d2 (δ<1.25²)':<20} {m['all_d2']*100:>15.2f}%  {m['out_d2']*100:>19.2f}%")
    print(f"  {'d3 (δ<1.25³)':<20} {m['all_d3']*100:>15.2f}%  {m['out_d3']*100:>19.2f}%")
    print(f"  {'arel':<20} {m['all_arel']:>16.4f}  {m['out_arel']:>20.4f}")
    print(f"  {'rmse (cm)':<20} {m['all_rmse']*100:>14.3f}  {m['out_rmse']*100:>18.3f}")
    print(f"  {'mae (cm)':<20} {m['all_mae']*100:>14.3f}  {m['out_mae']*100:>18.3f}")
    print(f"  {'median AE (cm)':<20} {m['all_median_ae']*100:>14.3f}  {m['out_median_ae']*100:>18.3f}")
    print(f"  {'silog':<20} {m['all_silog']:>16.4f}  {m['out_silog']:>20.4f}")
    print(f"  {'─'*55}")
    print(f"  [Cauchyho 95% CI  γ={ZED_GAMMA:.5f}]")
    print(f"    CI hranice        : [{m['ci_lo']:.5f}, {m['ci_hi']:.5f}]")
    print(f"    pixely v CI       : {m['ci_pct_within']:>7.2f}%")
    print(f"    medián |ε| mimo   : {m['ci_median_eps_out']:.5f}")
    print(f"  {'─'*55}")
    print(f"  [Vzdialenosť chýb voči modelu kamery  γ={ZED_GAMMA:.5f}]")
    print(f"    Medián |z| = |ε|/γ   : {m['median_abs_z']:>8.3f} ×γ")
    print(f"    Priemer |z|           : {m['mean_abs_z']:>8.3f} ×γ")
    print(f"    Pixely v ±1γ          : {m['z_within_1gamma_pct']:>7.2f}%")
    print(f"    Pixely v ±2γ          : {m['z_within_2gamma_pct']:>7.2f}%")
    print(f"    Mean log-likelihood   : {m['mean_log_lik']:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Načítanie dát
# ─────────────────────────────────────────────────────────────────────────────
imgs_rgb = load_rgbd_images(gt_data_dir, suffix=rgbd_suffix, max_imgs=max_imgs)

estimated_depth_maps = {}
for nn_name in get_depth_estimation_network_names():
    depth_maps = load_estimated_depth_map(depth_estimation_dir, nn_name, max_imgs=max_imgs)
    if depth_maps:
        estimated_depth_maps[nn_name] = depth_maps

# ─────────────────────────────────────────────────────────────────────────────
# Hlavná slučka
# ─────────────────────────────────────────────────────────────────────────────
all_networks_summary = []

for nn_name, depth_maps in estimated_depth_maps.items():
    print(f"\n{'━'*60}")
    print(f"  Network: {nn_name}")
    print(f"{'━'*60}")

    frame_size      = depth_maps[0].shape[::-1]
    k_source_scaled = scale_intrinsics(k_source.copy(), (3840, 2160), frame_size)
    k_target_scaled = scale_intrinsics(k_target.copy(), (1280, 720),  frame_size)

    per_image_rows = []

    for img_gt, depth_est in tqdm(zip(imgs_rgb, depth_maps), desc=f"[{nn_name}]"):
        image_id = img_gt.get_image_number()
        depth_gt = resize_depth_safe(img_gt.get_depth(), depth_est.shape)

        pred_warped, _ = warp_depth_to_target(
            source_depth      = depth_est,
            k_source          = k_source_scaled,
            d_source          = d_source,
            k_target          = k_target_scaled,
            d_target          = d_target,
            t_target_source   = transform_target_from_source,
            source_depth_scale= 1.0,
            target_hw         = (depth_gt.shape[0], depth_gt.shape[1]),
        )

        mask = (
            np.isfinite(pred_warped) & (pred_warped > 0) &
            np.isfinite(depth_gt)   & (depth_gt   > 0)
        )

        if not mask.any():
            continue

        m = compute_metrics(
            gt         = depth_gt,
            pred       = pred_warped,
            mask       = mask,
            gamma      = ZED_GAMMA,
            confidence = CONFIDENCE,
        )

        per_image_rows.append({"image_id": image_id, "nn_name": nn_name, **m})

    if not per_image_rows:
        print("  Žiadne platné pixely — preskočené.")
        continue

    # ── Per-image CSV ─────────────────────────────────────────────────────
    df_img = pd.DataFrame(per_image_rows)
    df_img.to_csv(metrics_out_dir / f"{nn_name}_per_image.csv", index=False)

    # ── Agregácia (mean cez všetky snímky) ────────────────────────────────
    skip_agg = {"image_id", "nn_name", "ci_lo", "ci_hi"}
    agg = {"nn_name": nn_name}
    for col in df_img.columns:
        if col not in skip_agg:
            agg[col] = float(df_img[col].mean())

    agg["ci_lo"] = float(df_img["ci_lo"].iloc[0])
    agg["ci_hi"] = float(df_img["ci_hi"].iloc[0])

    agg["zed_gamma"]       = ZED_GAMMA
    agg["confidence"]      = CONFIDENCE

    all_networks_summary.append(agg)
    print_metrics(agg, nn_name)
    df_net = pd.DataFrame([agg])
    df_net.to_csv(metrics_out_dir / f"{nn_name}_summary.csv", index=False)
    print(f"  [✓] Uložené → {metrics_out_dir / f'{nn_name}_summary.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# Globálny súhrn
# ─────────────────────────────────────────────────────────────────────────────
if all_networks_summary:
    df_global = pd.DataFrame(all_networks_summary)
    global_csv = metrics_out_dir / "all_networks_summary.csv"
    df_global.to_csv(global_csv, index=False)

    print(f"\n{'═'*60}")
    print("  GLOBÁLNY SÚHRN — VŠETKY SIETE")
    print(f"{'═'*60}")
    report_cols = ["nn_name",
                   "all_d1", "all_arel", "all_rmse", "all_median_ae", "all_silog",
                   "out_d1", "out_arel", "out_rmse", "out_median_ae",
                   "ci_pct_within", ]

    print(df_global[report_cols].to_string(index=False))
    print(f"\n  [✓] Globálny súhrn → {global_csv}")