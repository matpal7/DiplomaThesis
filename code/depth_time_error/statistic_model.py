import cv2
import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import sobel
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
parent_dir = Path(__file__).resolve().parents[3]
date       = "09042026"
base_dir   = parent_dir / "datasets" / f"dataset_{date}" / "cameras_statistic_model"
out_dir    = parent_dir / "out" / f"out_{date}" / "cameras_statistic_model"
out_dir.mkdir(parents=True, exist_ok=True)

CAMERAS = ["zed", "realsense"]


# ── Models ───────────────────────────────────────────────────────────────────
def power_law(Z, alpha, beta):
    return alpha * np.power(Z, beta)

def exp_model(Z, a, b):
    return a * np.exp(b * Z)


# ── Scene statistics ─────────────────────────────────────────────────────────
def compute_scene_stats(depth_dir: Path, pattern: str) -> dict | None:
    depth_files = sorted(depth_dir.glob(pattern))
    if len(depth_files) < 2:
        return None

    depths = []
    for f in depth_files:
        d = np.load(f).astype(np.float32)
        depths.append(d)

    ref_shape = depths[0].shape
    for i, d in enumerate(depths[1:], 1):
        if d.shape != ref_shape:
            depths[i] = cv2.resize(d, (ref_shape[1], ref_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

    depths = np.stack(depths, axis=0)   # (N, H, W)
    N      = depths.shape[0]
    depths[depths <= 0] = np.nan        # neplatné pixely → NaN
    valid  = np.isfinite(depths)

    # ── Per-frame stats ───────────────────────────────────────────────────────
    frame_stats = []
    for i in range(N):
        v   = valid[i]
        d_v = depths[i][v]
        frame_stats.append({
            "file":             depth_files[i].name,
            "valid_px":         int(v.sum()),
            "mean_m":           float(np.mean(d_v))   if v.any() else None,
            "median_m":         float(np.median(d_v)) if v.any() else None,
            "std_m":            float(np.std(d_v))    if v.any() else None,
            "mean_rel_std_pct": float(np.mean(
                                    np.std(depths[:, v], axis=0) /
                                    np.mean(depths[:, v], axis=0)) * 100
                                ) if v.any() else None,
        })

    # ── Consecutive-frame differences ─────────────────────────────────────────
    consec_diffs = []
    for i in range(N - 1):
        both = valid[i] & valid[i + 1]
        if not both.any():
            consec_diffs.append({
                "from": depth_files[i].name, "to": depth_files[i+1].name,
                "valid_px": 0,
                "mean_abs_diff_m": None, "median_abs_diff_m": None,
                "max_abs_diff_m":  None, "mean_signed_diff_m": None,
                "mean_rel_diff_pct": None, "median_rel_diff_pct": None,
            })
            continue
        diff     = depths[i+1][both] - depths[i][both]
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / depths[i][both] * 100
        consec_diffs.append({
            "from":                depth_files[i].name,
            "to":                  depth_files[i+1].name,
            "valid_px":            int(both.sum()),
            "mean_abs_diff_m":     float(np.mean(abs_diff)),
            "median_abs_diff_m":   float(np.median(abs_diff)),
            "max_abs_diff_m":      float(np.max(abs_diff)),
            "mean_signed_diff_m":  float(np.mean(diff)),
            "mean_rel_diff_pct":   float(np.mean(rel_diff)),
            "median_rel_diff_pct": float(np.median(rel_diff)),
        })

    # ── Per-pixel temporal statistics ─────────────────────────────────────────
    all_valid_mask    = valid.all(axis=0)
    pixel_std_map     = np.full(ref_shape, np.nan, dtype=np.float32)
    pixel_mean_map    = np.full(ref_shape, np.nan, dtype=np.float32)
    pixel_rel_std_map = np.full(ref_shape, np.nan, dtype=np.float32)

    if all_valid_mask.any():
        stacked                           = depths[:, all_valid_mask]
        pixel_std_map[all_valid_mask]     = stacked.std(axis=0, ddof=1)
        pixel_mean_map[all_valid_mask]    = stacked.mean(axis=0)
        pixel_rel_std_map[all_valid_mask] = (
            pixel_std_map[all_valid_mask] / pixel_mean_map[all_valid_mask] * 100
        )

    # ── Edge masking (pre bin_stats) ──────────────────────────────────────────
    gx       = sobel(pixel_mean_map, axis=1)
    gy       = sobel(pixel_mean_map, axis=0)
    edge_mag = np.hypot(gx, gy)
    edge_mask = edge_mag < np.percentile(edge_mag[np.isfinite(edge_mag)], 80)

    noise_valid = (
        all_valid_mask          &
        np.isfinite(pixel_std_map)  &
        (pixel_std_map > 0)     &
        (pixel_mean_map > 0.3)  &   # min. platná hĺbka ZED Mini
        (pixel_mean_map < 10.0) &   # max. platná hĺbka
        edge_mask
    )

    # ── Depth-binned stats (pre model fitting) ────────────────────────────────
    depth_bins = np.arange(0.25, 12.0, 0.25)
    bin_stats  = []
    if noise_valid.any():
        z_flat       = pixel_mean_map[noise_valid]
        rel_std_flat = pixel_rel_std_map[noise_valid]
        abs_std_flat = pixel_std_map[noise_valid]
        for b_lo, b_hi in zip(depth_bins[:-1], depth_bins[1:]):
            mask_bin = (z_flat >= b_lo) & (z_flat < b_hi)
            if mask_bin.sum() < 10:
                continue
            bin_stats.append({
                "z_center_m":         float((b_lo + b_hi) / 2),
                "n_pixels":           int(mask_bin.sum()),
                "mean_abs_std_m":     float(np.mean(abs_std_flat[mask_bin])),
                "median_abs_std_m":   float(np.median(abs_std_flat[mask_bin])),
                "mean_rel_std_pct":   float(np.mean(rel_std_flat[mask_bin])),
                "median_rel_std_pct": float(np.median(rel_std_flat[mask_bin])),
            })

    return {
        "n_frames":            N,
        "frame_stats":         frame_stats,
        "consec_diffs":        consec_diffs,
        "pixels_valid_all":    int(all_valid_mask.sum()),
        "mean_pixel_std_m":    float(np.nanmean(pixel_std_map)),
        "median_pixel_std_m":  float(np.nanmedian(pixel_std_map)),
        "max_pixel_std_m":     float(np.nanmax(pixel_std_map)) if all_valid_mask.any() else None,
        "pct_stable_1cm":      float(np.nanmean(pixel_std_map < 0.01)) * 100,
        "pct_stable_5cm":      float(np.nanmean(pixel_std_map < 0.05)) * 100,
        "mean_rel_std_pct":    float(np.nanmean(pixel_rel_std_map)),
        "median_rel_std_pct":  float(np.nanmedian(pixel_rel_std_map)),
        "pct_stable_rel_1pct": float(np.nanmean(pixel_rel_std_map < 1.0)) * 100,
        "pct_stable_rel_5pct": float(np.nanmean(pixel_rel_std_map < 5.0)) * 100,
        "bin_stats":           bin_stats,
    }


# ── Global cross-scene statistics ────────────────────────────────────────────
def build_global_stats(all_results: dict) -> dict:
    return {
        "n_scenes":            len(all_results),
        "total_frames":        sum(r["n_frames"]         for r in all_results.values()),
        "total_valid_px":      sum(r["pixels_valid_all"] for r in all_results.values()),
        "global_mean_std_m":   float(np.mean([r["mean_pixel_std_m"]    for r in all_results.values()])),
        "global_median_std_m": float(np.mean([r["median_pixel_std_m"]  for r in all_results.values()])),
        "global_max_std_m":    float(np.max( [r["max_pixel_std_m"]     for r in all_results.values()])),
        "global_mean_rel_pct": float(np.mean([r["mean_rel_std_pct"]    for r in all_results.values()])),
        "global_med_rel_pct":  float(np.mean([r["median_rel_std_pct"]  for r in all_results.values()])),
        "pct_stable_1cm":      float(np.mean([r["pct_stable_1cm"]      for r in all_results.values()])),
        "pct_stable_5cm":      float(np.mean([r["pct_stable_5cm"]      for r in all_results.values()])),
        "pct_stable_rel_1pct": float(np.mean([r["pct_stable_rel_1pct"] for r in all_results.values()])),
        "pct_stable_rel_5pct": float(np.mean([r["pct_stable_rel_5pct"] for r in all_results.values()])),
        "best_scene":  min(all_results, key=lambda k: all_results[k]["median_pixel_std_m"]),
        "worst_scene": max(all_results, key=lambda k: all_results[k]["median_pixel_std_m"]),
    }


# ── Model fitting: power-law + exponential, with R² ─────────────────────────
def fit_depth_models(all_results: dict) -> dict:
    all_bins = [b for r in all_results.values() for b in r["bin_stats"]]
    if len(all_bins) < 3:
        return {}

    z_vals    = np.array([b["z_center_m"]              for b in all_bins])
    s_abs     = np.array([b["median_abs_std_m"]         for b in all_bins])
    s_rel_raw = np.array([b["median_rel_std_pct"] / 100 for b in all_bins])
    w_vals    = np.array([b["n_pixels"]                 for b in all_bins], dtype=float)

    def _fit_and_r2(fn, x, y, p0, bounds):
        popt, pcov = curve_fit(fn, x, y, sigma=1/w_vals, p0=p0,
                               bounds=bounds, maxfev=10000)
        perr  = np.sqrt(np.diag(pcov))
        r2    = r2_score(y, fn(x, *popt))
        return popt, perr, r2

    # ── Power-law: σ_rel(Z) = α · Z^β  (relative, raw ratio) ────────────────
    popt_pl, perr_pl, r2_pl = _fit_and_r2(
        power_law, z_vals, s_rel_raw, p0=[0.005, 1.0],
        bounds=([0, 0.5], [np.inf, 5.0])
    )
    t_stat = (popt_pl[1] - 2.0) / perr_pl[1]

    # ── Exponential: σ_abs(Z) = a · exp(b · Z)  (absolute, metres) ───────────
    popt_ex, perr_ex, r2_ex = _fit_and_r2(
        exp_model, z_vals, s_abs, p0=[0.001, 0.3],
        bounds=([0, 0], [np.inf, np.inf])
    )

    return {
        # ── relative power-law ────────────────────────────────────────────────
        "rel_alpha":             float(popt_pl[0]),
        "rel_alpha_std":         float(perr_pl[0]),
        "rel_beta":              float(popt_pl[1]),
        "rel_beta_std":          float(perr_pl[1]),
        "rel_beta_t_vs_2":       float(t_stat),
        "rel_beta_reject_2":     bool(abs(t_stat) > 2.0),
        "rel_r2":                float(r2_pl),
        "rel_sigma_at_1m_pct":   float(power_law(1.0, *popt_pl) * 100),
        "rel_sigma_at_3m_pct":   float(power_law(3.0, *popt_pl) * 100),
        "rel_sigma_at_5m_pct":   float(power_law(5.0, *popt_pl) * 100),
        # ── absolute exponential ──────────────────────────────────────────────
        "exp_a":                 float(popt_ex[0]),
        "exp_a_std":             float(perr_ex[0]),
        "exp_b":                 float(popt_ex[1]),
        "exp_b_std":             float(perr_ex[1]),
        "exp_r2":                float(r2_ex),
        "exp_sigma_at_1m_m":     float(exp_model(1.0, *popt_ex)),
        "exp_sigma_at_3m_m":     float(exp_model(3.0, *popt_ex)),
        "exp_sigma_at_5m_m":     float(exp_model(5.0, *popt_ex)),
        # ── which model fits better ───────────────────────────────────────────
        "better_model":          "power_law" if r2_pl >= r2_ex else "exponential",
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_depth_model(all_results: dict, model: dict, camera: str, out_path: Path) -> None:
    all_bins = [b for r in all_results.values() for b in r["bin_stats"]]
    if not all_bins or not model:
        return

    z_vals = np.array([b["z_center_m"]              for b in all_bins])
    s_abs  = np.array([b["median_abs_std_m"]         for b in all_bins])
    s_rel  = np.array([b["median_rel_std_pct"] / 100 for b in all_bins])
    Z_plot = np.linspace(z_vals.min(), z_vals.max(), 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Depth Noise Model — {camera.upper()}", fontsize=13)

    # Left: relative power-law
    ax = axes[0]
    ax.scatter(z_vals, s_rel * 100, s=25, alpha=0.6, label='Binned median σ_rel')
    ax.plot(Z_plot,
            power_law(Z_plot, model["rel_alpha"], model["rel_beta"]) * 100,
            color='red',
            label=f'Power-law: {model["rel_alpha"]:.5f}·Z^{model["rel_beta"]:.3f}'
                  f'  R²={model["rel_r2"]:.3f}')
    ax.set_xlabel('Depth Z [m]'); ax.set_ylabel('σ_rel [%]')
    ax.set_title('Relative noise  σ_rel(Z) = α·Z^β')
    ax.legend(); ax.grid()

    # Right: absolute exponential
    ax = axes[1]
    ax.scatter(z_vals, s_abs * 100, s=25, alpha=0.6, label='Binned median σ_abs')
    ax.plot(Z_plot,
            exp_model(Z_plot, model["exp_a"], model["exp_b"]) * 100,
            color='darkorange',
            label=f'Exp: {model["exp_a"]:.5f}·exp({model["exp_b"]:.3f}·Z)'
                  f'  R²={model["exp_r2"]:.3f}')
    ax.set_xlabel('Depth Z [m]'); ax.set_ylabel('σ_abs [cm]')
    ax.set_title('Absolute noise  σ_abs(Z) = a·exp(b·Z)')
    ax.legend(); ax.grid()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [✓] Plot saved → {out_path}")


# ── Console printing ──────────────────────────────────────────────────────────
def print_scene(scene_name: str, r: dict, camera: str) -> None:
    W = 84
    print(f"\n{'█' * W}")
    print(f"  SCENE: {scene_name}  ({r['n_frames']} frames, camera={camera})")
    print(f"{'█' * W}")
    print(f"\n  {'File':<38} {'Mean':>7} {'Median':>7} {'Std':>7} {'RelStd%':>8} {'Valid px':>10}")
    print("  " + "─" * (W - 2))
    for s in r["frame_stats"]:
        print(f"  {s['file']:<38} {s['mean_m']:>7.3f} {s['median_m']:>7.3f} "
              f"{s['std_m']:>7.3f} {s['mean_rel_std_pct']:>7.2f}% {s['valid_px']:>10,}")

    print(f"\n  {'From → To':<44} {'MeanΔ':>7} {'MedΔ':>6} {'MaxΔ':>7} {'SignedΔ':>9} {'RelΔ%':>7} {'MedRelΔ%':>9}")
    print("  " + "─" * (W - 2))
    for d in r["consec_diffs"]:
        label = f"{d['from']} → {d['to']}"
        print(f"  {label:<44} {d['mean_abs_diff_m']:>7.3f} "
              f"{d['median_abs_diff_m']:>6.3f} {d['max_abs_diff_m']:>7.3f} "
              f"{d['mean_signed_diff_m']:>+9.4f} {d['mean_rel_diff_pct']:>6.2f}% "
              f"{d['median_rel_diff_pct']:>8.2f}%")

    print(f"\n  Pixels valid in ALL frames : {r['pixels_valid_all']:,}")
    print(f"  Mean per-pixel std         : {r['mean_pixel_std_m']:.4f} m")
    print(f"  Median per-pixel std       : {r['median_pixel_std_m']:.4f} m")
    print(f"  Max per-pixel std          : {r['max_pixel_std_m']:.4f} m")
    print(f"  Stable within 1 cm / 5 cm : {r['pct_stable_1cm']:.1f}% / {r['pct_stable_5cm']:.1f}%")
    print(f"  Mean / Median rel. std     : {r['mean_rel_std_pct']:.2f}% / {r['median_rel_std_pct']:.2f}%")
    print(f"  Stable within 1% / 5% rel : {r['pct_stable_rel_1pct']:.1f}% / {r['pct_stable_rel_5pct']:.1f}%")

    if r["bin_stats"]:
        print(f"\n  Depth-binned relative std:")
        print(f"  {'Z center':>10} {'N pixels':>10} {'MeanStd(m)':>12} {'MedStd(m)':>11} {'MeanRel%':>10} {'MedRel%':>9}")
        print("  " + "─" * (W - 2))
        for b in r["bin_stats"]:
            print(f"  {b['z_center_m']:>10.2f} {b['n_pixels']:>10,} "
                  f"{b['mean_abs_std_m']:>12.4f} {b['median_abs_std_m']:>11.4f} "
                  f"{b['mean_rel_std_pct']:>10.2f} {b['median_rel_std_pct']:>9.2f}")


def print_summary(camera: str, g: dict, model: dict) -> None:
    W = 84
    print(f"\n{'═' * W}")
    print(f"  CROSS-SCENE SUMMARY {camera.upper()}")
    print(f"{'═' * W}")
    print(f"  Scenes: {g['n_scenes']}  |  Frames: {g['total_frames']}  |  Valid px: {g['total_valid_px']:,}")
    print(f"  Global mean std        : {g['global_mean_std_m']:.4f} m")
    print(f"  Global median std      : {g['global_median_std_m']:.4f} m")
    print(f"  Global max std         : {g['global_max_std_m']:.4f} m")
    print(f"  Global mean rel. std   : {g['global_mean_rel_pct']:.2f}%")
    print(f"  Global median rel. std : {g['global_med_rel_pct']:.2f}%")
    print(f"  Stable ≤1cm / ≤5cm    : {g['pct_stable_1cm']:.1f}% / {g['pct_stable_5cm']:.1f}%")
    print(f"  Stable ≤1% / ≤5% rel  : {g['pct_stable_rel_1pct']:.1f}% / {g['pct_stable_rel_5pct']:.1f}%")
    print(f"  Best  scene            : {g['best_scene']}")
    print(f"  Worst scene            : {g['worst_scene']}")
    if model:
        print(f"\n  ── Power-law  σ_rel(Z) = α · Z^β  [raw ratio] ──────────────────────────")
        print(f"  α = {model['rel_alpha']:.8f} ± {model['rel_alpha_std']:.8f}")
        print(f"  β = {model['rel_beta']:.3f} ± {model['rel_beta_std']:.3f}")
        print(f"  R² = {model['rel_r2']:.4f}")
        print(f"    → Z=1m: {model['rel_sigma_at_1m_pct']:.3f}%")
        print(f"    → Z=3m: {model['rel_sigma_at_3m_pct']:.3f}%")
        print(f"    → Z=5m: {model['rel_sigma_at_5m_pct']:.3f}%")
        print(f"\n  ── Exponential  σ_abs(Z) = a · exp(b · Z)  [metres] ────────────────────")
        print(f"  a = {model['exp_a']:.8f} ± {model['exp_a_std']:.8f}")
        print(f"  b = {model['exp_b']:.4f} ± {model['exp_b_std']:.4f}")
        print(f"  R² = {model['exp_r2']:.4f}")
        print(f"    → Z=1m: {model['exp_sigma_at_1m_m']*100:.3f} cm")
        print(f"    → Z=3m: {model['exp_sigma_at_3m_m']*100:.3f} cm")
        print(f"    → Z=5m: {model['exp_sigma_at_5m_m']*100:.3f} cm")
        print(f"\n  Better model (by R²): {model['better_model'].upper()}")
    print(f"{'═' * W}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
scene_dirs  = sorted(d for d in base_dir.iterdir() if d.is_dir())
all_cameras = {}

for camera in CAMERAS:
    pattern     = f"*{camera}_depth.npy"
    all_results = {}
    skipped     = []

    for scene_dir in scene_dirs:
        depth_dir = scene_dir / "depth"
        if not depth_dir.exists():
            skipped.append((scene_dir.name, "no depth/ subfolder"))
            continue
        result = compute_scene_stats(depth_dir, pattern)
        if result is None:
            skipped.append((scene_dir.name, f"fewer than 2 {camera} depth maps"))
            continue
        all_results[scene_dir.name] = result
        # print_scene(scene_dir.name, result, camera)   # odkomentuj pre detail

    if not all_results:
        print(f"[{camera}] No results found — skipping.")
        continue

    global_stats = build_global_stats(all_results)
    model        = fit_depth_models(all_results)
    print_summary(camera, global_stats, model)

    # ── Per-camera outputs ────────────────────────────────────────────────────
    camera_out = out_dir / camera
    camera_out.mkdir(parents=True, exist_ok=True)

    plot_depth_model(all_results, model, camera,
                     camera_out / "depth_noise_model.png")

    global_summary = {
        "camera": camera, "date": date,
        "global": {k: v for k, v in global_stats.items() if k != "per_scene"},
        "model":  model,
    }
    with open(camera_out / "global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    full_stats = {
        "camera": camera, "date": date,
        "scenes": {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in all_results.items()
        },
    }
    with open(camera_out / "per_scene_stats.json", "w") as f:
        json.dump(full_stats, f, indent=2)

    all_cameras[camera] = {"global": global_stats, "model": model}

    if skipped:
        print(f"  [{camera}] Skipped:")
        for name, reason in skipped:
            print(f"    ✗  {name}  — {reason}")

# ── Combined summary ──────────────────────────────────────────────────────────
if all_cameras:
    combined = {
        "date": date,
        "cameras": {
            cam: {
                "global": {k: v for k, v in d["global"].items() if k != "per_scene"},
                "model":  d["model"],
            }
            for cam, d in all_cameras.items()
        }
    }
    with open(out_dir / "cameras_combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[✓] Saved combined summary → {out_dir / 'cameras_combined_summary.json'}")
    print(f"[✓] Per-camera outputs     → {out_dir}/<camera>/")