import cv2
import numpy as np
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
parent_dir = Path(__file__).resolve().parents[3]
base_dir   = parent_dir / "datasets" / "dataset_02042026" / "stereo_4k_calibration"
camera     = "zed"
pattern    = f"*{camera}_depth.npy"

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_scene_stats(depth_dir: Path) -> dict | None:
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

    depths = np.stack(depths, axis=0)
    N      = depths.shape[0]
    valid  = np.isfinite(depths) & (depths > 0)

    # per-frame
    frame_stats = []
    for i in range(N):
        v = valid[i]
        d_v = depths[i][v]
        mean_z = float(np.mean(d_v)) if v.any() else np.nan
        std_z  = float(np.std(d_v))  if v.any() else np.nan
        frame_stats.append({
            "file":         depth_files[i].name,
            "valid_px":     int(v.sum()),
            "mean_m":       mean_z,
            "median_m":     float(np.median(d_v)) if v.any() else np.nan,
            "std_m":        std_z,
            # relative std per pixel: std(Z) / Z for each pixel, then averaged
            "mean_rel_std_pct": float(np.mean(np.std(depths[:, v], axis=0) /
                                              np.mean(depths[:, v], axis=0)) * 100)
                                 if v.any() else np.nan,
        })

    # consecutive diffs
    consec_diffs = []
    for i in range(N - 1):
        both = valid[i] & valid[i + 1]
        if not both.any():
            consec_diffs.append({
                "from": depth_files[i].name, "to": depth_files[i+1].name,
                "valid_px": 0,
                "mean_abs_diff_m": np.nan, "median_abs_diff_m": np.nan,
                "max_abs_diff_m":  np.nan, "mean_signed_diff_m": np.nan,
                "mean_rel_diff_pct": np.nan, "median_rel_diff_pct": np.nan,
            })
            continue
        diff      = depths[i+1][both] - depths[i][both]
        abs_diff  = np.abs(diff)
        ref_depth = depths[i][both]  # Z_i as reference
        rel_diff  = abs_diff / ref_depth * 100  # as percentage

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

    # global stability — per-pixel std and relative std across all frames
    all_valid_mask    = valid.all(axis=0)
    pixel_std_map     = np.full(ref_shape, np.nan, dtype=np.float32)
    pixel_mean_map    = np.full(ref_shape, np.nan, dtype=np.float32)
    pixel_rel_std_map = np.full(ref_shape, np.nan, dtype=np.float32)

    if all_valid_mask.any():
        stacked_valid        = depths[:, all_valid_mask]
        pixel_std_map[all_valid_mask]  = stacked_valid.std(axis=0)
        pixel_mean_map[all_valid_mask] = stacked_valid.mean(axis=0)
        # relative std = σ(Z) / mean(Z) per pixel
        pixel_rel_std_map[all_valid_mask] = (
            pixel_std_map[all_valid_mask] / pixel_mean_map[all_valid_mask] * 100
        )

    # depth-binned relative std for model fitting (Z vs σ_rel)
    depth_bins = np.arange(0.5, 6.0, 0.25)  # 0.5m to 6m in 0.25m bins
    bin_stats  = []
    if all_valid_mask.any():
        z_flat        = pixel_mean_map[all_valid_mask]
        rel_std_flat  = pixel_rel_std_map[all_valid_mask]
        abs_std_flat  = pixel_std_map[all_valid_mask]
        for b_lo, b_hi in zip(depth_bins[:-1], depth_bins[1:]):
            mask_bin = (z_flat >= b_lo) & (z_flat < b_hi)
            if mask_bin.sum() < 10:
                continue
            bin_stats.append({
                "z_center_m":        float((b_lo + b_hi) / 2),
                "n_pixels":          int(mask_bin.sum()),
                "mean_abs_std_m":    float(np.mean(abs_std_flat[mask_bin])),
                "median_abs_std_m":  float(np.median(abs_std_flat[mask_bin])),
                "mean_rel_std_pct":  float(np.mean(rel_std_flat[mask_bin])),
                "median_rel_std_pct":float(np.median(rel_std_flat[mask_bin])),
            })

    return {
        "n_frames":            N,
        "frame_stats":         frame_stats,
        "consec_diffs":        consec_diffs,
        "pixels_valid_all":    int(all_valid_mask.sum()),
        "mean_pixel_std_m":    float(np.nanmean(pixel_std_map)),
        "median_pixel_std_m":  float(np.nanmedian(pixel_std_map)),
        "max_pixel_std_m":     float(np.nanmax(pixel_std_map)) if all_valid_mask.any() else np.nan,
        "pct_stable_1cm":      float(np.nanmean(pixel_std_map < 0.01)) * 100,
        "pct_stable_5cm":      float(np.nanmean(pixel_std_map < 0.05)) * 100,
        # relative stability
        "mean_rel_std_pct":    float(np.nanmean(pixel_rel_std_map)),
        "median_rel_std_pct":  float(np.nanmedian(pixel_rel_std_map)),
        "pct_stable_rel_1pct": float(np.nanmean(pixel_rel_std_map < 1.0)) * 100,
        "pct_stable_rel_5pct": float(np.nanmean(pixel_rel_std_map < 5.0)) * 100,
        "bin_stats":           bin_stats,
        # store maps for cross-scene aggregation if needed
        "_pixel_rel_std_map":  pixel_rel_std_map,
        "_pixel_std_map":      pixel_std_map,
    }


def print_scene(scene_name: str, r: dict) -> None:
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
    print(f"  Stable within 1 cm         : {r['pct_stable_1cm']:.1f}%")
    print(f"  Stable within 5 cm         : {r['pct_stable_5cm']:.1f}%")
    print(f"  Mean relative std          : {r['mean_rel_std_pct']:.2f}%")
    print(f"  Median relative std        : {r['median_rel_std_pct']:.2f}%")
    print(f"  Stable within 1% relative  : {r['pct_stable_rel_1pct']:.1f}%")
    print(f"  Stable within 5% relative  : {r['pct_stable_rel_5pct']:.1f}%")

    if r["bin_stats"]:
        print(f"\n  Depth-binned relative std (for Z² model fit):")
        print(f"  {'Z center':>10} {'N pixels':>10} {'MeanStd(m)':>12} {'MedStd(m)':>11} {'MeanRel%':>10} {'MedRel%':>9}")
        print("  " + "─" * (W - 2))
        for b in r["bin_stats"]:
            print(f"  {b['z_center_m']:>10.2f} {b['n_pixels']:>10,} "
                  f"{b['mean_abs_std_m']:>12.4f} {b['median_abs_std_m']:>11.4f} "
                  f"{b['mean_rel_std_pct']:>10.2f} {b['median_rel_std_pct']:>9.2f}")


# ── Main: iterate over all scenes ────────────────────────────────────────────
scene_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())

if not scene_dirs:
    raise RuntimeError(f"No scene subdirectories found in {base_dir}")

all_results = {}
skipped     = []

for scene_dir in scene_dirs:
    depth_dir = scene_dir / "depth"
    if not depth_dir.exists():
        skipped.append((scene_dir.name, "no depth/ subfolder"))
        continue

    result = compute_scene_stats(depth_dir)
    if result is None:
        skipped.append((scene_dir.name, f"fewer than 2 {camera} depth maps"))
        continue

    all_results[scene_dir.name] = result
    print_scene(scene_dir.name, result)

# ── Cross-scene summary ───────────────────────────────────────────────────────
if all_results:
    W = 84
    print(f"\n{'═' * W}")
    print("  CROSS-SCENE SUMMARY")
    print(f"{'═' * W}")
    print(f"  {'Scene':<20} {'Frames':>6} {'MeanStd':>9} {'MedStd':>9} "
          f"{'MeanRel%':>9} {'Stbl1cm':>8} {'Stbl5cm':>8} {'Stbl1%':>7} {'Stbl5%':>7}")
    print("  " + "─" * (W - 2))

    for name, r in all_results.items():
        print(f"  {name:<20} {r['n_frames']:>6} "
              f"{r['mean_pixel_std_m']:>9.4f} {r['median_pixel_std_m']:>9.4f} "
              f"{r['mean_rel_std_pct']:>8.2f}% "
              f"{r['pct_stable_1cm']:>7.1f}% {r['pct_stable_5cm']:>7.1f}% "
              f"{r['pct_stable_rel_1pct']:>6.1f}% {r['pct_stable_rel_5pct']:>6.1f}%")

    total_frames   = sum(r["n_frames"]        for r in all_results.values())
    total_valid_px = sum(r["pixels_valid_all"] for r in all_results.values())
    n_scenes       = len(all_results)

    global_mean_std      = float(np.mean([r["mean_pixel_std_m"]    for r in all_results.values()]))
    global_med_std       = float(np.mean([r["median_pixel_std_m"]  for r in all_results.values()]))
    global_max_std       = float(np.max( [r["max_pixel_std_m"]     for r in all_results.values()]))
    global_stbl_1cm      = float(np.mean([r["pct_stable_1cm"]      for r in all_results.values()]))
    global_stbl_5cm      = float(np.mean([r["pct_stable_5cm"]      for r in all_results.values()]))
    global_mean_rel      = float(np.mean([r["mean_rel_std_pct"]    for r in all_results.values()]))
    global_med_rel       = float(np.mean([r["median_rel_std_pct"]  for r in all_results.values()]))
    global_stbl_rel_1pct = float(np.mean([r["pct_stable_rel_1pct"] for r in all_results.values()]))
    global_stbl_rel_5pct = float(np.mean([r["pct_stable_rel_5pct"] for r in all_results.values()]))

    print("  " + "─" * (W - 2))
    print(f"  {'GLOBAL':<20} {total_frames:>6} "
          f"{global_mean_std:>9.4f} {global_med_std:>9.4f} "
          f"{global_mean_rel:>8.2f}% "
          f"{global_stbl_1cm:>7.1f}% {global_stbl_5cm:>7.1f}% "
          f"{global_stbl_rel_1pct:>6.1f}% {global_stbl_rel_5pct:>6.1f}%")

    print(f"\n  Scenes evaluated       : {n_scenes}")
    print(f"  Total frames           : {total_frames}")
    print(f"  Total valid pixels     : {total_valid_px:,}")
    print(f"  Global mean std        : {global_mean_std:.4f} m")
    print(f"  Global median std      : {global_med_std:.4f} m")
    print(f"  Global max std         : {global_max_std:.4f} m")
    print(f"  Global mean rel. std   : {global_mean_rel:.2f}%")
    print(f"  Global median rel. std : {global_med_rel:.2f}%")
    print(f"  Stable within 1 cm     : {global_stbl_1cm:.1f}%")
    print(f"  Stable within 5 cm     : {global_stbl_5cm:.1f}%")
    print(f"  Stable within 1% rel.  : {global_stbl_rel_1pct:.1f}%")
    print(f"  Stable within 5% rel.  : {global_stbl_rel_5pct:.1f}%")

    best  = min(all_results, key=lambda k: all_results[k]["median_pixel_std_m"])
    worst = max(all_results, key=lambda k: all_results[k]["median_pixel_std_m"])
    print(f"\n  Most stable scene      : {best}  "
          f"(median std = {all_results[best]['median_pixel_std_m']:.4f} m)")
    print(f"  Least stable scene     : {worst}  "
          f"(median std = {all_results[worst]['median_pixel_std_m']:.4f} m)")
    print(f"{'═' * W}\n")

if skipped:
    print("  Skipped scenes:")
    for name, reason in skipped:
        print(f"    ✗  {name}  — {reason}")