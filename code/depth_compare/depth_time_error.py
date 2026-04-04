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
        frame_stats.append({
            "file":     depth_files[i].name,
            "valid_px": int(v.sum()),
            "mean_m":   float(np.mean(depths[i][v]))   if v.any() else np.nan,
            "median_m": float(np.median(depths[i][v])) if v.any() else np.nan,
            "std_m":    float(np.std(depths[i][v]))    if v.any() else np.nan,
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
                "max_abs_diff_m": np.nan,  "mean_signed_diff_m": np.nan,
            })
            continue
        diff     = depths[i+1][both] - depths[i][both]
        abs_diff = np.abs(diff)
        consec_diffs.append({
            "from":               depth_files[i].name,
            "to":                 depth_files[i+1].name,
            "valid_px":           int(both.sum()),
            "mean_abs_diff_m":    float(np.mean(abs_diff)),
            "median_abs_diff_m":  float(np.median(abs_diff)),
            "max_abs_diff_m":     float(np.max(abs_diff)),
            "mean_signed_diff_m": float(np.mean(diff)),
        })

    # global stability
    all_valid_mask = valid.all(axis=0)
    pixel_std_map  = np.full(ref_shape, np.nan, dtype=np.float32)
    if all_valid_mask.any():
        pixel_std_map[all_valid_mask] = depths[:, all_valid_mask].std(axis=0)

    return {
        "n_frames":           N,
        "frame_stats":        frame_stats,
        "consec_diffs":       consec_diffs,
        "pixels_valid_all":   int(all_valid_mask.sum()),
        "mean_pixel_std_m":   float(np.nanmean(pixel_std_map)),
        "median_pixel_std_m": float(np.nanmedian(pixel_std_map)),
        "max_pixel_std_m":    float(np.nanmax(pixel_std_map)) if all_valid_mask.any() else np.nan,
        "pct_stable_1cm":     float(np.nanmean(pixel_std_map < 0.01)) * 100,
        "pct_stable_5cm":     float(np.nanmean(pixel_std_map < 0.05)) * 100,
    }


def print_scene(scene_name: str, r: dict) -> None:
    W = 74
    print(f"\n{'█' * W}")
    print(f"  SCENE: {scene_name}  ({r['n_frames']} frames, camera={camera})")
    print(f"{'█' * W}")

    print(f"\n  {'File':<38} {'Mean':>7} {'Median':>7} {'Std':>7} {'Valid px':>10}")
    print("  " + "─" * (W - 2))
    for s in r["frame_stats"]:
        print(f"  {s['file']:<38} {s['mean_m']:>7.3f} {s['median_m']:>7.3f} "
              f"{s['std_m']:>7.3f} {s['valid_px']:>10,}")

    print(f"\n  {'From → To':<52} {'MeanΔ':>7} {'MedΔ':>6} {'MaxΔ':>7} {'SignedΔ':>9}")
    print("  " + "─" * (W - 2))
    for d in r["consec_diffs"]:
        label = f"{d['from']} → {d['to']}"
        print(f"  {label:<52} {d['mean_abs_diff_m']:>7.3f} "
              f"{d['median_abs_diff_m']:>6.3f} {d['max_abs_diff_m']:>7.3f} "
              f"{d['mean_signed_diff_m']:>+9.4f}")

    print(f"\n  Pixels valid in ALL frames : {r['pixels_valid_all']:,}")
    print(f"  Mean per-pixel std         : {r['mean_pixel_std_m']:.4f} m")
    print(f"  Median per-pixel std       : {r['median_pixel_std_m']:.4f} m")
    print(f"  Max per-pixel std          : {r['max_pixel_std_m']:.4f} m")
    print(f"  Stable within 1 cm         : {r['pct_stable_1cm']:.1f}%")
    print(f"  Stable within 5 cm         : {r['pct_stable_5cm']:.1f}%")


# ── Main: iterate over all scenes ────────────────────────────────────────────
scene_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())

if not scene_dirs:
    raise RuntimeError(f"No scene subdirectories found in {base_dir}")

all_results   = {}   # scene_name → stats dict
skipped       = []

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
    W = 74
    print(f"\n{'═' * W}")
    print("  CROSS-SCENE SUMMARY")
    print(f"{'═' * W}")
    print(f"  {'Scene':<20} {'Frames':>6} {'MeanStd':>9} {'MedStd':>9} "
          f"{'Stbl1cm':>8} {'Stbl5cm':>8} {'ValidPx':>10}")
    print("  " + "─" * (W - 2))

    for name, r in all_results.items():
        print(f"  {name:<20} {r['n_frames']:>6} "
              f"{r['mean_pixel_std_m']:>9.4f} {r['median_pixel_std_m']:>9.4f} "
              f"{r['pct_stable_1cm']:>7.1f}% {r['pct_stable_5cm']:>7.1f}% "
              f"{r['pixels_valid_all']:>10,}")

    # ── Global aggregate across ALL scenes ───────────────────────────────────
    # Weight each scene equally (simple mean across scenes)
    total_frames      = sum(r["n_frames"]           for r in all_results.values())
    total_valid_px    = sum(r["pixels_valid_all"]    for r in all_results.values())
    n_scenes          = len(all_results)

    global_mean_std   = float(np.mean([r["mean_pixel_std_m"]   for r in all_results.values()]))
    global_med_std    = float(np.mean([r["median_pixel_std_m"] for r in all_results.values()]))
    global_max_std    = float(np.max( [r["max_pixel_std_m"]    for r in all_results.values()]))
    global_stbl_1cm   = float(np.mean([r["pct_stable_1cm"]     for r in all_results.values()]))
    global_stbl_5cm   = float(np.mean([r["pct_stable_5cm"]     for r in all_results.values()]))

    print("  " + "─" * (W - 2))
    print(f"  {'GLOBAL (mean over scenes)':<20} {total_frames:>6} "
          f"{global_mean_std:>9.4f} {global_med_std:>9.4f} "
          f"{global_stbl_1cm:>7.1f}% {global_stbl_5cm:>7.1f}% "
          f"{total_valid_px:>10,}")

    print(f"\n  Scenes evaluated       : {n_scenes}")
    print(f"  Total frames           : {total_frames}")
    print(f"  Total valid pixels     : {total_valid_px:,}")
    print(f"  Global mean std        : {global_mean_std:.4f} m")
    print(f"  Global median std      : {global_med_std:.4f} m")
    print(f"  Global max std         : {global_max_std:.4f} m")
    print(f"  Stable within 1 cm     : {global_stbl_1cm:.1f}%")
    print(f"  Stable within 5 cm     : {global_stbl_5cm:.1f}%")

    # best / worst scene
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