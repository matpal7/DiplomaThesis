from pathlib import Path
import numpy as np
import scipy.stats as stats
from scipy.ndimage import generic_gradient_magnitude, sobel
import matplotlib.pyplot as plt
import json

CAMERA_COLORS = {
    "zed":       {"scatter": "steelblue",  "line": "navy"},
    "realsense": {"scatter": "darkorange", "line": "saddlebrown"},
}

DISTRIBUTIONS = {
    "norm":     stats.norm,
    "laplace":  stats.laplace,
    "cauchy":   stats.cauchy,
    "t":        stats.t,
    "logistic": stats.logistic,
}

PARAM_NAMES = {
    "norm":     ["loc", "scale"],
    "laplace":  ["loc", "scale"],
    "cauchy":   ["loc", "scale"],
    "t":        ["df", "loc", "scale"],
    "logistic": ["loc", "scale"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
parent_dir = Path(__file__).resolve().parents[3]
date       = "24042026"
base_dir   = parent_dir / "datasets" / f"dataset_{date}" / "camera_stats_model"
out_dir    = parent_dir / "out" / f"out_{date}" / "cameras_statistic_model"
out_dir.mkdir(parents=True, exist_ok=True)

rng        = np.random.default_rng(42)
scene_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
print(scene_dirs)
camera_data = {}
CAMERAS     = ["zed", "realsense"]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):   # zachytí float32, float64, atď.
            return float(obj)
        if isinstance(obj, np.integer):    # zachytí int32, int64, atď.
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2: Reference depth + relative error (per scene)
# ─────────────────────────────────────────────────────────────────────────────
def compute_relative_errors(depth_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_ref = np.nanmean(depth_stack, axis=0)                        # (H, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = (depth_stack - z_ref[np.newaxis]) / z_ref[np.newaxis]
    return z_ref, rel_err


# ─────────────────────────────────────────────────────────────────────────────
# Edge mask — exclude flying pixels at depth discontinuities
# ─────────────────────────────────────────────────────────────────────────────
def compute_edge_mask(z_ref: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    z_filled = np.where(np.isfinite(z_ref), z_ref, 0.0)
    grad     = generic_gradient_magnitude(z_filled, sobel).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_grad = grad / np.where(z_filled > 0, z_filled, np.nan)
    return (rel_grad < threshold) & np.isfinite(z_ref)             # True = planar


# ─────────────────────────────────────────────────────────────────────────────
# Load depth stacks — compute ε and z̄ per scene, then pool
# ─────────────────────────────────────────────────────────────────────────────
for camera in CAMERAS:
    pattern    = f"depth/*{camera}_depth.npy"
    all_eps    = []
    all_z_flat = []

    for scene in scene_dirs:
        files = sorted(scene.glob(pattern))
        if not files:
            continue
        frames      = [np.load(f).astype(np.float32) for f in files]
        scene_stack = np.stack(frames, axis=0)          # (N, H, W)
        scene_stack[scene_stack <= 0] = np.nan

        z_ref, rel_err = compute_relative_errors(scene_stack)

        # Edge mask — exclude pixels at depth discontinuities
        flat_mask      = compute_edge_mask(z_ref)       # (H, W)
        flat_mask_3d   = np.broadcast_to(flat_mask[np.newaxis], rel_err.shape)

        # Pair ε with its z̄
        z_ref_expanded = np.broadcast_to(z_ref[np.newaxis], rel_err.shape)

        eps_flat = rel_err[flat_mask_3d].ravel()
        z_flat   = z_ref_expanded[flat_mask_3d].ravel()
        valid    = np.isfinite(eps_flat) & np.isfinite(z_flat)

        all_eps.append(eps_flat[valid])
        all_z_flat.append(z_flat[valid])

    if not all_eps:
        print(f"[{camera}] No data found — skipping.")
        continue

    eps_v = np.concatenate(all_eps)
    z_v   = np.concatenate(all_z_flat)

    camera_data[camera] = {"eps_v": eps_v, "z_v": z_v}
    print(f"[{camera}] Valid ε samples (planar only): {len(eps_v):,}")


# ─────────────────────────────────────────────────────────────────────────────
# Distribution fitting + KS test
# ─────────────────────────────────────────────────────────────────────────────
def test_distributions(eps_v: np.ndarray) -> dict:
    results = {}
    for name, dist in DISTRIBUTIONS.items():
        params        = dist.fit(eps_v, floc=0.0)
        ks_stat, ks_p = stats.kstest(eps_v, name, args=params)
        results[name] = dict(params=params, ks_stat=ks_stat, ks_p=ks_p)
    return dict(sorted(results.items(), key=lambda x: x[1]["ks_stat"]))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def _plot_panels(ax, eps_v, best_name, best_dist_result, colors):
    dist    = DISTRIBUTIONS[best_name]
    params  = best_dist_result["params"]
    ks_stat = best_dist_result["ks_stat"]

    x_min, x_max = (-0.04, 0.04)
    eps_clipped  = eps_v[(eps_v >= x_min) & (eps_v <= x_max)]

    counts, edges = np.histogram(eps_clipped, bins=120, density=True)
    centers       = 0.5 * (edges[:-1] + edges[1:])
    bin_width     = edges[1] - edges[0]
    x_g = np.linspace(x_min, x_max, 300)
    y_g = dist.pdf(x_g, *params)

    ax.bar(centers, counts, width=bin_width,
           color=colors["scatter"], alpha=0.70, label="Histogram ε")
    ax.plot(x_g, y_g, linewidth=2.5, color=colors["line"],
            label=f"Fitovanie: {best_name}")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Relatívna chyba ε")
    ax.set_ylabel("Hustota pravdepodobnosti")
    ax.set_title(f"Rozdelenie relatívnej chyby — najlepší fit: {best_name}")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97, f"KS štatistika={ks_stat:.4f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


def plot_single(eps_v, best_name, best_dist_result, camera, out_path):
    colors = CAMERA_COLORS[camera]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Model šumu hĺbky — {camera.upper()}", fontsize=13, fontweight="bold")
    _plot_panels(ax, eps_v, best_name, best_dist_result, colors)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  [✓] Graf uložený → {out_path}")

def plot_eps_vs_depth(eps_v, z_v, camera, dist_results, out_path):
    """
    Scatter plot ε vs z̄ s:
    - binovanými Cauchyho γ hodnotami namiesto std
    - vyznačeným 95% CI pásmom z globálneho modelu
    """
    colors = CAMERA_COLORS[camera]

    # ── Subsampling pre scatter ───────────────────────────────────────────
    MAX_SCATTER = 100_000
    if len(eps_v) > MAX_SCATTER:
        idx   = rng.choice(len(eps_v), size=MAX_SCATTER, replace=False)
        eps_s = eps_v[idx]
        z_s   = z_v[idx]
    else:
        eps_s, z_s = eps_v, z_v

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(z_s, eps_s, s=1, alpha=0.10, color=colors["scatter"], rasterized=True)

    # ── Binované Cauchyho γ — správna náhrada za std ──────────────────────
    bin_edges  = np.percentile(z_s, np.linspace(0, 100, 50))
    bin_idx    = np.digitize(z_s, bin_edges)
    bin_centers, medians, gammas = [], [], []

    for i in range(1, len(bin_edges)):
        mask = bin_idx == i
        if mask.sum() < 30:
            continue
        bin_centers.append(0.5 * (bin_edges[i-1] + bin_edges[i]))
        medians.append(np.median(eps_s[mask]))

        # Cauchyho γ pre tento bin — MLE fit s floc=0
        _, gamma_bin = stats.cauchy.fit(eps_s[mask], floc=0)
        gammas.append(gamma_bin)

    bin_centers = np.array(bin_centers)
    medians     = np.array(medians)
    gammas      = np.array(gammas)

    # ── 95% CI pásmo z binovaných γ ──────────────────────────────────────
    k      = np.tan(0.95 * np.pi / 2)   # ≈ 12.706 — Cauchyho kvantil
    ci_lo  = -k * gammas
    ci_hi  = +k * gammas

    ax.fill_between(bin_centers, ci_lo, ci_hi,
                    color=colors["line"], alpha=0.15, label="95% CI pásmo (Cauchy γ)")
    ax.plot(bin_centers, medians, color=colors["line"], linewidth=2, label="Medián ε")

    # ── Globálny 95% CI z fitovaného modelu ──────────────────────────────
    best_name   = next(iter(dist_results))
    best_params = dist_results[best_name]["params"]
    if best_name == "cauchy":
        gamma_global = best_params[-1]   # scale parameter
        lo_global    = -k * gamma_global
        hi_global    = +k * gamma_global
        ax.axhline(lo_global, color="red", linewidth=1.2, linestyle="--",
                   label=f"Globálny 95% CI (γ={gamma_global:.5f})")
        ax.axhline(hi_global, color="red", linewidth=1.2, linestyle="--")

    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    ax.set_xlabel("Referenčná hĺbka z̄ (m)")
    ax.set_ylabel("Relatívna chyba ε")
    ax.set_title(f"Hĺbkovo závislá chyba — {camera.upper()}")
    ax.set_ylim(-0.08, 0.08)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Scatter plot uložený → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
results = {}

for camera, data in camera_data.items():
    print(f"\n{'='*60}")
    print(f"  Camera: {camera.upper()}")
    print(f"{'='*60}")

    eps_v = data["eps_v"]
    z_v   = data["z_v"]
    print(f"  Valid samples (planar): {len(eps_v):,}")
    print(f"  Depth range: {z_v.min():.3f}m — {z_v.max():.3f}m")

    MAX_SAMPLES = 5_000_000
    if len(eps_v) > MAX_SAMPLES:
        idx        = rng.choice(len(eps_v), size=MAX_SAMPLES, replace=False)
        eps_sample = eps_v[idx]
        z_sample   = z_v[idx]           # ← keep pairing intact
        print(f"  Subsampled to: {MAX_SAMPLES:,} for fitting")
    else:
        eps_sample = eps_v
        z_sample   = z_v

    dist_results = test_distributions(eps_sample)
    print(f"\n  Distribution ranking (KS stat — lower is better):")
    print(f"  {'Distribution':<14} {'KS stat':>10}  {'p-value':>12}  {'verdict'}")
    print(f"  {'-'*55}")
    for dname, dr in dist_results.items():
        verdict = "✓ not rejected" if dr['ks_p'] > 0.05 else "✗ rejected"
        print(f"  {dname:<14} {dr['ks_stat']:>10.4f}  {dr['ks_p']:>12.4e}  {verdict}")

    best_name        = next(iter(dist_results))
    best_dist_result = dist_results[best_name]

    out_path = out_dir / f"{camera}_error_model.png"
    plot_single(eps_sample, best_name, best_dist_result, camera, out_path)

    scatter_path = out_dir / f"{camera}_eps_vs_depth.png"
    plot_eps_vs_depth(eps_v, z_v, camera, dist_results, scatter_path)

    results[camera] = dict(dist_ranking=dist_results, best=best_name,
                           z_range=(float(z_v.min()), float(z_v.max())))



# ─────────────────────────────────────────────────────────────────────────────
# Model summary — all distributions, all parameters
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  MODEL SUMMARY — ALL DISTRIBUTIONS")
print(f"{'='*60}")

summary_to_save = {}

for cam, r in results.items():
    print(f"\n  [{cam.upper()}]  depth range: {r['z_range'][0]:.2f}m — {r['z_range'][1]:.2f}m")
    print(f"  {'Distribution':<14} {'KS stat':>10}  {'Parameters'}")
    print(f"  {'-'*70}")
    for dname, dr in r["dist_ranking"].items():
        param_labels = PARAM_NAMES.get(dname, [f"p{i}" for i in range(len(dr["params"]))])
        param_str    = "  ".join(f"{k}={v:.5f}" for k, v in zip(param_labels, dr["params"]))
        print(f"  {dname:<14} {dr['ks_stat']:>10.4f}  {param_str}")

    # ── Najlepšie rozdelenie ──────────────────────────────────────────────
    best_name = r["best"]
    best_dr   = r["dist_ranking"][best_name]
    param_labels = PARAM_NAMES.get(best_name, [f"p{i}" for i in range(len(best_dr["params"]))])

    summary_to_save[cam] = {
        "best_distribution": best_name,
        "ks_stat":           best_dr["ks_stat"],
        "ks_p":              best_dr["ks_p"],
        "z_range_m":         list(r["z_range"]),
        "params":            dict(zip(param_labels, [float(p) for p in best_dr["params"]])),
    }

# ── Uloženie do JSON ──────────────────────────────────────────────────────
json_path = out_dir / "best_distribution_models.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary_to_save, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
print(f"\n  [✓] Najlepšie modely uložené → {json_path}")