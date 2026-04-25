# ─────────────────────────────────────────────────────────────────────────────
# Model ZED kamery — Cauchyho rozdelenie s loc=0
# ─────────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

import numpy as np
from scipy import stats

CONFIDENCE  = 0.95

def load_zed_gamma(json_path: Path, camera: str = "zed") -> float:
    """
    Načíta scale parameter (γ) Cauchyho rozdelenia pre danú kameru
    z best_distribution_models.json.

    json_path : cesta k súboru best_distribution_models.json
    camera    : kľúč kamery ("zed" alebo "realsense")
    """
    with open(json_path, "r", encoding="utf-8") as f:
        models = json.load(f)

    cam_model = models.get(camera)
    if cam_model is None:
        raise KeyError(f"Kamera '{camera}' sa nenašla v {json_path}")

    best_dist = cam_model["best_distribution"]
    if best_dist != "cauchy":
        raise ValueError(
            f"Najlepšie rozdelenie pre '{camera}' je '{best_dist}', nie 'cauchy'. "
            f"Skontroluj model alebo uprav funkciu."
        )

    return float(cam_model["params"]["scale"])


def is_within_ci(
    eps: np.ndarray,
    z_ref: np.ndarray,
    gamma: float,
    confidence: float = CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Skontroluje, či sa relatívne chyby nachádzajú v 95% CI Cauchyho modelu.

    Parametre
    ---------
    eps        : (N,) — pole relatívnych chýb ε = (z_pred - z_ref) / z_ref
    z_ref      : (N,) — pole referenčných hĺbok (ground truth)
    gamma      : scale parameter Cauchyho rozdelenia
    confidence : požadovaná hladina spoľahlivosti (default 0.95)

    Návratová hodnota
    -----------------
    within_ci  : (N,) bool — True ak ε leží v CI
    ci_bounds  : (2,) float — (ci_lo, ci_hi) hranice intervalu
    """
    valid = np.isfinite(eps) & np.isfinite(z_ref) & (z_ref > 0)

    ci_lo, ci_hi = stats.cauchy.interval(confidence, loc=0, scale=gamma)

    within_ci = np.zeros(len(eps), dtype=bool)
    within_ci[valid] = (eps[valid] >= ci_lo) & (eps[valid] <= ci_hi)

    return within_ci, np.array([ci_lo, ci_hi])


if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "24042026"
    json_path = parent_dir / "out" / f"out_{date}" / "cameras_statistic_model" / "best_distribution_models.json"

    ZED_GAMMA = load_zed_gamma(json_path, camera="zed")
    print(f"ZED γ (načítané): {ZED_GAMMA:.5f}")

    eps_lo, eps_hi = stats.cauchy.interval(CONFIDENCE, loc=0, scale=ZED_GAMMA)
    print(f"ZED 95% CI pre ε: [{eps_lo:.5f}, {eps_hi:.5f}]")