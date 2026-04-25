# ─────────────────────────────────────────────────────────────────────────────
# Model ZED kamery — Cauchyho rozdelenie s loc=0
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy import stats

ZED_GAMMA   = 0.00297   # scale parameter z fitovaného Cauchyho rozdelenia
CONFIDENCE  = 0.95

# 95% interval pre ε (Cauchyho kvantil)
eps_lo, eps_hi = stats.cauchy.interval(CONFIDENCE, loc=0, scale=ZED_GAMMA)
print(f"ZED 95% CI pre ε: [{eps_lo:.5f}, {eps_hi:.5f}]")
# → [-0.03770, +0.03770]


def evaluate_depth_predictions(z_zed: np.ndarray,
                                z_pred: np.ndarray,
                                gamma: float = ZED_GAMMA,
                                confidence: float = CONFIDENCE) -> dict:
    """
    Porovnaj predikované hĺbky z neurónovej siete so ZED GT.

    z_zed  : (H, W) alebo (N,) — ZED hĺbková mapa (GT)
    z_pred : (H, W) alebo (N,) — predikovaná hĺbková mapa
    """
    # Platné pixely — obe mapy musia mať platnú hodnotu
    valid = np.isfinite(z_zed) & np.isfinite(z_pred) & (z_zed > 0)

    z_ref  = z_zed[valid]
    z_hat  = z_pred[valid]

    # Relatívna chyba predikcie voči ZED
    eps = (z_hat - z_ref) / z_ref

    # 95% CI hranice z Cauchyho modelu
    lo, hi = stats.cauchy.interval(confidence, loc=0, scale=gamma)

    # Maska akceptovateľných pixelov
    accepted = (eps >= lo) & (eps <= hi)

    # Štatistiky
    n_total    = len(eps)
    n_accepted = accepted.sum()
    pct        = 100 * n_accepted / n_total

    # Absolútna chyba
    abs_err = np.abs(z_hat - z_ref)

    return dict(
        eps          = eps,
        accepted     = accepted,
        n_total      = n_total,
        n_accepted   = n_accepted,
        pct_accepted = pct,
        mae          = float(np.mean(abs_err)),
        rmse         = float(np.sqrt(np.mean(abs_err**2))),
        median_eps   = float(np.median(np.abs(eps))),
        ci_lo        = lo,
        ci_hi        = hi,
    )


def print_evaluation(result: dict, model_name: str = "MDE"):
    print(f"\n  [{model_name}] — Vyhodnotenie voči ZED GT (Cauchyho model, 95% CI)")
    print(f"  {'─'*55}")
    print(f"  Platné pixely    : {result['n_total']:>12,}")
    print(f"  95% CI (ε)       : [{result['ci_lo']:.5f}, {result['ci_hi']:.5f}]")
    print(f"  Akceptované      : {result['n_accepted']:>12,}  ({result['pct_accepted']:.1f}%)")
    print(f"  MAE              : {result['mae']*100:.3f} cm")
    print(f"  RMSE             : {result['rmse']*100:.3f} cm")
    print(f"  Medián |ε|       : {result['median_eps']:.5f}  ({result['median_eps']*100:.3f}%)")