import numpy as np
from collections import defaultdict
from functools import partial



def delta(arr1, arr2, exponent):
    inlier = np.maximum(arr1 / arr2, arr2 / arr1)
    return (inlier < 1.25 ** exponent).astype(np.float32).mean()

def tau(arr1, arr2, perc):
    inlier = np.maximum(arr1 / arr2, arr2 / arr1)
    return (inlier < (1.0 + perc)).astype(np.float32).mean()

def ssi(arr1, arr2):
    stability_mat = 1e-9 * np.eye(2)
    arr2_one = np.stack([arr2, np.ones_like(arr2)], axis=1)   # (N,2)
    scale_shift = np.linalg.inv(arr2_one.T @ arr2_one + stability_mat) @ (
        arr2_one.T @ arr1[:, np.newaxis]
    )
    scale, shift = scale_shift[0, 0], scale_shift[1, 0]
    return arr2 * scale + shift

def si(arr1, arr2):
    return arr2 * np.median(arr1) / np.median(arr2)

def arel(arr1, arr2):
    arr2 = arr2 * np.median(arr1) / np.median(arr2)
    return (np.abs(arr1 - arr2) / arr1).mean()

def d_auc(arr1, arr2):
    exponents = np.linspace(0.01, 5.0, num=100)
    deltas = [delta(arr1, arr2, e) for e in exponents]
    return float(np.trapezoid(deltas, exponents) / 5.0)

DICT_METRICS = {
    "d1":        partial(delta, exponent=1.0),
    "d2":        partial(delta, exponent=2.0),
    "d3":        partial(delta, exponent=3.0),
    "rmse":      lambda gt, pred: np.sqrt(((gt - pred) ** 2).mean()),
    "rmselog":   lambda gt, pred: np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean()),
    "median_abs_error": lambda gt, pred: float(np.median(np.abs(gt - pred))),
    "arel":      lambda gt, pred: (np.abs(gt - pred) / gt).mean(),
    "sqrel":     lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10":     lambda gt, pred: np.abs(np.log10(pred) - np.log10(gt)).mean(),
    "silog":     lambda gt, pred: 100 * np.std(np.log(pred) - np.log(gt)),
    "medianlog": lambda gt, pred: 100 * float(np.abs(np.median(np.log(pred) - np.log(gt)))),
    "d_auc":     d_auc,
    "tau":       partial(tau, perc=0.03),
}


def eval_depth(
    gts: list[np.ndarray],      # list of (H,W) float32 arrays
    preds: list[np.ndarray],    # list of (H,W) float32 arrays
    masks: list[np.ndarray],    # list of (H,W) bool arrays
    max_depth: float = None,
) -> dict[str, np.ndarray]:

    summary_metrics = defaultdict(list)

    for gt, pred, mask in zip(gts, preds, masks):
        # pred = resize_pred(pred, gt.shape[:2])

        if max_depth is not None:
            mask = mask & (gt <= max_depth)

        gt_m, pred_m = gt[mask], pred[mask]

        for name, fn in DICT_METRICS.items():
            if name in ["tau", "d1", "arel"]:
                for rescale_fn in [ssi, si]:
                    key = f"{name}_{rescale_fn.__name__}"
                    summary_metrics[key].append(fn(gt_m, rescale_fn(gt_m, pred_m)))
            summary_metrics[name].append(fn(gt_m, pred_m))

    return {name: np.array(vals) for name, vals in summary_metrics.items()}

def eval_depth_single(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    max_depth: float = None,
) -> dict[str, float]:
    """Evaluate a single GT/pred pair. Returns flat dict of metric → float."""
    if max_depth is not None:
        mask = mask & (gt <= max_depth)

    gt_m   = gt[mask].astype(np.float64)
    pred_m = pred[mask].astype(np.float64)

    if len(gt_m) == 0:
        return {}

    result = {}
    for name, fn in DICT_METRICS.items():
        result[name] = fn(gt_m, pred_m)
        if name in ["tau", "d1", "arel"]:
            for rescale_fn in [ssi, si]:
                key = f"{name}_{rescale_fn.__name__}"
                try:
                    result[key] = fn(gt_m, rescale_fn(gt_m, pred_m))
                except Exception:
                    result[key] = float("nan")
    return result