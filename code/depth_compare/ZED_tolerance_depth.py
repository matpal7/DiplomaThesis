import numpy as np

ZED_MINI_BASELINE_M    = 0.063      # 63 mm
ZED_MINI_FOCAL_PX      = 688.79      # approximate, replace with your calibrated value
ZED_MINI_DISP_SIGMA_PX = 0.2

def zed_mini_depth_sigma(z: np.ndarray) -> np.ndarray:
    """
    Returns the expected 1-sigma depth error of ZED Mini at depth z (in meters).
    Based on stereo disparity error propagation:
        sigma_z = (z^2 * sigma_disp) / (f * b)
    """
    c = ZED_MINI_DISP_SIGMA_PX / (ZED_MINI_FOCAL_PX * ZED_MINI_BASELINE_M)
    return c * z ** 2


def within_sensor_tolerance(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    n_sigma: float = 1.0
) -> dict:
    """
    Checks whether per-pixel network error |pred - gt| is within
     ZED Mini's own sensor uncertainty at that depth.

    Parameters
    ----------
    gt     : (H, W) ground truth depth in meters
    pred   : (H, W) predicted depth in meters
    mask   : (H, W) bool — valid pixels to evaluate

    Returns
    -------
    dict with scalar summary metrics
    """

    gt_m   = gt[mask].astype(np.float64)
    pred_m = pred[mask].astype(np.float64)

    abs_error   = np.abs(pred_m - gt_m)

    sensor_sigma = zed_mini_depth_sigma(gt_m)       # expected ZED error at each depth

    print("TOLERANCE STAS")
    print(sensor_sigma.shape)
    print(sensor_sigma)
    print(abs_error.shape)

    tolerance    = n_sigma * sensor_sigma            # allowed band = n * sigma

    within = abs_error <= tolerance                  # bool per pixel
    print("Within:", within)

    # how many sigma each pixel's error represents
    error_in_sigmas = abs_error / np.maximum(sensor_sigma, 1e-9)

    dict_results = {
        f"within_{n_sigma}sigma":        float(within.mean()),           # 0–1, higher better
        f"within_{n_sigma}sigma_pct":    float(within.mean() * 100),     # % form
        "mean_error_in_sigmas":          float(error_in_sigmas.mean()),   # < 1.0 = better than sensor
        "median_error_in_sigmas":        float(np.median(error_in_sigmas)),
        "sensor_sigma_mean_m":           float(sensor_sigma.mean()),      # reference: what ZED expects
        "abs_error_mean_m":              float(abs_error.mean()),
    }

    return dict_results

def sensor_filtered_errors(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    n_sigma: float = 1.0,
) -> dict:
    """
    Computes error metrics ONLY on pixels whose |pred - gt| exceeds
    the ZED Mini sensor tolerance (n_sigma * sigma_ZED).
    Pixels within tolerance are treated as 'sensor-explainable' and excluded.

    Returns metrics on the 'genuine network errors' subset.
    """
    gt_m   = gt[mask].astype(np.float64)
    pred_m = pred[mask].astype(np.float64)

    abs_error    = np.abs(pred_m - gt_m)
    sensor_sigma = zed_mini_depth_sigma(gt_m)
    tolerance    = n_sigma * sensor_sigma

    within  = abs_error <= tolerance
    outside = ~within                        # pixels with genuine network error

    n_total   = len(gt_m)
    n_outside = int(outside.sum())

    base = {
        "n_total_px":          n_total,
        "n_outside_tolerance": n_outside,
        "pct_outside":         float(n_outside / n_total * 100) if n_total > 0 else 0.0,
    }

    if n_outside == 0:
        return {**base,
                "filtered_mean_abs_error_m":   0.0,
                "filtered_median_abs_error_m": 0.0,
                "filtered_rmse_m":             0.0,
                "filtered_mean_error_sigmas":  0.0,
                "filtered_arel":               0.0}

    ae_out      = abs_error[outside]
    gt_out      = gt_m[outside]
    sigmas_out  = ae_out / np.maximum(sensor_sigma[outside], 1e-9)

    return {
        **base,
        "filtered_mean_abs_error_m":   float(ae_out.mean()),
        "filtered_median_abs_error_m": float(np.median(ae_out)),
        "filtered_rmse_m":             float(np.sqrt((ae_out ** 2).mean())),
        "filtered_mean_error_sigmas":  float(sigmas_out.mean()),
        "filtered_arel":               float((ae_out / np.maximum(gt_out, 1e-9)).mean()),
    }
if __name__ == '__main__':
    gt_m = np.array([2.0, 4.0])
    pred = np.array([2.01, 1.001])
    mask = np.array([True, True])
    print(within_sensor_tolerance(gt_m, pred, mask))