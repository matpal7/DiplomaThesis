from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_per_image_results(
    out_root: Path,
    image_id: str,
    pred_warped_m: np.ndarray,
    gt_depth_m: np.ndarray,
    depth_est_m: np.ndarray,
    valid_mask: np.ndarray,
    metrics: dict[str, Any],
    comparison_vis: np.ndarray | None = None,
) -> None:
    """
    Save all results for one image into its own folder.
    """
    image_dir = out_root / "per_image" / str(image_id)
    image_dir.mkdir(parents=True, exist_ok=True)

    np.save(image_dir / "pred_warped_m.npy", pred_warped_m.astype(np.float32))
    np.save(image_dir / "gt_depth_m.npy", gt_depth_m.astype(np.float32))
    np.save(image_dir / "depth_est_m.npy", depth_est_m.astype(np.float32))
    np.save(image_dir / "valid_mask.npy", valid_mask.astype(np.uint8))

    with (image_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if comparison_vis is not None:
        cv2.imwrite(str(image_dir / "comparison.png"), comparison_vis)


def summarize_metrics(all_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate metrics over all images.
    """
    if not all_metrics:
        return {"num_images": 0}

    numeric_keys = [
        "num_valid_pixels",
        "mae_m",
        "rmse_m",
        "median_abs_error_m",
        "abs_rel",
        "delta_1_25",
        "delta_1_25_sq",
        "delta_1_25_cu",
    ]

    summary: dict[str, Any] = {
        "num_images": len(all_metrics),
        "mean": {},
        "median": {},
        "std": {},
    }

    for key in numeric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if not values:
            continue

        arr = np.asarray(values, dtype=np.float64)
        summary["mean"][key] = float(np.mean(arr))
        summary["median"][key] = float(np.median(arr))
        summary["std"][key] = float(np.std(arr))

    return summary


def save_summary_results(
    out_root: Path,
    all_metrics: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save summary.json and metrics_all.csv for the whole experiment.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    summary = summarize_metrics(all_metrics)
    if metadata is not None:
        summary["metadata"] = metadata

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if all_metrics:
        fieldnames = sorted({k for row in all_metrics for k in row.keys()})
        with (out_root / "metrics_all.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)