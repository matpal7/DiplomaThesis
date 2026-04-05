import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class RunStats:
    run_name:             str
    date:                 str
    model_ckpt:           str
    network_type:         str                # "stereo" | "mono"
    input_resolution:     tuple[int, int]
    inference_resolution: tuple[int, int]
    scale:                int
    n_images:             int        = 0
    total_time_s:         float      = 0.0
    mean_time_s:          float      = 0.0
    min_time_s:           float      = 0.0
    max_time_s:           float      = 0.0
    per_image_stats:      list[dict] = field(default_factory=list)
    timestamp:            str        = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def save_run_stats(stats: RunStats, out_dir: Path) -> None:
    """
    Finalises aggregate timing fields from per_image_stats,
    then writes the full RunStats to <out_dir>/run_stats.json.
    """
    if not stats.per_image_stats:
        logging.warning("No per-image stats recorded — skipping save.")
        return

    times = [e["time_s"] for e in stats.per_image_stats]
    stats.total_time_s = round(sum(times), 4)
    stats.mean_time_s  = round(sum(times) / len(times), 4)
    stats.min_time_s   = round(min(times), 4)
    stats.max_time_s   = round(max(times), 4)

    stats_path = out_dir / "run_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)

    logging.info("Run stats saved → %s", stats_path)
    logging.info(
        "Timing  total=%.2fs  mean=%.3fs  min=%.3fs  max=%.3fs",
        stats.total_time_s, stats.mean_time_s,
        stats.min_time_s,   stats.max_time_s,
    )