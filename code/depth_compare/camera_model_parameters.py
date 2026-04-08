import json
from pathlib import Path

def load_sensor_model(camera_stats_dir: Path, camera: str) -> dict | None:
    """Load fitted power model for a given camera from global_summary.json."""
    path = camera_stats_dir / camera / "global_summary.json"
    if not path.exists():
        print(f"[WARN] No sensor model found at {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    model = data.get("model", {})
    if not model or "rel_alpha" not in model:
        print(f"[WARN] No valid model in {path}")
        return None
    return model

if __name__ == '__main__':
    parent_dir = Path(__file__).resolve().parents[3]
    date = "09042026"
    base_dir = parent_dir / "datasets" / f"dataset_{date}" / "cameras_statistic_model"
    out_dir = parent_dir / "out" / f"out_{date}" / "cameras_statistic_model"

    sensor_model = load_sensor_model(out_dir, "zed")
    alpha = sensor_model["rel_alpha"]
    beta = sensor_model["rel_beta"]
    print(f"Loaded sensor model: σ_rel(Z) = {alpha:.8f} · Z^{beta:.3f}")