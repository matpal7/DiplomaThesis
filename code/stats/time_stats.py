import json
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
from tabulate import tabulate

from code.prepare_paths import get_depth_estimation_network

parent_dir = Path(__file__).resolve().parents[3]
date = "24042026"

nn_names = get_depth_estimation_network()

rows = []

for nn_key, nn_value in nn_names.items():
    out_dir = parent_dir / "out" / f"out_{date}" / "depth_estimation" / nn_key
    stats_path = out_dir / "run_stats.json"

    nn_label =  nn_value

    if not stats_path.exists():
        print(f"Skipping {nn_label}: missing {stats_path}")
        continue

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    times = np.array([e["time_s"] for e in data["per_image_stats"]])
    mad = np.median(np.abs(times - np.median(times)))

    rows.append({
        "Sieť":        nn_label,
        "N":           len(times),
        "Min (s)":     f"{times.min():.4f}",
        "Max (s)":     f"{times.max():.4f}",
        "Mean (s)":    f"{times.mean():.4f}",
        "Median (s)":  f"{np.median(times):.4f}",
        "Std (s)":     f"{times.std(ddof=1):.4f}",
        "MAD (s)":     f"{mad:.4f}",
    })

df = pd.DataFrame(rows)

# --- Výpis do konzoly ---
print("\n=== Štatistika časov inferencie ===\n")
print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))

# --- Uloženie do CSV ---
csv_path = parent_dir / "out" / f"out_{date}" / "inference_time_stats.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\nUložené: {csv_path}")