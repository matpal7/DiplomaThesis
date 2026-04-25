import json
from pathlib import Path

import plotly.graph_objects as go

from code.prepare_paths import get_depth_estimation_network_names

parent_dir = Path(__file__).resolve().parents[3]
date = "24042026"

nn_names = get_depth_estimation_network_names()

fig = go.Figure()

max_len = 0

for nn in nn_names:
    out_dir = parent_dir / "out" / f"out_{date}" / "depth_estimation" / nn
    stats_path = out_dir / "run_stats.json"

    arr = nn.split("_")
    nn_label = arr[0]
    if nn_label == "DepthAnything3":
        nn_label += " " + arr[1]

    if not stats_path.exists():
        print(f"Skipping {nn_label}: missing {stats_path}")
        continue

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    times = [e["time_s"] for e in stats["per_image_stats"]]
    labels = [
        e.get("image_name", e.get("filename", str(i + 1)))
        for i, e in enumerate(stats["per_image_stats"])
    ]
    x = list(range(1, len(times) + 1))

    max_len = max(max_len, len(times))

    mean_t = stats["mean_time_s"]
    min_t = stats["min_time_s"]
    max_t = stats["max_time_s"]

    fig.add_trace(go.Scatter(
        x=x,
        y=times,
        mode="lines",
        name=nn_label,
        customdata=labels,
        hovertemplate=(
            "<b>Network:</b> " + stats.get("network_type", nn_label) +
            "<br><b>Image:</b> %{customdata}" +
            "<br><b>Index:</b> %{x}" +
            "<br><b>Time:</b> %{y:.4f}s<extra></extra>"
        ),
    ))

    print(
        f"{nn_label}: mean={mean_t:.4f}s  min={min_t:.4f}s  max={max_t:.4f}s"
    )

fig.update_layout(
    title="Graf času inferencie pre jednotlivé obrázky",
    xaxis_title="Poradové číslo vstupného obrázka",
    yaxis_title="Čas inferencie (s)",
    template="plotly_white",
    showlegend=True,
    width=1200,
    height=600,
)

fig.update_xaxes(tickmode="linear", dtick=10, range=[1, max_len])
fig.update_yaxes(tickformat=".3f", dtick=0.25)

fig.show()