from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from code.prepare_paths import (
    get_depth_estimation_network,
    prepare_depth_comparison_paths,
)

parent_dir = Path(__file__).resolve().parents[3]
date = "24042026"
rgbd_suffix = "zed"

(_, _, _, _, _, depth_comparison_dir) = prepare_depth_comparison_paths(
    parent_dir, date, rgbd_suffix
)
metrics_out_dir = depth_comparison_dir / "metrics_cauchy"



def display_name(nn: str) -> str:
    return nn.split("_")[0] if "_" in nn else nn


metric_col = "all_AbsRel"
fig = go.Figure()
max_len = 0

nn_names = get_depth_estimation_network()

for nn_key, nn_value in nn_names.items():
    csv_path = metrics_out_dir / f"{nn_key}_per_image.csv"
    if not csv_path.exists():
        print(f"Skipping {nn_key}: missing {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if metric_col not in df.columns:
        print(f"Skipping {nn_key}: column '{metric_col}' not found")
        continue

    values = df[metric_col].values
    image_ids = df["image_id"].astype(str).tolist()
    x = list(range(1, len(values) + 1))
    max_len = max(max_len, len(values))

    mean_v = float(df[metric_col].mean())
    median_v = float(df[metric_col].median())

    fig.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines",
            name=display_name(nn_value),
            customdata=image_ids,
            hovertemplate=(
                f"<b>Sieť:</b> {display_name(nn_value)}"
                "<br><b>Obrázok:</b> %{customdata}"
                "<br><b>Index:</b> %{x}"
                "<br><b>Absolútne relatívna chyba (arel):</b> %{y:.4f}<extra></extra>"
            ),
        )
    )

    print(
        f"{nn_value}: mean={mean_v:.4f}  median={median_v:.4f}  "
        f"min={df[metric_col].min():.4f}  max={df[metric_col].max():.4f}"
    )

fig.update_layout(
    title="Absolútne relatívna chyba hĺbky pre jednotlivé vstupné snímky",
    xaxis_title="Poradové číslo vstupnej snímky",
    yaxis_title="Absolútne relatívna chyba (arel)",
    template="plotly_white",
    showlegend=True,
    width=1200,
    height=600,
)

fig.update_xaxes(tickmode="linear", dtick=20, range=[1, max_len])
fig.update_yaxes(tickformat=".2f")

out_path = metrics_out_dir / "chart_eval_arel_per_image.html"
# fig.write_html(str(out_path))
# print(f"Saved → {out_path}")
fig.show()
