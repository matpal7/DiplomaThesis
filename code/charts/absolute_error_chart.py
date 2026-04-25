from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from code.prepare_paths import get_depth_estimation_network_names, prepare_depth_comparison_paths

parent_dir = Path(__file__).resolve().parents[3]
date       = "13042026"
rgbd_suffix = "zed"

(_, _, _, _, _, depth_comparison_dir) = prepare_depth_comparison_paths(
    parent_dir, date, rgbd_suffix
)
metrics_out_dir = depth_comparison_dir / "metrics"

nn_names = get_depth_estimation_network_names()

# ── helper: strip model variant suffix for display ───────────────────────────
def display_name(nn: str) -> str:
    return nn.split("_")[0] if "_" in nn else nn

# ── build one figure per metric, or a single figure with dropdown ─────────────
METRICS = {
    "eval_median_abs_error": {
        "label": "Mediánová absolútna chyba (m)",
        "title": "Mediánová absolútna chyba hĺbky pre jednotlivé vstupné obrázky",
        "scale": 1.0,
    },
    "eval_arel": {
        "label": "Relatívna absolútna chyba (arel)",
        "title": "Relatívna absolútna chyba hĺbky pre jednotlivé vstupné obrázky",
        "scale": 1.0,
    },
}

for metric_col, cfg in METRICS.items():
    fig = go.Figure()
    max_len = 0

    for nn_name in nn_names:
        csv_path = metrics_out_dir / f"{nn_name}_per_image_metrics.csv"
        if not csv_path.exists():
            print(f"Skipping {nn_name}: missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if metric_col not in df.columns:
            print(f"Skipping {nn_name}: column '{metric_col}' not found")
            continue

        values = df[metric_col].values * cfg["scale"]
        image_ids = df["image_id"].astype(str).tolist()
        x = list(range(1, len(values) + 1))
        max_len = max(max_len, len(values))

        mean_v   = float(df[metric_col].mean())
        median_v = float(df[metric_col].median())

        fig.add_trace(go.Scatter(
            x=x,
            y=values,
            mode="lines",
            name=display_name(nn_name),
            customdata=image_ids,
            hovertemplate=(
                f"<b>Sieť:</b> {display_name(nn_name)}"
                "<br><b>Obrázok:</b> %{customdata}"
                "<br><b>Index:</b> %{x}"
                f"<br><b>{cfg['label']}:</b> %{{y:.4f}}<extra></extra>"
            ),
        ))

        print(
            f"{nn_name}: mean={mean_v:.4f}  median={median_v:.4f}  "
            f"min={df[metric_col].min():.4f}  max={df[metric_col].max():.4f}"
        )

    fig.update_layout(
        title=cfg["title"],
        xaxis_title="Poradové číslo vstupného obrázka",
        yaxis_title=cfg["label"],
        template="plotly_white",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.update_xaxes(tickmode="linear", dtick=5, range=[1, max_len])
    fig.update_yaxes(tickformat=".4f")

    out_path = metrics_out_dir / f"chart_{metric_col}_per_image.html"
    fig.write_html(str(out_path))
    print(f"Saved → {out_path}")
    fig.show()