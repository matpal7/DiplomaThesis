from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go

from code.prepare_paths import (
    get_depth_estimation_network,
    prepare_depth_comparison_paths,
)

# -----------------------------------------------------------------------------
# Konfigurácia
# -----------------------------------------------------------------------------
parent_dir = Path(__file__).resolve().parents[3]
date = "24042026"
rgbd_suffix = "zed"
metric_col = "all_AbsRel"

(_, _, _, _, _, depth_comparison_dir) = prepare_depth_comparison_paths(
    parent_dir, date, rgbd_suffix
)
metrics_out_dir = depth_comparison_dir / "metrics_cauchy"


def display_name(nn: str) -> str:
    return nn.split("_")[0] if "_" in nn else nn


# -----------------------------------------------------------------------------
# Vytvorenie grafu
# -----------------------------------------------------------------------------
fig = go.Figure()
nn_names = get_depth_estimation_network()

for nn_key, nn_value in nn_names.items():
    # -------------------------------------------------------------------------
    # Načítanie metrík chyby pre jednotlivé snímky
    # -------------------------------------------------------------------------
    csv_path = metrics_out_dir / f"{nn_key}_per_image.csv"
    if not csv_path.exists():
        print(f"Preskakujem {nn_key}: chýba súbor s metrikami {csv_path}")
        continue

    df_metrics = pd.read_csv(csv_path)

    if metric_col not in df_metrics.columns:
        print(f"Preskakujem {nn_key}: stĺpec '{metric_col}' nebol nájdený")
        continue

    # -------------------------------------------------------------------------
    # Načítanie času inferencie pre jednotlivé snímky
    # -------------------------------------------------------------------------
    stats_path = parent_dir / "out" / f"out_{date}" / "depth_estimation" / nn_key / "run_stats.json"
    if not stats_path.exists():
        print(f"Preskakujem {nn_key}: chýba súbor s časmi inferencie {stats_path}")
        continue

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    per_image_stats = stats.get("per_image_stats", [])
    if len(per_image_stats) == 0:
        print(f"Preskakujem {nn_key}: neboli nájdené štatistiky pre jednotlivé snímky")
        continue

    # -------------------------------------------------------------------------
    # Kontrola kompatibility dĺžok
    # -------------------------------------------------------------------------
    if len(df_metrics) != len(per_image_stats):
        print(
            f"Upozornenie pre {nn_key}: počet riadkov metrík ({len(df_metrics)}) "
            f"sa nerovná počtu časových záznamov ({len(per_image_stats)}). "
            f"Použije sa kratšia dĺžka."
        )

    n = min(len(df_metrics), len(per_image_stats))
    df_metrics = df_metrics.iloc[:n].copy()
    per_image_stats = per_image_stats[:n]

    # -------------------------------------------------------------------------
    # Spojenie podľa poradia snímok
    # -------------------------------------------------------------------------
    runtimes = [entry["time_s"] for entry in per_image_stats]
    image_labels = [
        entry.get("image_name", entry.get("filename", str(i + 1)))
        for i, entry in enumerate(per_image_stats)
    ]

    df_metrics["runtime_s"] = runtimes
    df_metrics["image_label"] = image_labels
    df_metrics["plot_index"] = list(range(1, n + 1))

    # -------------------------------------------------------------------------
    # Pridanie bodov do grafu
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_metrics["runtime_s"],
            y=df_metrics[metric_col],
            mode="markers",
            name=display_name(nn_value),
            customdata=df_metrics[["image_id", "image_label", "plot_index"]].values,
            hovertemplate=(
                f"<b>Sieť:</b> {display_name(nn_value)}"
                "<br><b>ID snímky:</b> %{customdata[0]}"
                "<br><b>Názov snímky:</b> %{customdata[1]}"
                "<br><b>Poradie:</b> %{customdata[2]}"
                "<br><b>Čas inferencie:</b> %{x:.4f} s"
                "<br><b>Relatívna absolútna chyba:</b> %{y:.4f}"
                "<extra></extra>"
            ),
            marker=dict(size=8, opacity=0.7),
        )
    )

    print(
        f"{display_name(nn_value)}: "
        f"priemerný čas inferencie = {df_metrics['runtime_s'].mean():.4f} s, "
        f"priemerná relatívna absolútna chyba = {df_metrics[metric_col].mean():.4f}"
    )

# -----------------------------------------------------------------------------
# Nastavenie vzhľadu grafu
# -----------------------------------------------------------------------------
fig.update_layout(
    title="Čas inferencie a relatívna absolútna chyba pre jednotlivé snímky",
    xaxis_title="Čas inferencie jednej snímky [s]",
    yaxis_title="Relatívna absolútna chyba",
    template="plotly_white",
    width=1200,
    height=700,
    showlegend=True,
)

fig.update_xaxes(tickformat=".1f")
fig.update_yaxes(tickformat=".2f")

# Voliteľné uloženie
out_path = metrics_out_dir / "scatter_runtime_vs_absrel_per_image.html"
# fig.write_html(str(out_path))
# print(f"Uložené -> {out_path}")

fig.show()