from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from code.prepare_paths import get_depth_estimation_network_names

DATE = "24042026"
RGBD_SUFFIX = "zed"
METRICS_SUBDIR = "metrics_ci_extended"


def _nn_label(nn_name: str) -> str:
    parts = nn_name.split("_")
    if len(parts) == 1:
        return nn_name
    if parts[0] == "DepthAnything3" and len(parts) > 1:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def _load_metrics(metrics_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    per_image: dict[str, pd.DataFrame] = {}

    for nn_name in get_depth_estimation_network_names():
        per_image_csv = metrics_dir / f"{nn_name}_per_image.csv"
        if not per_image_csv.exists():
            continue

        df = pd.read_csv(per_image_csv)
        if df.empty:
            continue

        df = df.sort_values("image_id").reset_index(drop=True)
        df["frame_idx"] = range(1, len(df) + 1)
        df["nn_name"] = nn_name
        df["nn_label"] = _nn_label(nn_name)
        per_image[nn_name] = df

    global_summary_csv = metrics_dir / "all_networks_summary.csv"
    if global_summary_csv.exists():
        global_summary = pd.read_csv(global_summary_csv)
        global_summary["nn_label"] = global_summary["nn_name"].map(_nn_label)
    else:
        global_summary = pd.DataFrame()

    return per_image, global_summary


def plot_ci_extended_charts(metrics_dir: Path, out_dir: Path) -> None:
    per_image_data, global_summary = _load_metrics(metrics_dir)
    if not per_image_data:
        print(f"[WARN] No per-image metrics found in: {metrics_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames_df = pd.concat(per_image_data.values(), ignore_index=True)

    fig1 = px.line(
        all_frames_df,
        x="frame_idx",
        y="median_abs_z",
        color="nn_label",
        hover_data=["image_id", "nn_name"],
        title="Median |z| per frame (lower is better)",
        labels={"frame_idx": "Frame index", "median_abs_z": "Median |z|", "nn_label": "Network"},
    )
    fig1.update_layout(template="plotly_white", width=1200, height=520)
    fig1.write_html(out_dir / "01_median_abs_z_per_frame.html")

    fig2 = px.line(
        all_frames_df,
        x="frame_idx",
        y="within_ci_pct",
        color="nn_label",
        hover_data=["image_id", "nn_name"],
        title="Within-camera 95% CI percentage per frame (higher is better)",
        labels={"frame_idx": "Frame index", "within_ci_pct": "Within CI (%)", "nn_label": "Network"},
    )
    fig2.update_layout(template="plotly_white", width=1200, height=520)
    fig2.write_html(out_dir / "02_within_ci_pct_per_frame.html")

    box_df = all_frames_df[["nn_label", "all_AbsRel", "out_AbsRel"]].melt(
        id_vars="nn_label",
        value_vars=["all_AbsRel", "out_AbsRel"],
        var_name="region",
        value_name="AbsRel",
    )
    box_df["region"] = box_df["region"].map({"all_AbsRel": "All pixels", "out_AbsRel": "Outside-CI pixels"})

    fig3 = px.box(
        box_df,
        x="nn_label",
        y="AbsRel",
        color="region",
        points=False,
        title="AbsRel distribution across frames (all pixels vs outside-CI pixels)",
        labels={"nn_label": "Network", "AbsRel": "AbsRel", "region": "Region"},
    )
    fig3.update_layout(template="plotly_white", width=1200, height=560)
    fig3.write_html(out_dir / "03_absrel_box_all_vs_outside_ci.html")

    if not global_summary.empty:
        ranking_cols = ["nn_label", "median_abs_z", "within_ci_pct", "mean_log_lik", "all_AbsRel", "all_RMSE"]
        available_cols = [c for c in ranking_cols if c in global_summary.columns]
        ranking = global_summary[available_cols].copy()

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="median_abs_z", x=ranking["nn_label"], y=ranking["median_abs_z"]))
        fig4.add_trace(go.Bar(name="within_ci_pct", x=ranking["nn_label"], y=ranking["within_ci_pct"]))
        fig4.add_trace(go.Bar(name="mean_log_lik", x=ranking["nn_label"], y=ranking["mean_log_lik"]))
        fig4.update_layout(
            title="Global summary comparison (CI-aware metrics)",
            xaxis_title="Network",
            yaxis_title="Metric value",
            barmode="group",
            template="plotly_white",
            width=1200,
            height=560,
        )
        fig4.write_html(out_dir / "04_global_summary_ci_metrics.html")

        ranking.to_csv(out_dir / "global_ranking_table.csv", index=False)

    print(f"[INFO] Saved charts to: {out_dir}")


def main() -> None:
    parent_dir = Path(__file__).resolve().parents[3]
    metrics_dir = parent_dir / "out" / f"out_{DATE}" / "depth_comparison" / RGBD_SUFFIX / METRICS_SUBDIR
    charts_dir = metrics_dir / "charts"

    plot_ci_extended_charts(metrics_dir=metrics_dir, out_dir=charts_dir)


if __name__ == "__main__":
    main()
