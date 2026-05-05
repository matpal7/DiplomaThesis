from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


OBJECT_TRANSLATIONS_SK = {
    "apple": "jablko",
    "chips_box": "krabica čipsov",
    "lemon": "citrón",
    "orange": "pomaranč",
    "rubiks_cube": "Rubikova kocka",
    "scissors": "nožnice",
    "wood_block": "drevený kváder",
}

OBJECT_ORDER_SK = [
    "jablko",
    "citrón",
    "pomaranč",
    "Rubikova kocka",
    "krabica čipsov",
    "drevený kváder",
    "nožnice",
]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chýba vstupný súbor: {path}")
    return pd.read_csv(path)


def _save_figure(fig: go.Figure, html_path: Path, png_path: Path) -> None:
    fig.write_html(html_path)
    print(f"[INFO] Saved → {html_path}")


def _add_slovak_object_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["object_sk"] = df["object"].map(OBJECT_TRANSLATIONS_SK).fillna(df["object"])

    df["object_sk"] = pd.Categorical(
        df["object_sk"],
        categories=OBJECT_ORDER_SK,
        ordered=True,
    )

    return df


def print_summary_per_scene_object(summary_df: pd.DataFrame) -> None:
    required_columns = {
        "scene",
        "object",
        "n_frames",
        "err3d_mean_m",
        "err3d_median_m",
        "err3d_std_m",
    }

    missing = required_columns - set(summary_df.columns)
    if missing:
        raise ValueError(f"V CSV súbore chýbajú stĺpce: {sorted(missing)}")

    df = _add_slovak_object_names(summary_df)
    df = df.sort_values(["scene", "object_sk"])

    print("\nSúhrn 3D chyby podľa scény a objektu")
    print("─" * 95)
    print(
        f"{'Scéna':<12} "
        f"{'Objekt':<18} "
        f"{'Framy':>8} "
        f"{'Priemer [cm]':>14} "
        f"{'Medián [cm]':>14} "
        f"{'Std [cm]':>12}"
    )
    print("─" * 95)

    for _, row in df.iterrows():
        print(
            f"{row['scene']:<12} "
            f"{str(row['object_sk']):<18} "
            f"{int(row['n_frames']):>8} "
            f"{row['err3d_mean_m'] * 100.0:>14.2f} "
            f"{row['err3d_median_m'] * 100.0:>14.2f} "
            f"{row['err3d_std_m'] * 100.0:>12.2f}"
        )

    print("─" * 95)


def build_heatmap(
    summary_df: pd.DataFrame,
    out_dir: Path,
    metric: str = "err3d_mean_m",
) -> None:
    summary_df = _add_slovak_object_names(summary_df)

    heat = summary_df.pivot(
        index="object_sk",
        columns="scene",
        values=metric,
    )

    heat = heat.dropna(how="all")

    fig = px.imshow(
        heat * 100.0,
        labels={
            "color": "Priemerná 3D chyba [cm]",
            "x": "Scéna",
            "y": "Objekt",
        },
        text_auto=".1f",
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        title="Heatmapa priemernej 3D chyby (objekt × scéna)",
        template="plotly_white",
        width=1100,
        height=700,
    )

    html_path = out_dir / "heatmap_err3d_object_scene.html"
    png_path = out_dir / "heatmap_err3d_object_scene.png"
    _save_figure(fig, html_path, png_path)


def build_reprojection_frame_chart_all_scenes(
    all_frames_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    df = all_frames_df.copy()

    if df.empty:
        raise ValueError("Vstupný dataframe neobsahuje žiadne dáta.")

    df = _add_slovak_object_names(df)
    df = df.sort_values(["scene", "object_sk", "frame_id"])

    fig = px.line(
        df,
        x="frame_id",
        y="err_2d_px",
        color="object_sk",
        facet_col="scene",
        facet_col_wrap=2,
        markers=True,
        category_orders={
            "object_sk": OBJECT_ORDER_SK,
        },
        labels={
            "frame_id": "Frame ID",
            "err_2d_px": "Reprojekčná chyba [px]",
            "object_sk": "Objekt",
            "scene": "Scéna",
        },
        title="Reprojekčná chyba cez frame id pre všetky scény",
    )

    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=900,
        legend_title="Objekt",
    )

    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)

    html_path = out_dir / "reprojection_error_all_scenes.html"
    png_path = out_dir / "reprojection_error_all_scenes.png"
    _save_figure(fig, html_path, png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vizualizácie pre compare_6D_pose výstupy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/out_24042026/pose_estimation/results_comparison"),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_001",
        help="Scéna pre frame chart reprojekčnej chyby.",
    )

    args = parser.parse_args()

    parent_dir = Path(__file__).resolve().parents[3]
    camera = "zed"

    root = parent_dir / args.root / camera
    out_dir = root / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_csv(root / "summary_per_scene_object.csv")
    all_frames_df = _load_csv(root / "all_frames.csv")

    print_summary_per_scene_object(summary_df)

    build_heatmap(summary_df, out_dir)
    build_reprojection_frame_chart_all_scenes(all_frames_df, out_dir)


if __name__ == "__main__":
    main()