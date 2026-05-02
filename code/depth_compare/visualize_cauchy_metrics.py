from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATE = "24042026"
RGBD_SUFFIX = "zed"
METRICS_SUBDIR = "metrics_cauchy"


def _lineplot_per_image(df: pd.DataFrame, metric: str, title: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(12, 5))

    for nn_name in sorted(df["nn_name"].unique()):
        nn_df = df[df["nn_name"] == nn_name].sort_values("image_id")
        plt.plot(nn_df["image_id"], nn_df[metric], label=nn_name, linewidth=1.6)

    plt.xlabel("image_id")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _build_all_per_image_df(base_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(base_dir.glob("*_per_image.csv")):
        df = pd.read_csv(csv_path)
        if "nn_name" not in df.columns:
            nn_name = csv_path.name.replace("_per_image.csv", "")
            df["nn_name"] = nn_name
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No per-image CSV files found in {base_dir}")

    merged = pd.concat(rows, ignore_index=True)
    if "image_id" not in merged.columns:
        raise ValueError("Per-image CSV files must contain image_id column.")
    return merged


def main() -> None:
    parent_dir = Path(__file__).resolve().parents[3]
    base_dir = parent_dir / "out" / f"out_{DATE}" / "depth_comparison" / f"{RGBD_SUFFIX}" / METRICS_SUBDIR

    df_per_image = _build_all_per_image_df(base_dir)

    figs_dir = base_dir / "figures_line"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _lineplot_per_image(
        df_per_image,
        metric="median_abs_z",
        title="Vývoj median |z| naprieč datasetom (nižšie je lepšie)",
        out_path=figs_dir / "line_median_abs_z.png",
        ylabel="median |z|",
    )
    _lineplot_per_image(
        df_per_image,
        metric="within_ci_pct",
        title="Vývoj podielu pixelov v 95% Cauchy intervale naprieč datasetom",
        out_path=figs_dir / "line_within_ci_pct.png",
        ylabel="within_ci_pct [%]",
    )
    _lineplot_per_image(
        df_per_image,
        metric="all_AbsRel",
        title="Vývoj AbsRel naprieč datasetom (nižšie je lepšie)",
        out_path=figs_dir / "line_absrel.png",
        ylabel="AbsRel",
    )
    _lineplot_per_image(
        df_per_image,
        metric="mean_log_lik",
        title="Vývoj mean log-likelihood podľa Cauchy modelu naprieč datasetom",
        out_path=figs_dir / "line_mean_log_lik.png",
        ylabel="mean log-likelihood",
    )

    print(f"[INFO] Saved line charts to {figs_dir}")


if __name__ == "__main__":
    main()
