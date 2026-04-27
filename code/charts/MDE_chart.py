import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy import stats as scipy_stats

from code.prepare_paths import get_depth_estimation_network_names

parent_dir = Path(__file__).resolve().parents[3]
date       = "24042026"

metrics_dir = parent_dir / "out" / f"out_{date}" / "depth_comparison" / "zed" / "metrics_ci"

CI_95_DELTA = scipy_stats.cauchy.ppf(0.995, loc=0, scale=1.0)  # ≈ 19.08

nn_names = get_depth_estimation_network_names()

COLORS = px.colors.qualitative.Plotly


def nn_label(nn: str) -> str:
    arr = nn.split("_")
    label = arr[0]
    if label == "DepthAnything3":
        label += " " + arr[1]
    return label


# ─────────────────────────────────────────────────────────────────────────────
# Načítanie dát
# ─────────────────────────────────────────────────────────────────────────────
per_image_data = {}   # nn_label → DataFrame
summary_data   = {}   # nn_label → dict (jedna riadka)

for nn in nn_names:
    label      = nn_label(nn)
    per_csv    = metrics_dir / f"{nn}_per_image.csv"
    summ_csv   = metrics_dir / f"{nn}_summary.csv"

    if not per_csv.exists():
        print(f"Skipping {label}: missing {per_csv}")
        continue

    per_image_data[label] = pd.read_csv(per_csv)

    if summ_csv.exists():
        summary_data[label] = pd.read_csv(summ_csv).iloc[0].to_dict()

if not per_image_data:
    print("Žiadne dáta — skontroluj cestu metrics_dir.")
    exit()

print(f"Načítané siete: {list(per_image_data.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Graf 1 — cauchy_delta_mean per-image (časový priebeh)
# ─────────────────────────────────────────────────────────────────────────────
fig1 = go.Figure()

max_len = 0

for i, (label, df) in enumerate(per_image_data.items()):
    x       = list(range(1, len(df) + 1))
    max_len = max(max_len, len(df))
    color   = COLORS[i % len(COLORS)]

    fig1.add_trace(go.Scatter(
        x          = x,
        y          = df["cauchy_delta_mean"].tolist(),
        mode       = "lines",
        name       = label,
        line       = dict(color=color),
        hovertemplate=(
            f"<b>Sieť:</b> {label}" +
            "<br><b>Snímka:</b> %{x}" +
            "<br><b>δ mean:</b> %{y:.2f}×<extra></extra>"
        ),
    ))

fig1.add_hline(
    y            = CI_95_DELTA,
    line_dash    = "dash",
    line_color   = "red",
    line_width   = 1.5,
    annotation_text      = f"95% CI hranica ZED  (δ = {CI_95_DELTA:.2f}×)",
    annotation_position  = "top right",
    annotation_font_color= "red",
)

fig1.update_layout(
    title       = "Vzdialenosť chýb sietí od modelu ZED — cauchy_delta_mean per snímku",
    xaxis_title = "Poradové číslo snímky",
    yaxis_title = "δ = |ε| / γ  (×)",
    template    = "plotly_white",
    showlegend  = True,
    width       = 1200,
    height      = 550,
)
fig1.update_xaxes(tickmode="linear", dtick=10, range=[1, max_len])
fig1.update_yaxes(tickformat=".1f")

fig1.show()


# ─────────────────────────────────────────────────────────────────────────────
# Graf 2 — Box plot cauchy_delta_mean cez snímky
# ─────────────────────────────────────────────────────────────────────────────
fig2 = go.Figure()

for i, (label, df) in enumerate(per_image_data.items()):
    color = COLORS[i % len(COLORS)]
    fig2.add_trace(go.Box(
        y              = df["cauchy_delta_mean"].tolist(),
        name           = label,
        marker_color   = color,
        boxmean        = "sd",
        hovertemplate  = (
            f"<b>{label}</b><br>"
            "Medián: %{median:.2f}×<br>"
            "Q1: %{q1:.2f}×<br>"
            "Q3: %{q3:.2f}×<extra></extra>"
        ),
    ))

fig2.add_hline(
    y            = CI_95_DELTA,
    line_dash    = "dash",
    line_color   = "red",
    line_width   = 1.5,
    annotation_text      = f"95% CI hranica ZED  (δ = {CI_95_DELTA:.2f}×)",
    annotation_position  = "top right",
    annotation_font_color= "red",
)

fig2.update_layout(
    title       = "Distribúcia cauchy_delta_mean — porovnanie sietí",
    xaxis_title = "Sieť",
    yaxis_title = "δ = |ε| / γ  (×)",
    template    = "plotly_white",
    showlegend  = False,
    width       = 1000,
    height      = 550,
)
fig2.update_yaxes(tickformat=".1f")

fig2.show()


# ─────────────────────────────────────────────────────────────────────────────
# Graf 3 — % pixelov v CI per-image (časový priebeh)
# ─────────────────────────────────────────────────────────────────────────────
fig3 = go.Figure()

for i, (label, df) in enumerate(per_image_data.items()):
    x     = list(range(1, len(df) + 1))
    color = COLORS[i % len(COLORS)]

    fig3.add_trace(go.Scatter(
        x          = x,
        y          = df["ci_pct_within"].tolist(),
        mode       = "lines",
        name       = label,
        line       = dict(color=color),
        hovertemplate=(
            f"<b>Sieť:</b> {label}" +
            "<br><b>Snímka:</b> %{x}" +
            "<br><b>% v CI:</b> %{y:.2f}%<extra></extra>"
        ),
    ))

fig3.update_layout(
    title       = "Podiel pixelov v 95% CI ZED — per snímku",
    xaxis_title = "Poradové číslo snímky",
    yaxis_title = "Pixely v CI  (%)",
    template    = "plotly_white",
    showlegend  = True,
    width       = 1200,
    height      = 500,
)
fig3.update_xaxes(tickmode="linear", dtick=10, range=[1, max(len(df) for df in per_image_data.values())])
fig3.update_yaxes(tickformat=".1f", range=[0, 35])

fig3.show()


# ─────────────────────────────────────────────────────────────────────────────
# Graf 4 — Súhrnný bar chart (delta_mean, delta_median, delta_p95 side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
if summary_data:
    labels_s  = list(summary_data.keys())
    delta_m   = [summary_data[l]["cauchy_delta_mean"]   for l in labels_s]
    delta_med = [summary_data[l]["cauchy_delta_median"] for l in labels_s]
    delta_p95 = [summary_data[l]["cauchy_delta_p95"]    for l in labels_s]

    fig4 = go.Figure()

    fig4.add_trace(go.Bar(
        name = "δ mean",
        x    = labels_s,
        y    = delta_m,
        hovertemplate = "<b>%{x}</b><br>δ mean: %{y:.2f}×<extra></extra>",
    ))
    fig4.add_trace(go.Bar(
        name = "δ medián",
        x    = labels_s,
        y    = delta_med,
        hovertemplate = "<b>%{x}</b><br>δ medián: %{y:.2f}×<extra></extra>",
    ))
    fig4.add_trace(go.Bar(
        name = "δ p95",
        x    = labels_s,
        y    = delta_p95,
        hovertemplate = "<b>%{x}</b><br>δ p95: %{y:.2f}×<extra></extra>",
    ))

    fig4.add_hline(
        y            = CI_95_DELTA,
        line_dash    = "dash",
        line_color   = "red",
        line_width   = 1.5,
        annotation_text      = f"95% CI hranica ZED  (δ = {CI_95_DELTA:.2f}×)",
        annotation_position  = "top right",
        annotation_font_color= "red",
    )

    fig4.update_layout(
        title       = "Súhrnné porovnanie sietí — vzdialenosť od modelu ZED",
        xaxis_title = "Sieť",
        yaxis_title = "δ = |ε| / γ  (×)",
        barmode     = "group",
        template    = "plotly_white",
        showlegend  = True,
        width       = 1100,
        height      = 550,
    )
    fig4.update_yaxes(tickformat=".1f")

    fig4.show()