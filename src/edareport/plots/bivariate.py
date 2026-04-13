"""Bivariate plots — correlation heatmap + scatter (sampled)."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from edareport._types import ReportData


def build_bivariate_plots(
    df: pd.DataFrame,
    report_data: ReportData,
) -> dict[str, str]:
    """Return dict {plot_key: plotly_json_string}."""
    result: dict[str, str] = {}

    if report_data.correlation_matrix:
        result["correlation_heatmap"] = _correlation_heatmap(
            report_data.correlation_matrix
        ).to_json()

    # Scatter hanya untuk top-3 pasang korelasi tertinggi
    for i, (col_a, col_b, r) in enumerate(report_data.top_correlations[:3]):
        if col_a in df.columns and col_b in df.columns:
            key = f"scatter_{i}_{col_a}_vs_{col_b}"
            result[key] = _scatter(df, col_a, col_b, r).to_json()

    return result


def _correlation_heatmap(
    corr_matrix: dict[str, dict[str, float]]
) -> go.Figure:
    cols = list(corr_matrix.keys())
    z = [[corr_matrix[row][col] for col in cols] for row in cols]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=cols,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont_size=10,
            showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        )
    )
    n = len(cols)
    size = max(320, min(600, n * 45))
    fig.update_layout(
        title=dict(text="Correlation matrix", font_size=13),
        height=size,
        width=size,
        margin=dict(l=80, r=20, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
        font_family="system-ui, sans-serif",
        font_size=11,
    )
    return fig


def _scatter(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    r: float,
    max_points: int = 2000,
) -> go.Figure:
    sample = df[[col_a, col_b]].dropna()
    if len(sample) > max_points:
        sample = sample.sample(n=max_points, random_state=42)

    fig = go.Figure(
        go.Scatter(
            x=sample[col_a],
            y=sample[col_b],
            mode="markers",
            marker=dict(size=4, color="#534AB7", opacity=0.5),
        )
    )
    fig.update_layout(
        title=dict(text=f"{col_a} vs {col_b}  (r={r:+.2f})", font_size=13),
        xaxis_title=col_a,
        yaxis_title=col_b,
        height=280,
        margin=dict(l=50, r=20, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="system-ui, sans-serif",
        font_size=11,
    )
    return fig