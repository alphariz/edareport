"""Univariate plots — histogram, boxplot, bar chart via plotly.graph_objects."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from edareport._types import ColumnProfile


def build_univariate_plots(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    max_bar_categories: int = 20,
) -> dict[str, str]:
    """Return dict {col_name: plotly_json_string}.

    JSON string langsung di-embed ke HTML via Plotly.react() — tidak ada base64,
    tidak ada kaleido. File HTML tetap kecil.
    """
    result: dict[str, str] = {}
    for cp in profiles:
        fig = None
        if cp.dtype == "numeric":
            fig = _histogram_boxplot(df[cp.name], cp)
        elif cp.dtype == "categorical":
            fig = _bar_chart(cp, max_bar_categories)

        if fig is not None:
            result[cp.name] = fig.to_json()

    return result


# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------


def _histogram_boxplot(series: pd.Series, cp: ColumnProfile) -> go.Figure:  # type: ignore[type-arg]
    clean = series.dropna()
    # Konversi ke list supaya Plotly serialize sebagai JSON array biasa
    # bukan binary format (dtype+bdata) yang tidak kompatibel dengan CDN JS
    x_values = clean.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=x_values,
            name="distribution",
            nbinsx=min(50, max(10, int(len(clean) ** 0.5))),
            marker_color="#378ADD",
            opacity=0.8,
            showlegend=False,
        )
    )
    stats_text = (
        f"mean={cp.mean:.3g} | median={cp.median:.3g}<br>"
        f"std={cp.std:.3g} | outliers={cp.n_outliers}"
    )
    fig.update_layout(
        **_base_layout(cp.name),
        xaxis_title=cp.name,
        yaxis_title="count",
        annotations=[
            {
                "x": 0.98,
                "y": 0.95,
                "xref": "paper",
                "yref": "paper",
                "text": stats_text,
                "showarrow": False,
                "font_size": 11,
                "align": "right",
                "bgcolor": "rgba(255,255,255,0.7)",
            }
        ],
    )
    return fig


def _bar_chart(cp: ColumnProfile, max_categories: int) -> go.Figure | None:
    if not cp.top_values:
        return None
    items = list(cp.top_values.items())[:max_categories]
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#1D9E75",
            opacity=0.85,
        )
    )
    fig.update_layout(
        **_base_layout(cp.name),
        xaxis_title=cp.name,
        yaxis_title="count",
        xaxis_tickangle=-35,
    )
    return fig


def _base_layout(title: str) -> dict:
    return {
        "title": {"text": title, "font_size": 13},
        "height": 280,
        "margin": {"l": 40, "r": 20, "t": 40, "b": 60},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_family": "system-ui, sans-serif",
        "font_size": 11,
    }
