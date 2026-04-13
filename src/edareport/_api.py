"""generate_report() — single entry point public API."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from edareport._types import ReportData
from edareport.profiler.core import DataFrameProfiler


def generate_report(
    df: pd.DataFrame,
    *,
    title: str = "EDA Report",
    output: Literal["html", "widget", "data"] = "html",
    theme: Literal["light", "dark"] = "light",
    sample_size: int | None = 50_000,
    max_categories: int = 50,
) -> "Report":
    """Generate EDA report dari DataFrame.

    Args:
        df: Input DataFrame (pandas).
        title: Judul report.
        output: Format output — "html" (file/string), "widget" (Jupyter),
                atau "data" (ReportData mentah, tanpa render).
        theme: Tema visual "light" atau "dark".
        sample_size: Batas baris untuk visualisasi berat. None = pakai semua.
        max_categories: Batas n_unique sebelum kolom dianggap high-cardinality.

    Returns:
        Report object dengan method .show() dan .save(path).

    Example:
        >>> import pandas as pd
        >>> from edareport import generate_report
        >>> df = pd.read_csv("data.csv")
        >>> report = generate_report(df, title="Sales EDA", output="html")
        >>> report.save("report.html")
    """
    profiler = DataFrameProfiler(max_categories=max_categories)
    report_data = profiler.profile(df, title=title)

    return Report(
        data=report_data,
        output=output,
        theme=theme,
        sample_size=sample_size,
        source_df=df,
    )


class Report:
    """Wrapper hasil generate_report — lazy render."""

    def __init__(
        self,
        data: ReportData,
        output: str,
        theme: str,
        sample_size: int | None,
        source_df: pd.DataFrame,
    ) -> None:
        self.data = data
        self._output = output
        self._theme = theme
        self._sample_size = sample_size
        self._df = source_df
        self._rendered: str | None = None

    def _render(self) -> str:
        """Lazy render ke HTML string."""
        if self._rendered is not None:
            return self._rendered

        from edareport.plots.univariate import build_univariate_plots
        from edareport.plots.bivariate import build_bivariate_plots
        from edareport.renderers.html import HtmlRenderer

        df_sample = (
            self._df.sample(n=min(self._sample_size, len(self._df)), random_state=42)
            if self._sample_size and len(self._df) > self._sample_size
            else self._df
        )

        uni_plots = build_univariate_plots(df_sample, self.data.columns)
        bi_plots = build_bivariate_plots(df_sample, self.data)

        renderer = HtmlRenderer(theme=self._theme)
        self._rendered = renderer.render(self.data, uni_plots, bi_plots)
        return self._rendered

    def save(self, path: str = "report.html") -> None:
        """Simpan report ke file HTML."""
        html = self._render()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved → {path}")

    def show(self) -> None:
        """Tampilkan report di Jupyter/Colab (widget) atau browser (HTML)."""
        if self._output == "widget":
            self._show_widget()
        else:
            self._show_browser()

    def _show_widget(self) -> None:
        from edareport.renderers.widget import WidgetRenderer
        renderer = WidgetRenderer()
        renderer.display(self._render())

    def _show_browser(self) -> None:
        import tempfile, webbrowser, os
        html = self._render()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            tmp_path = f.name
        webbrowser.open(f"file://{os.path.abspath(tmp_path)}")

    def __repr__(self) -> str:
        return (
            f"Report(title={self.data.title!r}, "
            f"rows={self.data.n_rows}, cols={self.data.n_cols})"
        )