"""Unit tests untuk plots univariate + bivariate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from edareport._types import ColumnProfile
from edareport.plots.bivariate import _scatter
from edareport.plots.univariate import build_univariate_plots


def _make_cp(**kwargs) -> ColumnProfile:
    defaults = {
        "name": "x",
        "dtype": "numeric",
        "pandas_dtype": "float64",
        "n_total": 10,
        "n_missing": 0,
        "n_unique": 10,
        "missing_pct": 0.0,
    }
    defaults.update(kwargs)
    return ColumnProfile(**defaults)


class TestUnivariatePlots:
    def test_datetime_skipped(self):
        """continue path — baris 28: datetime kolom tidak masuk result."""
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5)})
        cp = _make_cp(name="ts", dtype="datetime", pandas_dtype="datetime64[ns]")
        result = build_univariate_plots(df, [cp])
        assert "ts" not in result

    def test_text_skipped(self):
        """continue path — baris 28: text kolom tidak masuk result."""
        df = pd.DataFrame({"desc": ["hello world this is a long text"] * 5})
        cp = _make_cp(name="desc", dtype="text", pandas_dtype="object")
        result = build_univariate_plots(df, [cp])
        assert "desc" not in result

    def test_bar_chart_empty_top_values(self):
        """top_values kosong — baris 73: return Figure() kosong."""
        df = pd.DataFrame({"cat": ["a", "b"]})
        cp = _make_cp(name="cat", dtype="categorical", pandas_dtype="object", top_values={})
        result = build_univariate_plots(df, [cp])
        # Figure kosong tetap di-return sebagai JSON string
        assert "cat" not in result  # skip karena to_json() figure kosong


class TestBivariatePlots:
    def test_scatter_sampling(self):
        """Baris 79: df > max_points → di-sample."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 3000),
                "b": rng.normal(0, 1, 3000),
            }
        )
        fig = _scatter(df, "a", "b", r=0.5, max_points=500)
        # Jumlah titik di scatter harus <= max_points
        n_points = len(fig.data[0].x)
        assert n_points <= 500
