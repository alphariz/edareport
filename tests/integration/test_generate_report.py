"""Integration tests — generate_report() end-to-end."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from edareport import generate_report


@pytest.fixture(scope="module")
def df_mixed() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "salary": rng.normal(50_000, 15_000, n),
            "dept": np.random.choice(["Eng", "Sales", "HR", "Finance"], n),
            "score": rng.uniform(0, 100, n),
            "active": np.random.choice([True, False], n),
        }
    )


class TestReportObject:
    def test_returns_report(self, df_mixed: pd.DataFrame) -> None:
        from edareport._api import Report

        report = generate_report(df_mixed, title="Test")
        assert isinstance(report, Report)

    def test_data_fields(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed)
        assert report.data.n_rows == 500
        assert report.data.n_cols == 5
        assert len(report.data.columns) == 5

    def test_repr(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed, title="My Report")
        assert "My Report" in repr(report)
        assert "rows=500" in repr(report)


@pytest.mark.slow
class TestHtmlOutput:
    def test_save_creates_file(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed, title="Integration Test")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            assert os.path.exists(path)

    def test_html_size_under_5mb(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            size_mb = os.path.getsize(path) / 1024**2
            assert size_mb < 5.0, f"HTML too large: {size_mb:.2f} MB"

    def test_html_contains_plotly(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "plotly" in content.lower()

    def test_html_contains_title(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed, title="My Custom Title")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "My Custom Title" in content

    def test_html_contains_column_names(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            for col in df_mixed.columns:
                assert col in content, f"Column '{col}' not found in HTML"

    def test_dark_theme(self, df_mixed: pd.DataFrame) -> None:
        report = generate_report(df_mixed, theme="dark")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            report.save(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert 'data-theme="dark"' in content

    def test_render_cache(self, df_mixed: pd.DataFrame) -> None:
        """Render kedua harus return objek yang sama — tidak re-render."""
        report = generate_report(df_mixed)
        first = report._render()
        second = report._render()
        assert first is second


@pytest.mark.slow
class TestEdgeCases:
    def test_all_numeric(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({f"v{i}": rng.normal(i, 1, 200) for i in range(5)})
        report = generate_report(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.html")
            report.save(path)
            assert os.path.exists(path)

    def test_all_categorical(self) -> None:
        df = pd.DataFrame(
            {
                "a": ["x", "y", "z"] * 50,
                "b": ["p", "q"] * 75,
            }
        )
        report = generate_report(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.html")
            report.save(path)
            assert os.path.exists(path)

    def test_with_nulls(self, df_with_nulls: pd.DataFrame) -> None:
        report = generate_report(df_with_nulls)
        assert len(report.data.warnings) > 0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.html")
            report.save(path)
            assert os.path.exists(path)

    def test_sample_size_applied(self) -> None:
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 10_000),
                "y": rng.normal(1, 2, 10_000),
            }
        )
        report = generate_report(df, sample_size=500)
        # Profile pakai full df
        assert report.data.n_rows == 10_000
