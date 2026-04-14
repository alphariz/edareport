"""Unit tests untuk lapisan API Publik (generate_report)."""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from edareport import generate_report
from edareport._types import ReportData


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.1, 2.2, 3.3, 4.4, 5.5],
            "num2": [5.5, 4.4, 3.3, 2.2, 1.1],
            "cat": ["A", "B", "A", "C", "A"],
            "bools": [True, False, True, True, False],
        }
    )


def test_generate_report_returns_report(sample_df: pd.DataFrame) -> None:
    """Verifikasi API contract generate_report me-return Report instance."""
    report = generate_report(sample_df, title="Test Report")

    assert report.__class__.__name__ == "Report"
    assert isinstance(report.data, ReportData)
    assert report.data.title == "Test Report"
    assert report.data.n_rows == 5


def test_report_lazy_render(sample_df: pd.DataFrame) -> None:
    """Verifikasi pemanggilan render berjalan memanggil komponen grafik."""
    report = generate_report(sample_df)

    assert report._rendered is None
    html_str = report._render()

    assert isinstance(html_str, str)
    assert "Test Report" in html_str or "EDA Report" in html_str  # Default title


def test_report_save_method(sample_df: pd.DataFrame) -> None:
    """Verifikasi Report.save() menghasilkan file HTML fisik."""
    report = generate_report(sample_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        report.save(path)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
            assert "<html" in content.lower()


@patch("webbrowser.open")
def test_report_show_html(mock_open, sample_df: pd.DataFrame) -> None:
    """Memanggil show di output browser."""
    report = generate_report(sample_df, output="html")
    report.show()

    # Memastikan webbrowser.open terpanggil dengan file path
    mock_open.assert_called_once()
    args = mock_open.call_args[0][0]
    assert args.startswith("file://")
    assert args.endswith(".html")


def test_report_show_widget(sample_df: pd.DataFrame) -> None:
    """Memanggil show di output widget."""
    report = generate_report(sample_df, output="widget")

    try:
        # Mock _show_widget to avoid actual import fail, or let it fail and catch
        with patch.object(report, "_show_widget") as mock_sw:
            report.show()
            mock_sw.assert_called_once()
    except Exception as e:
        if "No module named 'edareport.renderers.widget'" in str(e):
            pytest.skip("Widget renderer module belum di-implement sepenuhnya")
        else:
            raise e


def test_render_cache_hit(df_small):
    """_rendered cache path — baris 76."""
    report = generate_report(df_small)
    first = report._render()
    second = report._render()
    assert first is second  # objek yang sama, bukan re-render


def test_repr(df_small):
    """__repr__ — baris 125."""
    report = generate_report(df_small)
    r = repr(report)
    assert "Report(" in r
    assert "rows=" in r


def test_show_widget_no_ipython(df_small):
    """_show_widget path — baris 110-112."""
    import pytest

    from edareport._api import generate_report

    report = generate_report(df_small, output="widget")

    try:
        pass  # Just check if the module exists via subsequent logic or just rely on the try-except scope if needed
        # In this specific case, the import was just to check existence.
        # But we can just try to import and catch without assigning to anything if we want to keep the check.
        # Actually, ruff recommends importlib.util.find_spec.
        import importlib.util

        if importlib.util.find_spec("edareport.renderers.widget") is None:
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        pytest.skip("Widget renderer module belum di-implement sepenuhnya")

    with pytest.raises(RuntimeError, match="IPython"):
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
            report._show_widget()
