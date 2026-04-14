"""Unit tests — DataFrameProfiler: dtype detection, missing count, unique count."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edareport.profiler.core import DataFrameProfiler

# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def profiler() -> DataFrameProfiler:
    return DataFrameProfiler()


@pytest.fixture()
def df_mixed() -> pd.DataFrame:
    """DataFrame dengan berbagai tipe kolom untuk test dtype detection."""
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "a", "b"],
            "bool_col": [True, False, True, False, True],
            "date_col": pd.date_range("2024-01-01", periods=5, freq="D"),
        }
    )


@pytest.fixture()
def df_nulls() -> pd.DataFrame:
    """DataFrame dengan pola null berbeda per kolom."""
    return pd.DataFrame(
        {
            "no_null": [1, 2, 3, 4, 5],
            "some_null": [1.0, np.nan, 3.0, np.nan, 5.0],
            "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "cat_null": ["a", None, "b", None, "c"],
        }
    )


# ── dtype detection ────────────────────────────────────────────────────


class TestDtypeDetection:
    """Verifikasi _detect_dtype mengembalikan kategori semantik yang benar."""

    def test_int_is_numeric(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series([1, 2, 3], name="x")
        assert profiler._detect_dtype(s) == "numeric"

    def test_float_is_numeric(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series([1.1, 2.2, 3.3], name="x")
        assert profiler._detect_dtype(s) == "numeric"

    def test_datetime_is_datetime(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series(pd.date_range("2024-01-01", periods=3), name="x")
        assert profiler._detect_dtype(s) == "datetime"

    def test_bool_is_categorical(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series([True, False, True], dtype="bool", name="x")
        assert profiler._detect_dtype(s) == "categorical"

    def test_low_cardinality_string_is_categorical(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series(["a", "b", "c", "a", "b"], name="x")
        assert profiler._detect_dtype(s) == "categorical"

    def test_high_cardinality_long_string_is_text(self, profiler: DataFrameProfiler) -> None:
        """String panjang rata-rata > 50 char → text."""
        long_text = "x" * 60
        s = pd.Series([long_text] * 100, name="x")
        assert profiler._detect_dtype(s) == "text"

    def test_high_cardinality_short_string_is_text(self, profiler: DataFrameProfiler) -> None:
        """n_unique / n_total > 0.5 dan n_unique > max_categories → text."""
        profiler_strict = DataFrameProfiler(max_categories=5)
        s = pd.Series([f"id_{i}" for i in range(100)], name="x")
        assert profiler_strict._detect_dtype(s) == "text"

    def test_categorical_dtype_is_categorical(self, profiler: DataFrameProfiler) -> None:
        s = pd.Series(pd.Categorical(["low", "mid", "high", "low"]), name="x")
        assert profiler._detect_dtype(s) == "categorical"

    def test_full_profile_dtypes(self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame) -> None:
        """profile() harus menghasilkan dtype yang benar di ColumnProfile."""
        result = profiler.profile(df_mixed)
        col_map = {cp.name: cp.dtype for cp in result.columns}

        assert col_map["int_col"] == "numeric"
        assert col_map["float_col"] == "numeric"
        assert col_map["str_col"] == "categorical"
        # bool bisa jadi numeric (numpy bool → numeric dtype di pandas)
        assert col_map["bool_col"] in ("numeric", "categorical")
        assert col_map["date_col"] == "datetime"

    def test_pandas_dtype_preserved(
        self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame
    ) -> None:
        """pandas_dtype harus menyimpan dtype asli pandas (e.g. 'int64')."""
        result = profiler.profile(df_mixed)
        col_map = {cp.name: cp.pandas_dtype for cp in result.columns}

        assert "int" in col_map["int_col"]
        assert "float" in col_map["float_col"]
        assert col_map["str_col"] in ("object", "string", "str")


# ── missing count ───────────────────────────────────────────────────────


class TestMissingCount:
    """Verifikasi n_missing dan missing_pct dihitung dengan benar."""

    def test_no_nulls(self, profiler: DataFrameProfiler, df_nulls: pd.DataFrame) -> None:
        result = profiler.profile(df_nulls)
        cp = next(c for c in result.columns if c.name == "no_null")
        assert cp.n_missing == 0
        assert cp.missing_pct == 0.0

    def test_some_nulls(self, profiler: DataFrameProfiler, df_nulls: pd.DataFrame) -> None:
        result = profiler.profile(df_nulls)
        cp = next(c for c in result.columns if c.name == "some_null")
        assert cp.n_missing == 2
        assert cp.missing_pct == pytest.approx(0.4, abs=1e-4)

    def test_all_nulls(self, profiler: DataFrameProfiler, df_nulls: pd.DataFrame) -> None:
        result = profiler.profile(df_nulls)
        cp = next(c for c in result.columns if c.name == "all_null")
        assert cp.n_missing == 5
        assert cp.missing_pct == pytest.approx(1.0, abs=1e-4)

    def test_string_none_is_null(self, profiler: DataFrameProfiler, df_nulls: pd.DataFrame) -> None:
        """None di kolom object harus terhitung sebagai missing."""
        result = profiler.profile(df_nulls)
        cp = next(c for c in result.columns if c.name == "cat_null")
        assert cp.n_missing == 2

    def test_n_total_equals_dataframe_len(
        self, profiler: DataFrameProfiler, df_nulls: pd.DataFrame
    ) -> None:
        result = profiler.profile(df_nulls)
        for cp in result.columns:
            assert cp.n_total == len(df_nulls)

    def test_missing_pct_formula(self, profiler: DataFrameProfiler) -> None:
        """missing_pct = n_missing / n_total, dibulatkan 4 desimal."""
        df = pd.DataFrame({"x": [1, np.nan, np.nan, 4, 5, 6, 7, np.nan, 9, 10]})
        result = profiler.profile(df)
        cp = result.columns[0]
        assert cp.n_missing == 3
        assert cp.missing_pct == pytest.approx(3 / 10, abs=1e-4)


# ── unique count ────────────────────────────────────────────────────────


class TestUniqueCount:
    """Verifikasi n_unique dihitung dengan dropna=True."""

    def test_all_unique(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = profiler.profile(df)
        assert result.columns[0].n_unique == 5

    def test_all_same(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame({"x": [7, 7, 7, 7, 7]})
        result = profiler.profile(df)
        assert result.columns[0].n_unique == 1

    def test_unique_ignores_nan(self, profiler: DataFrameProfiler) -> None:
        """nunique(dropna=True) → NaN tidak dihitung sebagai unique value."""
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, np.nan, 3.0]})
        result = profiler.profile(df)
        assert result.columns[0].n_unique == 3

    def test_string_unique(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame({"x": ["a", "b", "a", "c", "b"]})
        result = profiler.profile(df)
        assert result.columns[0].n_unique == 3

    def test_mixed_df_unique_counts(
        self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame
    ) -> None:
        result = profiler.profile(df_mixed)
        col_map = {cp.name: cp.n_unique for cp in result.columns}

        assert col_map["int_col"] == 5
        assert col_map["float_col"] == 5
        assert col_map["str_col"] == 3  # a, b, c
        assert col_map["bool_col"] == 2  # True, False
        assert col_map["date_col"] == 5


# ── ReportData structure ────────────────────────────────────────────────


class TestReportDataStructure:
    """Verifikasi profile() mengembalikan ReportData yang lengkap."""

    def test_shape_metadata(self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame) -> None:
        result = profiler.profile(df_mixed, title="Test Report")
        assert result.title == "Test Report"
        assert result.n_rows == 5
        assert result.n_cols == 5
        assert result.memory_mb > 0

    def test_column_count_matches(
        self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame
    ) -> None:
        result = profiler.profile(df_mixed)
        assert len(result.columns) == df_mixed.shape[1]

    def test_column_names_match(self, profiler: DataFrameProfiler, df_mixed: pd.DataFrame) -> None:
        result = profiler.profile(df_mixed)
        names = [cp.name for cp in result.columns]
        assert names == list(df_mixed.columns)


# ── edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases & error handling."""

    def test_empty_df_raises(self, profiler: DataFrameProfiler) -> None:
        with pytest.raises(ValueError, match="empty"):
            profiler.profile(pd.DataFrame())

    def test_non_dataframe_raises(self, profiler: DataFrameProfiler) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            profiler.profile([1, 2, 3])  # type: ignore[arg-type]

    def test_single_column_df(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame({"only": [1, 2, 3]})
        result = profiler.profile(df)
        assert result.n_cols == 1
        assert result.columns[0].name == "only"

    def test_single_row_df(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame({"a": [42], "b": ["hello"]})
        result = profiler.profile(df)
        assert result.n_rows == 1
        assert result.columns[0].n_unique == 1

    def test_session_fixture_compat(
        self, profiler: DataFrameProfiler, df_small: pd.DataFrame
    ) -> None:
        """Pastikan profiler bisa memproses fixture session df_small dari conftest."""
        result = profiler.profile(df_small)
        assert result.n_rows == 100
        assert result.n_cols == 6


# ── numeric stats (describe, outliers) ──────────────────────────────────


class TestNumericStats:
    """Verifikasi perhitungan statistik numerik tersampel dari _numeric_stats."""

    def test_numeric_stats_basic(self, profiler: DataFrameProfiler) -> None:
        """Memastikan mean, std, quantiles, min max tersedia secara komplit."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = profiler._numeric_stats(s)

        assert stats["mean"] == 3.0
        assert stats["std"] == pytest.approx(1.5811388, abs=1e-5)
        assert stats["min"] == 1.0
        assert stats["q25"] == 2.0
        assert stats["median"] == 3.0
        assert stats["q75"] == 4.0
        assert stats["max"] == 5.0
        assert stats["n_outliers"] == 0

    def test_numeric_stats_outlier(self, profiler: DataFrameProfiler) -> None:
        """Memastikan filter outlier vektorisasi (1.5 IQR) bekerja proporsional."""
        # Kuartil: q25=2, q75=4 -> IQR=2. Bounds: (2 - 3, 4 + 3) -> (-1, 7)
        s = pd.Series([2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 10.0, -5.0])
        stats = profiler._numeric_stats(s)

        assert stats["n_outliers"] == 2  # 10.0 dan -5.0 adalah outlier

    def test_numeric_stats_single_value(self, profiler: DataFrameProfiler) -> None:
        """Edge case: jika panjang series = 1, std seharusnya 0 (awalnya NaN)."""
        s = pd.Series([42.0])
        stats = profiler._numeric_stats(s)

        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0
        assert stats["n_outliers"] == 0


# ── categorical stats ───────────────────────────────────────────────────


class TestCategoricalStats:
    """Verifikasi statistik kategori (top-k dan high cardinality)."""

    def test_top_values_limit(self, profiler: DataFrameProfiler) -> None:
        """Memastikan top_values maksimal dibatasi 10 data."""
        s = pd.Series([f"cat_{i}" for i in range(20)])
        stats = profiler._categorical_stats(s)

        assert len(stats["top_values"]) == 10
        assert stats["is_high_cardinality"] is True

    def test_low_cardinality(self, profiler: DataFrameProfiler) -> None:
        """Rasio keunikan yang kecil mengatur is_high_cardinality gampang diidentifikasi."""
        s = pd.Series(["a", "a", "b", "b", "a"])
        stats = profiler._categorical_stats(s)

        assert stats["top_values"] == {"a": 3, "b": 2}
        assert stats["is_high_cardinality"] is False


# ── correlations ────────────────────────────────────────────────────────


class TestCorrelations:
    """Verifikasi matriks korelasi dan pengekstrasian top korelasi."""

    def test_correlation_matrix_and_top(self, profiler: DataFrameProfiler) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],  # corr = 1.0 dengan 'a'
                "c": [5, 4, 3, 2, 1],  # corr = -1.0 dengan 'a'
                "d": ["a", "b", "c", "d", "e"],  # non-numeric, akan diabaikan
            }
        )
        corr_dict, top_corrs = profiler._compute_correlations(df)

        assert "a" in corr_dict
        assert "b" in corr_dict["a"]
        assert corr_dict["a"]["b"] == 1.0

        assert len(top_corrs) == 3
        # Mengembalikan list of tuple(col1, col2, score)
        # Expected pairs: (a,b), (a,c), (b,c)
        assert top_corrs[0][2] == 1.0 or top_corrs[0][2] == -1.0

    def test_single_numeric_column(self, profiler: DataFrameProfiler) -> None:
        """Jika df hanya punya satu numerik, kembalikan korelasi kosong."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        corr_dict, top_corrs = profiler._compute_correlations(df)

        assert corr_dict == {}
        assert top_corrs == []


def test_detect_dtype_other(profiler):
    """return 'other' path — baris 105."""
    # nullable integer → numeric lewat is_numeric_dtype
    # untuk memaksa 'other', pakai interval dtype
    s_interval = pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
    series = pd.Series(s_interval, name="interval_col")
    result = profiler._detect_dtype(series)
    assert result == "other"


def test_avg_len_exception_path(profiler):
    """Exception path avg_len — baris 97-98: astype str gagal → avg_len=0."""
    import unittest.mock as mock

    s = pd.Series(["a", "b", "c"] * 10, name="cat")
    with mock.patch(
        "pandas.Series.str", new_callable=mock.PropertyMock, side_effect=Exception("forced")
    ):
        result = profiler._detect_dtype(s)
    assert result in ("categorical", "text")  # tidak crash


def test_outlier_warning_above_threshold(profiler):
    """Outlier warning >5% — baris 181."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 100).tolist()
    # 10 outlier ekstrem → 10% > threshold 5%
    outliers = [9999.0] * 10
    df = pd.DataFrame({"v": base + outliers})
    report = profiler.profile(df)
    warning_texts = " ".join(report.warnings)
    assert "outlier" in warning_texts.lower()
