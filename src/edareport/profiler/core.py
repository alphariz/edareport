"""DataFrameProfiler — vectorized engine, zero .iterrows()."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from edareport._types import ColumnProfile, ReportData

# Threshold: n_unique / n_rows > ini → high cardinality
_HIGH_CARD_RATIO = 0.5
# Threshold: n_unique <= ini → pasti categorical
_MAX_UNIQUE_CATEGORICAL = 50
# Kolom dengan missing > threshold ini → warning
_MISSING_WARN_PCT = 0.3


class DataFrameProfiler:
    def __init__(self, max_categories: int = _MAX_UNIQUE_CATEGORICAL) -> None:
        self._max_categories = max_categories

    def profile(self, df: pd.DataFrame, title: str = "EDA Report") -> ReportData:
        """Profile seluruh DataFrame, return ReportData."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty.")

        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2

        column_profiles = [self._profile_column(df[col], n_rows) for col in df.columns]

        corr_matrix, top_corrs = self._compute_correlations(df)
        report_warnings = self._collect_warnings(column_profiles, df)

        return ReportData(
            title=title,
            n_rows=n_rows,
            n_cols=n_cols,
            memory_mb=round(memory_mb, 6),
            columns=column_profiles,
            correlation_matrix=corr_matrix,
            top_correlations=top_corrs,
            warnings=report_warnings,
        )

    # ------------------------------------------------------------------
    # Column profiling
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series, n_rows: int) -> ColumnProfile:  # type: ignore[type-arg]
        dtype_cat = self._detect_dtype(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        missing_pct = round(n_missing / n_rows, 4) if n_rows else 0.0

        base: dict[str, Any] = {
            "name": str(series.name),
            "dtype": dtype_cat,
            "pandas_dtype": str(series.dtype),
            "n_total": n_rows,
            "n_missing": n_missing,
            "n_unique": n_unique,
            "missing_pct": missing_pct,
        }

        if dtype_cat == "numeric":
            base.update(self._numeric_stats(series))
        elif dtype_cat == "categorical":
            base.update(self._categorical_stats(series))

        return ColumnProfile(**base)

    def _detect_dtype(self, series: pd.Series) -> str:  # type: ignore[type-arg]
        """Categorize kolom ke dalam 5 bucket semantik."""
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"

        # object / string — tentukan berdasarkan cardinality
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            n_unique = series.nunique(dropna=True)
            n_total = len(series.dropna())
            ratio = n_unique / n_total if n_total else 0
            # Heuristic: kalau rata-rata panjang string > 50 char → "text"
            try:
                avg_len = series.dropna().astype(str).str.len().mean()
            except Exception:
                avg_len = 0
            if avg_len > 50:
                return "text"
            if ratio > _HIGH_CARD_RATIO and n_unique > self._max_categories:
                return "text"
            return "categorical"

        return "other"

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:  # type: ignore[type-arg]
        clean = series.dropna()
        if clean.empty:
            return {}
        desc = clean.describe()
        q25 = float(desc["25%"])
        q75 = float(desc["75%"])
        iqr = q75 - q25
        n_outliers = int(((clean < q25 - 1.5 * iqr) | (clean > q75 + 1.5 * iqr)).sum())
        return {
            "mean": round(float(desc["mean"]), 6),
            "std": round(float(desc["std"]), 6) if not pd.isna(desc["std"]) else 0.0,
            "min": round(float(desc["min"]), 6),
            "q25": round(q25, 6),
            "median": round(float(desc["50%"]), 6),
            "q75": round(q75, 6),
            "max": round(float(desc["max"]), 6),
            "n_outliers": n_outliers,
        }

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:  # type: ignore[type-arg]
        n_unique = series.nunique(dropna=True)
        n_total = len(series.dropna())
        # top-10 value counts
        top_vc = series.value_counts(dropna=True).head(10)
        return {
            "top_values": {str(k): int(v) for k, v in top_vc.items()},
            "is_high_cardinality": (n_unique / n_total > _HIGH_CARD_RATIO if n_total else False),
        }

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _compute_correlations(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, dict[str, float]], list[tuple[str, str, float]]]:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return {}, []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = numeric_df.corr(method="pearson")

        corr_dict: dict[str, dict[str, float]] = {
            col: {c: round(float(v), 4) for c, v in row.items()}
            for col, row in corr.to_dict().items()
        }

        # Top-10 pairs (absolute value), exclude self-correlation
        pairs: list[tuple[str, str, float]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = corr.loc[a, b]
                if not np.isnan(val):
                    pairs.append((a, b, round(float(val), 4)))

        top_corrs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
        return corr_dict, top_corrs

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _collect_warnings(self, profiles: list[ColumnProfile], df: pd.DataFrame) -> list[str]:
        msgs: list[str] = []
        for cp in profiles:
            if cp.missing_pct > _MISSING_WARN_PCT:
                msgs.append(f"'{cp.name}': {cp.missing_pct:.0%} missing values")
            if cp.dtype == "numeric" and cp.n_outliers and cp.n_outliers > 0:
                pct = cp.n_outliers / cp.n_total
                if pct > 0.05:
                    msgs.append(f"'{cp.name}': {cp.n_outliers} outliers ({pct:.1%} of rows)")
            if cp.dtype == "categorical" and cp.is_high_cardinality:
                msgs.append(f"'{cp.name}': high cardinality ({cp.n_unique} unique values)")
        # Duplicate rows check
        n_dup = int(df.duplicated().sum())
        if n_dup > 0:
            msgs.append(f"{n_dup} duplicate rows detected")
        return msgs
