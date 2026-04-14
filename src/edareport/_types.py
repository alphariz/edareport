"""Public dataclasses — stable contract antara engine dan renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # "numeric" | "categorical" | "datetime" | "text" | "other"
    pandas_dtype: str  # dtype asli pandas, e.g. "float64", "object"
    n_total: int
    n_missing: int
    n_unique: int
    missing_pct: float

    # Numeric only (None untuk non-numeric)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    n_outliers: int | None = None  # IQR-based

    # Categorical only
    top_values: dict[str, int] = field(default_factory=dict)  # value → count, top-10
    is_high_cardinality: bool = False  # n_unique / n_total > 0.5


@dataclass
class ReportData:
    title: str
    n_rows: int
    n_cols: int
    memory_mb: float
    columns: list[ColumnProfile]
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    top_correlations: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (col_a, col_b, r)
    warnings: list[str] = field(default_factory=list)

    # Raw stats untuk renderer (opsional, diisi engine)
    _meta: dict[str, Any] = field(default_factory=dict, repr=False)
