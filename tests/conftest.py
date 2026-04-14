"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def df_small() -> pd.DataFrame:
    """100 baris, mix tipe kolom — cukup untuk unit test cepat."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, 100).astype(float),
            "salary": rng.normal(50_000, 15_000, 100),
            "department": rng.choice(["Engineering", "Sales", "HR", "Finance"], 100),
            "score": rng.uniform(0, 100, 100),
            "active": rng.choice([True, False], 100),
            "joined": pd.date_range("2020-01-01", periods=100, freq="W"),
        }
    )


@pytest.fixture(scope="session")
def df_medium() -> pd.DataFrame:
    """50_000 baris untuk benchmark & sampling test."""
    rng = np.random.default_rng(0)
    n = 50_000
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "value_a": rng.normal(0, 1, n),
            "value_b": rng.exponential(2, n),
            "value_c": rng.integers(0, 1000, n).astype(float),
            "category": rng.choice(list("ABCDE"), n),
            "label": rng.choice(["yes", "no"], n),
        }
    )


@pytest.fixture(scope="session")
def df_with_nulls() -> pd.DataFrame:
    """DataFrame dengan banyak missing values untuk test warnings."""
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(5, 2, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    # Masukkan null ~40%
    mask_x = rng.random(n) < 0.4
    mask_cat = rng.random(n) < 0.35
    df.loc[mask_x, "x"] = np.nan
    df.loc[mask_cat, "cat"] = np.nan
    return df
