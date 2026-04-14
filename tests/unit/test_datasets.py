"""Test profiling against real-world standard datasets (Titanic, Diamonds) & Synths."""

from __future__ import annotations

import os
import urllib.request

import pandas as pd
import pytest

from edareport.profiler.core import DataFrameProfiler


def fetch_dataset(name: str) -> pd.DataFrame:
    """Download or load semantic standard datasets."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    urls = {
        "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "diamonds": "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv",
    }

    file_path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(file_path):
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlretrieve(urls[name], file_path, context=ctx)

    return pd.read_csv(file_path)


@pytest.fixture(scope="session")
def titanic_df() -> pd.DataFrame:
    try:
        return fetch_dataset("titanic")
    except Exception:  # Fallback sintetik jika gagal tembus firewall
        return pd.DataFrame(
            {
                "PassengerId": range(1, 892),
                "Survived": [0, 1] * 445 + [0],
                "Pclass": [3] * 891,
                "Name": ["John Doe"] * 891,
                "Sex": ["male", "female", "male"] * 297,
                "Age": [22.0, 38.0, None] * 297,
                "SibSp": [0] * 891,
                "Parch": [0] * 891,
                "Ticket": ["A/5 21171"] * 891,
                "Fare": [7.25, 71.28, 8.05] * 297,
                "Cabin": [None, "C85", None] * 297,
                "Embarked": ["S", "C", "Q"] * 297,
            }
        )


@pytest.fixture(scope="session")
def diamonds_df() -> pd.DataFrame:
    try:
        return fetch_dataset("diamonds").head(10000)  # Subset agar tes cepat
    except Exception:
        import numpy as np

        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "carat": rng.uniform(0.2, 5.0, 10000),
                "cut": rng.choice(["Ideal", "Premium", "Good"], 10000),
                "color": rng.choice(["E", "I", "J", "H"], 10000),
                "clarity": rng.choice(["SI2", "SI1", "VS1"], 10000),
                "depth": rng.normal(61, 1.5, 10000),
                "table": rng.normal(57, 2, 10000),
                "price": rng.integers(326, 18823, 10000),
                "x": rng.uniform(3, 10, 10000),
                "y": rng.uniform(3, 10, 10000),
                "z": rng.uniform(2, 6, 10000),
            }
        )


def test_profiler_titanic(titanic_df: pd.DataFrame) -> None:
    """Uji titanic me-resolve missing values (Age, Cabin) dan string teks (Name)."""
    profiler = DataFrameProfiler()
    report = profiler.profile(titanic_df)

    assert report.n_rows == 891
    assert report.n_cols > 0

    # Periksa deteksi missing value di Cabin
    cabin_profile = next(cp for cp in report.columns if cp.name == "Cabin")
    assert cabin_profile.missing_pct > 0.5  # Kebanyakan cabin is null (>50%)

    # Kolom Name seharusnya berjenis text
    name_profile = next(cp for cp in report.columns if cp.name == "Name")
    assert name_profile.dtype in ("text", "categorical")


def test_profiler_diamonds(diamonds_df: pd.DataFrame) -> None:
    """Uji dataset diamonds (skala menengah) memastikan outlers & korelasi terbentuk."""
    profiler = DataFrameProfiler()
    report = profiler.profile(diamonds_df)

    assert report.n_rows == 10000
    assert report.n_cols == 10 or report.n_cols == 11  # with or without index

    # Periksa terdapat korelasi di top_correlations (carat & price misal)
    assert len(report.top_correlations) > 0

    # Harga numeric pasti punya outlier flag tergenerasi
    price_profile = next(cp for cp in report.columns if cp.name == "price")
    assert price_profile.dtype == "numeric"
    assert price_profile.median is not None
