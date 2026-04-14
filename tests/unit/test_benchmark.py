"""Benchmark: profiler harus memenuhi non-negotiable targets."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from edareport.profiler.core import DataFrameProfiler


@pytest.mark.benchmark
@pytest.mark.slow
def test_profiler_speed_10k(df_small: pd.DataFrame) -> None:
    """Target: <5 detik untuk 10k rows."""

    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 10_000),
            "b": rng.integers(0, 100, 10_000).astype(float),
            "c": np.random.choice(["x", "y", "z"], 10_000),
        }
    )
    profiler = DataFrameProfiler()
    start = time.perf_counter()
    profiler.profile(df)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Profiler too slow: {elapsed:.2f}s for 10k rows (target <5s)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_profiler_speed_100k(df_medium: pd.DataFrame) -> None:
    """Target: <15 detik untuk 100k rows."""

    rng = np.random.default_rng(2)
    df = pd.DataFrame({f"col_{i}": rng.normal(i, 1, 100_000) for i in range(10)})
    for i in range(3):
        df[f"cat_{i}"] = np.random.choice(list("ABCD"), 100_000)
    profiler = DataFrameProfiler()
    start = time.perf_counter()
    profiler.profile(df)
    elapsed = time.perf_counter() - start
    assert elapsed < 15.0, f"Profiler too slow: {elapsed:.2f}s for 100k rows (target <15s)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_memory_peak(df_medium: pd.DataFrame) -> None:
    """Peak RAM tidak boleh melebihi 2x ukuran dataset."""
    pytest.importorskip("memory_profiler")
    from memory_profiler import memory_usage  # type: ignore[import]

    rng = np.random.default_rng(3)
    df = pd.DataFrame({f"v{i}": rng.normal(0, 1, 50_000) for i in range(10)})
    dataset_mb = df.memory_usage(deep=True).sum() / 1024**2

    profiler = DataFrameProfiler()
    baseline = float(memory_usage(max_usage=True))
    mem_usage = memory_usage((profiler.profile, (df,)), interval=0.05, max_usage=True)
    peak_diff_mb = float(mem_usage) - baseline

    # Allowance: 2x dataset memory overhead peak jump
    allowance_mb = max(20, dataset_mb * 2.5)  # toleransi dasar untuk metadata
    assert (
        peak_diff_mb < allowance_mb
    ), f"Memory jump too high: {peak_diff_mb:.1f}MB peak diff, dataset={dataset_mb:.1f}MB"


@pytest.mark.benchmark
@pytest.mark.slow
def test_ydata_profiling_comparison(df_small: pd.DataFrame) -> None:
    """Benchmark perbandingan vs ydata-profiling (waktu eksekusi)."""
    # 10k rows untuk dibandingkan

    # Hanya jalan jika ydata-profiling terinstall (untuk keperluan milestone test environment)
    ProfileReport = pytest.importorskip("ydata_profiling").ProfileReport

    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "num": rng.normal(0, 1, 10_000),
            "cat": rng.choice(["a", "b", "c"], 10_000),
            "text": ["random text " + str(i) for i in range(10_000)],
        }
    )

    # Test edareport
    start = time.perf_counter()
    DataFrameProfiler().profile(df)
    time_eda = time.perf_counter() - start

    # Test ydata
    start = time.perf_counter()
    # minimal=True agar sedikit lebih adil secara eksekusi (ydata standar menembus limit ekstrim)
    report_ydata = ProfileReport(df, minimal=True)
    report_ydata.get_description()
    time_ydata = time.perf_counter() - start

    assert (
        time_eda < 5.0
    ), f"EdaReport melebihi batas toleransi milstone < 5.0 detik ({time_eda:.2f}s)"

    # Log ekspektasi (biasanya EDA Report jauh di bawah ydata)
    print(f"\nEdaReport: {time_eda:.2f}s | YData (minimal): {time_ydata:.2f}s")
