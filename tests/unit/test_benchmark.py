"""Benchmark: profiler harus memenuhi non-negotiable targets."""

from __future__ import annotations

import time

import pytest

from edareport.profiler.core import DataFrameProfiler


@pytest.mark.benchmark
@pytest.mark.slow
def test_profiler_speed_10k(df_small: "pd.DataFrame") -> None:
    """Target: <5 detik untuk 10k rows."""
    import pandas as pd, numpy as np
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 10_000),
        "b": rng.integers(0, 100, 10_000).astype(float),
        "c": np.random.choice(["x", "y", "z"], 10_000),
    })
    profiler = DataFrameProfiler()
    start = time.perf_counter()
    profiler.profile(df)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Profiler too slow: {elapsed:.2f}s for 10k rows (target <5s)"


@pytest.mark.benchmark
@pytest.mark.slow
def test_profiler_speed_100k(df_medium: "pd.DataFrame") -> None:
    """Target: <15 detik untuk 100k rows."""
    import pandas as pd, numpy as np
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
def test_memory_peak(df_medium: "pd.DataFrame") -> None:
    """Peak RAM tidak boleh melebihi 2x ukuran dataset."""
    pytest.importorskip("memory_profiler")
    from memory_profiler import memory_usage  # type: ignore[import]
    import pandas as pd, numpy as np

    rng = np.random.default_rng(3)
    df = pd.DataFrame({f"v{i}": rng.normal(0, 1, 50_000) for i in range(10)})
    dataset_mb = df.memory_usage(deep=True).sum() / 1024**2

    profiler = DataFrameProfiler()
    mem_usage = memory_usage((profiler.profile, (df,)), interval=0.05, max_usage=True)
    peak_mb = float(mem_usage)

    # Allowance: baseline process memory (~50MB) + 2x dataset
    allowance_mb = 50 + dataset_mb * 2
    assert peak_mb < allowance_mb, (
        f"Memory too high: {peak_mb:.1f}MB peak, dataset={dataset_mb:.1f}MB"
    )