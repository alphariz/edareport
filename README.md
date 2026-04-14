# edareport

**EDA report paling cepat & ringan di Python — insight dalam detik, bukan menit.**

[![PyPI version](https://badge.fury.io/py/edafast.svg)](https://pypi.org/project/edafast/)
[![Python](https://img.shields.io/pypi/pyversions/edareport)](https://pypi.org/project/edafast/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`edareport` adalah Python library untuk generate EDA report super cepat — summary stats + interactive plots — dalam satu baris kode.

> *"Versi lebih ringan & cepat dari ydata-profiling / Sweetviz"*

**Target user:** Data Analyst, Data Scientist, Business Analyst yang ingin insight dalam detik, bukan menit atau jam.

## Install

```bash
pip install edafast
```

Optional: Polars backend (5–10x lebih cepat untuk dataset besar):

```bash
pip install "edareport[polars]"
```

## Quickstart

```python
import pandas as pd
from edareport import generate_report

df = pd.read_csv("data.csv")

# One-liner — generate & save HTML report
report = generate_report(df, title="Sales EDA")
report.save("report.html")

# Atau tampilkan langsung di Jupyter/Colab
report.show()
```

## Features

- **Super cepat** — 0.013s untuk 10k rows, 0.104s untuk 100k rows
- **Minimal dependencies** — hanya 5 package inti (pandas, numpy, scipy, plotly, jinja2)
- **Low memory** — peak RAM selalu < 1x ukuran dataset
- **Clean HTML** — self-contained, mudah di-share via email/Slack/Notion (< 5MB)
- **Full interactive** — Plotly native, berjalan tanpa server
- **Notebook ready** — Jupyter + Google Colab

## Benchmark

| Rows    | edareport | Target  | Status |
|---------|-----------|---------|--------|
| 10,000  | 0.013s    | < 5s    | ✅     |
| 50,000  | 0.054s    | —       | ✅     |
| 100,000 | 0.104s    | < 15s   | ✅     |

Peak RAM < 1x ukuran dataset (target < 2x). ✅

## API

```python
generate_report(
    df,                    # pandas DataFrame
    title="EDA Report",   # judul report
    output="html",        # "html" | "widget" | "data"
    theme="light",        # "light" | "dark"
    sample_size=50_000,   # batas baris untuk visualisasi berat
)
```

## Output

- `report.save("report.html")` — simpan ke file HTML
- `report.show()` — tampilkan di browser atau Jupyter widget
- `report.data` — akses `ReportData` dataclass mentah

## Development

```bash
git clone https://github.com/lancewrg/edareport
cd edareport
uv sync --extra dev
uv run pytest -m "not slow"
```

## License

MIT — see [LICENSE](LICENSE)
