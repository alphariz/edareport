"""edareport — EDA report paling cepat & ringan di Python."""

from edareport._api import generate_report
from edareport._types import ColumnProfile, ReportData

__version__ = "0.1.0"
__all__ = ["generate_report", "ReportData", "ColumnProfile"]