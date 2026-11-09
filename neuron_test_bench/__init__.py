"""Neuron test bench: read oscilloscope CSV exports and measure a spiking
neuron board's output (spike count, frequency, ISI, amplitude, width, jitter)
and produce a pass/fail report against expected values.

Stdlib + NumPy only (matplotlib is used only by the optional plot helper).
"""
from .parser import parse_csv, load_trace
from .analysis import detect_spikes, compute_metrics
from .report import check_pass_fail, format_report, write_report, write_report_csv

__all__ = [
    "parse_csv",
    "load_trace",
    "detect_spikes",
    "compute_metrics",
    "check_pass_fail",
    "format_report",
    "write_report",
    "write_report_csv",
]
__version__ = "1.0.0"
