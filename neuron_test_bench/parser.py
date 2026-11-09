"""Oscilloscope CSV parser.

Tolerant of:
  * an optional comment line starting with '#'
  * a header line naming the columns, e.g. ``Time (ms), Voltage (mV)``,
    ``time_s, voltage_V``, ``t, v`` -- units in the header are honoured
    (s / ms / us for time, V / mV / uV for voltage)
  * no header at all (columns are assumed to be time in seconds,
    voltage in volts, in that order)
  * extra columns (only the first two numeric columns are used)
  * blank lines and trailing whitespace

Returns times in seconds and voltages in volts as NumPy arrays.
"""
from __future__ import annotations

import re

import numpy as np

_TIME_UNITS = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
_VOLT_UNITS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6}


def _unit_factor(header: str, units: dict) -> float:
    """Extract a unit in parentheses/brackets/suffix from a header cell."""
    m = re.search(r"\(([^)]+)\)|\[([^\]]+)\]", header)
    token = (m.group(1) or m.group(2)) if m else header.strip().split()[-1]
    return units.get(token.strip().lower(), 1.0)


def _is_header(row: list[str]) -> bool:
    """A row is a header if any cell contains a letter (not a number)."""
    for cell in row:
        c = cell.strip()
        if not c:
            continue
        try:
            float(c)
        except ValueError:
            return True
    return False


def parse_csv(path: str):
    """Parse an oscilloscope CSV export.

    Returns ``(times_s, voltages_v)`` as float NumPy arrays, sorted by time.
    Raises ``ValueError`` if no numeric data rows are found.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw_lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]

    if not raw_lines:
        raise ValueError(f"{path}: no data found")

    time_factor = 1.0
    volt_factor = 1.0
    data_lines = raw_lines
    first = [c.strip() for c in raw_lines[0].split(",")]
    if _is_header(first):
        if len(first) >= 1:
            time_factor = _unit_factor(first[0], _TIME_UNITS)
        if len(first) >= 2:
            volt_factor = _unit_factor(first[1], _VOLT_UNITS)
        data_lines = raw_lines[1:]

    times, volts = [], []
    for line in data_lines:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2:
            continue
        try:
            t = float(cells[0])
            v = float(cells[1])
        except ValueError:
            continue  # skip stray non-numeric rows
        times.append(t * time_factor)
        volts.append(v * volt_factor)

    if not times:
        raise ValueError(f"{path}: no numeric data rows found")

    times = np.asarray(times, dtype=float)
    volts = np.asarray(volts, dtype=float)
    order = np.argsort(times)
    return times[order], volts[order]


def load_trace(path: str):
    """Alias for :func:`parse_csv` -- same return, clearer name in demos."""
    return parse_csv(path)
