"""Pass/fail report generation against expected values / tolerances."""
from __future__ import annotations

import csv
import math

# Default acceptance spec for the proof-of-concept board:
# a neuron firing regularly at ~20 Hz, ~80 mV spikes, ~1.2 ms wide, low jitter.
DEFAULT_SPEC = {
    "freq_min_hz": 15.0,
    "freq_max_hz": 25.0,
    "amplitude_min_mv": 60.0,
    "amplitude_max_mv": 100.0,
    "isi_cv_max": 0.15,
    "width_min_ms": 0.5,
    "width_max_ms": 2.5,
    "min_spikes": 5,
}


def _ok(value, lo, hi):
    return isinstance(value, float) and math.isfinite(value) and lo <= value <= hi


def check_pass_fail(metrics, spec=None):
    """Check measured metrics against a spec. Returns (overall_pass, checks).

    ``checks`` is a list of dicts: name, measured, expected (str), passed.
    """
    spec = {**DEFAULT_SPEC, **(spec or {})}
    checks = [
        {
            "name": "spike_count",
            "measured": metrics["spike_count"],
            "expected": f">= {spec['min_spikes']}",
            "passed": metrics["spike_count"] >= spec["min_spikes"],
        },
        {
            "name": "frequency_hz",
            "measured": metrics["frequency_hz"],
            "expected": f"{spec['freq_min_hz']}-{spec['freq_max_hz']} Hz",
            "passed": _ok(metrics["frequency_hz"], spec["freq_min_hz"], spec["freq_max_hz"]),
        },
        {
            "name": "amplitude_mv",
            "measured": metrics["amplitude_mean_mv"],
            "expected": f"{spec['amplitude_min_mv']}-{spec['amplitude_max_mv']} mV",
            "passed": _ok(metrics["amplitude_mean_mv"], spec["amplitude_min_mv"], spec["amplitude_max_mv"]),
        },
        {
            "name": "isi_cv_jitter",
            "measured": metrics["isi_cv"],
            "expected": f"<= {spec['isi_cv_max']}",
            "passed": _ok(metrics["isi_cv"], 0.0, spec["isi_cv_max"]),
        },
        {
            "name": "width_ms",
            "measured": metrics["width_mean_ms"],
            "expected": f"{spec['width_min_ms']}-{spec['width_max_ms']} ms",
            "passed": _ok(metrics["width_mean_ms"], spec["width_min_ms"], spec["width_max_ms"]),
        },
    ]
    return all(c["passed"] for c in checks), checks


def _fmt(value):
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.4g}"
    return str(value)


def format_report(name, metrics, passed, checks):
    lines = [
        f"Neuron test-bench report: {name}",
        f"Overall: {'PASS' if passed else 'FAIL'}",
        f"  spikes:            {metrics['spike_count']}",
        f"  duration:          {metrics['duration_s']:.3f} s",
        f"  frequency:         {metrics['frequency_hz']:.2f} Hz",
        f"  ISI mean / median: {_fmt(metrics['isi_mean_s'] * 1000)} / {_fmt(metrics['isi_median_s'] * 1000)} ms",
        f"  ISI std (jitter):  {_fmt(metrics['isi_std_s'] * 1000)} ms",
        f"  ISI CV (jitter):   {_fmt(metrics['isi_cv'])}",
        f"  amplitude (mean):  {_fmt(metrics['amplitude_mean_mv'])} mV",
        f"  width FWHM (mean): {_fmt(metrics['width_mean_ms'])} ms",
        f"  baseline:          {_fmt(metrics['baseline_v'] * 1000)} mV",
        "Checks:",
    ]
    for c in checks:
        lines.append(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {_fmt(c['measured'])} (expected {c['expected']})")
    return "\n".join(lines) + "\n"


def write_report(path, name, metrics, passed, checks):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_report(name, metrics, passed, checks))
    return path


def write_report_csv(path, rows):
    """Write one row per trace: name, metrics, overall pass/fail."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "spike_count", "frequency_hz", "isi_mean_ms",
                    "isi_cv", "amplitude_mv", "width_ms", "overall"])
        for name, metrics, passed, _checks in rows:
            w.writerow([
                name, metrics["spike_count"],
                f"{metrics['frequency_hz']:.3f}",
                _fmt(metrics["isi_mean_s"] * 1000),
                _fmt(metrics["isi_cv"]),
                _fmt(metrics["amplitude_mean_mv"]),
                _fmt(metrics["width_mean_ms"]),
                "PASS" if passed else "FAIL",
            ])
    return path
