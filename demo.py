"""Demo: run the test bench on every sample CSV in data/ and print/write
the pass/fail report.

Run:  python3 demo.py            (from the project root)
      python3 demo.py --plot     (also saves plots/good_neuron.png etc.)
Writes reports/report.txt and reports/report.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_test_bench import (  # noqa: E402
    check_pass_fail,
    compute_metrics,
    format_report,
    load_trace,
    write_report,
    write_report_csv,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPORTS = os.path.join(HERE, "reports")

SAMPLES = ["good_neuron.csv", "noisy_neuron.csv", "no_spikes.csv", "failing_neuron.csv"]


def run_all(plot=False):
    os.makedirs(REPORTS, exist_ok=True)
    rows = []
    full_text = []
    for name in SAMPLES:
        path = os.path.join(DATA, name)
        times, volts = load_trace(path)
        metrics = compute_metrics(times, volts, threshold_v=0.04, refractory_s=0.002)
        passed, checks = check_pass_fail(metrics)
        text = format_report(name, metrics, passed, checks)
        print(text)
        full_text.append(text)
        rows.append((name, metrics, passed, checks))
        if plot:
            _plot(name, times, volts, metrics)
    # Full multi-trace report file:
    with open(os.path.join(REPORTS, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(full_text))
    write_report_csv(os.path.join(REPORTS, "report.csv"), rows)
    print(f"Reports written to {REPORTS}/report.txt and {REPORTS}/report.csv")
    return rows


def _plot(name, times, volts, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = os.path.join(HERE, "plots")
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(times * 1000, volts * 1000, lw=0.6)
    for s in metrics["spikes"]:
        ax.plot(s["time_s"] * 1000, s["peak_v"] * 1000, "ro", ms=4)
    ax.axhline(40, color="gray", ls="--", lw=0.8, label="threshold 40 mV")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"{name}: {metrics['spike_count']} spikes, "
                 f"{metrics['frequency_hz']:.1f} Hz")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, name.replace(".csv", ".png"))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run_all(plot=args.plot)
