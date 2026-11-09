"""Spike detection and metrics for a spiking-neuron voltage trace.

Detection: upward threshold crossing, then the peak within the refractory
window after the crossing is the spike; crossings closer together than
``refractory_s`` are merged (the larger peak wins). Width is the full width
at half maximum (FWHM), found by linear interpolation of the half-height
crossings on either side of the peak.
"""
from __future__ import annotations

import numpy as np


def detect_spikes(times, voltages, threshold_v, refractory_s=0.002):
    """Detect spikes.

    Parameters
    ----------
    times, voltages : array-like, seconds / volts
    threshold_v : float, volts above baseline the trace must cross
    refractory_s : float, minimum separation between spikes (seconds)

    Returns a list of dicts, one per spike, with keys
    ``time_s`` (peak time), ``peak_v``, ``amplitude_v`` (peak - baseline),
    ``width_s`` (FWHM, ``nan`` if the half-height crossings are not found).
    Baseline is the median of the trace.
    """
    times = np.asarray(times, dtype=float)
    volts = np.asarray(voltages, dtype=float)
    if len(times) < 3:
        return []
    baseline = float(np.median(volts))
    level = baseline + threshold_v

    above = volts > level
    # Upward crossings: index i where trace goes from <= level to > level.
    crossings = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if len(crossings) == 0:
        return []

    spikes = []
    last_peak_time = -np.inf
    for cross in crossings:
        # Search for the peak from the crossing until the trace falls back
        # below the level, or the refractory window ends -- whichever is later
        # bounded by the refractory window so a wide spike is not re-found.
        end = cross
        while end + 1 < len(volts) and volts[end + 1] > level:
            end += 1
        # Extend the search at least to the refractory window so the true
        # peak is found even if the trace dips briefly.
        search_end = min(len(volts), int(cross + max(1, refractory_s / _dt(times))))
        peak_idx = cross + int(np.argmax(volts[cross:search_end])) if search_end > cross else cross
        # Prefer the peak in the above-threshold run if it is larger.
        run_peak = cross + int(np.argmax(volts[cross:end + 1]))
        if volts[run_peak] >= volts[peak_idx]:
            peak_idx = run_peak

        peak_time = float(times[peak_idx])
        if peak_time - last_peak_time < refractory_s and spikes:
            # Merge: keep the larger spike.
            if volts[peak_idx] > spikes[-1]["peak_v"]:
                spikes[-1] = _spike_record(times, volts, peak_idx, baseline)
                last_peak_time = peak_time
            continue
        spikes.append(_spike_record(times, volts, peak_idx, baseline))
        last_peak_time = peak_time
    return spikes


def _dt(times):
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 1e-4


def _spike_record(times, volts, peak_idx, baseline):
    peak_v = float(volts[peak_idx])
    amp = peak_v - baseline
    width = _fwhm(times, volts, peak_idx, baseline, amp)
    return {
        "time_s": float(times[peak_idx]),
        "peak_v": peak_v,
        "amplitude_v": amp,
        "width_s": width,
    }


def _fwhm(times, volts, peak_idx, baseline, amplitude):
    """Full width at half maximum by linear interpolation."""
    if amplitude <= 0:
        return float("nan")
    half = baseline + amplitude / 2.0
    # Left crossing: walk back from the peak until below half.
    left = None
    for i in range(peak_idx, 0, -1):
        if volts[i - 1] <= half < volts[i] or volts[i - 1] < half <= volts[i]:
            frac = (half - volts[i - 1]) / (volts[i] - volts[i - 1]) if volts[i] != volts[i - 1] else 0.0
            left = times[i - 1] + frac * (times[i] - times[i - 1])
            break
    right = None
    for i in range(peak_idx, len(volts) - 1):
        if volts[i] >= half > volts[i + 1] or volts[i] > half >= volts[i + 1]:
            frac = (volts[i] - half) / (volts[i] - volts[i + 1]) if volts[i] != volts[i + 1] else 0.0
            right = times[i] + frac * (times[i + 1] - times[i])
            break
    if left is None or right is None:
        return float("nan")
    return float(right - left)


def compute_metrics(times, voltages, threshold_v=0.04, refractory_s=0.002):
    """Compute all headline metrics for a trace.

    Returns a dict with spike_count, duration_s, frequency_hz,
    isi_mean_s / isi_median_s / isi_std_s / isi_cv (std/mean, jitter),
    amplitude_mean_v / amplitude_mean_mv, width_mean_s / width_mean_ms,
    baseline_v, plus the raw ``spikes`` list. Empty traces yield 0 spikes
    and ``nan`` for the undefined averages -- never an exception.
    """
    times = np.asarray(times, dtype=float)
    volts = np.asarray(voltages, dtype=float)
    spikes = detect_spikes(times, volts, threshold_v, refractory_s)
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    count = len(spikes)

    if count >= 2:
        isi = np.diff([s["time_s"] for s in spikes])
        isi_mean = float(np.mean(isi))
        isi_median = float(np.median(isi))
        isi_std = float(np.std(isi, ddof=1)) if len(isi) >= 2 else 0.0
        isi_cv = isi_std / isi_mean if isi_mean > 0 else float("nan")
    else:
        isi_mean = isi_median = isi_std = isi_cv = float("nan")

    amps = [s["amplitude_v"] for s in spikes]
    widths = [s["width_s"] for s in spikes if np.isfinite(s["width_s"])]
    return {
        "spike_count": count,
        "duration_s": duration,
        "frequency_hz": count / duration if duration > 0 else 0.0,
        "isi_mean_s": isi_mean,
        "isi_median_s": isi_median,
        "isi_std_s": isi_std,
        "isi_cv": isi_cv,
        "amplitude_mean_v": float(np.mean(amps)) if amps else float("nan"),
        "amplitude_mean_mv": float(np.mean(amps)) * 1000.0 if amps else float("nan"),
        "width_mean_s": float(np.mean(widths)) if widths else float("nan"),
        "width_mean_ms": float(np.mean(widths)) * 1000.0 if widths else float("nan"),
        "baseline_v": float(np.median(volts)) if len(volts) else float("nan"),
        "spikes": spikes,
    }
