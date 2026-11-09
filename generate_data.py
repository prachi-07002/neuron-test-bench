"""Generate the sample oscilloscope CSVs in data/.

Ground truth (all traces 2.0 s long, sampled at 20 kHz, Gaussian spikes
with sigma = 0.5 ms -> FWHM = 2.355 * 0.5 = 1.18 ms):
  good_neuron.csv    -- 40 spikes, exactly 20 Hz, 80 mV, 1 mV RMS noise
  noisy_neuron.csv   -- ~40 spikes, ~20 Hz with jittered ISIs (std 12 ms),
                        70-90 mV amplitudes, 5 mV RMS noise
  no_spikes.csv      -- baseline + 2 mV RMS noise, no spikes
  failing_neuron.csv -- 16 spikes, 8 Hz, 45 mV (below the 60 mV spec floor):
                        fails frequency and amplitude (jitter is in spec)

Run:  python3 generate_data.py
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FS = 20_000            # Hz
DURATION = 2.0         # s
SIGMA = 0.0005         # s, Gaussian spike width -> FWHM 1.18 ms
BASELINE = 0.0


def _trace(spike_times, amplitudes, noise_mv, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    v = np.full_like(t, BASELINE) + rng.normal(0.0, noise_mv / 1000.0, size=len(t))
    for st, amp_mv in zip(spike_times, amplitudes):
        v += (amp_mv / 1000.0) * np.exp(-0.5 * ((t - st) / SIGMA) ** 2)
    return t, v


def _write(name, t, v):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Time (ms), Voltage (mV)\n")
        for ti, vi in zip(t, v):
            fh.write(f"{ti * 1000:.3f},{vi * 1000:.4f}\n")
    print(f"wrote {path} ({len(t)} rows)")


def main():
    os.makedirs(DATA, exist_ok=True)

    # Good: perfectly regular 20 Hz, spikes at 25 ms + k*50 ms.
    good_times = 0.025 + 0.05 * np.arange(40)
    t, v = _trace(good_times, [80.0] * 40, noise_mv=1.0, seed=1)
    _write("good_neuron.csv", t, v)

    # Noisy/jittered: ISIs drawn around 50 ms (std 8 ms), amplitudes 70-90 mV.
    rng = np.random.default_rng(7)
    isi = rng.normal(0.05, 0.012, size=39)
    noisy_times = np.concatenate([[0.025], 0.025 + np.cumsum(isi)])
    noisy_times = noisy_times[noisy_times < DURATION - 0.01]
    amps = rng.uniform(70.0, 90.0, size=len(noisy_times))
    t, v = _trace(noisy_times, amps, noise_mv=5.0, seed=2)
    _write("noisy_neuron.csv", t, v)

    # No spikes.
    t, v = _trace([], [], noise_mv=2.0, seed=3)
    _write("no_spikes.csv", t, v)

    # Failing: 8 Hz, 35 mV, jittered ISIs (std 20 ms).
    isi = rng.normal(0.125, 0.020, size=15)
    fail_times = np.concatenate([[0.05], 0.05 + np.cumsum(isi)])
    fail_times = fail_times[fail_times < DURATION - 0.01]
    t, v = _trace(fail_times, [45.0] * len(fail_times), noise_mv=2.0, seed=4)
    _write("failing_neuron.csv", t, v)


if __name__ == "__main__":
    main()
