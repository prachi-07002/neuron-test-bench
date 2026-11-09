"""Unit tests for neuron-test-bench. Synthetic traces have known ground
truth (Gaussian spikes, sigma 0.5 ms -> FWHM 1.18 ms), so metrics are
checked against it within tolerance. Run: python3 -m pytest tests/ -q
or: python3 -m unittest discover -s tests -v
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_test_bench import (  # noqa: E402
    check_pass_fail,
    compute_metrics,
    detect_spikes,
    parse_csv,
)
from neuron_test_bench.report import DEFAULT_SPEC  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
FS = 20_000
SIGMA = 0.0005
FWHM_MS = 2.35482 * SIGMA * 1000  # 1.177 ms


def synthetic(spike_times, amplitudes_mv, noise_mv=1.0, seed=0, duration=2.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / FS)
    v = rng.normal(0.0, noise_mv / 1000.0, size=len(t))
    for st, amp in zip(spike_times, amplitudes_mv):
        v += (amp / 1000.0) * np.exp(-0.5 * ((t - st) / SIGMA) ** 2)
    return t, v


class TestParser(unittest.TestCase):
    def test_reads_good_sample(self):
        t, v = parse_csv(os.path.join(DATA, "good_neuron.csv"))
        self.assertEqual(len(t), 40_000)          # 2 s at 20 kHz
        self.assertAlmostEqual(t[0], 0.0, places=3)
        self.assertAlmostEqual(t[-1], 1.99995, places=4)
        self.assertAlmostEqual(float(np.max(v)), 0.080, delta=0.005)  # ~80 mV

    def test_reads_noisy_and_no_spike_samples(self):
        for name in ("noisy_neuron.csv", "no_spikes.csv", "failing_neuron.csv"):
            t, v = parse_csv(os.path.join(DATA, name))
            self.assertEqual(len(t), 40_000)
            self.assertTrue(np.all(np.diff(t) > 0))

    def test_header_units_ms_mv(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("Time (ms), Voltage (mV)\n0,0\n10,50\n20,0\n")
            path = fh.name
        t, v = parse_csv(path)
        self.assertAlmostEqual(t[1], 0.010)
        self.assertAlmostEqual(v[1], 0.050)

    def test_no_header_seconds_volts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("0.0,0.0\n0.01,0.05\n0.02,0.0\n")
            path = fh.name
        t, v = parse_csv(path)
        self.assertAlmostEqual(t[1], 0.01)
        self.assertAlmostEqual(v[1], 0.05)

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("Time (ms), Voltage (mV)\n")
            path = fh.name
        with self.assertRaises(ValueError):
            parse_csv(path)


class TestDetection(unittest.TestCase):
    def test_known_spike_count(self):
        times = 0.025 + 0.05 * np.arange(40)
        t, v = synthetic(times, [80.0] * 40)
        spikes = detect_spikes(t, v, threshold_v=0.04)
        self.assertEqual(len(spikes), 40)

    def test_refractory_merges_close_crossings(self):
        # Two spikes 1 ms apart with a 5 ms refractory window -> counted once.
        t, v = synthetic([0.5, 0.501], [80.0, 80.0])
        spikes = detect_spikes(t, v, threshold_v=0.04, refractory_s=0.005)
        self.assertEqual(len(spikes), 1)

    def test_no_spikes_detected(self):
        t, v = synthetic([], [], noise_mv=2.0)
        self.assertEqual(detect_spikes(t, v, threshold_v=0.04), [])


class TestMetrics(unittest.TestCase):
    def test_good_trace_matches_ground_truth(self):
        times = 0.025 + 0.05 * np.arange(40)
        t, v = synthetic(times, [80.0] * 40)
        m = compute_metrics(t, v, threshold_v=0.04)
        self.assertEqual(m["spike_count"], 40)
        self.assertAlmostEqual(m["frequency_hz"], 20.0, delta=0.5)
        self.assertAlmostEqual(m["isi_mean_s"] * 1000, 50.0, delta=1.0)
        self.assertAlmostEqual(m["isi_median_s"] * 1000, 50.0, delta=1.0)
        self.assertAlmostEqual(m["amplitude_mean_mv"], 80.0, delta=3.0)
        self.assertAlmostEqual(m["width_mean_ms"], FWHM_MS, delta=0.3)

    def test_jitter_low_for_regular_high_for_jittered(self):
        regular_times = 0.025 + 0.05 * np.arange(40)
        t, v = synthetic(regular_times, [80.0] * 40)
        m_reg = compute_metrics(t, v, threshold_v=0.04)

        rng = np.random.default_rng(7)
        isi = rng.normal(0.05, 0.008, size=39)
        jit_times = np.concatenate([[0.025], 0.025 + np.cumsum(isi)])
        t, v = synthetic(jit_times, [80.0] * len(jit_times), noise_mv=5.0, seed=2)
        m_jit = compute_metrics(t, v, threshold_v=0.04)

        self.assertLess(m_reg["isi_cv"], 0.05)
        self.assertGreater(m_jit["isi_cv"], 0.10)
        self.assertGreater(m_jit["isi_cv"], m_reg["isi_cv"])

    def test_no_spike_trace_fails_gracefully(self):
        t, v = synthetic([], [], noise_mv=2.0)
        m = compute_metrics(t, v, threshold_v=0.04)
        self.assertEqual(m["spike_count"], 0)
        self.assertEqual(m["frequency_hz"], 0.0)
        self.assertTrue(math.isnan(m["isi_mean_s"]))
        self.assertTrue(math.isnan(m["amplitude_mean_mv"]))


class TestPassFail(unittest.TestCase):
    def _metrics_for(self, name):
        t, v = parse_csv(os.path.join(DATA, name))
        return compute_metrics(t, v, threshold_v=0.04)

    def test_good_trace_passes(self):
        passed, checks = check_pass_fail(self._metrics_for("good_neuron.csv"))
        self.assertTrue(passed)
        self.assertTrue(all(c["passed"] for c in checks))

    def test_noisy_trace_passes_or_fails_only_on_jitter(self):
        # Noisy trace is within freq/amplitude spec; only jitter may fail.
        passed, checks = check_pass_fail(self._metrics_for("noisy_neuron.csv"))
        by_name = {c["name"]: c["passed"] for c in checks}
        self.assertTrue(by_name["frequency_hz"])
        self.assertTrue(by_name["amplitude_mv"])
        self.assertFalse(by_name["isi_cv_jitter"])  # CV ~0.16 > 0.15 limit

    def test_failing_trace_fails(self):
        passed, checks = check_pass_fail(self._metrics_for("failing_neuron.csv"))
        self.assertFalse(passed)
        by_name = {c["name"]: c["passed"] for c in checks}
        self.assertFalse(by_name["frequency_hz"])   # 8 Hz < 15 Hz floor
        self.assertFalse(by_name["amplitude_mv"])   # 45 mV < 60 mV floor

    def test_no_spike_trace_fails(self):
        passed, _ = check_pass_fail(self._metrics_for("no_spikes.csv"))
        self.assertFalse(passed)

    def test_spec_is_customisable(self):
        m = self._metrics_for("failing_neuron.csv")
        passed, _ = check_pass_fail(m, spec={"freq_min_hz": 5.0, "freq_max_hz": 10.0,
                                              "amplitude_min_mv": 20.0, "amplitude_max_mv": 50.0,
                                              "isi_cv_max": 0.5, "width_min_ms": 0.1,
                                              "width_max_ms": 5.0, "min_spikes": 5})
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
