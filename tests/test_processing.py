"""Έλεγχοι μονάδας: python -m unittest discover -s tests -v"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ble_proximity import analyze, storage
from ble_proximity.config import (APPROACHING, FAR, MEDIUM, NEAR, NOT_DETECTED, RECEDING, STEADY,
                                  Settings, Thresholds)
from ble_proximity.models import Measurement
from ble_proximity.processing import (DeviceTracker, ExponentialSmoother, MovingAverage,
                                      ProximityClassifier, TrendDetector, estimate_distance,
                                      linear_slope, smooth_series)


class SmoothingTests(unittest.TestCase):
    def test_moving_average(self):
        ma = MovingAverage(3)
        self.assertEqual([ma.update(x) for x in (3, 6, 9, 12)], [3, 4.5, 6, 9])

    def test_ema(self):
        ema = ExponentialSmoother(0.5)
        self.assertEqual([ema.update(x) for x in (10, 20, 20)], [10, 15, 17.5])
        with self.assertRaises(ValueError):
            ExponentialSmoother(0)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        raw = -70 + rng.normal(0, 4, 2000)
        sm = np.array(smooth_series(raw, "ema", 0.2))
        self.assertLess(sm[50:].std(), raw.std() / 2)

    def test_reset_after_gap(self):
        out = smooth_series([-50, -50, -90], "ema", 0.1, times=[0, 1, 20], gap_s=5)
        self.assertEqual(out[-1], -90)


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.thr = Thresholds(near_dbm=-60, far_dbm=-75, hysteresis_db=4)

    def test_basic_classes(self):
        for v, exp in ((-55, NEAR), (-65, MEDIUM), (-80, FAR)):
            self.assertEqual(ProximityClassifier(self.thr).classify(v), exp)

    def test_hysteresis_prevents_flicker(self):
        clf = ProximityClassifier(self.thr)
        self.assertEqual(clf.classify(-59), NEAR)
        self.assertEqual(clf.classify(-61), NEAR)      # μέσα στη ζώνη υστέρησης
        self.assertEqual(clf.classify(-62.5), MEDIUM)  # πέρα από -60 - 2
        self.assertEqual(clf.classify(-59.5), MEDIUM)  # χρειάζεται ≥ -58 για επιστροφή
        self.assertEqual(clf.classify(-57.5), NEAR)


class TrendTests(unittest.TestCase):
    def run_series(self, slope):
        td = TrendDetector(window_s=6, slope_threshold=0.4, min_samples=8)
        label = None
        for i in range(80):
            t = i * 0.1
            label, _ = td.update(t, -70 + slope * t)
        return label

    def test_directions(self):
        self.assertEqual(self.run_series(+2.0), APPROACHING)
        self.assertEqual(self.run_series(-2.0), RECEDING)
        self.assertEqual(self.run_series(0.0), STEADY)

    def test_linear_slope(self):
        self.assertAlmostEqual(linear_slope([0, 1, 2, 3], [1, 3, 5, 7]), 2.0)


class TrackerTests(unittest.TestCase):
    def test_lost_device_is_not_far(self):
        tr = DeviceTracker("AA", "x", Settings(lost_timeout_s=5), Thresholds())
        for i in range(20):
            tr.process(Measurement(100 + i * 0.2, "AA", "x", -55))
        self.assertEqual(tr.state(104.5).proximity, NEAR)
        st = tr.state(120)
        self.assertFalse(st.detected)
        self.assertEqual(st.proximity, NOT_DETECTED)   # όχι «μακριά»

    def test_reprocess_with_new_settings(self):
        s = Settings()
        tr = DeviceTracker("AA", None, s, Thresholds())
        for i, v in enumerate([-50, -90, -50, -90]):
            tr.process(Measurement(i * 0.1, "AA", None, v))
        s.smoothing, s.ma_window = "ma", 2
        tr.reprocess()
        self.assertEqual(tr.samples[-1].smoothed, -70)

    def test_distance_model(self):
        self.assertAlmostEqual(estimate_distance(-60, -60, 2), 1.0)
        self.assertAlmostEqual(estimate_distance(-80, -60, 2), 10.0)
        self.assertIsNone(estimate_distance(-60, None, 2))


class StorageTests(unittest.TestCase):
    def test_roundtrip(self):
        ms = [Measurement(1000.5 + i, "AA:BB", "Δοκιμή", -60 - i, None) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            storage.write_measurements(p, ms, extra={"distance_m": 1, "condition": "los", "rep": 1})
            back = storage.read_measurements(p)
        self.assertEqual([m.rssi for m in back], [m.rssi for m in ms])
        self.assertEqual(back[0].name, "Δοκιμή")


class AnalysisTests(unittest.TestCase):
    def test_fit_thresholds_separates_classes(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(-55, 2, 300), rng.normal(-68, 2, 300), rng.normal(-82, 2, 300)])
        labels = np.repeat([0, 1, 2], 300)
        near, far, ba = analyze.fit_thresholds(values, labels)
        self.assertTrue(-63 < near < -58, near)
        self.assertTrue(-78 < far < -72, far)
        self.assertGreater(ba, 0.9)

    def test_full_pipeline_on_synthetic_data(self):
        from tests.make_synthetic_data import generate
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "data", Path(d) / "out"
            generate(data)
            code = analyze.main(["--data", str(data), "--out", str(out), "--no-write-thresholds"])
            self.assertEqual(code, 0)
            self.assertTrue((out / "results.md").exists())
            self.assertTrue((out / "confusion_matrices.png").exists())


if __name__ == "__main__":
    unittest.main()
