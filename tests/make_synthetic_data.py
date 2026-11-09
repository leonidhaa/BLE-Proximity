"""Δημιουργία ΣΥΝΘΕΤΙΚΩΝ πειραματικών αρχείων για δοκιμή της ανάλυσης.

ΔΕΝ αντικαθιστούν τις πραγματικές μετρήσεις της εργασίας· γράφονται σε
ξεχωριστό φάκελο (π.χ. data/synthetic) για να ελεγχθεί ότι η αλυσίδα
ανάλυσης λειτουργεί πριν από τα πειράματα.

    python -m tests.make_synthetic_data --out data/synthetic
    python -m ble_proximity.analyze --data data/synthetic --out analysis/synthetic --no-write-thresholds
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from ble_proximity.models import Measurement
from ble_proximity.storage import write_measurements

CONDITION_OFFSET = {"los": 0.0, "obstacle": -7.0, "orient90": -3.5}


def generate(out: Path, seed: int = 1, duration: float = 62.0, rate: float = 6.0) -> int:
    rng = random.Random(seed)
    a, n = -58.0, 2.3
    count = 0
    t_base = 1_700_000_000.0
    for cond, off in CONDITION_OFFSET.items():
        for d in (1, 2, 3, 5):
            for rep in (1, 2, 3):
                bias = rng.gauss(0, 1.0)  # μικρή διαφορά μεταξύ επαναλήψεων
                ms, t = [], t_base
                while t - t_base < duration:
                    t += rng.expovariate(rate)
                    rssi = a - 10 * n * math.log10(d) + off + bias + rng.gauss(0, 3.5)
                    if rng.random() < 0.05:
                        rssi -= rng.uniform(5, 12)
                    ms.append(Measurement(t, "00:00:5E:00:53:01", "SyntheticBeacon", int(round(rssi))))
                write_measurements(out / f"{cond}_d{d}m_rep{rep}.csv", ms,
                                   extra={"distance_m": d, "condition": cond, "rep": rep, "note": "synthetic"})
                t_base += 1000
                count += 1
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    print(f"Δημιουργήθηκαν {generate(args.out, args.seed)} συνθετικά αρχεία στο {args.out}")
