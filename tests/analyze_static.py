"""Στατιστικά σταθερότητας για καταγραφή ακίνητων συσκευών (από tests/record_all.py).

    python -m tests.analyze_static data/static/static_60s.csv --out analysis/static
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ble_proximity.processing import smooth_series  # noqa: E402
from ble_proximity.storage import read_measurements  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-samples", type=int, default=30)
    a = p.parse_args()
    ms = read_measurements(a.csv)
    t_start, t_end = ms[0].t, ms[-1].t
    by_dev = defaultdict(list)
    names = {}
    for m in ms:
        by_dev[m.address].append(m)
        if m.name:
            names[m.address] = m.name
    rows = []
    for addr, lst in by_dev.items():
        t = np.array([m.t for m in lst])
        r = np.array([m.rssi for m in lst], dtype=float)
        ema = np.array(smooth_series(r, "ema", 0.2))
        ma = np.array(smooth_series(r, "ma", 10))
        warm = (t - t[0]) >= 5
        rows.append({"address": addr, "name": names.get(addr), "n": len(lst),
                     "rate": len(lst) / (t_end - t_start), "mean": float(r.mean()),
                     "median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()),
                     "std_raw": float(r.std()),
                     "std_ema": float(ema[warm].std()) if warm.sum() > 2 else float("nan"),
                     "std_ma": float(ma[warm].std()) if warm.sum() > 2 else float("nan"),
                     "max_gap": float(np.diff(t).max()) if len(t) > 1 else float("nan"),
                     "t": t - t_start, "rssi": r, "ema": ema, "ma": ma})
    rows.sort(key=lambda d: -d["n"])
    a.out.mkdir(parents=True, exist_ok=True)

    plot = [d for d in rows if d["n"] >= a.min_samples][:4]
    fig, axes = plt.subplots(len(plot), 1, figsize=(10, 2.6 * len(plot)), sharex=True, squeeze=False)
    for ax, d in zip(axes[:, 0], plot):
        ax.plot(d["t"], d["rssi"], ".", ms=4, color="#9e9e9e", label="Αρχικό RSSI")
        ax.plot(d["t"], d["ma"], "-", lw=1.4, color="#ef6c00", label="Κινητός μέσος N=10")
        ax.plot(d["t"], d["ema"], "-", lw=2, color="#1565c0", label="EMA α=0.2")
        ax.set_title(f"{d['name'] or '(χωρίς όνομα)'} [{d['address']}] — {d['n']} πακέτα, "
                     f"Τ.Α. αρχικού {d['std_raw']:.1f} dB → EMA {d['std_ema']:.1f} dB", fontsize=9)
        ax.set_ylabel("RSSI (dBm)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=3, loc="lower right")
    axes[-1, 0].set_xlabel("Χρόνος (s)")
    fig.suptitle("Πραγματική καταγραφή 60 s — ακίνητες συσκευές")
    fig.tight_layout()
    fig.savefig(a.out / "static_raw_vs_smoothed.png", dpi=130)
    plt.close(fig)

    summary = [{k: v for k, v in d.items() if k not in ("t", "rssi", "ema", "ma")} for d in rows]
    (a.out / "static_summary.json").write_text(json.dumps(
        {"file": str(a.csv), "duration_s": t_end - t_start, "devices": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for d in summary:
        print(f"{d['address']}  {str(d['name'])[:24]:<24} n={d['n']:4d} {d['rate']:.2f}/s "
              f"mean={d['mean']:.1f} std={d['std_raw']:.2f} ema={d['std_ema']:.2f} ma={d['std_ma']:.2f} "
              f"gap={d['max_gap']:.1f}")


if __name__ == "__main__":
    main()
