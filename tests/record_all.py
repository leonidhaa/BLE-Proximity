"""Καταγραφή ΟΛΩΝ των ανιχνεύσιμων συσκευών BLE για ορισμένο χρόνο (χωρίς ετικέτα απόστασης).

Χρήσιμο για μετρήσεις σταθερότητας σε ακίνητες συσκευές άγνωστης απόστασης.

    python -m tests.record_all --seconds 60 --out data/static/all_devices.csv
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from ble_proximity.models import Measurement
from ble_proximity.storage import write_measurements


async def run(seconds: float) -> list[Measurement]:
    from bleak import BleakScanner

    out: list[Measurement] = []

    def cb(dev, adv) -> None:
        out.append(Measurement(time.time(), dev.address, adv.local_name or dev.name, adv.rssi, adv.tx_power))

    async with BleakScanner(detection_callback=cb, scanning_mode="active"):
        await asyncio.sleep(seconds)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ms = asyncio.run(run(a.seconds))
    n = write_measurements(a.out, ms)
    print(f"{n} μετρήσεις από {len({m.address for m in ms})} συσκευές -> {a.out}")
