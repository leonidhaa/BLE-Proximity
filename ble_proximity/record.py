"""Καταγραφή πειραματικών μετρήσεων από τη γραμμή εντολών.

Παραδείγματα:
    # 1) Εύρεση της συσκευής δοκιμών
    python -m ble_proximity.record --list

    # 2) Καταγραφή 60 s σε απόσταση 1 m, με οπτική επαφή, 1η επανάληψη
    python -m ble_proximity.record --target "MyBeacon" --distance 1 --condition los --rep 1

Το αρχείο αποθηκεύεται στο data/experiments/<συνθήκη>_d<απόσταση>m_rep<επανάληψη>.csv
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

from .config import CONDITION_LABELS, EXPERIMENTS_DIR
from .models import Measurement
from .storage import write_measurements

MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


async def list_devices(seconds: float) -> None:
    from bleak import BleakScanner

    seen: dict[str, dict] = {}

    def cb(dev, adv) -> None:
        e = seen.setdefault(dev.address, {"name": None, "rssi": []})
        name = adv.local_name or dev.name
        if name:
            e["name"] = name
        e["rssi"].append(adv.rssi)

    print(f"Σάρωση BLE για {seconds:g} s…")
    async with BleakScanner(detection_callback=cb, scanning_mode="active"):
        await asyncio.sleep(seconds)
    if not seen:
        print("Δεν βρέθηκαν συσκευές. Είναι ενεργό το Bluetooth;")
        return
    rows = sorted(seen.items(), key=lambda kv: -statistics.median(kv[1]["rssi"]))
    print(f"\n{'Διεύθυνση':<20}{'RSSI(διάμ.)':>12}{'Πακέτα':>8}{'Πακ./s':>8}   Όνομα")
    print("-" * 78)
    for addr, e in rows:
        print(f"{addr:<20}{statistics.median(e['rssi']):>12.0f}{len(e['rssi']):>8}"
              f"{len(e['rssi']) / seconds:>8.1f}   {e['name'] or '(χωρίς όνομα)'}")
    print("\nΧρησιμοποιήστε τη διεύθυνση ή μέρος του ονόματος στο --target.")


async def record(target: str, duration: float, countdown: int) -> tuple[list[Measurement], set[str]]:
    from bleak import BleakScanner

    by_address = bool(MAC_RE.match(target))
    wanted = target.upper().replace("-", ":")
    out: list[Measurement] = []
    addresses: set[str] = set()
    state = {"recording": False}

    def cb(dev, adv) -> None:
        if not state["recording"]:
            return
        name = adv.local_name or dev.name
        if by_address:
            ok = dev.address.upper() == wanted
        else:
            ok = name is not None and target.lower() in name.lower()
        if ok:
            out.append(Measurement(time.time(), dev.address, name, adv.rssi, adv.tx_power))
            addresses.add(dev.address)

    async with BleakScanner(detection_callback=cb, scanning_mode="active"):
        for i in range(countdown, 0, -1):
            print(f"Έναρξη σε {i}…", flush=True)
            await asyncio.sleep(1)
        state["recording"] = True
        t0 = time.time()
        print("Καταγραφή — μην μετακινείτε τη συσκευή.")
        while (elapsed := time.time() - t0) < duration:
            await asyncio.sleep(min(1.0, duration - elapsed))
            last = f"{out[-1].rssi} dBm" if out else "—"
            print(f"\r  {min(time.time() - t0, duration):5.0f}/{duration:g} s | δείγματα: {len(out):5d}"
                  f" | τελευταίο RSSI: {last:>8}   ", end="", flush=True)
        state["recording"] = False
    print()
    end = t0 + duration
    return [m for m in out if m.t <= end], addresses


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    p = argparse.ArgumentParser(prog="python -m ble_proximity.record",
                                description="Καταγραφή πειραματικών μετρήσεων RSSI μίας συσκευής BLE.")
    p.add_argument("--list", action="store_true", help="σάρωση και εμφάνιση των κοντινών συσκευών")
    p.add_argument("--seconds", type=float, default=10, help="διάρκεια σάρωσης για --list (s)")
    p.add_argument("--target", help="διεύθυνση (AA:BB:..) ή μέρος του ονόματος της συσκευής")
    p.add_argument("--distance", type=float, help="πραγματική απόσταση σε μέτρα")
    p.add_argument("--condition", help="συνθήκη: " + ", ".join(f"{k} ({v})" for k, v in CONDITION_LABELS.items()))
    p.add_argument("--rep", type=int, help="αριθμός επανάληψης (1, 2, 3, …)")
    p.add_argument("--duration", type=float, default=60, help="διάρκεια καταγραφής σε s (προεπ. 60)")
    p.add_argument("--countdown", type=int, default=5, help="αντίστροφη μέτρηση πριν την έναρξη (s)")
    p.add_argument("--note", default="", help="προαιρετική σημείωση (π.χ. «τοίχος γυψοσανίδας»)")
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR, help="φάκελος αποθήκευσης")
    p.add_argument("--force", action="store_true", help="αντικατάσταση υπάρχοντος αρχείου")
    a = p.parse_args(argv)

    if a.list:
        asyncio.run(list_devices(a.seconds))
        return 0

    missing = [opt for opt, v in (("--target", a.target), ("--distance", a.distance),
                                   ("--condition", a.condition), ("--rep", a.rep)) if v is None]
    if missing:
        p.error("λείπουν τα ορίσματα: " + ", ".join(missing))
    if a.distance <= 0:
        p.error("η απόσταση πρέπει να είναι θετική")
    if a.duration < 60:
        print("Προσοχή: η εκφώνηση ζητά τουλάχιστον 60 s ανά συνθήκη.")

    condition = re.sub(r"[^A-Za-z0-9_-]", "_", a.condition)
    path = a.out / f"{condition}_d{a.distance:g}m_rep{a.rep}.csv"
    if path.exists() and not a.force:
        print(f"Το αρχείο {path} υπάρχει ήδη. Χρησιμοποιήστε --force για αντικατάσταση.")
        return 1

    print(f"Συσκευή: {a.target} | απόσταση {a.distance:g} m | συνθήκη {condition} | επανάληψη {a.rep}")
    ms, addresses = asyncio.run(record(a.target, a.duration, a.countdown))
    if not ms:
        print("Δεν λήφθηκε καμία μέτρηση από τη συσκευή. Ελέγξτε το --target "
              "(δείτε: python -m ble_proximity.record --list).")
        return 2
    if len(addresses) > 1:
        print(f"Προσοχή: ταίριαξαν {len(addresses)} διευθύνσεις ({', '.join(sorted(addresses))}). "
              "Ίσως η συσκευή αλλάζει διεύθυνση ή υπάρχουν πολλές με παρόμοιο όνομα.")

    write_measurements(path, ms, extra={"distance_m": f"{a.distance:g}", "condition": condition,
                                        "rep": a.rep, "note": a.note})
    rssi = [m.rssi for m in ms]
    dur = ms[-1].t - ms[0].t if len(ms) > 1 else 0
    print(f"Αποθηκεύτηκε: {path}")
    print(f"  δείγματα {len(ms)} · {len(ms) / a.duration:.1f} πακέτα/s · μέσο {statistics.mean(rssi):.1f} dBm"
          f" · διάμεσος {statistics.median(rssi):.0f} dBm"
          f" · τυπ. απόκλιση {statistics.pstdev(rssi):.1f} dB · κάλυψη {dur:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
