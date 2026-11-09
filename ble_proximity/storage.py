"""Αποθήκευση και ανάγνωση μετρήσεων σε αρχεία CSV."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from .config import Settings
from .models import Measurement

RAW_FIELDS = ["t_unix", "time_iso", "address", "name", "rssi", "tx_power"]
SESSION_FIELDS = ["t_unix", "time_iso", "address", "name", "rssi", "rssi_smoothed",
                  "proximity", "trend", "slope_db_s", "distance_m", "smoothing"]

# utf-8-sig: τα ελληνικά εμφανίζονται σωστά και όταν το αρχείο ανοίγει στο Excel
ENCODING = "utf-8-sig"


def iso_time(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def _fmt(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def write_measurements(path: Path, measurements: Iterable[Measurement],
                       extra: Mapping[str, object] | None = None) -> int:
    """Γράφει αρχικές μετρήσεις. Το `extra` προσθέτει σταθερές στήλες
    (π.χ. απόσταση/συνθήκη/επανάληψη σε πειραματικές καταγραφές)."""
    extra = dict(extra or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding=ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS + list(extra))
        w.writeheader()
        for m in measurements:
            w.writerow({"t_unix": f"{m.t:.3f}", "time_iso": iso_time(m.t), "address": m.address,
                        "name": m.name or "", "rssi": m.rssi,
                        "tx_power": "" if m.tx_power is None else m.tx_power, **extra})
            n += 1
    return n


def write_session(path: Path, tracker, settings: Settings) -> int:
    """Εξάγει τη συνεδρία παρακολούθησης μίας συσκευής (αρχικές + επεξεργασμένες τιμές)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = settings.smoothing_description()
    with path.open("w", newline="", encoding=ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        w.writeheader()
        for s in tracker.samples:
            w.writerow({"t_unix": f"{s.t:.3f}", "time_iso": iso_time(s.t), "address": tracker.address,
                        "name": tracker.name or "", "rssi": s.rssi, "rssi_smoothed": _fmt(s.smoothed),
                        "proximity": s.proximity, "trend": s.trend, "slope_db_s": _fmt(s.slope, 3),
                        "distance_m": _fmt(s.distance_m), "smoothing": desc})
    return len(tracker.samples)


def read_measurements(path: Path) -> list[Measurement]:
    """Διαβάζει μετρήσεις από οποιοδήποτε CSV της εφαρμογής (αρχικές, συνεδρία ή πείραμα)."""
    out: list[Measurement] = []
    with Path(path).open(newline="", encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        missing = {"t_unix", "address", "rssi"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Λείπουν στήλες από το CSV: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                tx = row.get("tx_power") or ""
                out.append(Measurement(float(row["t_unix"]), row["address"], row.get("name") or None,
                                       int(float(row["rssi"])), int(float(tx)) if tx else None))
            except (TypeError, ValueError):
                continue  # παράλειψη κατεστραμμένων γραμμών
    out.sort(key=lambda m: m.t)
    return out
