"""Βασικοί τύποι δεδομένων που ανταλλάσσονται μεταξύ των τμημάτων της εφαρμογής."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    """Μία λήψη BLE advertisement από μία συσκευή."""

    t: float                 # χρόνος λήψης (Unix time, δευτερόλεπτα)
    address: str             # αναγνωριστικό συσκευής (διεύθυνση Bluetooth)
    name: str | None         # όνομα, αν διατίθεται
    rssi: int                # ισχύς λαμβανόμενου σήματος (dBm)
    tx_power: int | None = None  # ισχύς εκπομπής, αν τη δηλώνει η συσκευή


@dataclass(frozen=True)
class SourceError:
    """Σφάλμα από την πηγή μετρήσεων (π.χ. απενεργοποιημένο Bluetooth)."""

    message: str


@dataclass(frozen=True)
class SourceFinished:
    """Η πηγή ολοκλήρωσε (π.χ. τέλος επαναπροβολής)."""

    message: str = ""
