"""Πηγές μετρήσεων: πραγματική σάρωση BLE (bleak), προσομοίωση, επαναπροβολή CSV.

Κάθε πηγή τρέχει σε δικό της νήμα και τοποθετεί αντικείμενα `Measurement`
(ή `SourceError` / `SourceFinished`) σε μια ουρά που διαβάζει η εφαρμογή.
"""

from __future__ import annotations

import asyncio
import math
import queue
import random
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import storage
from .models import Measurement, SourceError, SourceFinished


class _ThreadedSource:
    def __init__(self, out: queue.Queue):
        self.out = out
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:  # pragma: no cover - υλοποιείται από τις υποκλάσεις
        raise NotImplementedError


class BleakSource(_ThreadedSource):
    """Συνεχής σάρωση BLE advertisements μέσω της βιβλιοθήκης bleak."""

    def __init__(self, out: queue.Queue, scanning_mode: str = "active"):
        super().__init__(out)
        self.scanning_mode = scanning_mode

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # π.χ. Bluetooth απενεργοποιημένο / χωρίς προσαρμογέα
            self.out.put(SourceError(f"{type(exc).__name__}: {exc}"))

    async def _main(self) -> None:
        from bleak import BleakScanner

        def on_advertisement(device, adv) -> None:
            self.out.put(Measurement(time.time(), device.address, adv.local_name or device.name,
                                     adv.rssi, adv.tx_power))

        scanner = BleakScanner(detection_callback=on_advertisement, scanning_mode=self.scanning_mode)
        await scanner.start()
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            await scanner.stop()


class SimulatedSource(_ThreadedSource):
    """Συνθετικές μετρήσεις για επίδειξη/δοκιμή χωρίς πραγματική συσκευή.

    Η «Δοκιμαστική συσκευή» πλησιάζει από 6 m σε 0.7 m, μένει κοντά,
    απομακρύνεται και για λίγα δευτερόλεπτα βγαίνει εκτός εμβέλειας.
    Χρησιμοποιούνται διευθύνσεις του εύρους τεκμηρίωσης 00:00:5E:00:53:xx.
    """

    TARGET = "00:00:5E:00:53:01"
    CYCLE_S = 80.0

    @staticmethod
    def target_distance(elapsed: float) -> float | None:
        x = elapsed % SimulatedSource.CYCLE_S
        if x < 25:
            return 6.0 - (6.0 - 0.7) * x / 25
        if x < 35:
            return 0.7
        if x < 62:
            return 0.7 + (6.0 - 0.7) * (x - 35) / 27
        if x < 74:
            return None  # εκτός εμβέλειας → καμία λήψη
        return 6.0

    def _run(self) -> None:
        rng = random.Random()
        a, n = -59.0, 2.2
        statics = [("00:00:5E:00:53:02", "[SIM] Ακουστικά", -71, 0.35),
                   ("00:00:5E:00:53:03", None, -86, 0.2),
                   ("00:00:5E:00:53:04", "[SIM] Smartwatch", -64, 0.3)]
        t0 = time.time()
        while not self._stop.wait(0.1):
            now = time.time()
            d = self.target_distance(now - t0)
            if d is not None and rng.random() < 0.8:
                rssi = a - 10 * n * math.log10(d) + rng.gauss(0, 3.5)
                if rng.random() < 0.05:
                    rssi -= rng.uniform(5, 12)  # στιγμιαία εξασθένηση (fading)
                self.out.put(Measurement(now, self.TARGET, "[SIM] Δοκιμαστική συσκευή",
                                         int(round(max(rssi, -100))), -59))
            for addr, name, base, p in statics:
                if rng.random() < p:
                    self.out.put(Measurement(now, addr, name, int(round(base + rng.gauss(0, 3))), None))


class ReplaySource(_ThreadedSource):
    """Επαναπροβολή αποθηκευμένης συνεδρίας με τον αρχικό ρυθμό (× speed)."""

    def __init__(self, out: queue.Queue, path: Path, speed: float = 1.0):
        super().__init__(out)
        self.path = Path(path)
        self.speed = max(0.1, float(speed))

    def _run(self) -> None:
        try:
            ms = storage.read_measurements(self.path)
        except Exception as exc:
            self.out.put(SourceError(f"Αδυναμία ανάγνωσης {self.path.name}: {exc}"))
            return
        if not ms:
            self.out.put(SourceError(f"Το αρχείο {self.path.name} δεν περιέχει μετρήσεις."))
            return
        t_first = ms[0].t
        start = time.time()
        for m in ms:
            delay = start + (m.t - t_first) / self.speed - time.time()
            if delay > 0 and self._stop.wait(delay):
                return
            if self._stop.is_set():
                return
            # Νέα χρονοσφραγίδα, ώστε ο έλεγχος «δεν ανιχνεύεται» να λειτουργεί κανονικά.
            self.out.put(replace(m, t=time.time()))
        self.out.put(SourceFinished("Η επαναπροβολή ολοκληρώθηκε."))
