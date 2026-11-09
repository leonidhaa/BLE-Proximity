"""Ηχητική ένδειξη αναζήτησης (προαιρετική επέκταση).

Όσο πιο κοντά η συσκευή, τόσο πιο συχνά και πιο οξέα «μπιπ»· όταν η τάση
δείχνει ότι πλησιάζετε, ο τόνος ανεβαίνει λίγο.
"""

from __future__ import annotations

import sys
import threading

from .config import APPROACHING, FAR, MEDIUM, NEAR, RECEDING

try:
    import winsound
except ImportError:  # εκτός Windows
    winsound = None


class Beeper:
    LEVELS = {NEAR: (1400, 0.35), MEDIUM: (1000, 0.9), FAR: (700, 1.8)}  # (Hz, διάστημα s)

    def __init__(self) -> None:
        self.enabled = False
        self._proximity: str | None = None
        self._trend: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="Beeper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def set_state(self, proximity: str | None, trend: str | None) -> None:
        self._proximity, self._trend = proximity, trend

    def _beep(self, freq: int, ms: int) -> None:
        if winsound is not None:
            winsound.Beep(freq, ms)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            level = self.LEVELS.get(self._proximity) if self.enabled else None
            if level is None:
                self._stop.wait(0.2)
                continue
            freq, interval = level
            if self._trend == APPROACHING:
                freq += 250
            elif self._trend == RECEDING:
                freq -= 150
            try:
                self._beep(freq, 70)
            except RuntimeError:
                pass
            self._stop.wait(interval)
