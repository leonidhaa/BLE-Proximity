"""Επεξεργασία μετρήσεων RSSI: εξομάλυνση, κατάταξη εγγύτητας, εκτίμηση τάσης.

Η ίδια λογική χρησιμοποιείται από τη γραφική εφαρμογή και από το σενάριο
ανάλυσης, ώστε τα πειραματικά αποτελέσματα να αντιστοιχούν σε αυτό που
βλέπει ο χρήστης.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import (APPROACHING, FAR, MEDIUM, NEAR, NOT_DETECTED, RECEDING,
                     STEADY, TREND_WAIT, UNKNOWN, Settings, Thresholds)
from .models import Measurement


# --------------------------------------------------------------------------- #
# Εξομάλυνση
# --------------------------------------------------------------------------- #
class MovingAverage:
    """Απλός κινητός μέσος όρος των τελευταίων N δειγμάτων."""

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self._buf: deque[float] = deque(maxlen=self.window)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()


class ExponentialSmoother:
    """Εκθετική εξομάλυνση: y[k] = α·x[k] + (1−α)·y[k−1]."""

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("Ο συντελεστής α πρέπει να είναι στο (0, 1].")
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, x: float) -> float:
        x = float(x)
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def make_smoother(method: str, param: float):
    if method == "ema":
        return ExponentialSmoother(float(param))
    if method == "ma":
        return MovingAverage(int(param))
    raise ValueError(f"Άγνωστη μέθοδος εξομάλυνσης: {method}")


def smooth_series(values: Iterable[float], method: str, param: float,
                  times: Sequence[float] | None = None,
                  gap_s: float | None = None) -> list[float]:
    """Εξομαλύνει μια σειρά τιμών. Αν δοθούν χρόνοι και `gap_s`, το φίλτρο
    μηδενίζεται μετά από κενό λήψης μεγαλύτερο του `gap_s`."""
    smoother = make_smoother(method, param)
    out: list[float] = []
    prev_t = None
    for i, v in enumerate(values):
        if times is not None and gap_s is not None:
            t = times[i]
            if prev_t is not None and t - prev_t > gap_s:
                smoother.reset()
            prev_t = t
        out.append(smoother.update(v))
    return out


# --------------------------------------------------------------------------- #
# Κατάταξη εγγύτητας
# --------------------------------------------------------------------------- #
def classify_value(value: float, near_dbm: float, far_dbm: float) -> str:
    """Κατάταξη χωρίς υστέρηση."""
    if value >= near_dbm:
        return NEAR
    if value >= far_dbm:
        return MEDIUM
    return FAR


class ProximityClassifier:
    """Κατάταξη με υστέρηση: για να αλλάξει κατηγορία, η τιμή πρέπει να
    ξεπεράσει το όριο κατά hysteresis/2, ώστε να μην «τρεμοπαίζει» η ένδειξη
    όταν το σήμα κυμαίνεται γύρω από ένα όριο."""

    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.state: str | None = None

    def reset(self) -> None:
        self.state = None

    def classify(self, value: float) -> str:
        t = self.thresholds
        m = t.hysteresis_db / 2.0
        near_thr, far_thr = t.near_dbm, t.far_dbm
        if self.state == NEAR:
            near_thr -= m
        elif self.state == MEDIUM:
            near_thr += m
            far_thr -= m
        elif self.state == FAR:
            far_thr += m
        self.state = classify_value(value, near_thr, far_thr)
        return self.state


# --------------------------------------------------------------------------- #
# Τάση (λειτουργία αναζήτησης)
# --------------------------------------------------------------------------- #
def linear_slope(ts: Sequence[float], ys: Sequence[float]) -> float:
    """Κλίση ευθείας ελαχίστων τετραγώνων (μονάδες y ανά μονάδα t)."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    my = sum(ys) / n
    num = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    den = sum((t - mt) ** 2 for t in ts)
    return num / den if den > 0 else 0.0


class TrendDetector:
    """Εκτιμά την τάση του εξομαλυμένου RSSI στα τελευταία `window_s` δευτερόλεπτα.
    Αυξανόμενο RSSI → πιθανότατα πλησιάζετε, μειούμενο → απομακρύνεστε."""

    def __init__(self, window_s: float, slope_threshold: float, min_samples: int):
        self.window_s = window_s
        self.slope_threshold = slope_threshold
        self.min_samples = min_samples
        self._buf: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, y: float) -> tuple[str, float | None]:
        self._buf.append((t, y))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        span = self._buf[-1][0] - self._buf[0][0]
        if len(self._buf) < self.min_samples or span < 0.5 * self.window_s:
            return TREND_WAIT, None
        t0 = self._buf[0][0]
        slope = linear_slope([p[0] - t0 for p in self._buf], [p[1] for p in self._buf])
        if slope >= self.slope_threshold:
            return APPROACHING, slope
        if slope <= -self.slope_threshold:
            return RECEDING, slope
        return STEADY, slope


# --------------------------------------------------------------------------- #
# Εκτίμηση απόστασης (προαιρετική επέκταση)
# --------------------------------------------------------------------------- #
def estimate_distance(rssi: float, rssi_at_1m: float | None, n: float | None) -> float | None:
    """Μοντέλο log-distance: d = 10^((A − RSSI) / (10·n))."""
    if rssi_at_1m is None or n is None or n <= 0:
        return None
    return 10 ** ((rssi_at_1m - rssi) / (10 * n))


# --------------------------------------------------------------------------- #
# Παρακολούθηση μίας συσκευής
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedSample:
    t: float
    rssi: int
    smoothed: float
    proximity: str
    trend: str
    slope: float | None
    distance_m: float | None
    new_segment: bool = False   # πρώτο δείγμα μετά από διακοπή λήψης


@dataclass
class TrackerState:
    detected: bool
    proximity: str
    trend: str
    rssi: int | None
    smoothed: float | None
    slope: float | None
    distance_m: float | None
    last_seen: float | None
    rate_hz: float
    n_samples: int


class DeviceTracker:
    """Συλλέγει και επεξεργάζεται τις μετρήσεις μίας επιλεγμένης συσκευής."""

    RATE_WINDOW_S = 5.0

    def __init__(self, address: str, name: str | None, settings: Settings, thresholds: Thresholds):
        self.address = address
        self.name = name
        self.settings = settings
        self.thresholds = thresholds
        self.raw: list[Measurement] = []
        self.samples: list[ProcessedSample] = []
        self._init_pipeline()

    # -- pipeline ---------------------------------------------------------------
    def _init_pipeline(self) -> None:
        s = self.settings
        self.smoother = make_smoother(s.smoothing, s.smoothing_param)
        self.classifier = ProximityClassifier(self.thresholds)
        self.trend = TrendDetector(s.trend_window_s, s.trend_slope_db_s, s.trend_min_samples)
        self._last_t: float | None = None

    def _compute(self, m: Measurement) -> ProcessedSample:
        new_segment = False
        if self._last_t is None or m.t - self._last_t > self.settings.lost_timeout_s:
            # νέα λήψη μετά από διακοπή: τα παλιά δεδομένα δεν περιγράφουν την τωρινή θέση
            self.smoother.reset()
            self.classifier.reset()
            self.trend.reset()
            new_segment = self._last_t is not None
        self._last_t = m.t
        sm = self.smoother.update(m.rssi)
        prox = self.classifier.classify(sm)
        trend, slope = self.trend.update(m.t, sm)
        dist = estimate_distance(sm, self.thresholds.rssi_at_1m, self.thresholds.path_loss_n)
        return ProcessedSample(m.t, m.rssi, sm, prox, trend, slope, dist, new_segment)

    def process(self, m: Measurement) -> ProcessedSample:
        if m.name and not self.name:
            self.name = m.name
        self.raw.append(m)
        sample = self._compute(m)
        self.samples.append(sample)
        return sample

    def reprocess(self) -> None:
        """Επανυπολογισμός όλου του ιστορικού (π.χ. μετά από αλλαγή ρυθμίσεων)."""
        self._init_pipeline()
        self.samples = [self._compute(m) for m in self.raw]

    def set_thresholds(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds
        self.reprocess()

    # -- κατάσταση ----------------------------------------------------------------
    @property
    def t0(self) -> float | None:
        return self.samples[0].t if self.samples else None

    def is_detected(self, now: float) -> bool:
        return self._last_t is not None and now - self._last_t <= self.settings.lost_timeout_s

    def rate_hz(self, now: float) -> float:
        cutoff = now - self.RATE_WINDOW_S
        count = 0
        for s in reversed(self.samples):
            if s.t < cutoff:
                break
            count += 1
        return count / self.RATE_WINDOW_S

    def state(self, now: float) -> TrackerState:
        last = self.samples[-1] if self.samples else None
        detected = self.is_detected(now)
        if last is None:
            return TrackerState(False, NOT_DETECTED, UNKNOWN, None, None, None, None, None, 0.0, 0)
        return TrackerState(
            detected=detected,
            # Διακοπή λήψης ≠ «μακριά»: εμφανίζεται ρητά «δεν ανιχνεύεται».
            proximity=last.proximity if detected else NOT_DETECTED,
            trend=last.trend if detected else UNKNOWN,
            rssi=last.rssi,
            smoothed=last.smoothed,
            slope=last.slope if detected else None,
            distance_m=last.distance_m if detected else None,
            last_seen=last.t,
            rate_hz=self.rate_hz(now),
            n_samples=len(self.samples),
        )
