"""Ρυθμίσεις, διαδρομές αρχείων, ετικέτες και όρια εγγύτητας."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"   # αρχεία πειραματικών μετρήσεων
SESSIONS_DIR = DATA_DIR / "sessions"         # εξαγωγές συνεδριών από την εφαρμογή
ANALYSIS_DIR = PROJECT_ROOT / "analysis"     # γραφήματα & αποτελέσματα ανάλυσης
THRESHOLDS_FILE = PROJECT_ROOT / "config" / "thresholds.json"

# --- Ετικέτες εγγύτητας -------------------------------------------------------
NEAR = "κοντά"
MEDIUM = "μεσαία απόσταση"
FAR = "μακριά"
NOT_DETECTED = "δεν ανιχνεύεται"
UNKNOWN = "—"

# --- Ετικέτες λειτουργίας αναζήτησης (τάση σήματος) ---------------------------
APPROACHING = "πιθανότατα πλησιάζετε"
RECEDING = "πιθανότατα απομακρύνεστε"
STEADY = "χωρίς σαφή μεταβολή"
TREND_WAIT = "συλλογή δεδομένων…"

# Ονομασίες πειραματικών συνθηκών (για γραφήματα/αναφορά)
CONDITION_LABELS = {
    "los": "Οπτική επαφή",
    "obstacle": "Με εμπόδιο",
    "orient90": "Προσανατολισμός 90°",
    "orient180": "Προσανατολισμός 180°",
}


def condition_label(code: str) -> str:
    return CONDITION_LABELS.get(code, code)


@dataclass
class Settings:
    """Παράμετροι επεξεργασίας που μπορεί να αλλάξει ο χρήστης."""

    smoothing: str = "ema"          # "ema" (εκθετική) ή "ma" (κινητός μέσος όρος)
    ema_alpha: float = 0.2          # συντελεστής εκθετικής εξομάλυνσης (0 < α ≤ 1)
    ma_window: int = 10             # παράθυρο κινητού μέσου όρου (δείγματα)
    lost_timeout_s: float = 5.0     # μετά από τόσα s χωρίς λήψη → «δεν ανιχνεύεται»
    trend_window_s: float = 6.0     # παράθυρο εκτίμησης τάσης (s)
    trend_slope_db_s: float = 0.4   # ελάχιστη κλίση (dB/s) για «πλησιάζετε/απομακρύνεστε»
    trend_min_samples: int = 8      # ελάχιστα δείγματα στο παράθυρο τάσης
    plot_window_s: float = 60.0     # εύρος χρόνου στο γράφημα

    @property
    def smoothing_param(self) -> float:
        return self.ema_alpha if self.smoothing == "ema" else self.ma_window

    def smoothing_description(self) -> str:
        if self.smoothing == "ema":
            return f"EMA a={self.ema_alpha:g}"
        return f"MA N={self.ma_window}"


@dataclass
class Thresholds:
    """Όρια κατάταξης εγγύτητας στο εξομαλυμένο RSSI.

    εξομαλυμένο ≥ near_dbm            → κοντά
    far_dbm ≤ εξομαλυμένο < near_dbm  → μεσαία απόσταση
    εξομαλυμένο < far_dbm             → μακριά

    Οι προεπιλεγμένες τιμές είναι ενδεικτικές· τα τελικά όρια προκύπτουν από
    τις πειραματικές μετρήσεις με `python -m ble_proximity.analyze`.
    """

    near_dbm: float = -60.0
    far_dbm: float = -75.0
    hysteresis_db: float = 2.0
    calibrated: bool = False
    source: str = "προεπιλογή (μη βαθμονομημένα)"
    # Μοντέλο απώλειας διαδρομής: RSSI(d) = A - 10·n·log10(d)
    rssi_at_1m: float | None = None
    path_loss_n: float | None = None

    @classmethod
    def load(cls, path: Path = THRESHOLDS_FILE) -> "Thresholds":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path = THRESHOLDS_FILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def describe(self) -> str:
        return (f"Όρια: κοντά ≥ {self.near_dbm:.1f} dBm · μακριά < {self.far_dbm:.1f} dBm "
                f"· υστέρηση {self.hysteresis_db:g} dB — {self.source}")
