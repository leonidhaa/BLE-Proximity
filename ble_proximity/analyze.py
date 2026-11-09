"""Ανάλυση πειραματικών μετρήσεων.

1. Φορτώνει τα αρχεία του data/experiments (από `ble_proximity.record`).
2. Εφαρμόζει την ίδια εξομάλυνση με την εφαρμογή.
3. Προσδιορίζει τα όρια «κοντά/μεσαία/μακριά» ΜΟΝΟ από τις επαναλήψεις ρύθμισης
   (προεπ. 1 και 2), μεγιστοποιώντας την ισοσταθμισμένη ακρίβεια (balanced accuracy).
4. Αξιολογεί σε ΔΙΑΦΟΡΕΤΙΚΕΣ επαναλήψεις (προεπ. 3): ορθότητα & σταθερότητα,
   αρχικές έναντι εξομαλυμένων μετρήσεων, ανά συνθήκη.
5. Προαιρετικά: μοντέλο απόστασης log-distance και σύγκριση με ταξινομητή ML.
6. Γράφει γραφήματα + analysis/results.md και αποθηκεύει τα όρια στο
   config/thresholds.json, ώστε να τα χρησιμοποιεί η εφαρμογή.

Εκτέλεση:  python -m ble_proximity.analyze
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import (ANALYSIS_DIR, EXPERIMENTS_DIR, FAR, MEDIUM, NEAR, THRESHOLDS_FILE,  # noqa: E402
                     Thresholds, condition_label)
from .processing import ProximityClassifier, classify_value, smooth_series  # noqa: E402

CLASSES = [NEAR, MEDIUM, FAR]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


# =========================================================================== δεδομένα
@dataclass
class Run:
    path: Path
    condition: str
    distance: float
    rep: int
    t: np.ndarray
    rssi: np.ndarray
    smoothed: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def rate(self) -> float:
        return (len(self.t) - 1) / self.duration if self.duration > 0 else float("nan")

    @property
    def label(self) -> str:
        return f"{condition_label(self.condition)}, {self.distance:g} m, επαν. {self.rep}"


def load_runs(data_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        with p.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        need = {"t_unix", "rssi", "distance_m", "condition", "rep"}
        if not rows or not need <= rows[0].keys():
            print(f"  παράλειψη {p.name}: δεν είναι αρχείο πειράματος")
            continue
        t = np.array([float(r["t_unix"]) for r in rows])
        rssi = np.array([float(r["rssi"]) for r in rows])
        order = np.argsort(t)
        runs.append(Run(p, rows[0]["condition"], float(rows[0]["distance_m"]), int(rows[0]["rep"]),
                        t[order], rssi[order]))
    return runs


def true_class(distance: float, near_max: float, medium_max: float) -> int:
    if distance <= near_max:
        return CLASS_IDX[NEAR]
    if distance <= medium_max:
        return CLASS_IDX[MEDIUM]
    return CLASS_IDX[FAR]


def after_warmup(run: Run, warmup_s: float) -> np.ndarray:
    return (run.t - run.t[0]) >= warmup_s


# =========================================================================== μετρικές
def classify_array(values: np.ndarray, near: float, far: float) -> np.ndarray:
    return np.where(values >= near, 0, np.where(values >= far, 1, 2))


def classify_sequence(values: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Κατάταξη με υστέρηση — ακριβώς όπως στην εφαρμογή."""
    clf = ProximityClassifier(thresholds)
    return np.array([CLASS_IDX[clf.classify(float(v))] for v in values], dtype=int)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) == 0:
        return float("nan")
    return float(np.mean([np.mean(p[y == c] == c) for c in classes]))


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    np.add.at(m, (y, p), 1)
    return m


def switches_per_minute(pred: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0 or len(pred) < 2:
        return float("nan")
    return float(np.count_nonzero(np.diff(pred))) / (duration_s / 60.0)


def fit_thresholds(values: np.ndarray, labels: np.ndarray, step: float = 0.5) -> tuple[float, float, float]:
    """Εξαντλητική αναζήτηση (near, far) με far < near που μεγιστοποιεί τη
    balanced accuracy. Σε ισοβαθμία επιλέγεται το κέντρο της βέλτιστης περιοχής."""
    lo = np.floor(values.min()) - step
    hi = np.ceil(values.max()) + step
    grid = np.arange(lo, hi + step / 2, step)
    best = -1.0
    best_pairs: list[tuple[float, float]] = []
    for i, near in enumerate(grid):
        is_near = values >= near
        for far in grid[:i]:
            pred = np.where(is_near, 0, np.where(values >= far, 1, 2))
            ba = balanced_accuracy(labels, pred)
            if ba > best + 1e-12:
                best, best_pairs = ba, [(near, far)]
            elif abs(ba - best) <= 1e-12:
                best_pairs.append((near, far))
    near = float(np.median([p[0] for p in best_pairs]))
    far = float(np.median([p[1] for p in best_pairs]))
    if far >= near:  # ασφάλεια: διατήρηση έγκυρης διάταξης
        near, far = best_pairs[len(best_pairs) // 2]
    return round(near * 2) / 2, round(far * 2) / 2, best


def evaluate(runs: list[Run], thresholds: Thresholds, near_max: float, medium_max: float,
             warmup_s: float) -> dict:
    ys, pr, ps = [], [], []
    per_run, per_cond = [], {}
    for r in runs:
        mask = after_warmup(r, warmup_s)
        if mask.sum() < 2:
            continue
        y = np.full(mask.sum(), true_class(r.distance, near_max, medium_max))
        p_raw = classify_sequence(r.rssi, thresholds)[mask]
        p_sm = classify_sequence(r.smoothed, thresholds)[mask]
        dur = float(r.t[mask][-1] - r.t[mask][0])
        ys.append(y), pr.append(p_raw), ps.append(p_sm)
        item = {"run": r.label, "condition": r.condition, "distance": r.distance, "rep": r.rep,
                "acc_raw": float(np.mean(p_raw == y)), "acc_smoothed": float(np.mean(p_sm == y)),
                "switches_raw": switches_per_minute(p_raw, dur),
                "switches_smoothed": switches_per_minute(p_sm, dur)}
        per_run.append(item)
        c = per_cond.setdefault(r.condition, {"y": [], "raw": [], "sm": []})
        c["y"].append(y), c["raw"].append(p_raw), c["sm"].append(p_sm)
    if not ys:
        return {}
    y, p_raw, p_sm = np.concatenate(ys), np.concatenate(pr), np.concatenate(ps)

    def summary(p: np.ndarray, key: str) -> dict:
        return {"accuracy": float(np.mean(p == y)), "balanced_accuracy": balanced_accuracy(y, p),
                "confusion": confusion(y, p).tolist(),
                "switches_per_min": float(np.nanmean([it[f"switches_{key}"] for it in per_run]))}

    cond_summary = {}
    for cond, c in per_cond.items():
        cy = np.concatenate(c["y"])
        cond_summary[cond] = {"raw": float(np.mean(np.concatenate(c["raw"]) == cy)),
                              "smoothed": float(np.mean(np.concatenate(c["sm"]) == cy))}
    return {"raw": summary(p_raw, "raw"), "smoothed": summary(p_sm, "smoothed"),
            "per_condition": cond_summary, "per_run": per_run, "n_samples": int(len(y))}


# ================================================================ μοντέλο απόστασης
def fit_distance_model(runs: list[Run]) -> tuple[float, float] | None:
    """RSSI(d) = A − 10·n·log10(d), προσαρμογή στις διαμέσους κάθε εκτέλεσης."""
    if len({r.distance for r in runs}) < 2:
        return None
    x = np.log10([r.distance for r in runs])
    y = np.array([np.median(r.smoothed) for r in runs])
    slope, intercept = np.polyfit(x, y, 1)
    n = -slope / 10.0
    if n <= 0:
        return None
    return float(intercept), float(n)


def distance_errors(runs: list[Run], a: float, n: float, warmup_s: float) -> dict:
    errs, rows = [], []
    for r in runs:
        sm = r.smoothed[after_warmup(r, warmup_s)]
        d_hat = 10 ** ((a - sm) / (10 * n))
        errs.append(np.abs(d_hat - r.distance))
        rows.append({"run": r.label, "distance": r.distance, "median_estimate": float(np.median(d_hat))})
    e = np.concatenate(errs) if errs else np.array([])
    return {"mae_m": float(e.mean()) if len(e) else float("nan"),
            "median_ae_m": float(np.median(e)) if len(e) else float("nan"), "per_run": rows}


# ============================================================== μηχανική μάθηση
def window_features(r: Run, win: float, step: float):
    tt = r.t - r.t[0]
    feats, ends = [], []
    e = win
    while e <= tt[-1] + 1e-9:
        m = (tt > e - win) & (tt <= e)
        if m.sum() >= 3:
            v = r.rssi[m]
            feats.append([v.mean(), np.median(v), v.std(), v.min(), v.max(), m.sum() / win])
            ends.append(int(np.flatnonzero(m)[-1]))
        e += step
    return feats, ends


def ml_comparison(cal: list[Run], test: list[Run], thr: Thresholds, near_max: float,
                  medium_max: float, win: float = 3.0, step: float = 1.0) -> dict | None:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None
    Xc, yc = [], []
    for r in cal:
        f, _ = window_features(r, win, step)
        Xc += f
        yc += [true_class(r.distance, near_max, medium_max)] * len(f)
    Xt, yt, base = [], [], []
    for r in test:
        f, ends = window_features(r, win, step)
        Xt += f
        yt += [true_class(r.distance, near_max, medium_max)] * len(f)
        base += [CLASS_IDX[classify_value(r.smoothed[i], thr.near_dbm, thr.far_dbm)] for i in ends]
    if len(set(yc)) < 2 or not Xt:
        return None
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0)
    clf.fit(np.array(Xc), np.array(yc))
    p = clf.predict(np.array(Xt))
    yt_a, base_a = np.array(yt), np.array(base)
    return {"window_s": win, "n_train": len(Xc), "n_test": len(Xt),
            "ml_accuracy": float(np.mean(p == yt_a)), "ml_balanced_accuracy": balanced_accuracy(yt_a, p),
            "baseline_accuracy": float(np.mean(base_a == yt_a)),
            "baseline_balanced_accuracy": balanced_accuracy(yt_a, base_a),
            "features": ["μέσος", "διάμεσος", "τυπ. απόκλιση", "ελάχιστο", "μέγιστο", "πακέτα/s"]}


# ===================================================================== γραφήματα
def plot_rssi_by_distance(runs: list[Run], out: Path) -> None:
    conds = sorted({r.condition for r in runs})
    dists = sorted({r.distance for r in runs})
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plt.cm.tab10.colors
    for ci, cond in enumerate(conds):
        data, pos = [], []
        for di, d in enumerate(dists):
            vals = np.concatenate([r.rssi for r in runs if r.condition == cond and r.distance == d] or [np.array([])])
            if len(vals):
                data.append(vals)
                pos.append(di + (ci - (len(conds) - 1) / 2) * width)
        if data:
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=colors[ci % 10], alpha=0.6)
            ax.plot([], [], "s", color=colors[ci % 10], alpha=0.6, label=condition_label(cond))
    ax.set_xticks(range(len(dists)), [f"{d:g} m" for d in dists])
    ax.set_xlabel("Απόσταση")
    ax.set_ylabel("Αρχικό RSSI (dBm)")
    ax.set_title("Κατανομή RSSI ανά απόσταση και συνθήκη (όλες οι επαναλήψεις)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_raw_vs_smoothed(runs: list[Run], out: Path, smoothing: str) -> None:
    cond = "los" if any(r.condition == "los" for r in runs) else runs[0].condition
    chosen = []
    for d in sorted({r.distance for r in runs if r.condition == cond}):
        cands = sorted([r for r in runs if r.condition == cond and r.distance == d], key=lambda r: -r.rep)
        chosen.append(cands[0])
    chosen = chosen[:4]
    cols = 2 if len(chosen) > 1 else 1
    rows = (len(chosen) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False, sharey=True)
    for ax, r in zip(axes.flat, chosen):
        tt = r.t - r.t[0]
        ax.plot(tt, r.rssi, ".", ms=3, color="#9e9e9e", label="Αρχικό")
        ax.plot(tt, r.smoothed, "-", lw=2, color="#1565c0", label=f"Εξομαλυμένο ({smoothing})")
        ax.set_title(r.label, fontsize=10)
        ax.set_xlabel("Χρόνος (s)")
        ax.set_ylabel("RSSI (dBm)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Αρχικές έναντι εξομαλυμένων μετρήσεων")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_std(runs: list[Run], out: Path, warmup_s: float) -> None:
    conds = sorted({r.condition for r in runs})
    raw = [np.mean([r.rssi[after_warmup(r, warmup_s)].std() for r in runs if r.condition == c]) for c in conds]
    sm = [np.mean([r.smoothed[after_warmup(r, warmup_s)].std() for r in runs if r.condition == c]) for c in conds]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, raw, 0.4, label="Αρχικό", color="#9e9e9e")
    ax.bar(x + 0.2, sm, 0.4, label="Εξομαλυμένο", color="#1565c0")
    ax.set_xticks(x, [condition_label(c) for c in conds])
    ax.set_ylabel("Μέση τυπική απόκλιση εντός εκτέλεσης (dB)")
    ax.set_title("Διακύμανση RSSI σε σταθερή απόσταση")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusions(ev: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, title in ((axes[0], "raw", "Αρχικό RSSI"), (axes[1], "smoothed", "Εξομαλυμένο RSSI")):
        m = np.array(ev[key]["confusion"])
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.0%}\n({m[i, j]})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Ένδειξη εφαρμογής")
        ax.set_ylabel("Πραγματική κατηγορία")
        ax.set_title(f"{title} — ακρίβεια {ev[key]['accuracy']:.0%}")
    fig.suptitle("Πίνακες σύγχυσης στις επαναλήψεις αξιολόγησης")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_threshold_fit(values: np.ndarray, labels: np.ndarray, thr: Thresholds, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(np.floor(values.min()) - 1, np.ceil(values.max()) + 2, 1)
    colors = {0: "#2e7d32", 1: "#ef6c00", 2: "#c62828"}
    for c in range(3):
        v = values[labels == c]
        if len(v):
            ax.hist(v, bins=bins, alpha=0.5, color=colors[c], label=f"Πραγματικά «{CLASSES[c]}»", density=True)
    ax.axvline(thr.near_dbm, color=colors[0], ls="--", lw=2, label=f"Όριο κοντά = {thr.near_dbm:g} dBm")
    ax.axvline(thr.far_dbm, color=colors[2], ls="--", lw=2, label=f"Όριο μακριά = {thr.far_dbm:g} dBm")
    ax.set_xlabel("Εξομαλυμένο RSSI (dBm)")
    ax.set_ylabel("Πυκνότητα")
    ax.set_title("Προσδιορισμός ορίων από τις επαναλήψεις ρύθμισης")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distance_model(fit_runs: list[Run], test_runs: list[Run], a: float, n: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r.distance for r in fit_runs], [np.median(r.smoothed) for r in fit_runs], "o",
            label="Ρύθμιση (διάμεσος ανά εκτέλεση)")
    ax.plot([r.distance for r in test_runs], [np.median(r.smoothed) for r in test_runs], "x",
            label="Αξιολόγηση (διάμεσος ανά εκτέλεση)")
    d = np.linspace(0.3, max(r.distance for r in fit_runs + test_runs) * 1.2, 200)
    ax.plot(d, a - 10 * n * np.log10(d), "-", label=f"RSSI = {a:.1f} − 10·{n:.2f}·log10(d)")
    ax.set_xlabel("Απόσταση (m)")
    ax.set_ylabel("Εξομαλυμένο RSSI (dBm)")
    ax.set_title("Μοντέλο απώλειας διαδρομής (log-distance)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ===================================================================== αναφορά
def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return "—" if x != x else f"{x:.1%}"


def write_report(path: Path, ctx: dict) -> None:
    runs: list[Run] = ctx["runs"]
    a = ctx["args"]
    thr: Thresholds = ctx["thresholds"]
    ev, ev_cal = ctx["eval"], ctx["eval_cal"]
    L = [f"# Αποτελέσματα πειραματικής αξιολόγησης",
         "", f"Δημιουργήθηκε: {datetime.now():%Y-%m-%d %H:%M} · δεδομένα: `{a.data}`",
         "", f"Εξομάλυνση: **{ctx['smoothing']}** · προθέρμανση φίλτρου: {a.warmup:g} s · "
             f"υστέρηση: {thr.hysteresis_db:g} dB",
         f"Πραγματικές κατηγορίες: κοντά ≤ {a.near_max:g} m, μεσαία ≤ {a.medium_max:g} m, μακριά > {a.medium_max:g} m",
         f"Επαναλήψεις ρύθμισης ορίων: {sorted(ctx['cal_reps'])} · επαναλήψεις αξιολόγησης: {sorted(ctx['test_reps'])}",
         ""]
    if ctx["warnings"]:
        L += ["## Έλεγχοι πληρότητας", ""] + [f"- ⚠ {w}" for w in ctx["warnings"]] + [""]

    L += ["## 1. Περιγραφικά στατιστικά ανά εκτέλεση", "",
          md_table(["Συνθήκη", "Απόσταση", "Επαν.", "Δείγματα", "Διάρκεια (s)", "Πακέτα/s",
                    "Μέσο RSSI", "Διάμεσος", "Τ.Α. αρχικού", "Τ.Α. εξομαλ."],
                   [[condition_label(r.condition), f"{r.distance:g} m", r.rep, len(r.t), f"{r.duration:.0f}",
                     f"{r.rate:.1f}", f"{r.rssi.mean():.1f}", f"{np.median(r.rssi):.0f}",
                     f"{r.rssi.std():.2f}", f"{r.smoothed.std():.2f}"]
                    for r in sorted(runs, key=lambda r: (r.condition, r.distance, r.rep))]), ""]

    L += ["## 2. Όρια εγγύτητας", "",
          f"Τα όρια επιλέχθηκαν με εξαντλητική αναζήτηση (βήμα 0.5 dB) ώστε να μεγιστοποιείται η "
          f"ισοσταθμισμένη ακρίβεια στις επαναλήψεις ρύθμισης ({ctx['fit_desc']}).", "",
          f"- **κοντά**: εξομαλυμένο RSSI ≥ **{thr.near_dbm:g} dBm**",
          f"- **μεσαία απόσταση**: {thr.far_dbm:g} ≤ RSSI < {thr.near_dbm:g} dBm",
          f"- **μακριά**: RSSI < **{thr.far_dbm:g} dBm**",
          f"- Balanced accuracy στα δεδομένα ρύθμισης: {pct(ctx['fit_ba'])}", "",
          "![Όρια](threshold_fit.png)", ""]

    def ev_table(e: dict) -> str:
        return md_table(["Μετρική", "Αρχικό RSSI", "Εξομαλυμένο RSSI"],
                        [["Ακρίβεια", pct(e["raw"]["accuracy"]), pct(e["smoothed"]["accuracy"])],
                         ["Ισοσταθμισμένη ακρίβεια", pct(e["raw"]["balanced_accuracy"]),
                          pct(e["smoothed"]["balanced_accuracy"])],
                         ["Αλλαγές ένδειξης ανά λεπτό (σταθερότητα)", f"{e['raw']['switches_per_min']:.1f}",
                          f"{e['smoothed']['switches_per_min']:.1f}"]])

    L += ["## 3. Αξιολόγηση (επαναλήψεις που ΔΕΝ χρησιμοποιήθηκαν για τα όρια)", "",
          f"Δείγματα αξιολόγησης: {ev['n_samples']}", "", ev_table(ev), "",
          "Ακρίβεια ανά συνθήκη:", "",
          md_table(["Συνθήκη", "Αρχικό", "Εξομαλυμένο"],
                   [[condition_label(c), pct(v["raw"]), pct(v["smoothed"])] for c, v in ev["per_condition"].items()]),
          "", "Ανά εκτέλεση:", "",
          md_table(["Εκτέλεση", "Ακρ. αρχικού", "Ακρ. εξομαλ.", "Αλλαγές/λεπτό αρχ.", "Αλλαγές/λεπτό εξομ."],
                   [[it["run"], pct(it["acc_raw"]), pct(it["acc_smoothed"]), f"{it['switches_raw']:.1f}",
                     f"{it['switches_smoothed']:.1f}"] for it in ev["per_run"]]),
          "", "![Πίνακες σύγχυσης](confusion_matrices.png)", "",
          "Για σύγκριση, στα δεδομένα ρύθμισης (εντός δείγματος):", "", ev_table(ev_cal), ""]

    L += ["## 4. Αρχικές έναντι εξομαλυμένων μετρήσεων", "",
          "![RSSI ανά απόσταση](rssi_by_distance.png)", "",
          "![Αρχικό vs εξομαλυμένο](raw_vs_smoothed.png)", "", "![Τυπική απόκλιση](std_comparison.png)", ""]

    dm = ctx.get("distance")
    L += ["## 5. Εκτίμηση απόστασης σε μέτρα (προαιρετική επέκταση)", ""]
    if dm:
        L += [f"Μοντέλο: RSSI(d) = A − 10·n·log10(d), με **A = {dm['a']:.1f} dBm** και **n = {dm['n']:.2f}** "
              f"(προσαρμογή στις επαναλήψεις ρύθμισης, {dm['fit_desc']}).", "",
              f"Μέσο απόλυτο σφάλμα στις επαναλήψεις αξιολόγησης: **{dm['errors']['mae_m']:.2f} m** "
              f"(διάμεσο {dm['errors']['median_ae_m']:.2f} m).", "",
              md_table(["Εκτέλεση", "Πραγματική", "Διάμεση εκτίμηση"],
                       [[r["run"], f"{r['distance']:g} m", f"{r['median_estimate']:.2f} m"]
                        for r in dm["errors"]["per_run"]]),
              "", "![Μοντέλο απόστασης](distance_model.png)", ""]
    else:
        L += ["Δεν ήταν δυνατή η προσαρμογή (απαιτούνται τουλάχιστον δύο διαφορετικές αποστάσεις).", ""]

    ml = ctx.get("ml")
    L += ["## 6. Σύγκριση με μηχανική μάθηση (προαιρετική επέκταση)", ""]
    if ml:
        L += [f"Random Forest σε παράθυρα {ml['window_s']:g} s με χαρακτηριστικά: {', '.join(ml['features'])}. "
              f"Εκπαίδευση: {ml['n_train']} παράθυρα (επαναλήψεις ρύθμισης), αξιολόγηση: {ml['n_test']} παράθυρα.", "",
              md_table(["Μέθοδος", "Ακρίβεια", "Ισοσταθμισμένη ακρίβεια"],
                       [["Όρια σε εξομαλυμένο RSSI (βασική)", pct(ml["baseline_accuracy"]),
                         pct(ml["baseline_balanced_accuracy"])],
                        ["Random Forest", pct(ml["ml_accuracy"]), pct(ml["ml_balanced_accuracy"])]]), ""]
    else:
        L += ["Δεν εκτελέστηκε (λείπει το scikit-learn ή δεν υπάρχουν αρκετά δεδομένα).", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ========================================================================= main
def _parse_reps(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="python -m ble_proximity.analyze", description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=EXPERIMENTS_DIR, help="φάκελος πειραματικών CSV")
    p.add_argument("--out", type=Path, default=ANALYSIS_DIR, help="φάκελος αποτελεσμάτων")
    p.add_argument("--cal-reps", default="1,2", help="επαναλήψεις για ρύθμιση ορίων (προεπ. 1,2)")
    p.add_argument("--test-reps", default="3", help="επαναλήψεις για αξιολόγηση (προεπ. 3)")
    p.add_argument("--near-max", type=float, default=1.5, help="μέγιστη απόσταση κατηγορίας «κοντά» (m)")
    p.add_argument("--medium-max", type=float, default=3.5, help="μέγιστη απόσταση «μεσαίας» (m)")
    p.add_argument("--smoothing", choices=["ema", "ma"], default="ema")
    p.add_argument("--param", type=float, default=None, help="α για EMA (0.2) ή N για MA (10)")
    p.add_argument("--hysteresis", type=float, default=2.0, help="υστέρηση κατάταξης (dB)")
    p.add_argument("--warmup", type=float, default=2.0, help="αγνόηση των πρώτων s κάθε εκτέλεσης")
    p.add_argument("--fit-conditions", default="all",
                   help="συνθήκες για τα όρια, π.χ. «los» ή «los,obstacle» (προεπ. όλες)")
    p.add_argument("--thresholds-file", type=Path, default=THRESHOLDS_FILE)
    p.add_argument("--no-write-thresholds", action="store_true", help="να μην ενημερωθεί το thresholds.json")
    a = p.parse_args(argv)

    param = a.param if a.param is not None else (0.2 if a.smoothing == "ema" else 10)
    smoothing = f"EMA α={param:g}" if a.smoothing == "ema" else f"κινητός μέσος N={int(param)}"
    cal_reps, test_reps = _parse_reps(a.cal_reps), _parse_reps(a.test_reps)
    if cal_reps & test_reps:
        p.error("οι επαναλήψεις ρύθμισης και αξιολόγησης πρέπει να είναι διαφορετικές")

    print(f"Φόρτωση μετρήσεων από {a.data} …")
    runs = load_runs(a.data)
    if not runs:
        print("Δεν βρέθηκαν πειραματικές μετρήσεις. Καταγράψτε πρώτα με: python -m ble_proximity.record")
        return 1
    for r in runs:
        r.smoothed = np.array(smooth_series(r.rssi.tolist(), a.smoothing, param))

    warnings = []
    for r in runs:
        if r.duration < 59:
            warnings.append(f"{r.path.name}: διάρκεια {r.duration:.0f} s (< 60 s)")
    combos = {(r.condition, r.distance) for r in runs}
    for cond, d in sorted(combos):
        reps = sorted({r.rep for r in runs if r.condition == cond and r.distance == d})
        if len(reps) < 3:
            warnings.append(f"{condition_label(cond)}, {d:g} m: {len(reps)} επανάληψη/εις ({reps}) — ζητούνται 3")
    for w in warnings:
        print("  ⚠", w)

    cal = [r for r in runs if r.rep in cal_reps]
    test = [r for r in runs if r.rep in test_reps]
    if not cal or not test:
        print(f"Χρειάζονται εκτελέσεις τόσο για ρύθμιση {sorted(cal_reps)} όσο και για αξιολόγηση {sorted(test_reps)}.")
        return 1
    fit_conds = None if a.fit_conditions == "all" else set(a.fit_conditions.split(","))
    fit_runs = [r for r in cal if fit_conds is None or r.condition in fit_conds]
    if not fit_runs:
        print("Δεν υπάρχουν εκτελέσεις ρύθμισης για τις συνθήκες", a.fit_conditions)
        return 1
    fit_desc = "όλες οι συνθήκες" if fit_conds is None else "συνθήκες: " + ", ".join(map(condition_label, sorted(fit_conds)))

    vals, labs = [], []
    for r in fit_runs:
        m = after_warmup(r, a.warmup)
        vals.append(r.smoothed[m])
        labs.append(np.full(m.sum(), true_class(r.distance, a.near_max, a.medium_max)))
    values, labels = np.concatenate(vals), np.concatenate(labs)
    near, far, fit_ba = fit_thresholds(values, labels)

    los_cal = [r for r in cal if r.condition == "los"] or cal
    model = fit_distance_model(los_cal)
    thr = Thresholds(near_dbm=near, far_dbm=far, hysteresis_db=a.hysteresis, calibrated=True,
                     source=f"βαθμονόμηση {datetime.now():%Y-%m-%d}, επαναλήψεις {sorted(cal_reps)}, {smoothing}",
                     rssi_at_1m=round(model[0], 2) if model else None,
                     path_loss_n=round(model[1], 3) if model else None)

    ev = evaluate(test, thr, a.near_max, a.medium_max, a.warmup)
    ev_cal = evaluate(cal, thr, a.near_max, a.medium_max, a.warmup)
    distance = None
    if model:
        los_test = [r for r in test if r.condition == "los"] or test
        distance = {"a": model[0], "n": model[1], "errors": distance_errors(los_test, *model, a.warmup),
                    "fit_desc": "οπτική επαφή" if los_cal is not cal else "όλες οι συνθήκες"}
    ml = ml_comparison(cal, test, thr, a.near_max, a.medium_max)

    a.out.mkdir(parents=True, exist_ok=True)
    plot_rssi_by_distance(runs, a.out / "rssi_by_distance.png")
    plot_raw_vs_smoothed(runs, a.out / "raw_vs_smoothed.png", smoothing)
    plot_std(runs, a.out / "std_comparison.png", a.warmup)
    plot_threshold_fit(values, labels, thr, a.out / "threshold_fit.png")
    if ev:
        plot_confusions(ev, a.out / "confusion_matrices.png")
    if model:
        plot_distance_model(los_cal, [r for r in test if r.condition == "los"] or test, *model,
                            a.out / "distance_model.png")

    ctx = {"runs": runs, "args": a, "thresholds": thr, "eval": ev, "eval_cal": ev_cal, "smoothing": smoothing,
           "cal_reps": cal_reps, "test_reps": test_reps, "warnings": warnings, "fit_ba": fit_ba,
           "fit_desc": fit_desc, "distance": distance, "ml": ml}
    write_report(a.out / "results.md", ctx)
    (a.out / "results.json").write_text(json.dumps(
        {"thresholds": thr.__dict__, "fit_balanced_accuracy": fit_ba, "evaluation": ev,
         "evaluation_calibration": ev_cal, "distance_model": distance, "ml": ml, "warnings": warnings},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if not a.no_write_thresholds:
        thr.save(a.thresholds_file)

    print(f"\nΕκτελέσεις: {len(runs)} (ρύθμιση {len(cal)}, αξιολόγηση {len(test)})")
    print(f"Όρια: κοντά ≥ {near:g} dBm, μακριά < {far:g} dBm (balanced acc. ρύθμισης {fit_ba:.1%})")
    if ev:
        print(f"Αξιολόγηση — ακρίβεια: αρχικό {ev['raw']['accuracy']:.1%}, εξομαλυμένο {ev['smoothed']['accuracy']:.1%}"
              f" | αλλαγές/λεπτό: {ev['raw']['switches_per_min']:.1f} → {ev['smoothed']['switches_per_min']:.1f}")
    if distance:
        print(f"Μοντέλο απόστασης: A={model[0]:.1f} dBm, n={model[1]:.2f}, MAE={distance['errors']['mae_m']:.2f} m")
    if ml:
        print(f"ML (Random Forest): {ml['ml_accuracy']:.1%} έναντι βασικής μεθόδου {ml['baseline_accuracy']:.1%}")
    print(f"Αποτελέσματα: {a.out / 'results.md'}")
    if not a.no_write_thresholds:
        print(f"Τα όρια αποθηκεύτηκαν στο {a.thresholds_file} (η εφαρμογή τα φορτώνει αυτόματα).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
