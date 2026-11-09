"""Γραφική εφαρμογή (tkinter + matplotlib).

Αριστερά: λίστα ανιχνευμένων συσκευών BLE.
Δεξιά: παρακολούθηση επιλεγμένης συσκευής — εγγύτητα, λειτουργία αναζήτησης,
γράφημα RSSI στον χρόνο (αρχικό και εξομαλυμένο).
"""

from __future__ import annotations

import argparse
import math
import queue
import re
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import __version__, storage  # noqa: E402
from .config import (APPROACHING, DATA_DIR, FAR, MEDIUM, NEAR, NOT_DETECTED,  # noqa: E402
                     RECEDING, SESSIONS_DIR, STEADY, THRESHOLDS_FILE, UNKNOWN,
                     Settings, Thresholds)
from .models import Measurement, SourceError, SourceFinished  # noqa: E402
from .processing import DeviceTracker  # noqa: E402
from .sound import Beeper  # noqa: E402
from .sources import BleakSource, ReplaySource, SimulatedSource  # noqa: E402

PROX_COLORS = {NEAR: "#2e7d32", MEDIUM: "#ef6c00", FAR: "#c62828",
               NOT_DETECTED: "#616161", UNKNOWN: "#616161"}
TREND_STYLE = {  # (κείμενο, χρώμα κειμένου, φόντο)
    APPROACHING: ("▲  " + APPROACHING, "#1b5e20", "#e8f5e9"),
    RECEDING: ("▼  " + RECEDING, "#b71c1c", "#ffebee"),
    STEADY: ("►  " + STEADY, "#37474f", "#eceff1"),
}
SMOOTHING_CHOICES = {"Εκθετική (EMA)": "ema", "Κινητός μέσος όρος": "ma"}


@dataclass
class DeviceInfo:
    address: str
    name: str | None
    rssi: int
    first_seen: float
    last_seen: float
    count: int = 1


def _hhmmss(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")[:40] or "device"


class App:
    HISTORY_PER_DEVICE = 600  # τελευταίες μετρήσεις κάθε συσκευής (για άμεσο ιστορικό όταν επιλεγεί)

    def __init__(self, root: tk.Tk, mode: str = "ble", replay: Path | None = None,
                 track: str | None = None):
        self.root = root
        self.mode = tk.StringVar(value=mode)
        self.settings = Settings()
        self.thresholds = Thresholds.load()
        self.queue: queue.Queue = queue.Queue()
        self.source = None
        self.devices: dict[str, DeviceInfo] = {}
        self.history: dict[str, deque[Measurement]] = {}
        self.all_log: deque[Measurement] = deque(maxlen=500_000)
        self.tracker: DeviceTracker | None = None
        self.auto_track: str | None = track
        self.replay_speed = tk.DoubleVar(value=1.0)
        self.beeper = Beeper()
        self._last_table = 0.0
        self._last_plot = 0.0

        root.title(f"Ανίχνευση εγγύτητας BLE — v{__version__}")
        root.geometry("1320x780")
        root.minsize(1050, 620)
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.beeper.start()
        root.after(100, self._poll)
        if replay:
            root.after(300, lambda: self.start_replay(replay))

    # ======================================================================= UI
    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("Treeview", rowheight=24)

        self._build_menu()

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        self.btn_start = ttk.Button(toolbar, text="▶  Έναρξη σάρωσης", command=self.start_scan)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(toolbar, text="■  Διακοπή", command=self.stop_scan, state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(toolbar, text="Πηγή:").pack(side="left")
        self.rb_ble = ttk.Radiobutton(toolbar, text="Bluetooth", value="ble", variable=self.mode)
        self.rb_ble.pack(side="left", padx=4)
        self.rb_sim = ttk.Radiobutton(toolbar, text="Προσομοίωση", value="simulate", variable=self.mode)
        self.rb_sim.pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="Εξαγωγή CSV…", command=self.export_tracked).pack(side="left")
        ttk.Button(toolbar, text="Επαναπροβολή…", command=self.start_replay).pack(side="left", padx=6)
        self.sound_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Ηχητική ένδειξη αναζήτησης", variable=self.sound_var,
                        command=self._toggle_sound).pack(side="right")

        self.status = tk.StringVar(value="Έτοιμο. Πατήστε «Έναρξη σάρωσης».")
        ttk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken",
                  padding=(8, 3)).pack(fill="x", side="bottom")

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=(10, 4, 4, 4))
        paned.add(left, weight=2)
        paned.add(right, weight=3)
        self._build_device_list(left)
        self._build_tracking_panel(right)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Εξαγωγή παρακολουθούμενης συσκευής (CSV)…", command=self.export_tracked)
        m_file.add_command(label="Εξαγωγή όλων των μετρήσεων (CSV)…", command=self.export_all)
        m_file.add_separator()
        m_file.add_command(label="Επαναπροβολή αποθηκευμένης συνεδρίας…", command=self.start_replay)
        m_speed = tk.Menu(m_file, tearoff=False)
        for s in (1.0, 2.0, 5.0, 10.0):
            m_speed.add_radiobutton(label=f"{s:g}×", value=s, variable=self.replay_speed)
        m_file.add_cascade(label="Ταχύτητα επαναπροβολής", menu=m_speed)
        m_file.add_separator()
        m_file.add_command(label="Έξοδος", command=self.on_close)
        menubar.add_cascade(label="Αρχείο", menu=m_file)

        m_set = tk.Menu(menubar, tearoff=False)
        m_set.add_command(label="Επαναφόρτωση ορίων (config/thresholds.json)", command=self.reload_thresholds)
        m_set.add_command(label="Καθαρισμός λίστας συσκευών", command=self.clear_devices)
        menubar.add_cascade(label="Ρυθμίσεις", menu=m_set)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="Σχετικά", command=self._about)
        menubar.add_cascade(label="Βοήθεια", menu=m_help)
        self.root.config(menu=menubar)

    def _build_device_list(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Ανιχνευμένες συσκευές BLE", style="Title.TLabel").pack(anchor="w")
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(4, 4))
        ttk.Label(bar, text="Φίλτρο:").pack(side="left")
        self.filter_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.filter_var, width=12).pack(side="left", padx=(4, 8))
        self.named_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Μόνο με όνομα", variable=self.named_only).pack(side="left")
        self.sort_var = tk.StringVar(value="RSSI")
        ttk.Combobox(bar, textvariable=self.sort_var, state="readonly", width=12,
                     values=["RSSI", "Όνομα", "Τελευταία ανίχνευση"]).pack(side="right")
        ttk.Label(bar, text="Ταξινόμηση:").pack(side="right", padx=(8, 4))

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        cols = ("name", "address", "rssi", "seen")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        for col, text, width, anchor in (("name", "Όνομα", 190, "w"),
                                         ("address", "Αναγνωριστικό", 150, "w"),
                                         ("rssi", "RSSI", 70, "e"),
                                         ("seen", "Τελευταία ανίχνευση", 150, "w")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "name"))
        self.tree.tag_configure("stale", foreground="#9e9e9e")
        self.tree.tag_configure("tracked", background="#e3f2fd")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda _e: self.track_selected())

        bottom = ttk.Frame(parent)
        bottom.pack(fill="x", pady=(6, 0))
        ttk.Button(bottom, text="Παρακολούθηση επιλεγμένης  →", command=self.track_selected).pack(side="left")
        self.count_var = tk.StringVar(value="0 συσκευές")
        ttk.Label(bottom, textvariable=self.count_var, style="Small.TLabel").pack(side="right")

    def _build_tracking_panel(self, p: ttk.Frame) -> None:
        self.tracked_title = tk.StringVar(
            value="Καμία συσκευή υπό παρακολούθηση — επιλέξτε μία από τη λίστα (διπλό κλικ).")
        ttk.Label(p, textvariable=self.tracked_title, style="Title.TLabel").pack(anchor="w")

        ind = ttk.Frame(p)
        ind.pack(fill="x", pady=(8, 4))
        ind.columnconfigure(0, weight=1)
        ind.columnconfigure(1, weight=2)
        ttk.Label(ind, text="Εγγύτητα", style="Small.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(ind, text="Λειτουργία αναζήτησης (τάση σήματος)", style="Small.TLabel").grid(
            row=0, column=1, sticky="w", padx=(10, 0))
        self.prox_label = tk.Label(ind, text=UNKNOWN, font=("Segoe UI", 20, "bold"), fg="white",
                                   bg=PROX_COLORS[UNKNOWN], pady=10)
        self.prox_label.grid(row=1, column=0, sticky="nsew")
        self.trend_label = tk.Label(ind, text=UNKNOWN, font=("Segoe UI", 15, "bold"), fg="#37474f",
                                    bg="#eceff1", pady=14)
        self.trend_label.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        self.details_var = tk.StringVar(value="")
        ttk.Label(p, textvariable=self.details_var, font=("Consolas", 10), justify="left").pack(
            anchor="w", pady=(6, 4))

        sett = ttk.LabelFrame(p, text="Επεξεργασία μετρήσεων", padding=(8, 4))
        sett.pack(fill="x", pady=(2, 4))
        ttk.Label(sett, text="Εξομάλυνση:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="Εκθετική (EMA)")
        cb = ttk.Combobox(sett, textvariable=self.method_var, state="readonly", width=20,
                          values=list(SMOOTHING_CHOICES))
        cb.grid(row=0, column=1, padx=(4, 12))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_method_changed())
        self.param_caption = tk.StringVar(value="α:")
        ttk.Label(sett, textvariable=self.param_caption).grid(row=0, column=2, sticky="e")
        self.param_var = tk.StringVar(value=f"{self.settings.ema_alpha:g}")
        self.param_spin = ttk.Spinbox(sett, textvariable=self.param_var, width=6,
                                      from_=0.05, to=1.0, increment=0.05)
        self.param_spin.grid(row=0, column=3, padx=(4, 12))
        ttk.Button(sett, text="Εφαρμογή", command=self.apply_settings).grid(row=0, column=4, rowspan=2,
                                                                             sticky="ns")
        ttk.Label(sett, text="«Δεν ανιχνεύεται» μετά από (s):").grid(row=1, column=0, columnspan=2,
                                                                     sticky="w", pady=(4, 0))
        self.timeout_var = tk.StringVar(value=f"{self.settings.lost_timeout_s:g}")
        ttk.Spinbox(sett, textvariable=self.timeout_var, width=6, from_=1, to=60,
                    increment=1).grid(row=1, column=3, padx=(4, 12), pady=(4, 0))

        self.thr_var = tk.StringVar(value=self.thresholds.describe())
        ttk.Label(p, textvariable=self.thr_var, style="Small.TLabel").pack(anchor="w", pady=(0, 2))

        self.fig = Figure(figsize=(6.5, 3.6), dpi=100, layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Χρόνος από την έναρξη παρακολούθησης (s)")
        self.ax.set_ylabel("RSSI (dBm)")
        self.ax.grid(alpha=0.3)
        (self.raw_line,) = self.ax.plot([], [], ".", ms=3.5, color="#9e9e9e", label="Αρχικό RSSI")
        (self.sm_line,) = self.ax.plot([], [], "-", lw=2.2, color="#1565c0", label="Εξομαλυμένο RSSI")
        self.near_line = self.ax.axhline(self.thresholds.near_dbm, ls="--", lw=1, color=PROX_COLORS[NEAR],
                                         label="Όριο «κοντά»")
        self.far_line = self.ax.axhline(self.thresholds.far_dbm, ls="--", lw=1, color=PROX_COLORS[FAR],
                                        label="Όριο «μακριά»")
        self.ax.set_xlim(0, self.settings.plot_window_s)
        self.ax.set_ylim(-100, -40)
        self.ax.legend(loc="upper left", fontsize=8, ncol=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=p)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 0))

    # ================================================================ πηγές
    def _set_running(self, running: bool) -> None:
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal" if running else "disabled")
        for rb in (self.rb_ble, self.rb_sim):
            rb.config(state="disabled" if running else "normal")

    def start_scan(self) -> None:
        if self.source is not None:
            return
        if self.mode.get() == "simulate":
            self.source = SimulatedSource(self.queue)
            self.status.set("Σάρωση (προσομοίωση) σε εξέλιξη…")
        else:
            self.source = BleakSource(self.queue)
            self.status.set("Σάρωση Bluetooth LE σε εξέλιξη…")
        self.source.start()
        self._set_running(True)

    def stop_scan(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None
            self.status.set("Η λήψη διακόπηκε.")
        self._set_running(False)

    def start_replay(self, path: Path | None = None) -> None:
        if path is None:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            chosen = filedialog.askopenfilename(
                title="Επιλογή συνεδρίας για επαναπροβολή", initialdir=str(SESSIONS_DIR),
                filetypes=[("CSV", "*.csv"), ("Όλα τα αρχεία", "*.*")])
            if not chosen:
                return
            path = Path(chosen)
        try:
            ms = storage.read_measurements(path)
        except Exception as exc:
            messagebox.showerror("Επαναπροβολή", f"Αδυναμία ανάγνωσης αρχείου:\n{exc}")
            return
        if not ms:
            messagebox.showwarning("Επαναπροβολή", "Το αρχείο δεν περιέχει μετρήσεις.")
            return
        self.stop_scan()
        self.clear_devices()
        self.auto_track = Counter(m.address for m in ms).most_common(1)[0][0]
        speed = self.replay_speed.get()
        self.source = ReplaySource(self.queue, path, speed)
        self.source.start()
        self._set_running(True)
        self.status.set(f"Επαναπροβολή: {path.name} ({len(ms)} μετρήσεις, {speed:g}×)")

    # ============================================================ δεδομένα
    def _poll(self) -> None:
        try:
            for _ in range(5000):
                item = self.queue.get_nowait()
                if isinstance(item, Measurement):
                    self._on_measurement(item)
                elif isinstance(item, SourceError):
                    self._on_source_error(item)
                elif isinstance(item, SourceFinished):
                    self.source = None
                    self._set_running(False)
                    self.status.set(item.message)
        except queue.Empty:
            pass
        now = time.time()
        if now - self._last_table >= 1.0:
            self._last_table = now
            self._refresh_table(now)
        if now - self._last_plot >= 0.4:
            self._last_plot = now
            self._update_tracked_view(now)
        self._poll_id = self.root.after(100, self._poll)

    def _on_measurement(self, m: Measurement) -> None:
        d = self.devices.get(m.address)
        if d is None:
            self.devices[m.address] = DeviceInfo(m.address, m.name, m.rssi, m.t, m.t)
            self.history[m.address] = deque(maxlen=self.HISTORY_PER_DEVICE)
        else:
            d.rssi, d.last_seen, d.count = m.rssi, m.t, d.count + 1
            if m.name:
                d.name = m.name
        self.history[m.address].append(m)
        self.all_log.append(m)
        if self.tracker is not None and m.address == self.tracker.address:
            self.tracker.process(m)
        elif self.tracker is None and self.auto_track and m.address.upper() == self.auto_track.upper():
            self.track(m.address)

    def _on_source_error(self, err: SourceError) -> None:
        self.stop_scan()
        self.status.set("Σφάλμα πηγής μετρήσεων.")
        messagebox.showerror(
            "Σφάλμα Bluetooth",
            f"{err.message}\n\nΒεβαιωθείτε ότι το Bluetooth είναι ενεργοποιημένο στις ρυθμίσεις "
            "των Windows. Για δοκιμή χωρίς συσκευή επιλέξτε «Προσομοίωση».")

    def clear_devices(self) -> None:
        self.devices.clear()
        self.history.clear()
        self.tracker = None
        self.tree.delete(*self.tree.get_children())
        self.tracked_title.set("Καμία συσκευή υπό παρακολούθηση — επιλέξτε μία από τη λίστα (διπλό κλικ).")
        self.raw_line.set_data([], [])
        self.sm_line.set_data([], [])
        self.prox_label.config(text=UNKNOWN, bg=PROX_COLORS[UNKNOWN])
        self.trend_label.config(text=UNKNOWN, fg="#37474f", bg="#eceff1")
        self.details_var.set("")
        self.beeper.set_state(None, None)
        self.canvas.draw_idle()

    # =========================================================== λίστα
    def _refresh_table(self, now: float) -> None:
        flt = self.filter_var.get().strip().lower()
        named = self.named_only.get()
        timeout = self.settings.lost_timeout_s
        items = [d for d in self.devices.values()
                 if (not named or d.name)
                 and (not flt or flt in (d.name or "").lower() or flt in d.address.lower())]
        sort = self.sort_var.get()
        if sort == "Όνομα":
            items.sort(key=lambda d: (d.name is None, (d.name or "").lower(), d.address))
        elif sort == "Τελευταία ανίχνευση":
            items.sort(key=lambda d: -d.last_seen)
        else:  # RSSI: οι ενεργές πρώτα, κατά φθίνον RSSI
            items.sort(key=lambda d: (now - d.last_seen > timeout, -d.rssi))

        wanted = {d.address for d in items}
        for iid in self.tree.get_children():
            if iid not in wanted:
                self.tree.delete(iid)
        tracked = self.tracker.address if self.tracker else None
        for idx, d in enumerate(items):
            age = max(0.0, now - d.last_seen)
            values = (d.name or "(χωρίς όνομα)", d.address, f"{d.rssi} dBm",
                      f"{_hhmmss(d.last_seen)}  (πριν {age:.0f} s)")
            tags = []
            if age > timeout:
                tags.append("stale")
            if d.address == tracked:
                tags.append("tracked")
            if self.tree.exists(d.address):
                self.tree.item(d.address, values=values, tags=tags)
                self.tree.move(d.address, "", idx)
            else:
                self.tree.insert("", idx, iid=d.address, values=values, tags=tags)
        active = sum(1 for d in self.devices.values() if now - d.last_seen <= timeout)
        self.count_var.set(f"Εμφανίζονται {len(items)} · ενεργές {active} · σύνολο {len(self.devices)}")

    def track_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Παρακολούθηση", "Επιλέξτε πρώτα μια συσκευή από τη λίστα.")
            return
        self.track(sel[0])

    def track(self, address: str) -> None:
        info = self.devices.get(address)
        self.tracker = DeviceTracker(address, info.name if info else None, self.settings, self.thresholds)
        for m in self.history.get(address, ()):
            self.tracker.process(m)
        self.auto_track = None
        name = self.tracker.name or "(χωρίς όνομα)"
        self.tracked_title.set(f"Παρακολούθηση: {name}  [{address}]")
        self.status.set(f"Παρακολούθηση της συσκευής {name} ({address}).")
        self._last_table = 0.0

    # ======================================================= παρακολούθηση
    def _update_tracked_view(self, now: float) -> None:
        tr = self.tracker
        if tr is None:
            return
        st = tr.state(now)
        self.prox_label.config(text=st.proximity, bg=PROX_COLORS.get(st.proximity, PROX_COLORS[UNKNOWN]))
        text, fg, bg = TREND_STYLE.get(st.trend, (st.trend, "#37474f", "#eceff1"))
        self.trend_label.config(text=text, fg=fg, bg=bg)
        if tr.name and "(χωρίς όνομα)" in self.tracked_title.get():
            self.tracked_title.set(f"Παρακολούθηση: {tr.name}  [{tr.address}]")

        if st.last_seen is None:
            self.details_var.set("Αναμονή για λήψη από τη συσκευή…")
        else:
            known = "" if st.detected else " (τελευταία)"
            slope = "—" if st.slope is None else f"{st.slope:+.2f} dB/s"
            if st.distance_m is not None:
                dist = f"~{st.distance_m:.1f} m (μοντέλο log-distance)"
            elif self.thresholds.rssi_at_1m is None:
                dist = "— (απαιτείται βαθμονόμηση)"
            else:
                dist = "—"
            self.details_var.set(
                f"RSSI τελευταίο : {st.rssi:>4} dBm    εξομαλυμένο: {st.smoothed:7.1f} dBm{known}\n"
                f"Κλίση τάσης    : {slope:<12}  ρυθμός λήψης: {st.rate_hz:4.1f} πακέτα/s\n"
                f"Εκτ. απόσταση  : {dist}\n"
                f"Τελευταία λήψη : {_hhmmss(st.last_seen)} (πριν {now - st.last_seen:.0f} s)"
                f"    δείγματα: {st.n_samples}")
        self.beeper.set_state(st.proximity if st.detected else None, st.trend)
        self._update_plot(now)

    def _update_plot(self, now: float) -> None:
        tr = self.tracker
        if tr is None or not tr.samples:
            return
        t0 = tr.t0
        window = self.settings.plot_window_s
        x_now = now - t0
        x_min = max(0.0, x_now - window)
        xs, raw, sm = [], [], []
        prev_t = None
        for s in tr.samples:
            x = s.t - t0
            if x < x_min - 1:
                prev_t = s.t
                continue
            if prev_t is not None and s.t - prev_t > self.settings.lost_timeout_s:
                xs.append(x)
                raw.append(math.nan)
                sm.append(math.nan)  # κενό στη γραμμή κατά τη διακοπή λήψης
            xs.append(x)
            raw.append(s.rssi)
            sm.append(s.smoothed)
            prev_t = s.t
        self.raw_line.set_data(xs, raw)
        self.sm_line.set_data(xs, sm)
        self.near_line.set_ydata([self.thresholds.near_dbm] * 2)
        self.far_line.set_ydata([self.thresholds.far_dbm] * 2)
        self.ax.set_xlim(x_min, max(x_now, x_min + 10))
        vals = [v for v in raw if not math.isnan(v)] + [self.thresholds.near_dbm, self.thresholds.far_dbm]
        self.ax.set_ylim(min(vals) - 4, max(vals) + 4)
        self.canvas.draw_idle()

    # ============================================================ ρυθμίσεις
    def _on_method_changed(self) -> None:
        if SMOOTHING_CHOICES[self.method_var.get()] == "ema":
            self.param_caption.set("α:")
            self.param_spin.config(from_=0.05, to=1.0, increment=0.05)
            self.param_var.set(f"{self.settings.ema_alpha:g}")
        else:
            self.param_caption.set("N:")
            self.param_spin.config(from_=2, to=100, increment=1)
            self.param_var.set(str(self.settings.ma_window))

    def apply_settings(self) -> None:
        method = SMOOTHING_CHOICES[self.method_var.get()]
        try:
            param = float(self.param_var.get().replace(",", "."))
            timeout = float(self.timeout_var.get().replace(",", "."))
            if method == "ema" and not 0 < param <= 1:
                raise ValueError("Το α πρέπει να είναι στο (0, 1].")
            if method == "ma" and not 1 <= param <= 500:
                raise ValueError("Το N πρέπει να είναι από 1 έως 500.")
            if timeout <= 0:
                raise ValueError("Ο χρόνος πρέπει να είναι θετικός.")
        except ValueError as exc:
            messagebox.showerror("Ρυθμίσεις", f"Μη έγκυρη τιμή: {exc}")
            return
        self.settings.smoothing = method
        if method == "ema":
            self.settings.ema_alpha = param
        else:
            self.settings.ma_window = int(param)
        self.settings.lost_timeout_s = timeout
        if self.tracker:
            self.tracker.reprocess()
        self.status.set(f"Εφαρμόστηκε εξομάλυνση {self.settings.smoothing_description()}, "
                        f"όριο «δεν ανιχνεύεται» {timeout:g} s.")

    def reload_thresholds(self) -> None:
        self.thresholds = Thresholds.load()
        if self.tracker:
            self.tracker.set_thresholds(self.thresholds)
        self.thr_var.set(self.thresholds.describe())
        if not THRESHOLDS_FILE.exists():
            messagebox.showinfo("Όρια", f"Δεν βρέθηκε το {THRESHOLDS_FILE}.\n"
                                        "Χρησιμοποιούνται τα προεπιλεγμένα όρια.\n"
                                        "Δημιουργείται με: python -m ble_proximity.analyze")
        self.status.set("Τα όρια επαναφορτώθηκαν.")

    def _toggle_sound(self) -> None:
        self.beeper.enabled = self.sound_var.get()

    # ============================================================== εξαγωγή
    def export_tracked(self) -> None:
        if not self.tracker or not self.tracker.samples:
            messagebox.showinfo("Εξαγωγή", "Δεν υπάρχουν μετρήσεις παρακολουθούμενης συσκευής.")
            return
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        default = (f"session_{_safe_filename(self.tracker.name or self.tracker.address)}_"
                   f"{datetime.now():%Y%m%d_%H%M%S}.csv")
        path = filedialog.asksaveasfilename(title="Εξαγωγή μετρήσεων", initialdir=str(SESSIONS_DIR),
                                            initialfile=default, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            n = storage.write_session(Path(path), self.tracker, self.settings)
            self.status.set(f"Αποθηκεύτηκαν {n} μετρήσεις στο {path}")

    def export_all(self) -> None:
        if not self.all_log:
            messagebox.showinfo("Εξαγωγή", "Δεν υπάρχουν μετρήσεις.")
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(title="Εξαγωγή όλων των μετρήσεων", initialdir=str(SESSIONS_DIR),
                                            initialfile=f"all_devices_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            n = storage.write_measurements(Path(path), list(self.all_log))
            self.status.set(f"Αποθηκεύτηκαν {n} μετρήσεις στο {path}")

    # ================================================================ λοιπά
    def _about(self) -> None:
        messagebox.showinfo("Σχετικά", f"Ανίχνευση και εκτίμηση εγγύτητας συσκευών BLE\nΈκδοση {__version__}\n\n"
                                       "Python · bleak · tkinter · matplotlib")

    def on_close(self) -> None:
        self.stop_scan()
        self.beeper.stop()
        if getattr(self, "_poll_id", None):
            self.root.after_cancel(self._poll_id)
        # quit() τερματίζει το mainloop· το παράθυρο καταστρέφεται μετά στο main(),
        # ώστε να μην εκτελεστούν εκκρεμείς χρονοπρογραμματισμένες ενημερώσεις.
        self.root.quit()


def _enable_dpi_awareness() -> None:
    try:  # καθαρότερη απεικόνιση σε οθόνες υψηλής ανάλυσης (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ble_proximity",
                                     description="Ανίχνευση και εκτίμηση εγγύτητας συσκευών BLE")
    parser.add_argument("--simulate", action="store_true", help="προσομοιωμένες συσκευές (χωρίς Bluetooth)")
    parser.add_argument("--replay", type=Path, help="επαναπροβολή αποθηκευμένης συνεδρίας CSV")
    parser.add_argument("--track", help="αυτόματη παρακολούθηση της συσκευής με αυτή τη διεύθυνση")
    parser.add_argument("--autostart", action="store_true", help="έναρξη σάρωσης με το άνοιγμα")
    args = parser.parse_args(argv)

    _enable_dpi_awareness()
    root = tk.Tk()
    app = App(root, mode="simulate" if args.simulate else "ble", replay=args.replay, track=args.track)
    if args.autostart and not args.replay:
        root.after(300, app.start_scan)
    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass
    return 0
