"""
gui.py
======
Haupt-GUI: tkinter-Fenster mit eingebettetem matplotlib-Plot und Animation.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

from .config import (
    ANIM_STEPS,
    BG,
    BLUE,
    BLUE2,
    BORDER,
    GHOST2,
    GRID_COLS,
    GRID_N,
    GRID_ROWS,
    GREEN,
    GREEN2,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    SUBTEXT,
    TEXT,
    YELLOW,
    GHOST,
)
from .models import simulate, tier_farbe, status_msg

# Gitter-Positionen (einmalig berechnet)
_gx = np.array([c for _ in range(GRID_ROWS) for c in range(GRID_COLS)], dtype=float)
_gy = np.array([r for r in range(GRID_ROWS) for _ in range(GRID_COLS)], dtype=float)


class PopulationSimulatorApp(tk.Tk):
    """Hauptfenster des Populationswachstum-Simulators."""

    APP_TITLE = "Populationswachstum - Tierreich"
    WINDOW_SIZE = "1120x760"
    MIN_SIZE = (980, 700)

    def __init__(self) -> None:
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(self.WINDOW_SIZE)
        self.minsize(*self.MIN_SIZE)
        self.configure(bg=BG)

        # --- Zustandsvariablen ---
        self._model = tk.StringVar(value="Logistisch")
        self._mode = tk.StringVar(value="Kontinuierlich")
        self._r = tk.DoubleVar(value=0.50)
        self._K = tk.DoubleVar(value=200.0)
        self._A = tk.DoubleVar(value=40.0)
        self._n0 = tk.DoubleVar(value=10.0)
        self._t_end = tk.DoubleVar(value=50.0)
        self._speed = tk.IntVar(value=3)

        # --- Animations-Zustand ---
        self._playing = False
        self._frame_idx = 0
        self._after_id: str | None = None
        self._t_data: np.ndarray | None = None
        self._N_data: np.ndarray | None = None
        self._K_val = 200.0
        self._A_val: float | None = None

        # --- matplotlib-Kuenstler (nach _setup_ax1 gueltig) ---
        self._cur_line = None
        self._cur_dot = None
        self._vline = None
        self._scatter = None

        # --- tkinter Info-Labels ---
        self._lbl_n: tk.Label | None = None
        self._lbl_pct: tk.Label | None = None
        self._lbl_t: tk.Label | None = None
        self._status_var: tk.StringVar | None = None
        self._status_lbl: tk.Label | None = None

        self._build_ui()
        self._recompute(reset_frame=True)

    # =========================================================================
    # UI-Aufbau
    # =========================================================================

    def _build_ui(self) -> None:
        self._build_header()

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        left = tk.Frame(body, bg=BG, width=296)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_plot(right)

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="x", padx=20, pady=12)

        tk.Label(
            inner,
            text="Populationswachstum - Tierreich",
            font=("Segoe UI", 17, "bold"),
            bg=PANEL,
            fg=TEXT,
        ).pack(side="left")
        tk.Label(
            inner,
            text="Beobachte, wie Tierpopulationen wachsen und schrumpfen!",
            font=("Segoe UI", 10),
            bg=PANEL,
            fg=SUBTEXT,
        ).pack(side="left", padx=14, pady=(5, 0))

        ctrl = tk.Frame(inner, bg=PANEL)
        ctrl.pack(side="right")
        self._ibtn(ctrl, "  Start  ", self._play, GREEN, GREEN2).pack(
            side="left", padx=(0, 6)
        )
        self._ibtn(ctrl, "  Pause  ", self._pause, YELLOW, "#d97706", TEXT).pack(
            side="left", padx=(0, 6)
        )
        self._ibtn(ctrl, "  Neu    ", self._reset_anim, ORANGE, "#ea580c").pack(
            side="left"
        )

    def _build_controls(self, parent: tk.Frame) -> None:
        # Modell & Modus
        self._sec(parent, "Welches Modell?")
        mf = self._last_sec

        self._btn_log = tk.Button(
            mf,
            text="Normales Wachstum",
            font=("Segoe UI", 11, "bold"),
            bg=BLUE,
            fg="white",
            activebackground=BLUE2,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            pady=9,
            command=lambda: self._set_model("Logistisch"),
        )
        self._btn_log.pack(fill="x", pady=(0, 7))

        self._btn_all = tk.Button(
            mf,
            text="Bedrohte Tierart (Allee-Effekt)",
            font=("Segoe UI", 11, "bold"),
            bg=GHOST2,
            fg=TEXT,
            activebackground=BORDER,
            activeforeground=TEXT,
            relief="flat",
            cursor="hand2",
            pady=9,
            command=lambda: self._set_model("Allee-Effekt"),
        )
        self._btn_all.pack(fill="x", pady=(0, 6))

        tk.Label(
            mf, text="Berechnungsmethode:", font=("Segoe UI", 9), bg=PANEL, fg=SUBTEXT
        ).pack(anchor="w", pady=(6, 2))
        rb = tk.Frame(mf, bg=PANEL)
        rb.pack(fill="x")
        for lbl in ("Kontinuierlich", "Diskret"):
            tk.Radiobutton(
                rb,
                text=lbl,
                variable=self._mode,
                value=lbl,
                font=("Segoe UI", 10),
                bg=PANEL,
                fg=TEXT,
                selectcolor=BLUE,
                activebackground=PANEL,
                cursor="hand2",
                command=self._on_param_change,
            ).pack(side="left", padx=(0, 14))

        # Parameter
        self._sec(parent, "Parameter einstellen")
        sf = self._last_sec

        self._kslider(sf, "Wachstumsgeschwindigkeit  r", self._r, 0.05, 3.0, 0.05)
        self._kslider(sf, "Maximale Anzahl Tiere  K", self._K, 20, 1000, 10)

        self._sl_A_frame = tk.Frame(sf, bg=PANEL)
        self._sl_A_frame.pack(fill="x")
        self._sl_A = self._kslider(
            self._sl_A_frame, "Mindestanzahl (Allee-Schwelle)  A", self._A, 5, 400, 5
        )

        self._kslider(sf, "Startzahl der Tiere  N0", self._n0, 1, 800, 1)
        self._kslider(sf, "Beobachtungszeit  t", self._t_end, 10, 300, 5)

        # Geschwindigkeit
        self._sec(parent, "Animationsgeschwindigkeit")
        af = self._last_sec
        sr = tk.Frame(af, bg=PANEL)
        sr.pack(fill="x")
        tk.Label(
            sr, text="<<<", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=SUBTEXT
        ).pack(side="left", padx=(0, 6))
        tk.Scale(
            sr,
            variable=self._speed,
            from_=1,
            to=10,
            orient="horizontal",
            bg=PANEL,
            fg=TEXT,
            highlightthickness=0,
            troughcolor=GHOST2,
            activebackground=BLUE,
            sliderrelief="flat",
            showvalue=False,
            bd=0,
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            sr, text=">>>", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=SUBTEXT
        ).pack(side="left", padx=(6, 0))

        self._update_model_btns()
        self._update_allee_state()

    def _build_plot(self, parent: tk.Frame) -> None:
        # Matplotlib Figure
        self._fig = Figure(facecolor=PANEL, dpi=100)
        gs = gridspec.GridSpec(
            2,
            1,
            figure=self._fig,
            height_ratios=[3, 2],
            hspace=0.42,
            left=0.09,
            right=0.97,
            top=0.94,
            bottom=0.08,
        )
        self._ax1 = self._fig.add_subplot(gs[0])
        self._ax2 = self._fig.add_subplot(gs[1])

        self._style_ax(self._ax1)
        self._init_animal_ax(self._ax2)

        pf = tk.Frame(
            parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        pf.pack(fill="both", expand=True, pady=(0, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Info-Leiste
        info = tk.Frame(
            parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        info.pack(fill="x", pady=(0, 6))
        ii = tk.Frame(info, bg=PANEL)
        ii.pack(fill="x", padx=16, pady=9)

        self._lbl_n = tk.Label(
            ii, text="Tiere:  -", font=("Segoe UI", 14, "bold"), bg=PANEL, fg=TEXT
        )
        self._lbl_n.pack(side="left", expand=True)

        tk.Frame(ii, bg=BORDER, width=1).pack(side="left", fill="y", padx=12)

        self._lbl_pct = tk.Label(
            ii, text="- %  von  K", font=("Segoe UI", 12), bg=PANEL, fg=SUBTEXT
        )
        self._lbl_pct.pack(side="left", expand=True)

        tk.Frame(ii, bg=BORDER, width=1).pack(side="left", fill="y", padx=12)

        self._lbl_t = tk.Label(
            ii, text="Zeit:  -", font=("Segoe UI", 12), bg=PANEL, fg=SUBTEXT
        )
        self._lbl_t.pack(side="right", expand=True)

        # Status-Meldung
        sb = tk.Frame(
            parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        sb.pack(fill="x")
        self._status_var = tk.StringVar(
            value="Druecke  Start  um die Simulation zu beginnen!"
        )
        self._status_lbl = tk.Label(
            sb,
            textvariable=self._status_var,
            font=("Segoe UI", 12, "bold"),
            bg=PANEL,
            fg=BLUE,
            pady=9,
        )
        self._status_lbl.pack()

    # =========================================================================
    # Widget-Helfer
    # =========================================================================

    def _sec(self, parent: tk.Frame, title: str) -> None:
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", pady=(0, 10))
        tk.Label(outer, text=title, font=("Segoe UI", 9, "bold"), bg=BG, fg=TEXT).pack(
            anchor="w", pady=(0, 3)
        )
        card = tk.Frame(
            outer, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        card.pack(fill="x")
        inner = tk.Frame(card, bg=PANEL)
        inner.pack(fill="x", padx=12, pady=10)
        self._last_sec: tk.Frame = inner

    def _kslider(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        res: float,
    ) -> tk.Scale:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(0, 8))
        hdr = tk.Frame(row, bg=PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text=label, font=("Segoe UI", 9, "bold"), bg=PANEL, fg=TEXT).pack(
            side="left"
        )
        fmt = ".0f" if res >= 1 else ".2f"
        val_lbl = tk.Label(
            hdr,
            text=f"{var.get():{fmt}}",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=BLUE,
        )
        val_lbl.pack(side="right")

        def on_slide(v: str) -> None:
            val_lbl.config(text=f"{float(v):{fmt}}")

        sl = tk.Scale(
            row,
            variable=var,
            from_=from_,
            to=to,
            resolution=res,
            orient="horizontal",
            bg=PANEL,
            fg=TEXT,
            highlightthickness=0,
            troughcolor=GHOST2,
            activebackground=BLUE,
            sliderrelief="flat",
            showvalue=False,
            bd=0,
            command=on_slide,
        )
        sl.pack(fill="x")
        sl.bind("<ButtonRelease-1>", lambda _e: self._on_param_change())
        return sl

    def _ibtn(
        self,
        parent: tk.Frame,
        text: str,
        cmd,
        color: str = BLUE,
        hover: str = BLUE2,
        fg: str = "white",
    ) -> tk.Button:
        b = tk.Button(
            parent,
            text=text,
            command=cmd,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=7,
            bd=0,
        )
        b.bind("<Enter>", lambda _e: b.config(bg=hover))
        b.bind("<Leave>", lambda _e: b.config(bg=color))
        return b

    # =========================================================================
    # Modell-Auswahl
    # =========================================================================

    def _set_model(self, model: str) -> None:
        self._model.set(model)
        self._update_model_btns()
        self._update_allee_state()
        self._on_param_change()

    def _update_model_btns(self) -> None:
        log = self._model.get() == "Logistisch"
        self._btn_log.config(
            bg=BLUE if log else GHOST2,
            fg="white" if log else TEXT,
            activebackground=BLUE2 if log else BORDER,
        )
        self._btn_all.config(
            bg=BLUE if not log else GHOST2,
            fg="white" if not log else TEXT,
            activebackground=BLUE2 if not log else BORDER,
        )

    def _update_allee_state(self) -> None:
        is_allee = self._model.get() == "Allee-Effekt"
        self._sl_A.config(state="normal" if is_allee else "disabled")
        fg_col = TEXT if is_allee else GHOST
        for child in self._sl_A_frame.winfo_children():
            for w in [child] + list(child.winfo_children()):
                try:
                    if isinstance(w, (tk.Label, tk.Scale)):
                        w.config(fg=fg_col)
                except tk.TclError:
                    pass

    # =========================================================================
    # Matplotlib-Axes
    # =========================================================================

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor("#f8faff")
        ax.grid(True, color="#e8f0fe", linewidth=1.3, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(BORDER)
        ax.tick_params(colors=SUBTEXT, labelsize=9)

    def _init_animal_ax(self, ax) -> None:
        ax.set_facecolor("#f0fdf4")
        ax.set_xlim(-0.6, GRID_COLS - 0.4)
        ax.set_ylim(-0.6, GRID_ROWS - 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Populationsfeld  -  jeder Punkt = 1 % der Kapazitaet K",
            fontsize=9.5,
            color=SUBTEXT,
            pad=5,
        )
        self._scatter = ax.scatter(
            _gx,
            _gy,
            s=370,
            c=[GHOST2] * GRID_N,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )

    def _setup_ax1(
        self,
        t: np.ndarray,
        N: np.ndarray,
        K: float,
        A: float | None,
        model: str,
        mode: str,
    ) -> None:
        ax = self._ax1
        ax.clear()
        self._style_ax(ax)

        y_max = max(float(N.max()), K) * 1.13
        ax.set_xlim(0, float(t[-1]))
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Zeit  t", fontsize=10, color=SUBTEXT, labelpad=4)
        ax.set_ylabel("Anzahl Tiere  N(t)", fontsize=10, color=SUBTEXT, labelpad=4)

        mode_lbl = "ODE (kontinuierlich)" if mode == "Kontinuierlich" else "Diskret"
        model_lbl = "Normales Wachstum" if model == "Logistisch" else "Bedrohte Tierart"
        ax.set_title(
            f"{model_lbl}  -  {mode_lbl}",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
            pad=8,
        )

        # Statischer Hintergrundverlauf
        ax.plot(t, N, color="#cbd5e1", lw=2.5, ls="--", zorder=1, alpha=0.55)
        ax.fill_between(t, 0, N, color="#cbd5e1", alpha=0.12, zorder=0)

        # Referenzlinien
        ax.axhline(
            K,
            color=PURPLE,
            lw=1.8,
            ls="--",
            alpha=0.75,
            zorder=2,
            label=f"Kapazitaet  K = {K:.0f}",
        )
        if A is not None:
            ax.axhline(
                A,
                color=RED,
                lw=1.8,
                ls=":",
                alpha=0.85,
                zorder=2,
                label=f"Allee-Schwelle  A = {A:.0f}",
            )
            ax.fill_between(t, 0, A, color=RED, alpha=0.06, zorder=0)

        leg = ax.legend(
            fontsize=9,
            frameon=True,
            framealpha=0.92,
            edgecolor=BORDER,
            loc="upper right",
        )
        for txt in leg.get_texts():
            txt.set_color(TEXT)

        # Animierte Kuenstler
        (self._cur_line,) = ax.plot([], [], lw=3.5, zorder=4, solid_capstyle="round")
        self._vline = ax.axvline(0, color=TEXT, lw=1.3, ls=":", alpha=0.4, zorder=5)
        (self._cur_dot,) = ax.plot(
            [], [], "o", ms=13, zorder=6, markeredgecolor="white", markeredgewidth=2.5
        )

    # =========================================================================
    # Simulation & Frame
    # =========================================================================

    def _on_param_change(self) -> None:
        self._pause()
        self._recompute(reset_frame=True)

    def _recompute(self, reset_frame: bool = False) -> None:
        model = self._model.get()
        mode = self._mode.get()
        r = self._r.get()
        K = self._K.get()
        A = self._A.get() if model == "Allee-Effekt" else None
        n0 = self._n0.get()
        t_end = self._t_end.get()

        self._K_val = K
        self._A_val = A

        try:
            t, N = simulate(model, mode, n0, t_end, r, K, A)
        except ValueError:
            return

        self._t_data = t
        self._N_data = N

        if reset_frame:
            self._frame_idx = 0

        self._setup_ax1(t, N, K, A, model, mode)
        self._draw_frame(self._frame_idx)

    def _draw_frame(self, i: int) -> None:
        t_data = self._t_data
        N_data = self._N_data
        cur_line = self._cur_line
        vline = self._vline
        cur_dot = self._cur_dot
        scatter = self._scatter
        lbl_n = self._lbl_n
        lbl_pct = self._lbl_pct
        lbl_t = self._lbl_t

        if (
            t_data is None
            or N_data is None
            or cur_line is None
            or vline is None
            or cur_dot is None
            or scatter is None
            or lbl_n is None
            or lbl_pct is None
            or lbl_t is None
        ):
            return

        i = min(i, len(t_data) - 1)
        K = self._K_val
        A = self._A_val
        N_val = float(N_data[i])
        t_val = float(t_data[i])
        col = tier_farbe(N_val, K, A)

        # Zeitreihe
        cur_line.set_data(t_data[: i + 1], N_data[: i + 1])
        cur_line.set_color(col)
        vline.set_xdata([t_val, t_val])
        cur_dot.set_data([t_val], [N_val])
        cur_dot.set_color(col)

        # Tier-Gitter
        n_show = min(GRID_N, max(0, round(GRID_N * N_val / K)))
        scatter.set_facecolors([col] * n_show + [GHOST2] * (GRID_N - n_show))

        # Info-Leiste
        pct = min(999, round(N_val / K * 100))
        lbl_n.config(text=f"Tiere:  {N_val:.0f}", fg=col)
        lbl_pct.config(text=f"{pct} %  von  K = {K:.0f}", fg=col)
        lbl_t.config(text=f"Zeit:  t = {t_val:.1f}", fg=SUBTEXT)

        # Status
        prev = float(N_data[i - 1]) if i > 0 else N_val
        txt, c = status_msg(N_val, K, A, prev)
        if self._status_var:
            self._status_var.set(txt)
        if self._status_lbl:
            self._status_lbl.config(fg=c)

        self._canvas.draw_idle()

    # =========================================================================
    # Animations-Loop
    # =========================================================================

    def _play(self) -> None:
        t_data = self._t_data
        if t_data is None:
            return
        if self._frame_idx >= len(t_data) - 1:
            self._frame_idx = 0
        self._playing = True
        self._loop()

    def _pause(self) -> None:
        self._playing = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _reset_anim(self) -> None:
        self._pause()
        self._frame_idx = 0
        self._draw_frame(0)

    def _loop(self) -> None:
        if not self._playing:
            return
        t_data = self._t_data
        if t_data is None:
            return

        speed = self._speed.get()
        self._frame_idx += speed

        if self._frame_idx >= len(t_data):
            self._frame_idx = len(t_data) - 1
            self._playing = False
            self._draw_frame(self._frame_idx)
            if self._status_var:
                cur = self._status_var.get()
                if "fertig" not in cur:
                    self._status_var.set(cur + "  |  Animation fertig! Neu starten.")
            return

        self._draw_frame(self._frame_idx)
        interval = max(16, 95 - speed * 8)
        self._after_id = self.after(interval, self._loop)
