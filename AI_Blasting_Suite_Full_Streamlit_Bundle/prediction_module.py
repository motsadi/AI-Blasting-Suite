#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modules/prediction_module.py
from typing import Dict, List, Optional, Tuple
import math
import threading
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as mtick


# ---------- CTk compatibility shim ----------
if not hasattr(ctk, "CTkCollapsibleFrame"):
    class CTkCollapsibleFrame(ctk.CTkFrame):
        def __init__(self, master=None, text: str = "", **kwargs):
            super().__init__(master, **kwargs)
            if text:
                ctk.CTkLabel(
                    self, text=text, font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=6, pady=(6, 2))
    ctk.CTkCollapsibleFrame = CTkCollapsibleFrame
# -------------------------------------------


# ---------- header resolver (to find slider ranges in df) ----------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

_INPUT_SYNS: Dict[str, List[str]] = {
    "hole depth (m)": ["hole depth (m)", "hole depth", "depth", "depthm", "hdepthm"],
    "hole diameter (mm)": ["hole diameter (mm)", "hole diameter", "diameter", "diametermm", "holedia"],
    "burden (m)": ["burden (m)", "burden", "bm"],
    "spacing (m)": ["spacing (m)", "spacing", "sm"],
    "stemming (m)": ["stemming (m)", "stemming"],
    "distance (m)": ["distance (m)", "distance", "monitor distance", "distancem", "r"],
    "powder factor (kg/m³)": ["powder factor (kg/m³)", "powder factor", "powderfactorkg/m3", "pf"],
    "rock density (t/m³)": ["rock density (t/m³)", "rock density", "density", "densityt/m3"],
    "linear charge (kg/m)": ["linear charge (kg/m)", "linear charge", "kg/m", "chargeperm"],
    "explosive mass (kg)": ["explosive mass (kg)", "explosive mass", "chargemass", "masskg"],
    "blast volume (m³)": ["blast volume (m³)", "blast volume", "volume"],
    "# holes": ["# holes", "number of holes", "noholes", "holes", "holescount"],
}

def _resolve(df_cols: List[str], pretty_names: List[str]) -> Dict[str, Optional[str]]:
    cols = list(df_cols)
    lower = {c.lower(): c for c in cols}
    norm  = {_norm(c): c for c in cols}
    out: Dict[str, Optional[str]] = {}
    for p in pretty_names:
        found = None
        if p in cols:
            found = p
        elif p.lower() in lower:
            found = lower[p.lower()]
        else:
            for s in _INPUT_SYNS.get(p.lower(), _INPUT_SYNS.get(p, [])):
                if s in cols:
                    found = s; break
                if s.lower() in lower:
                    found = lower[s.lower()]; break
                n = _norm(s)
                if n in norm:
                    found = norm[n]; break
        out[p] = found
    return out


class PredictionWindow(ctk.CTkToplevel):
    """
    Slider-based simultaneous prediction for all outputs using registry models + scaler,
    and empirical baselines (USBM: PPV/Airblast, Kuz–Ram 2005: Xm). Also shows a Rosin–Rammler
    fragmentation curve (CDF) using Xm + n (manual or Kuz–Ram estimate).
    """

    # ------------ empirical defaults ------------
    _DEF_K_PPV = 1000.0   # PPV = K * SD^-beta, SD=R/sqrt(Q_delay), PPV in mm/s
    _DEF_BETA  = 1.60
    _DEF_K_AIR = 170.0    # L = K_air + B_air*log10(Qd^(1/3)/R), L in dB
    _DEF_B_AIR = 20.0
    _DEF_A_KUZ = 22.0     # Rock factor A (site-calibrated)
    _DEF_RWS   = 115.0    # Relative Weight Strength (% of ANFO)
    _DEF_HPD   = 1.0      # holes per delay
    _DEF_N     = 1.8      # default manual RR uniformity index
    _DEF_X_OV  = 500.0    # oversize threshold (mm) for RR plot

    def __init__(self, master, registry, input_labels: List[str], outputs: List[str]):
        super().__init__(master)
        self.title("Simultaneous Prediction")
        self.geometry("1180x760")
        self.minsize(1060, 660)

        self.registry = registry
        self.input_labels = input_labels[:]
        self.outputs = outputs[:]

        # slider vars + meta
        self._vars: Dict[str, ctk.DoubleVar] = {}
        self._ranges: Dict[str, Tuple[float, float, float]] = {}

        # thresholds (optional)
        self._thresholds = {o: ctk.DoubleVar(value=0.0) for o in self.outputs}

        # empirical setting vars
        self._k_ppv  = ctk.DoubleVar(value=self._DEF_K_PPV)
        self._beta   = ctk.DoubleVar(value=self._DEF_BETA)
        self._k_air  = ctk.DoubleVar(value=self._DEF_K_AIR)
        self._b_air  = ctk.DoubleVar(value=self._DEF_B_AIR)
        self._a_kuz  = ctk.DoubleVar(value=self._DEF_A_KUZ)  # Rock factor A
        self._rws    = ctk.DoubleVar(value=self._DEF_RWS)    # Relative weight strength
        self._hpd    = ctk.DoubleVar(value=self._DEF_HPD)
        self._x_ov   = ctk.DoubleVar(value=self._DEF_X_OV)   # oversize threshold for RR plot

        # RR n control: mode + vars
        self._n_mode = ctk.StringVar(value="Estimate (Kuz–Ram)")
        self._n_manual = ctk.DoubleVar(value=self._DEF_N)
        self._n_display = ctk.StringVar(value="—")  # shows current n (est or manual)

        # export cache
        self._last_pred_row: Optional[pd.DataFrame] = None

        self._build_ui()
        self._init_sliders_from_dataset()

    # ---------- UI ----------
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Left column
        left = ctk.CTkFrame(self, width=410)
        left.grid(row=0, column=0, sticky="nsnsw", padx=(12, 6), pady=12)
        left.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(left, text="Inputs", font=ctk.CTkFont(size=16, weight="bold"))\
            .grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        # dataset tag
        self._ds_tag = ctk.CTkLabel(left, text="Dataset: (none)")
        self._ds_tag.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # sliders frame (scrollable)
        self._inputs_frame = ctk.CTkScrollableFrame(left, width=390, height=360)
        self._inputs_frame.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="nswe")

        # Empirical settings (scrollable)
        emp_wrap = ctk.CTkCollapsibleFrame(left, text="Empirical settings (USBM & Kuz–Ram)")
        emp_wrap.grid(row=3, column=0, padx=12, pady=(6, 6), sticky="we")

        self._empirical_frame = ctk.CTkScrollableFrame(emp_wrap, width=390, height=260)
        self._empirical_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self._enable_mousewheel(self._empirical_frame)

        # --- USBM ---
        self._row_labeled(self._empirical_frame, "K_ppv", self._k_ppv, 0)
        self._row_labeled(self._empirical_frame, "β (PPV exponent)", self._beta, 1)
        self._row_labeled(self._empirical_frame, "K_air (dB)", self._k_air, 2)
        self._row_labeled(self._empirical_frame, "B_air (dB/dec)", self._b_air, 3)

        # --- Kuz–Ram Xm & RR ---
        self._row_labeled(self._empirical_frame, "Rock factor A (Kuz–Ram)", self._a_kuz, 4)
        self._row_labeled(self._empirical_frame, "RWS (%)", self._rws, 5)
        self._row_labeled(self._empirical_frame, "Holes per delay (HPD)", self._hpd, 6)
        self._row_labeled(self._empirical_frame, "Oversize threshold (mm)", self._x_ov, 7)

        # n selector
        nrow = ctk.CTkFrame(self._empirical_frame); nrow.grid(row=8, column=0, sticky="we", pady=(8,2))
        nrow.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(nrow, text="Uniformity index n").grid(row=0, column=0, sticky="w", padx=(4, 6))
        opt = ctk.CTkOptionMenu(nrow, values=["Estimate (Kuz–Ram)", "Manual (n)"], variable=self._n_mode,
                                command=lambda *_: self._update_n_display())
        opt.grid(row=0, column=1, sticky="we", padx=(0, 4))

        nrow2 = ctk.CTkFrame(self._empirical_frame); nrow2.grid(row=9, column=0, sticky="we")
        nrow2.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(nrow2, text="Manual n").grid(row=0, column=0, sticky="w", padx=(4, 6))
        entn = ctk.CTkEntry(nrow2, width=120); entn.grid(row=0, column=1, sticky="e", padx=(0, 4))
        entn.insert(0, f"{self._n_manual.get():.4g}")
        entn.bind("<Return>", lambda _e: self._set_from_entry(entn, self._n_manual))
        entn.bind("<FocusOut>", lambda _e: self._set_from_entry(entn, self._n_manual))

        nrow3 = ctk.CTkFrame(self._empirical_frame); nrow3.grid(row=10, column=0, sticky="we")
        ctk.CTkLabel(nrow3, textvariable=self._n_display, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=4)

        # small note
        ctk.CTkLabel(emp_wrap,
            text=("PPV = K_ppv·(R/√Qd)^-β;  Air = K_air + B_air·log10(Qd^(1/3)/R)\n"
                  "Kuz–Ram mean size: Xm = A·K^-0.8·Q^(1/6)·(115/RWS)^(19/20)\n"
                  "RR CDF: P(x)=1-exp[-(x/λ)^n],  λ=Xm/Γ(1+1/n)"),
            wraplength=370, justify="left", font=ctk.CTkFont(size=11)
        ).pack(padx=8, pady=(0,6), anchor="w")

        # Buttons under everything
        btns = ctk.CTkFrame(left)
        btns.grid(row=4, column=0, padx=12, pady=(0, 6), sticky="we")
        ctk.CTkButton(btns, text="Use Medians", command=self._set_medians)\
            .pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="Reset Ranges", command=self._init_sliders_from_dataset)\
            .pack(side="left")

        # Right panel with tabs
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        topbar = ctk.CTkFrame(right)
        topbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        self._btn_predict = ctk.CTkButton(topbar, text="Predict (ML + Empirical)", command=self._on_predict)
        self._btn_predict.pack(side="left", padx=(0, 10))
        self._progress = ctk.CTkProgressBar(topbar, mode="indeterminate")
        self._progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._btn_export = ctk.CTkButton(topbar, text="Export Result CSV",
                                         state="disabled", command=self._export_results)
        self._btn_export.pack(side="left")

        # Tabs for plots
        self._tabs = ctk.CTkTabview(right)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        tab_pred = self._tabs.add("Outputs (Empirical vs ML)")
        tab_rr   = self._tabs.add("Fragmentation (RR Curve)")

        # Plot 1: bars
        host1 = tk.Frame(tab_pred); host1.pack(fill="both", expand=True, padx=6, pady=6)
        self._fig = Figure(figsize=(7.4, 4.6), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title("Predicted outputs — Empirical vs ML")
        self._ax.set_ylabel("Value")
        self._canvas = FigureCanvasTkAgg(self._fig, master=host1)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Plot 2: RR curve
        host2 = tk.Frame(tab_rr); host2.pack(fill="both", expand=True, padx=6, pady=6)
        self._fig_rr = Figure(figsize=(7.4, 4.6), dpi=100)
        self._ax_rr = self._fig_rr.add_subplot(111)
        self._ax_rr.set_title("Rosin–Rammler Fragmentation (CDF)")
        self._ax_rr.set_xlabel("Size (mm)")
        self._ax_rr.set_ylabel("Passing (%)")
        self._canvas_rr = FigureCanvasTkAgg(self._fig_rr, master=host2)
        self._canvas_rr.get_tk_widget().pack(fill="both", expand=True)

        # Thresholds & log
        bottom = ctk.CTkCollapsibleFrame(right, text="Thresholds & Alerts")
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        th_grid = ctk.CTkFrame(bottom); th_grid.pack(fill="x", padx=6, pady=(2, 6))
        for j, o in enumerate(self.outputs):
            ctk.CTkLabel(th_grid, text=o).grid(row=j, column=0, sticky="w", padx=(4, 6), pady=3)
            ctk.CTkEntry(th_grid, textvariable=self._thresholds[o], width=110)\
                .grid(row=j, column=1, padx=4, pady=3, sticky="w")

        self._log = ctk.CTkTextbox(bottom, height=140)
        self._log.pack(fill="x", padx=6, pady=(0, 8))

        self._rebuild_sliders({name: (0.0, 0.0, 1.0) for name in self.input_labels})

    # --- make CTkScrollableFrame mouse-wheel friendly on all OSes ---
    def _enable_mousewheel(self, sf: ctk.CTkScrollableFrame):
        canvas = getattr(sf, "_parent_canvas", None)
        if canvas is None:
            return
        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            step = -1 if delta > 0 else 1
            canvas.yview_scroll(step, "units")
        sf.bind("<Enter>", lambda _e: sf.bind_all("<MouseWheel>", _on_mousewheel))
        sf.bind("<Leave>", lambda _e: sf.unbind_all("<MouseWheel>"))
        sf.bind("<Enter>", lambda _e: (sf.bind_all("<Button-4>", lambda _e: canvas.yview_scroll(-1, "units")),
                                       sf.bind_all("<Button-5>", lambda _e: canvas.yview_scroll(1, "units"))))
        sf.bind("<Leave>", lambda _e: (sf.unbind_all("<Button-4>"),
                                       sf.unbind_all("<Button-5>")))

    def _set_from_entry(self, ent: ctk.CTkEntry, var: ctk.DoubleVar):
        try: var.set(float(ent.get()))
        except: pass
        ent.delete(0, "end"); ent.insert(0, f"{var.get():.4g}")
        self._update_n_display()

    def _row_labeled(self, parent, label: str, var: ctk.DoubleVar, row: int):
        fr = ctk.CTkFrame(parent); fr.grid(row=row, column=0, sticky="we", pady=2)
        fr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(fr, text=label).grid(row=0, column=0, sticky="w", padx=(4, 6))
        ent = ctk.CTkEntry(fr, width=120); ent.grid(row=0, column=1, sticky="e", padx=(0, 4))
        ent.insert(0, f"{var.get():.4g}")
        ent.bind("<Return>", lambda _e: self._set_from_entry(ent, var))
        ent.bind("<FocusOut>", lambda _e: self._set_from_entry(ent, var))

    def _rebuild_sliders(self, ranges: Dict[str, Tuple[float, float, float]]):
        for w in self._inputs_frame.winfo_children():
            w.destroy()
        self._vars.clear()
        self._ranges = dict(ranges)

        # Build rows + keep RR n updated when key sliders move
        for i, name in enumerate(self.input_labels):
            vmin, vmed, vmax = self._ranges.get(name, (0.0, 0.0, 1.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmed, vmax = 0.0, 0.0, 1.0
            var = ctk.DoubleVar(value=float(np.clip(vmed, vmin, vmax)))
            self._vars[name] = var

            ctk.CTkLabel(self._inputs_frame, text=name).grid(row=2*i, column=0, padx=(6, 6), pady=(8, 2), sticky="w")

            row = ctk.CTkFrame(self._inputs_frame); row.grid(row=2*i+1, column=0, sticky="we", padx=6)
            row.grid_columnconfigure(0, weight=1)

            slider = ctk.CTkSlider(row, from_=float(vmin), to=float(vmax),
                                   number_of_steps=200, variable=var,
                                   command=lambda _=None: self._live_update_title())
            slider.grid(row=0, column=0, sticky="we", padx=(0, 8))

            ent = ctk.CTkEntry(row, width=90)
            ent.grid(row=0, column=1)
            ent.insert(0, f"{var.get():.4g}")

            def bind_entry(e=ent, v=var, lo=vmin, hi=vmax):
                try: x = float(e.get())
                except Exception: return
                v.set(float(np.clip(x, lo, hi)))
                e.delete(0, "end"); e.insert(0, f"{v.get():.4g}")

            ent.bind("<Return>", lambda _ev, f=bind_entry: f())
            ent.bind("<FocusOut>", lambda _ev, f=bind_entry: f())

            def on_slider_change(_=None, e=ent, v=var):
                e.delete(0, "end"); e.insert(0, f"{v.get():.4g}")
                self._update_n_display()
            var.trace_add("write", lambda *_args, f=on_slider_change: f())

        self._update_n_display()
        self._live_update_title()

    def _live_update_title(self):
        if self.input_labels:
            first = self.input_labels[0]
            val = self._vars[first].get() if first in self._vars else None
            if val is not None:
                self.title(f"Simultaneous Prediction  —  {first}: {val:.3g}")

    # ---------- dataset -> slider ranges ----------
    def _init_sliders_from_dataset(self):
        df, path = self.registry.get_dataset()
        if path:
            self._ds_tag.configure(text=f"Dataset: {path.split('/')[-1]}")
        else:
            self._ds_tag.configure(text="Dataset: (none)")

        if df is None or df.empty:
            self._rebuild_sliders({name: (0.0, 0.0, 1.0) for name in self.input_labels})
            return

        mapping = _resolve(list(df.columns), self.input_labels)
        ranges: Dict[str, Tuple[float, float, float]] = {}
        for name in self.input_labels:
            col = mapping.get(name)
            if col is None:
                ranges[name] = (0.0, 0.0, 1.0); continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                ranges[name] = (0.0, 0.0, 1.0); continue
            lo = float(s.quantile(0.02)); hi = float(s.quantile(0.98)); md = float(s.median())
            if lo == hi: lo, hi = float(s.min()), float(s.max())
            if lo == hi: lo, hi = (md - 0.5 if np.isfinite(md) else 0.0), (md + 0.5 if np.isfinite(md) else 1.0)
            ranges[name] = (lo, md, hi)

        self._rebuild_sliders(ranges)

    def _set_medians(self):
        for name, var in self._vars.items():
            lo, md, hi = self._ranges.get(name, (0.0, 0.0, 1.0))
            var.set(float(np.clip(md, lo, hi)))

    # ---------- helpers to read current inputs ----------
    def _val(self, pretty: str) -> float:
        v = self._vars.get(pretty)
        return float(v.get()) if v is not None else 0.0

    def _derived_charge_volume(self) -> Tuple[float, float, float, float]:
        """
        Returns (charge_per_hole, total_mass, charge_per_delay, volume)
        """
        depth = self._val("Hole depth (m)")
        stem  = self._val("Stemming (m)")
        Lc    = max(0.0, depth - stem)  # charged length
        lin   = self._val("Linear charge (kg/m)")
        mass_per_hole_lin = lin * Lc

        mass_per_hole_exp = self._val("Explosive mass (kg)")
        n_holes = max(1, int(round(self._val("# Holes"))))

        vol = self._val("Blast volume (m³)")
        if vol <= 0.0:
            B = self._val("Burden (m)"); S = self._val("Spacing (m)")
            vol = max(0.0, B * S * depth * n_holes)

        PF_slider = self._val("Powder factor (kg/m³)")

        per_hole = 0.0
        for cand in (mass_per_hole_lin, mass_per_hole_exp):
            if cand > 0:
                per_hole = cand; break

        total_mass = per_hole * n_holes
        if PF_slider > 0.0 and vol > 0.0:
            total_mass = PF_slider * vol
            if n_holes > 0:
                per_hole = total_mass / n_holes

        HPD = max(1.0, float(self._hpd.get()))
        Qd = per_hole * HPD

        return per_hole, total_mass, Qd, vol

    # ---------- n (RR) estimation ----------
    def _estimate_n_rr(self) -> Optional[float]:
        """
        Kuz–Ram-style uniformity: n = 2.2 - 14*(d/B), d in meters.
        Clipped to [0.5, 3.5].
        """
        B = self._val("Burden (m)")
        d_mm = self._val("Hole diameter (mm)")
        if B <= 0 or d_mm <= 0:
            return None
        d_m = d_mm / 1000.0
        n_hat = 2.2 - 14.0 * (d_m / max(B, 1e-9))
        return float(np.clip(n_hat, 0.5, 3.5))

    def _current_n(self) -> Optional[float]:
        if self._n_mode.get().startswith("Manual"):
            return float(self._n_manual.get())
        # Estimate
        est = self._estimate_n_rr()
        if est is None:
            # fallback to manual value if estimate impossible
            return float(self._n_manual.get())
        return est

    def _update_n_display(self):
        n = self._current_n()
        if self._n_mode.get().startswith("Manual"):
            self._n_display.set(f"n (manual): {n:.3g}")
        else:
            if n is None:
                self._n_display.set("n (estimated): —")
            else:
                self._n_display.set(f"n (estimated): {n:.3g}")

    # ---------- empirical (USBM + Kuz–Ram) ----------
    @staticmethod
    def _safe_log10(x: float, eps: float = 1e-12) -> float:
        return math.log10(max(x, eps))

    def _empirical_predictions(self) -> Dict[str, float]:
        """
        - PPV (USBM): PPV = K_ppv * (R/sqrt(Qd))^-β
        - Airblast (USBM): L = K_air + B_air * log10(Qd^(1/3)/R)
        - Fragmentation mean size Xm (Kuz–Ram 2005 form you requested):
              Xm = A * K^-0.8 * Q^(1/6) * (115/RWS)^(19/20)
          K from PF slider if >0 else Q_total/V. Q is charge per hole (kg).
        """
        R = self._val("Distance (m)")
        per_hole, total_mass, Qd, V = self._derived_charge_volume()

        emp: Dict[str, float] = {}

        # Ground Vibration (PPV)
        if "Ground Vibration" in self.outputs and R > 0 and Qd > 0:
            SD = R / math.sqrt(Qd)
            emp["Ground Vibration"] = float(self._k_ppv.get()) * (SD ** (-float(self._beta.get())))

        # Airblast
        if "Airblast" in self.outputs and R > 0 and Qd > 0:
            emp["Airblast"] = float(self._k_air.get()) + float(self._b_air.get()) * self._safe_log10((Qd ** (1.0/3.0)) / R)

        # Kuz–Ram mean fragment size (Xm)
        if "Fragmentation" in self.outputs:
            A   = max(0.0, float(self._a_kuz.get()))
            RWS = max(1e-6, float(self._rws.get()))
            K_pf = self._val("Powder factor (kg/m³)")
            if (K_pf <= 0.0) and (V > 0.0):
                K_pf = total_mass / V  # fallback if PF slider is zero
            if K_pf > 0.0 and per_hole > 0.0:
                emp["Fragmentation"] = A * (K_pf ** -0.8) * (per_hole ** (1.0/6.0)) * ((115.0 / RWS) ** (19.0/20.0))

        return emp

    # ---------- prediction ----------
    def _on_predict(self):
        self._btn_predict.configure(state="disabled")
        self._progress.start()
        self._log.delete("1.0", "end")
        self._log.insert("end", "Running predictions (ML + Empirical)…\n")
        threading.Thread(target=self._worker_predict, daemon=True).start()

    def _worker_predict(self):
        try:
            emp = self._empirical_predictions()

            ml: Dict[str, Optional[float]] = {k: None for k in self.outputs}
            ready, err = self.registry.status()
            if ready and not err:
                scaler = self.registry.get_scaler()
                models = self.registry.get_models()
                if scaler is not None and models:
                    values = {lbl: float(self._vars[lbl].get()) for lbl in self.input_labels}
                    X = self.registry.ensure_feature_vector(values)
                    Xs = scaler.transform(X)
                    for o in self.outputs:
                        mdl = models.get(o)
                        if mdl is not None:
                            ml[o] = float(mdl.predict(Xs)[0])

            self._plot_predictions(ml, emp)
            self._plot_rr(emp)  # new RR curve

            lines = []
            alerts = []
            for o in self.outputs:
                ml_txt = "NA" if (ml.get(o) is None or not np.isfinite(ml.get(o))) else f"{ml[o]:.3f}"
                em_txt = "NA" if (emp.get(o) is None or not np.isfinite(emp.get(o, np.nan))) else f"{emp[o]:.3f}"
                lines.append(f"{o}: ML={ml_txt} | Emp={em_txt}")
                thr = float(self._thresholds[o].get())
                v_for_alert = None
                if ml.get(o) is not None and np.isfinite(ml[o]): v_for_alert = ml[o]
                elif o in emp and np.isfinite(emp[o]): v_for_alert = emp[o]
                if (thr and v_for_alert is not None and v_for_alert > thr):
                    alerts.append(f"{o} exceeds {thr:.3f} (={v_for_alert:.3f})")
            if alerts:
                lines.append("\nAlerts (checked against ML if present, otherwise empirical):")
                lines.extend([f"• {a}" for a in alerts])

            row = {lbl: float(self._vars[lbl].get()) for lbl in self.input_labels}
            for o in self.outputs:
                row[f"ML {o}"] = ml.get(o, np.nan)
                row[f"Empirical {o}"] = emp.get(o, np.nan)
            # also store RR parameters if computed
            ncur, xm, x50 = self._current_rr_summary(emp)
            row["RR n"] = ncur if ncur is not None else np.nan
            row["KuzRam Xm (mm)"] = xm if xm is not None else np.nan
            row["RR X50 (mm)"] = x50 if x50 is not None else np.nan

            self._last_pred_row = pd.DataFrame([row])
            self._btn_export.configure(state="normal")

            self._clear_and_write("\n".join(lines))

        except Exception as e:
            self._clear_and_write(f"Error: {e}")
        finally:
            self._progress.stop()
            self._btn_predict.configure(state="normal")

    # ---------- plotting ----------
    def _plot_predictions(self, ml: Dict[str, Optional[float]], emp: Dict[str, float]):
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        names = list(self.outputs)
        x = np.arange(len(names))
        width = 0.38

        ml_vals = [ml.get(n) if (ml.get(n) is not None and np.isfinite(ml.get(n))) else np.nan for n in names]
        em_vals = [emp.get(n, np.nan) for n in names]

        ax.bar(x - width/2, em_vals, width, label="Empirical", alpha=0.85)
        ax.bar(x + width/2, ml_vals, width, label="ML", alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(names, rotation=0)
        ax.set_ylabel("Predicted value")
        ax.set_title("Predicted outputs — Empirical vs ML")
        ax.legend(loc="best")

        for i, n in enumerate(names):
            thr = float(self._thresholds[n].get())
            if thr:
                ax.axhline(thr, linestyle="--", linewidth=1, alpha=0.8, color="#444")
                ax.text(i, thr, f" thr={thr:.3g} ", va="bottom", ha="center", fontsize=8, color="#333")

        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _current_rr_summary(self, emp: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (n, Xm, X50) if computable."""
        xm = emp.get("Fragmentation", None)
        n = self._current_n()
        if (xm is None) or (not np.isfinite(xm)) or (n is None) or (n <= 0):
            return None, None, None
        lam = xm / math.gamma(1.0 + 1.0/n)
        x50 = lam * (math.log(2.0) ** (1.0/n))
        return n, xm, x50

    def _plot_rr(self, emp: Dict[str, float]):
        self._fig_rr.clear()
        ax = self._fig_rr.add_subplot(111)
        ax.set_title("Rosin–Rammler Fragmentation (CDF)")
        ax.set_xlabel("Size (mm)")
        ax.set_ylabel("Passing (%)")

        n, xm, x50 = self._current_rr_summary(emp)
        if (n is None) or (xm is None):
            ax.text(0.5, 0.5, "Insufficient inputs for RR curve.\nCheck PF, per-hole charge, A, RWS, and n.",
                    ha="center", va="center", transform=ax.transAxes)
            self._fig_rr.tight_layout(); self._canvas_rr.draw_idle(); return

        lam = xm / math.gamma(1.0 + 1.0/n)

        # x-grid (mm), log-spaced
        x_ov = max(1e-9, float(self._x_ov.get()))
        xmax = max(6.0*xm, 1.5*x_ov, 10.0)
        xs = np.logspace(math.log10(max(0.1, xm/20.0)), math.log10(xmax), 240)
        P = 100.0 * (1.0 - np.exp(- (xs / lam) ** n))

        ax.plot(xs, P, lw=2.0, alpha=0.9)

        # markers: X20, X50, X80
        def x_at(pct):
            return lam * ((-math.log(1.0 - pct)) ** (1.0/n))
        x20 = x_at(0.20); x80 = x_at(0.80)

        for xv, label in [(x20, "X20"), (x50, "X50"), (x80, "X80")]:
            ax.axvline(xv, ls="--", lw=1.0, color="#777")
            ax.text(xv, 2, f"{label}={xv:.1f} mm", rotation=90, va="bottom", ha="right", fontsize=9, color="#444")

        # oversize %
        oversize = 100.0 * float(np.exp(- (x_ov / lam) ** n))
        ax.axvline(x_ov, ls=":", lw=1.0, color="#555")
        ax.text(x_ov, 60, f"Oversize@{x_ov:.0f}mm = {oversize:.1f}%", rotation=90,
                va="center", ha="right", fontsize=9, color="#333")

        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", axis="both", alpha=0.2)
        ax.text(0.02, 0.06, f"n = {n:.2f}   Xm = {xm:.1f} mm", transform=ax.transAxes, fontsize=10)

        self._fig_rr.tight_layout()
        self._canvas_rr.draw_idle()

    # ---------- export ----------
    def _export_results(self):
        if self._last_pred_row is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            self._last_pred_row.to_csv(path, index=False)
            messagebox.showinfo("Export", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save:\n{e}")

    # ---------- small I/O helpers ----------
    def _clear_and_write(self, text: str):
        self._log.delete("1.0", "end"); self._log.insert("end", text + "\n"); self._log.see("end")


# ---- Standalone quick test ----
if __name__ == "__main__":
    import os, joblib

    class _MockReg:
        def __init__(self):
            self._loaded = True
            self._err = None
            try:
                self._scaler = joblib.load("scaler1.joblib") if os.path.exists("scaler1.joblib") else None
            except Exception as e:
                self._scaler, self._err = None, str(e)
            self._models = {}
            for name in ["Fragmentation", "Ground Vibration", "Airblast"]:
                fname = f"random_forest_model_{name}.joblib"
                if os.path.exists(fname):
                    try:
                        self._models[name] = joblib.load(fname)
                    except Exception:
                        pass
            self._df, self._path = None, None
            for candidate in ["combinedv2Orapa.csv", "combinedv2Jwaneng.csv"]:
                if os.path.exists(candidate):
                    try:
                        self._df = pd.read_csv(candidate); self._path = os.path.abspath(candidate)
                        break
                    except Exception:
                        pass

        def status(self): return self._loaded, self._err
        def get_scaler(self): return self._scaler
        def get_models(self): return dict(self._models)
        def get_dataset(self): return self._df, self._path
        @staticmethod
        def ensure_feature_vector(values: Dict[str, float]):
            return np.asarray([float(values[k]) for k in INPUT_LABELS], dtype=float).reshape(1, -1)

    INPUT_LABELS = [
        "Hole depth (m)", "Hole diameter (mm)", "Burden (m)", "Spacing (m)",
        "Stemming (m)", "Distance (m)", "Powder factor (kg/m³)", "Rock density (t/m³)",
        "Linear charge (kg/m)", "Explosive mass (kg)", "Blast volume (m³)", "# Holes",
    ]
    OUTPUTS = ["Fragmentation", "Ground Vibration", "Airblast"]

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Prediction — Standalone")
    PredictionWindow(root, _MockReg(), INPUT_LABELS, OUTPUTS)
    root.mainloop()


# In[ ]:




