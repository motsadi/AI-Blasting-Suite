#!/usr/bin/env python
# coding: utf-8

# In[5]:


# flyrock.py — ML + Empirical flyrock predictor (CTk)
# Python 3.8/3.9 compatible

import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ---------- CTk compatibility shim ----------
if not hasattr(ctk, "CTkCollapsibleFrame"):
    class CTkCollapsibleFrame(ctk.CTkFrame):
        def __init__(self, master=None, text: str = "", **kwargs):
            super().__init__(master, **kwargs)
            if text:
                ctk.CTkLabel(
                    self,
                    text=text,
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=6, pady=(6, 2))
    ctk.CTkCollapsibleFrame = CTkCollapsibleFrame
# -------------------------------------------


# --------- header helpers & synonyms ---------
def _norm(s: str) -> str:
    """Normalize header: lowercase, strip punctuation/units, collapse spaces."""
    s = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s))
    s = " ".join(s.split())
    # remove very common unit tokens so matching is robust
    s = (s.replace(" kg/m3", "").replace(" kg m3", "").replace(" kg m^3", "")
           .replace(" kn/m3", "").replace(" kn m3", "")
           .replace("(m)", "").replace(" m", "").replace(" mm", "")
           .replace(" kpa", ""))
    return " ".join(s.split())

# feature/target synonym pools (normalized text)
BURDEN_SYNS   = ["burden", "b"]
SPACING_SYNS  = ["spacing", "s"]
STEMMING_SYNS = ["stemming", "stem"]
DIAM_SYNS     = ["hole diameter", "diameter", "hole dia", "bh dia", "bit diameter", "drill diameter", "d"]
BENCH_SYNS    = ["bench height", "bench", "height", "h"]
CHARGE_SYNS   = ["charge per delay", "charge", "explosive mass", "charge per hole"]
PF_SYNS       = ["powder factor", "specific charge", "pf", "q"]
ROCKD_SYNS    = ["rock density", "density", "t/m3", "tm3", "rho"]
SDOB_SYNS     = ["sdob", "s/b", "spacing to burden", "spacing burden ratio"]
LUNDBORG_SYNS = ["lundborg", "lundborg distance", "lundborg m"]

TARGET_SYNS   = ["flyrock", "fly rock", "flyrock distance", "throw", "max throw"]


def _resolve_one(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    cols = list(df_cols)
    norm_map = {_norm(c): c for c in cols}
    # exact/normalized
    for cand in candidates:
        nc = _norm(cand)
        if nc in norm_map:
            return norm_map[nc]
    # contains fallback
    for c in cols:
        cn = _norm(c)
        if any(_norm(cand) in cn for cand in candidates):
            return c
    return None


def _infer_target(df: pd.DataFrame) -> Optional[str]:
    name = _resolve_one(df.columns.tolist(), TARGET_SYNS)
    if name is not None:
        return name
    # fallback: last numeric with variance
    num = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    if num.shape[1] >= 1:
        last = num.columns[-1]
        if np.isfinite(num[last]).sum() >= 10 and float(num[last].std()) > 1e-9:
            return last
    return None


# --------- Data/model loader ---------
def _load_flyrock_csv(path: str) -> Tuple[pd.DataFrame, str]:
    last_err = None
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
            return df, path
        except Exception as e:
            last_err = e
    raise last_err


def _pick_numeric_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], str]:
    tgt = _infer_target(df)
    if tgt is None:
        raise ValueError("Could not infer flyrock target column. "
                         "Expected a column like: Flyrock, Flyrock (m), Throw, Max Throw, etc.")

    num = df.apply(pd.to_numeric, errors="coerce")
    y = num[tgt]
    X = num.drop(columns=[tgt]).copy()

    # ---- drop engineered or non-physical columns from ML features (no sliders for these) ----
    drop_ml = []
    for c in X.columns:
        cn = _norm(c)
        if any(_norm(s) in cn for s in LUNDBORG_SYNS + SDOB_SYNS):
            drop_ml.append(c)
    if drop_ml:
        X = X.drop(columns=drop_ml, errors="ignore")

    # keep columns with enough finite values
    valid_cols = [c for c in X.columns if np.isfinite(X[c]).sum() >= max(20, int(0.3 * len(X)))]
    X = X[valid_cols]

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask].copy()
    y = y[mask].copy()

    if X.shape[0] < 30 or X.shape[1] < 2:
        raise ValueError("Not enough clean rows/features to train after cleaning.")

    return X, y, X.columns.tolist(), tgt


# --------- Empirical relations ----------
def _lookup_norm(nrow: Dict[str, float], syns: List[str], default: float = np.nan) -> float:
    """Robust: exact normalized key OR substring match in any key."""
    # fast path: exact
    keys = list(nrow.keys())
    for s in syns:
        ns = _norm(s)
        if ns in nrow:
            return float(nrow[ns])
    # slow path: contains
    norm_syns = [_norm(s) for s in syns]
    for k in keys:
        if any(ns in k for ns in norm_syns):
            return float(nrow[k])
    return float(default)


def _derive_sdob_if_possible(vals: Dict[str, float]) -> Optional[float]:
    sd = _lookup_norm(vals, SDOB_SYNS, default=np.nan)
    if np.isfinite(sd):
        return float(sd)
    # Try r_u-like proxy: stemming / charge^(1/3)
    stem = _lookup_norm(vals, STEMMING_SYNS, default=np.nan)  # m
    qd   = _lookup_norm(vals, CHARGE_SYNS,   default=np.nan)  # kg/delay
    if np.isfinite(stem) and np.isfinite(qd) and qd > 0:
        return float(stem / (qd ** (1.0/3.0)))
    return None


def emp_lundborg_1981(vals: Dict[str, float]) -> float:
    """R = 143 * d_in * (q - 0.2),  d in inches, q in kg/m^3."""
    Dmm = _lookup_norm(vals, DIAM_SYNS, default=np.nan)
    q   = _lookup_norm(vals, PF_SYNS,   default=np.nan)
    if not np.isfinite(Dmm) or not np.isfinite(q):
        return float('nan')
    d_in = Dmm / 25.4
    return float(max(0.0, 143.0 * d_in * (q - 0.2)))


def emp_mckenzie_sdob(vals: Dict[str, float]) -> float:
    """R = 10 * d_mm^0.667 * SDoB^-2.167 * (rho/2.6)."""
    Dmm = _lookup_norm(vals, DIAM_SYNS,  default=np.nan)
    rho = _lookup_norm(vals, ROCKD_SYNS, default=2.6)
    sd  = _derive_sdob_if_possible(vals)
    if not np.isfinite(Dmm) or sd is None or sd <= 0:
        return float('nan')
    return float(10.0 * (max(Dmm, 0.0) ** 0.667) * (sd ** -2.167) * (rho / 2.6))


def emp_lundborg_legacy(vals: Dict[str, float]) -> float:
    """R = 30.745 * d_mm^0.66  (rough legacy form)."""
    Dmm = _lookup_norm(vals, DIAM_SYNS, default=np.nan)
    if not np.isfinite(Dmm):
        return float('nan')
    return float(30.745 * (max(Dmm, 0.0) ** 0.66))


def empirical_auto(vals: Dict[str, float]) -> Tuple[float, str]:
    """
    Prefer Lundborg (1981) when Diameter & PF are present.
    Else use McKenzie/SDoB when SDoB present or derivable.
    Else fall back to legacy.
    """
    L81 = emp_lundborg_1981(vals)
    if np.isfinite(L81):
        return L81, "Lundborg (1981): 143*d_in*(q-0.2)"
    MK = emp_mckenzie_sdob(vals)
    if np.isfinite(MK):
        return MK, "McKenzie/SDoB: 10*d_mm^0.667*SDoB^-2.167*(ρ/2.6)"
    LG = emp_lundborg_legacy(vals)
    return LG, "Legacy d-only: 30.745*d_mm^0.66"


# --------- UI Window ----------
class FlyrockWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Flyrock — ML + Empirical")
        self.geometry("1180x780")
        self.minsize(1080, 720)

        self.df = None
        self.model: Optional[RandomForestRegressor] = None
        self.target_name: Optional[str] = None
        self.X_cols: List[str] = []

        self.sliders: Dict[str, ctk.DoubleVar] = {}
        self.slider_bounds: Dict[str, Tuple[float, float]] = {}
        self.slider_value_vars: Dict[str, ctk.StringVar] = {}

        self._surf_x = ctk.StringVar(self, value="")
        self._surf_y = ctk.StringVar(self, value="")

        self._build_ui()

        try:
            if os.path.exists("flyrock_synth.csv"):
                self._load_path("flyrock_synth.csv")
        except Exception:
            pass

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left
        left = ctk.CTkFrame(self, width=360)
        left.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)
        left.grid_rowconfigure(99, weight=1)

        ctk.CTkLabel(left, text="Data", font=ctk.CTkFont(size=16, weight="bold"))\
            .grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")
        ctk.CTkButton(left, text="Load CSV", command=self._choose_csv)\
            .grid(row=1, column=0, padx=10, pady=4, sticky="we")

        self._pred_label = ctk.CTkLabel(left, text="Predicted flyrock: —", font=ctk.CTkFont(size=15, weight="bold"))
        self._pred_label.grid(row=2, column=0, padx=10, pady=(10, 4), sticky="w")

        self._emp_label = ctk.CTkLabel(left, text="Empirical estimate: —", font=ctk.CTkFont(size=13))
        self._emp_label.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="w")

        self._sliders_panel = ctk.CTkScrollableFrame(left, width=340, height=500)
        self._sliders_panel.grid(row=4, column=0, padx=10, pady=6, sticky="nswe")

        # Right / plots
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nswe", padx=(6, 12), pady=12)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        surface_ctrl = ctk.CTkFrame(right)
        surface_ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 6))
        ctk.CTkLabel(surface_ctrl, text="Surface axes:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        self._x_menu = ctk.CTkOptionMenu(surface_ctrl, values=[], variable=self._surf_x, command=lambda _v=None: self._draw_surface())
        self._x_menu.pack(side="left", padx=(0, 8))
        self._y_menu = ctk.CTkOptionMenu(surface_ctrl, values=[], variable=self._surf_y, command=lambda _v=None: self._draw_surface())
        self._y_menu.pack(side="left", padx=(0, 12))
        ctk.CTkButton(surface_ctrl, text="Redraw surface", command=self._draw_surface).pack(side="left")

        plot_host = ctk.CTkFrame(right)
        plot_host.grid(row=1, column=0, sticky="nswe", padx=10, pady=(0, 10))
        plot_host.grid_rowconfigure(0, weight=1); plot_host.grid_columnconfigure(0, weight=1)
        pack_parent = tk.Frame(plot_host)
        pack_parent.grid(row=0, column=0, sticky="nsew")

        self._fig = Figure(figsize=(7.8, 5.8), dpi=100)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._fig, master=pack_parent)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, pack_parent).update()

        note = ctk.CTkLabel(
            right,
            text="Empirical: auto-chooses Lundborg (1981), McKenzie/SDoB (SDoB derived if possible), or legacy d-only.",
            justify="left"
        )
        note.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))

    # ---------- data/model ----------
    def _choose_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return
        self._load_path(p)

    def _load_path(self, path: str):
        try:
            df, used_path = _load_flyrock_csv(path)
            X, y, cols, tgt = _pick_numeric_and_target(df)

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=500, random_state=42)
            rf.fit(Xtr, ytr)

            self.df = df
            self.model = rf
            self.target_name = tgt
            self.X_cols = cols

            # sliders
            self.sliders.clear(); self.slider_bounds.clear(); self.slider_value_vars.clear()
            for name in self.X_cols:
                col = pd.to_numeric(df[name], errors="coerce").dropna()
                if col.empty:
                    vmin, vmax, v0 = 0.0, 1.0, 0.5
                else:
                    vmin = float(col.quantile(0.02))
                    vmax = float(col.quantile(0.98))
                    if vmin == vmax:
                        vmin, vmax = float(col.min()), float(col.max())
                    v0 = float(col.median())
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax, v0 = 0.0, 1.0, 0.5
                self.slider_bounds[name] = (vmin, vmax)
                self.sliders[name] = ctk.DoubleVar(value=v0)
                self.slider_value_vars[name] = ctk.StringVar(value=self._fmt_val(v0))

            for w in self._sliders_panel.winfo_children():
                w.destroy()

            ctk.CTkLabel(self._sliders_panel, text="Adjust Inputs",
                         font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=8, pady=(4, 6))

            for name in self.X_cols:
                vmin, vmax = self.slider_bounds[name]
                var = self.sliders[name]
                sval = self.slider_value_vars[name]

                row = ctk.CTkFrame(self._sliders_panel)
                row.pack(fill="x", padx=8, pady=(8, 0))

                top = ctk.CTkFrame(row)
                top.pack(fill="x")
                ctk.CTkLabel(top, text=name).pack(side="left")
                ctk.CTkLabel(top, textvariable=sval).pack(side="right")

                def _mk_cmd(_var=var, _sval=sval):
                    def _on_move(_=None):
                        _sval.set(self._fmt_val(float(_var.get())))
                        self._update_prediction()
                    return _on_move

                s = ctk.CTkSlider(row, from_=vmin, to=vmax,
                                  number_of_steps=200 if vmax > vmin else 1,
                                  variable=var, command=_mk_cmd())
                s.pack(fill="x")

            self._x_menu.configure(values=self.X_cols)
            self._y_menu.configure(values=self.X_cols)
            if len(self.X_cols) >= 2:
                self._surf_x.set(self.X_cols[0])
                self._surf_y.set(self.X_cols[1])
                self._x_menu.set(self._surf_x.get()); self._y_menu.set(self._surf_y.get())

            messagebox.showinfo("Flyrock", f"Loaded: {used_path}\nTarget: {tgt}\nFeatures: {len(self.X_cols)}")
            self._update_prediction()
            self._draw_surface()

        except Exception as e:
            messagebox.showerror("Flyrock", f"Failed to load:\n{e}")

    # ---------- prediction ----------
    def _fmt_val(self, v: float) -> str:
        av = abs(v)
        if av >= 1000: return f"{v:,.0f}"
        if av >= 100:  return f"{v:,.1f}"
        if av >= 10:   return f"{v:,.2f}"
        return f"{v:,.3f}"

    def _current_feature_vector_df(self) -> Optional[pd.DataFrame]:
        """Return a single-row DataFrame with the exact training feature names."""
        if self.model is None or not self.X_cols:
            return None
        data = {name: float(self.sliders[name].get()) for name in self.X_cols}
        return pd.DataFrame([data], columns=self.X_cols)

    def _empirical_current(self) -> Tuple[float, str]:
        # Build a normalized map of slider values (key = normalized feature name)
        vals: Dict[str, float] = {}
        for name in self.X_cols:
            vals[_norm(name)] = float(self.sliders[name].get())
        return empirical_auto(vals)

    def _update_prediction(self):
        if self.model is None:
            self._pred_label.configure(text="Predicted flyrock: —")
            self._emp_label.configure(text="Empirical estimate: —")
            return

        Xstar_df = self._current_feature_vector_df()
        yhat = float(self.model.predict(Xstar_df)[0]) if Xstar_df is not None else float("nan")
        emp, method = self._empirical_current()

        self._pred_label.configure(text=f"Predicted flyrock: {self._fmt_val(yhat)} m")
        self._emp_label.configure(text=f"Empirical estimate: {self._fmt_val(emp)} m  ({method})")

    # ---------- surface ----------
    def _draw_surface(self):
        if self.model is None or not self.X_cols:
            return
        x_name = self._surf_x.get(); y_name = self._surf_y.get()
        if (x_name not in self.X_cols) or (y_name not in self.X_cols) or (x_name == y_name):
            return

        x_min, x_max = self.slider_bounds.get(x_name, (0.0, 1.0))
        y_min, y_max = self.slider_bounds.get(y_name, (0.0, 1.0))
        xs = np.linspace(x_min, x_max, 50)
        ys = np.linspace(y_min, y_max, 50)
        XX, YY = np.meshgrid(xs, ys)

        base_df = self._current_feature_vector_df()
        if base_df is None:
            return
        G = XX.size
        DM = np.repeat(base_df.values, G, axis=0)
        ix = self.X_cols.index(x_name); iy = self.X_cols.index(y_name)
        DM[:, ix] = XX.ravel(); DM[:, iy] = YY.ravel()
        DM_df = pd.DataFrame(DM, columns=self.X_cols)  # keep feature names → no sklearn warnings

        try:
            Z = self.model.predict(DM_df).reshape(XX.shape)
        except Exception:
            Z = np.zeros_like(XX)

        self._fig.clear()
        ax = self._fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XX, YY, Z, cmap="viridis", edgecolor="none", linewidth=0, antialiased=True, alpha=0.97)
        ax.set_xlabel(x_name); ax.set_ylabel(y_name); ax.set_zlabel(self.target_name or "Flyrock (m)")
        ax.set_title(f"Flyrock surface: {self.target_name or 'Flyrock (m)'} vs ({x_name}, {y_name})")
        self._fig.colorbar(surf, ax=ax, shrink=0.65, aspect=18)
        self._fig.tight_layout()
        self._canvas.draw_idle()


# ---------- Standalone run ----------
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Flyrock — Standalone")
    FlyrockWindow(root)
    root.mainloop()


# In[ ]:




