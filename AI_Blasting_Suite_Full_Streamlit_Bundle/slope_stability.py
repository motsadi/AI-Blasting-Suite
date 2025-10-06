#!/usr/bin/env python
# coding: utf-8

# In[2]:


# slope_stability.py  —  Stable / Failure classification (CTk + RandomForest)

import os
import math
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Arc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------- CTk compatibility shim ----------
# Some CustomTkinter versions don't include CTkCollapsibleFrame.
if not hasattr(ctk, "CTkCollapsibleFrame"):
    class CTkCollapsibleFrame(ctk.CTkFrame):
        def __init__(self, master=None, text: str = "", **kwargs):
            super().__init__(master, **kwargs)
            if text:
                ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=13, weight="bold"))\
                    .pack(anchor="w", padx=6, pady=(6, 2))
    ctk.CTkCollapsibleFrame = CTkCollapsibleFrame
# -------------------------------------------


# ---------- header normalization & mapping ----------
def _norm(name: str) -> str:
    s = str(name).strip().lower()
    greek = {"γ": "gamma", "𝛾": "gamma", "𝜸": "gamma", "𝝲": "gamma",
             "φ": "phi",   "ϕ": "phi",   "𝜑": "phi",   "𝝋": "phi",
             "β": "beta",  "𝛽": "beta",  "𝜷": "beta"}
    for g, rep in greek.items():
        s = s.replace(g, rep)
    s = (s.replace("kn/m3", "")
           .replace("kpa", "")
           .replace("(m)", "")
           .replace("°", "")
           .replace("/", " ")
           .replace("(", " ")
           .replace(")", " ")
           .replace("_", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return s


CANON: Dict[str, List[str]] = {
    "gamma_kN_m3": ["gamma", "unit weight", "gamma kn m3", "gamma knm3"],
    "c_kPa":       ["c", "cohesion"],
    "phi_deg":     ["phi", "friction angle", "phi deg"],
    "beta_deg":    ["beta", "slope angle", "beta deg"],
    "H_m":         ["h", "height", "h m"],
    "ru":          ["ru", "r u", "pore pressure ratio", "pore pressure ratio ru"],
    "status":      ["status", "class", "label"]
}


def _map_by_name(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    nmap = {c: _norm(c) for c in df.columns}
    for canon, options in CANON.items():
        for c, n in nmap.items():
            if n in options:
                rename[c] = canon
                break
    out = df.rename(columns=rename).copy()
    # drop obvious index-like column
    for c in list(out.columns):
        if _norm(c) in {"no", "#", "index", "id", "sr"}:
            out = out.drop(columns=[c])
    return out


def _map_by_position(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if _norm(out.columns[0]) in {"no", "#", "index", "id", "sr"} and out.shape[1] >= 8:
        out = out.drop(columns=[out.columns[0]])
    if out.shape[1] == 7:
        out.columns = ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]
    return out


def _load_slope_csv(path: str) -> pd.DataFrame:
    # robust read
    last_err: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err  # type: ignore[misc]

    orig_cols = list(df.columns)
    df1 = _map_by_name(df)
    needed = ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]
    if not all(c in df1.columns for c in needed):
        df2 = _map_by_position(df)
        if all(c in df2.columns for c in needed):
            df_use = df2
        else:
            raise ValueError(
                f"Missing required columns. Expected {needed}, "
                f"got {list(df1.columns)} from headers {orig_cols}"
            )
    else:
        df_use = df1

    for c in ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru"]:
        df_use[c] = pd.to_numeric(df_use[c], errors="coerce")

    status_map = {"stable": "stable", "failure": "failure", "failed": "failure", "unstable": "failure"}
    df_use["status"] = df_use["status"].astype(str).str.strip().str.lower().map(status_map)

    df_use = df_use.dropna(subset=["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]).copy()
    df_use["y"] = (df_use["status"] == "stable").astype(int)
    if df_use["y"].nunique() < 2:
        raise ValueError("Dataset contains only one class (all Stable or all Failure).")
    return df_use


# ---------- UI Window ----------
class SlopeStabilityWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Slope Stability — Stable / Failure (ML)")
        self.geometry("1100x740")
        self.minsize(1000, 680)

        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[Pipeline] = None  # StandardScaler + RandomForestClassifier

        self._build_ui()

        # auto-load if "slope data.csv" exists
        try:
            if os.path.exists("slope data.csv"):
                self._load_path("slope data.csv")
        except Exception:
            pass

    # ---- UI layout ----
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # left toolbar
        left = ctk.CTkFrame(self, width=320)
        left.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)
        left.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(left, text="Data", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkButton(left, text="Load CSV", command=self._load_csv_dialog)\
            .grid(row=1, column=0, padx=10, pady=4, sticky="we")

        ctk.CTkLabel(left, text="Parameters", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=2, column=0, padx=10, pady=(10, 4), sticky="w"
        )

        self._H   = ctk.DoubleVar(value=10.0)
        self._beta= ctk.DoubleVar(value=30.0)
        self._c   = ctk.DoubleVar(value=50.0)
        self._phi = ctk.DoubleVar(value=30.0)
        self._gam = ctk.DoubleVar(value=20.0)
        self._ru  = ctk.DoubleVar(value=0.2)
        self._B   = ctk.DoubleVar(value=4.0)  # bench length for sketch only

        def mk_slider(row, text, var, a, b, step):
            ctk.CTkLabel(left, text=text).grid(row=row, column=0, padx=10, pady=(8, 2), sticky="w")
            s = ctk.CTkSlider(
                left, from_=a, to=b,
                number_of_steps=max(int((b - a) / step), 1),
                variable=var,
                command=lambda _=None: self._update_prediction_and_plot()
            )
            s.grid(row=row + 1, column=0, padx=10, sticky="we")

        mk_slider(3,  "H (m)",            self._H,   1.0, 50.0, 0.5)
        mk_slider(5,  "β (deg)",          self._beta, 5.0, 80.0, 0.5)
        mk_slider(7,  "c (kPa)",          self._c,    1.0, 200.0, 0.5)
        mk_slider(9,  "φ (deg)",          self._phi,  5.0, 60.0, 0.5)
        mk_slider(11, "γ (kN/m³)",        self._gam, 14.0, 28.0, 0.1)
        mk_slider(13, "ru (–)",           self._ru,   0.0,  1.0, 0.01)
        mk_slider(15, "B (m) — sketch only", self._B, 0.0, 30.0, 0.5)

        # prediction label
        self._pred = ctk.CTkLabel(left, text="Prediction: —", font=ctk.CTkFont(size=15, weight="bold"))
        self._pred.grid(row=17, column=0, padx=10, pady=(10, 6), sticky="w")

        # right: plot area
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nswe", padx=(6, 12), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        host = tk.Frame(right)
        host.grid(row=0, column=0, sticky="nsew")

        self._fig = Figure(figsize=(7.8, 5.6), dpi=100)
        self._ax = self._fig.add_subplot(111)

        self._canvas = FigureCanvasTkAgg(self._fig, master=host)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, host).update()

        self._draw_slope()
        self._fig.tight_layout()

    # ---- data & model ----
    def _load_csv_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        try:
            df = _load_slope_csv(path)
            self.data = df

            # defaults from data medians
            self._H.set(float(np.median(df["H_m"])))
            self._beta.set(float(np.median(df["beta_deg"])))
            self._c.set(float(np.median(df["c_kPa"])))
            self._phi.set(float(np.median(df["phi_deg"])))
            self._gam.set(float(np.median(df["gamma_kN_m3"])))
            self._ru.set(float(np.median(df["ru"])))
            self._B.set(max(0.0, 0.4 * self._H.get()))

            # train Random Forest pipeline
            X = df[["H_m", "beta_deg", "c_kPa", "phi_deg", "gamma_kN_m3", "ru"]].values
            y = df["y"].values
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            self.model = Pipeline([
                ("sc", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced"))
            ]).fit(Xtr, ytr)

            messagebox.showinfo("Slope Stability", f"Loaded {len(df)} rows from:\n{path}")
            self._update_prediction_and_plot()

        except Exception as e:
            messagebox.showerror("Slope Stability", f"Failed to load:\n{e}")

    def _predict_prob(self) -> Optional[float]:
        if self.model is None:
            return None
        X = np.array([[self._H.get(), self._beta.get(), self._c.get(),
                       self._phi.get(), self._gam.get(), self._ru.get()]])
        p = float(self.model.predict_proba(X)[0, 1])  # P(stable)
        return p

    # ---- drawing ----
    def _draw_slope(self):
        self._ax.clear()
        H = self._H.get()
        beta = self._beta.get()
        B = self._B.get()

        beta_rad = math.radians(beta)
        run = H / max(math.tan(beta_rad), 1e-3)
        toe_x = B + run

        x = [0.0, 0.0, B, toe_x, 0.0]
        y = [0.0, H,   H,  0.0,  0.0]
        self._ax.fill(x, y, color="#d6d7db", zorder=1)
        self._ax.plot(x, y, "k-", linewidth=2, zorder=2)
        self._ax.plot([0, toe_x * 1.06], [0, 0], "k-", linewidth=1)

        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlim(-0.08 * toe_x, toe_x * 1.12)
        self._ax.set_ylim(-0.06 * H, H * 1.24)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        for spine in ["top", "right", "left", "bottom"]:
            self._ax.spines[spine].set_visible(False)
        # Y axis line
        self._ax.plot([0, 0], [0, H * 1.05], color="black", linewidth=1.5)

        # H arrow
        self._ax.annotate("", xy=(-0.05 * toe_x, H), xytext=(-0.05 * toe_x, 0),
                          arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2))
        self._ax.text(-0.07 * toe_x, H / 2, f"H = {H:.1f} m",
                      color="#1f77b4", rotation=90, va="center", ha="right", fontsize=10)

        # beta arc & label
        arc_r = 0.10 * min(H, run)
        arc = Arc((toe_x, 0), width=2 * arc_r, height=2 * arc_r,
                  angle=0, theta1=180 - beta, theta2=180, color="black", lw=1.5)
        self._ax.add_patch(arc)
        self._ax.text(toe_x - arc_r * 0.80, arc_r * 0.50, f"β = {beta:.1f}°", fontsize=10)

        # Bench annotation
        if B > 0:
            self._ax.annotate("", xy=(0, H * 1.06), xytext=(B, H * 1.06),
                              arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
            self._ax.text(B / 2, H * 1.08, f"B = {B:.1f} m", color="#555",
                          ha="center", va="bottom", fontsize=9)

        # prediction banner outside drawing area (upper right corner)
        p = self._predict_prob()
        if p is not None:
            label = "Stable" if p >= 0.5 else "Failure"
            col = "#1ca04a" if p >= 0.5 else "#c0392b"
            self._ax.text(toe_x * 0.65, H * 1.14, f"{label}  (P(stable)={p:.2f})",
                          fontsize=12, color="white",
                          bbox=dict(facecolor=col, edgecolor="none", boxstyle="round,pad=0.3"))

        self._canvas.draw_idle()

    def _update_prediction_and_plot(self):
        p = self._predict_prob()
        if p is None:
            self._pred.configure(text="Prediction: —")
        else:
            label = "🟢 Stable" if p >= 0.5 else "🔴 Failure"
            self._pred.configure(text=f"Prediction: {label}   (P(stable)={p:.2f})")
        self._draw_slope()


# ---- Standalone run for quick tests ----
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Slope Stability — Standalone")
    SlopeStabilityWindow(root)
    root.mainloop()


# In[ ]:




