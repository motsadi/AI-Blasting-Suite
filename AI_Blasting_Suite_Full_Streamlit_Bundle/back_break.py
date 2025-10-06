#!/usr/bin/env python
# coding: utf-8

# In[3]:


# back_break.py  (Python 3.8/3.9 compatible typing)

import os
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---- typing for 3.8/3.9 ----
from typing import Optional, List, Tuple, Dict

# ---------- CTk compatibility shim ----------
if not hasattr(ctk, "CTkCollapsibleFrame"):
    class CTkCollapsibleFrame(ctk.CTkFrame):
        def __init__(self, master=None, text: str = "", **kwargs):
            super().__init__(master, **kwargs)
            if text:
                ctk.CTkLabel(
                    self, text=text,
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=6, pady=(6, 2))
    ctk.CTkCollapsibleFrame = CTkCollapsibleFrame
# -------------------------------------------

# --------- helpers ---------
def _normalize_col(s: str) -> str:
    keep = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s))
    return " ".join(keep.split())

def _infer_target(df: pd.DataFrame) -> Optional[str]:
    """Try to find a Back Break target column."""
    names = list(df.columns)
    norm_map = {_normalize_col(c): c for c in names}
    candidates = [
        "backbreak", "back break", "bb",
        "backbreak m", "backbreak mm", "back break m", "back break mm"
    ]
    for key in candidates:
        if key in norm_map:
            return norm_map[key]
    # fallback: last column if numeric with variance
    last = names[-1]
    try:
        x = pd.to_numeric(df[last], errors="coerce")
        if x.notna().sum() >= 10 and float(x.std()) > 1e-9:
            return last
    except Exception:
        pass
    return None

# --------- window ---------
class BackbreakWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Back Break — Random Forest")
        self.geometry("1100x740")
        self.minsize(1000, 680)

        # data / model
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestRegressor] = None
        self.target_name: Optional[str] = None

        # UI state
        self.top_feats: List[Tuple[str, float, float]] = []  # (name, vmin, vmax)
        self._feat_vars: Dict[str, ctk.DoubleVar] = {}

        self._build_ui()

        if os.path.exists("Backbreak.csv"):
            try:
                self._load_path("Backbreak.csv")
            except Exception:
                pass

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # left panel
        left = ctk.CTkFrame(self, width=360)
        left.grid(row=0, column=0, sticky="nswe", padx=(12, 6), pady=12)
        left.grid_rowconfigure(99, weight=1)

        ctk.CTkLabel(left, text="Back Break — Data & Controls",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")
        ctk.CTkButton(left, text="Load CSV", command=self._choose_csv)\
            .grid(row=1, column=0, padx=10, pady=4, sticky="we")

        self._pred_lbl = ctk.CTkLabel(
            left, text="Predicted Back Break: —",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self._pred_lbl.grid(row=2, column=0, padx=10, pady=(10, 6), sticky="w")

        self._sliders_frame = ctk.CTkScrollableFrame(left, width=330, height=470)
        self._sliders_frame.grid(row=3, column=0, padx=10, pady=6, sticky="nswe")

        btns = ctk.CTkFrame(left)
        btns.grid(row=4, column=0, padx=10, pady=(6, 10), sticky="we")
        ctk.CTkButton(btns, text="Reset to Medians", command=self._reset_to_medians, width=150)\
            .pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="Predict Now", command=self._update_prediction, width=120)\
            .pack(side="left")

        # right (plot)
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nswe", padx=(6, 12), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        host = tk.Frame(right)
        host.grid(row=0, column=0, sticky="nsew")

        self._fig = Figure(figsize=(8.6, 5.6), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title("Random Forest — Feature Importance")
        self._ax.set_xlabel("Importance")
        self._fig.set_tight_layout(True)

        self._canvas = FigureCanvasTkAgg(self._fig, master=host)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, host).update()

    # ---- data/model ----
    def _choose_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p:
            self._load_path(p)

    def _load_path(self, path: str):
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns (features + target).")

            for c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except Exception:
                    pass

            tgt = _infer_target(df)
            if tgt is None:
                raise ValueError(
                    "Could not infer Back Break target column.\n"
                    "Hint: name your target like 'Backbreak'/'Back Break' or put it as the last numeric column."
                )

            num = df.select_dtypes(include=[np.number]).copy()
            if tgt not in num.columns:
                y = pd.to_numeric(df[tgt], errors="coerce")
                num[tgt] = y

            X = num.drop(columns=[tgt], errors="ignore").copy()
            y = pd.to_numeric(df[tgt], errors="coerce")

            mask = X.notna().all(axis=1) & y.notna()
            X, y = X[mask], y[mask]

            if X.shape[0] < 30 or X.shape[1] < 2:
                raise ValueError("Not enough clean rows or features to train Random Forest (need ≥30 rows, ≥2 features).")

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=500, random_state=42)
            model.fit(Xtr, ytr)

            self.df = df
            self.model = model
            self.target_name = tgt

            imp = model.feature_importances_
            order = np.argsort(imp)[::-1]
            feat_names = model.feature_names_in_.tolist()
            keep = [feat_names[i] for i in order[:min(6, len(order))]]

            self.top_feats = []
            for name in keep:
                col = pd.to_numeric(X[name], errors="coerce").dropna()
                vmin = float(col.quantile(0.02)) if col.size else 0.0
                vmax = float(col.quantile(0.98)) if col.size else 1.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = (float(col.min()), float(col.max())) if col.size else (0.0, 1.0)
                    if vmin == vmax:
                        vmax = vmin + 1.0
                self.top_feats.append((name, vmin, vmax))

            for w in self._sliders_frame.winfo_children():
                w.destroy()
            self._feat_vars.clear()

            row = 0
            ctk.CTkLabel(self._sliders_frame, text="Adjust Top Features",
                         font=ctk.CTkFont(size=15, weight="bold")).grid(row=row, column=0, padx=6, pady=(4, 8), sticky="w")
            row += 1

            for (name, vmin, vmax) in self.top_feats:
                var = ctk.DoubleVar(value=(vmin + vmax) / 2.0)
                self._feat_vars[name] = var

                lbl_frame = ctk.CTkFrame(self._sliders_frame, fg_color="transparent")
                lbl_frame.grid(row=row, column=0, padx=6, pady=(6, 2), sticky="we")
                ctk.CTkLabel(lbl_frame, text=name).pack(side="left")
                val_lbl = ctk.CTkLabel(lbl_frame, text=f"{var.get():.3g}")
                val_lbl.pack(side="right")
                row += 1

                def _mk_callback(vvar=var, vlabel=val_lbl):
                    def _cb(_=None):
                        vlabel.configure(text=f"{vvar.get():.3g}")
                        self._update_prediction()
                    return _cb

                slider = ctk.CTkSlider(
                    self._sliders_frame, from_=vmin, to=vmax, number_of_steps=100,
                    variable=var, command=_mk_callback()
                )
                slider.grid(row=row, column=0, padx=6, sticky="we")
                row += 1

            self._ax.clear()
            self._ax.barh(feat_names, imp)
            self._ax.set_title("Random Forest — Feature Importance")
            self._ax.set_xlabel("Importance")
            self._fig.set_tight_layout(True)
            self._canvas.draw_idle()

            messagebox.showinfo(
                "Back Break",
                f"Loaded: {os.path.basename(path)}\n"
                f"Target: {tgt}\n"
                f"Top slider features: {', '.join(n for n, _, _ in self.top_feats)}"
            )
            self._update_prediction()

        except Exception as e:
            messagebox.showerror("Back Break", f"Failed to load:\n{e}")

    def _reset_to_medians(self):
        if self.df is None or not self._feat_vars:
            return
        try:
            num = self.df.select_dtypes(include=[np.number]).copy()
            for name, var in self._feat_vars.items():
                series = pd.to_numeric(num[name], errors="coerce")
                med = float(series.median()) if series.notna().any() else var.get()
                var.set(med)
            self._update_prediction()
        except Exception:
            self._update_prediction()

    def _update_prediction(self):
        if self.model is None or self.df is None:
            return
        try:
            feats_all = self.model.feature_names_in_.tolist()
            vec = []
            num = self.df.select_dtypes(include=[np.number]).copy()
            for name in feats_all:
                if name in self._feat_vars:
                    v = float(self._feat_vars[name].get())
                else:
                    series = pd.to_numeric(num[name], errors="coerce")
                    v = float(series.median()) if series.notna().any() else 0.0
                vec.append(v)
            Xstar = np.array(vec, dtype=float).reshape(1, -1)
            yhat = float(self.model.predict(Xstar)[0])
            self._pred_lbl.configure(text=f"Predicted Back Break: {yhat:.3f}")
        except Exception:
            self._pred_lbl.configure(text="Predicted Back Break: —")

# ---- Standalone run for quick tests ----
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Back Break — Standalone")
    BackbreakWindow(root)
    root.mainloop()


# In[ ]:




