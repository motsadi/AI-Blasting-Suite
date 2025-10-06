#!/usr/bin/env python
# coding: utf-8

# In[4]:


# modules/feature_importance_module.py

import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------- CTk compatibility shim ----------
# Some CustomTkinter versions don't include CTkCollapsibleFrame.
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


def _norm(s: str) -> str:
    """Simple normalization: lowercase and strip non-alnum."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _resolve_map(df_cols, expected: List[str], synonyms_map: Dict[str, List[str]]):
    """
    Try to resolve each pretty/expected name to a real column from df.
    Returns list of column names or None per item if not found.
    """
    cols = list(df_cols)
    lower = {c.lower(): c for c in cols}
    norm = {_norm(c): c for c in cols}
    result = []
    for name in expected:
        if name in cols:
            result.append(name)
            continue
        if name.lower() in lower:
            result.append(lower[name.lower()])
            continue
        chosen = None
        syns = synonyms_map.get(name.lower(), [])
        for s in [name] + syns:
            if s in cols:
                chosen = s
                break
            if s.lower() in lower:
                chosen = lower[s.lower()]
                break
            if _norm(s) in norm:
                chosen = norm[_norm(s)]
                break
        result.append(chosen)
    return result


class FeatureImportanceWindow(ctk.CTkToplevel):
    def __init__(self, master, registry, input_labels, outputs):
        super().__init__(master)
        self.title("Feature Importance & PCA")
        self.geometry("1000x760")
        self.minsize(900, 660)

        self.registry = registry
        self.input_labels = input_labels[:]   # UI-pretty labels from main
        self.outputs = outputs[:]             # 3 outputs (by convention: last 3 if mapping fails)

        # rich synonyms for resilient header matching
        self._input_syn = {
            "hole depth (m)": ["hole depth", "depth", "holedepth"],
            "hole diameter (mm)": ["hole diameter", "diameter", "hole dia", "hole_diameter", "holediameter"],
            "burden (m)": ["burden"],
            "spacing (m)": ["spacing"],
            "stemming (m)": ["stemming"],
            "distance (m)": ["distance", "monitor distance", "distance m", "monitoring distance"],
            "powder factor (kg/m³)": ["powder factor", "powderfactor", "pf", "powder factor (kg/m3)"],
            "rock density (t/m³)": ["rock density", "density", "rock density (t/m3)"],
            "linear charge (kg/m)": ["linear charge", "linearcharge", "kg/m", "charge per metre", "charge/m"],
            "explosive mass (kg)": ["explosive mass", "charge mass", "explosivemass"],
            "blast volume (m³)": ["blast volume", "volume", "blast volume (m3)"],
            "# holes": ["number of holes", "no. holes", "holes", "holes count", "holes #"],
        }
        self._output_syn = {
            "fragmentation": ["mean fragmentation", "fragmentation", "p80", "frag", "fragmentation (mm)"],
            "ground vibration": ["ground vibration", "ppv", "peak particle velocity", "ppv (mm/s)", "ppv mms"],
            "airblast": ["airblast", "air blast", "air overpressure", "overpressure", "db", "air blast (db)"],
        }

        # UI state
        self._topk = ctk.IntVar(value=12)  # how many features to show per plot

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        head = ctk.CTkFrame(self)
        head.pack(fill="x", padx=14, pady=(12, 8))

        ctk.CTkLabel(
            head,
            text="Feature Importance (Random Forest) & PCA",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left")

        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkButton(
            controls, text="Compute RF Importance", command=self._do_importance
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            controls, text="Run PCA Analysis", command=self._do_pca
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(controls, text="Top-K features:").pack(side="left", padx=(20, 6))
        ctk.CTkSlider(
            controls,
            from_=5,
            to=30,
            number_of_steps=25,
            variable=self._topk,
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._msg = ctk.CTkTextbox(self, height=120)
        self._msg.pack(fill="x", padx=14, pady=(0, 8))
        self._msg.insert("end", "Tip: Load/confirm dataset in Data Manager. Inputs = first N−3 if names can't be mapped.\n")

        self._fig = Figure(figsize=(8.8, 6.0), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(0, 12))

    # ---------- helpers ----------
    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        cast = df.copy()
        for c in cols:
            cast[c] = pd.to_numeric(cast[c], errors="coerce")
        return cast.dropna(subset=cols)

    def _name_or_position_split(self, df: pd.DataFrame):
        """
        Try to map by header names & synonyms.
        If that fails, use position: inputs = first N−3, outputs = last 3.
        """
        in_res = _resolve_map(df.columns, self.input_labels, self._input_syn)
        out_res = _resolve_map(df.columns, self.outputs, self._output_syn)

        name_mode_ok = (not any(v is None for v in in_res)) and (not any(v is None for v in out_res))

        if name_mode_ok:
            X = df[in_res].copy()
            Y = df[out_res].copy()
            note = "Mapped by column names/synonyms."
        else:
            n = df.shape[1]
            if n < 4:
                raise ValueError("Dataset needs at least 4 columns to split inputs/outputs by position.")
            X = df.iloc[:, : n - 3].copy()
            Y = df.iloc[:, n - 3 :].copy()
            note = "Name mapping failed — used positional split (inputs = first N−3, outputs = last 3)."

        return X, Y, note

    # ---------- actions ----------
    def _do_importance(self):
        df, path = self.registry.get_dataset()
        if df is None or df.empty:
            messagebox.showwarning("Dataset", "No dataset loaded.")
            return
        try:
            Xraw, Yraw, note = self._name_or_position_split(df)

            # numeric + align rows
            Xnum = self._coerce_numeric(Xraw, Xraw.columns.tolist())
            Ynum = self._coerce_numeric(Yraw, Yraw.columns.tolist())
            work = Xnum.join(Ynum, how="inner")
            X = work.iloc[:, : Xraw.shape[1]].values
            ydf = work.iloc[:, Xraw.shape[1] :]

            if X.shape[0] < 20 or X.shape[1] < 1 or ydf.shape[1] < 1:
                raise ValueError("Not enough clean data after numeric coercion.")

            topk = int(self._topk.get())
            topk = max(5, min(topk, X.shape[1]))

            # draw per-output vertically
            self._fig.clear()
            nrows = ydf.shape[1]
            for i, col in enumerate(ydf.columns, 1):
                y = ydf[col].values
                rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                importances = rf.feature_importances_
                order = np.argsort(importances)[::-1]
                idx = order[:topk]
                ax = self._fig.add_subplot(nrows, 1, i)
                ax.barh(np.array(Xraw.columns)[idx][::-1], importances[idx][::-1])
                ax.set_title(f"Random Forest Feature Importance — {col}")
                ax.set_xlabel("Relative importance")
                ax.set_ylabel("Feature")

            self._fig.tight_layout()
            self._canvas.draw_idle()

            self._msg.delete("1.0", "end")
            self._msg.insert(
                "end",
                f"Dataset: {path}\n{note}\n"
                f"Rows used: {X.shape[0]} | Inputs: {Xraw.shape[1]} | Outputs: {ydf.shape[1]}\n"
                f"Plotted top-{topk} features for each output.\n",
            )

        except Exception as e:
            messagebox.showerror("Importance", f"Failed: {e}")

    def _do_pca(self):
        df, path = self.registry.get_dataset()
        if df is None or df.empty:
            messagebox.showwarning("Dataset", "No dataset loaded.")
            return
        try:
            # inputs only (same name/position logic)
            Xraw, _, note = self._name_or_position_split(df)
            Xraw = self._coerce_numeric(Xraw, Xraw.columns.tolist())

            if Xraw.shape[0] < 10 or Xraw.shape[1] < 2:
                raise ValueError("Not enough data for PCA (need ≥10 rows and ≥2 features).")

            scaler = StandardScaler()
            X = scaler.fit_transform(Xraw.values)

            pca = PCA()
            comps = pca.fit_transform(X)
            vr = pca.explained_variance_ratio_

            self._fig.clear()
            ax1 = self._fig.add_subplot(2, 1, 1)
            ax1.bar(range(1, len(vr) + 1), vr, align="center")
            ax1.step(range(1, len(vr) + 1), np.cumsum(vr), where="mid", linestyle="--")
            ax1.set_xlabel("Principal Component")
            ax1.set_ylabel("Variance Ratio")
            ax1.set_title("PCA — Explained Variance")

            ax2 = self._fig.add_subplot(2, 1, 2)
            ax2.scatter(comps[:, 0], comps[:, 1], alpha=0.65)
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_title("PC1 vs PC2")

            self._fig.tight_layout()
            self._canvas.draw_idle()

            self._msg.delete("1.0", "end")
            self._msg.insert(
                "end",
                f"PCA based on dataset: {path}\n{note}\n"
                f"Shape used: {Xraw.shape[0]} rows × {Xraw.shape[1]} inputs\n",
            )

        except Exception as e:
            messagebox.showerror("PCA", f"Failed: {e}")


# ---- Standalone run for quick tests ----
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Feature Importance — Standalone")
    # Minimal mock for registry: expect get_dataset()
    class _MockReg:
        def __init__(self):
            self._df = None
            self._path = None
            try:
                self._path = "combinedv2Orapa.csv"
                self._df = pd.read_csv(self._path)
            except Exception:
                pass
        def get_dataset(self):
            return self._df, self._path
    # Typical main-app labels:
    INPUT_LABELS = [
        "Hole depth (m)", "Hole diameter (mm)", "Burden (m)", "Spacing (m)",
        "Stemming (m)", "Distance (m)", "Powder factor (kg/m³)", "Rock density (t/m³)",
        "Linear charge (kg/m)", "Explosive mass (kg)", "Blast volume (m³)", "# Holes",
    ]
    OUTPUTS = ["Fragmentation", "Ground Vibration", "Airblast"]
    FeatureImportanceWindow(root, _MockReg(), INPUT_LABELS, OUTPUTS)
    root.mainloop()


# In[ ]:




