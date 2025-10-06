#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modules/param_optimization_module.py
# -*- coding: utf-8 -*-

import tkinter as tk  # pack-only container for canvas/toolbar
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.preprocessing import MinMaxScaler

# TensorFlow / Keras (TF 2.x)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Hover helpers
_HAVE_MPLCURSORS = True
try:
    import mplcursors  # type: ignore
except Exception:
    _HAVE_MPLCURSORS = False

from mpl_toolkits.mplot3d import proj3d


# ---------- small helpers ----------
def _minmax_tf(x, data_min, data_max, eps=1e-9):
    """Vectorised MinMax scaling inside TF to keep gradients."""
    rng = tf.convert_to_tensor(data_max - data_min + eps, dtype=tf.float32)
    return (x - tf.convert_to_tensor(data_min, tf.float32)) / rng

def _inv_minmax_np(x_scaled, data_min, data_max):
    return x_scaled * (data_max - data_min) + data_min


class ParamOptimWindow(ctk.CTkToplevel):
    """
    Builds a 3D optimisation surface for a chosen output versus two chosen inputs.
    Other inputs are optimised via vectorised gradient descent under data bounds.
    Includes:
      • Hover tooltips (shows optimised 'other' inputs at any grid point)
      • Inverse Design / Goal Seek (all inputs free): target output -> input recipe

    Assumptions:
      • Inputs  = first N−3 columns (numeric)
      • Outputs = last 3 columns (Fragmentation, Ground Vibration, Airblast)
    """

    def __init__(self, master, registry, _input_labels_unused, _outputs_unused):
        super().__init__(master)
        self.title("Parameter Optimisation")
        self.geometry("1160x820")
        self.minsize(1100, 760)

        self.registry = registry
        self._last_path: Optional[str] = None

        # names from dataset
        self.in_cols: List[str] = []
        self.out_cols: List[str] = []

        # selectors
        self._out_val = ctk.StringVar(self)
        self._x1_val  = ctk.StringVar(self)
        self._x2_val  = ctk.StringVar(self)

        # trained artefacts
        self._sx: Optional[MinMaxScaler] = None
        self._sy: Optional[MinMaxScaler] = None
        self._model: Optional[tf.keras.Model] = None

        # export/hover cache
        self._last_surface: Optional[Dict[str, Any]] = None   # {"x_name","y_name","z_name","X1","X2","Z","OTHERS",...}
        self._hover_points_artist = None
        self._cursor = None
        self._hover_annot = None
        self._hover_cid = None

        # inverse design controls
        self._target_val = ctk.StringVar(self, value="")
        self._tol_val = ctk.StringVar(self, value="1e-3")

        self._build_ui()
        self._load_cols_and_train(initial=True)
        self.bind("<FocusIn>", self._maybe_retrain_on_focus)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ctk.CTkFrame(self)
        root.pack(fill="both", expand=True, padx=12, pady=10)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(4, weight=1)

        # top controls
        row_top = ctk.CTkFrame(root)
        row_top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for i in range(12):
            row_top.grid_columnconfigure(i, weight=0)
        row_top.grid_columnconfigure(11, weight=1)

        ctk.CTkLabel(row_top, text="Output:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self._out_menu = ctk.CTkOptionMenu(row_top, values=[], variable=self._out_val, width=170)
        self._out_menu.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(row_top, text="Input 1 (X-axis):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, padx=6, pady=6, sticky="e")
        self._x1_menu = ctk.CTkOptionMenu(row_top, values=[], variable=self._x1_val, width=170)
        self._x1_menu.grid(row=0, column=3, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(row_top, text="Input 2 (Y-axis):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=4, padx=6, pady=6, sticky="e")
        self._x2_menu = ctk.CTkOptionMenu(row_top, values=[], variable=self._x2_val, width=170)
        self._x2_menu.grid(row=0, column=5, padx=6, pady=6, sticky="w")

        ctk.CTkButton(row_top, text="Refresh / Train",
                      command=lambda: self._load_cols_and_train(initial=False))\
            .grid(row=0, column=6, padx=10, pady=6, sticky="w")

        # buttons
        row_btns = ctk.CTkFrame(root)
        row_btns.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkButton(row_btns, text="Optimise & Plot Surface",
                      command=self._optimise_and_plot).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row_btns, text="Export surface CSV",
                      command=self._export_surface).pack(side="left", padx=8)

        # Inverse design section (always visible; no collapsible to avoid hidden height)
        inv = ctk.CTkFrame(root)
        inv.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 6))
        inv.grid_columnconfigure(0, weight=0)
        inv.grid_columnconfigure(1, weight=0)
        inv.grid_columnconfigure(2, weight=0)
        inv.grid_columnconfigure(3, weight=0)
        inv.grid_columnconfigure(4, weight=1)

        ctk.CTkLabel(inv, text="Inverse Design (Goal Seek)", font=ctk.CTkFont(size=13, weight="bold"))\
            .grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(8, 2))

        ctk.CTkLabel(inv, text="Target output:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=(8,6), pady=6, sticky="e")
        ctk.CTkEntry(inv, textvariable=self._target_val, width=140).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(inv, text="Tolerance (abs):", font=ctk.CTkFont(weight="bold")).grid(row=1, column=2, padx=6, pady=6, sticky="e")
        ctk.CTkEntry(inv, textvariable=self._tol_val, width=120).grid(row=1, column=3, padx=6, pady=6, sticky="w")

        ctk.CTkButton(inv, text="Run Goal Seek (All Inputs)", command=self._goal_seek)\
            .grid(row=1, column=4, padx=10, pady=6, sticky="w")

        # message
        self._msg = ctk.CTkTextbox(root, height=160, wrap="word")
        self._msg.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 6))
        self._msg.insert("end",
            "Creates an optimisation surface by minimising/maximising the chosen output.\n"
            "Other inputs are optimised via gradient descent within their observed bounds.\n\n"
            "Hover over the surface to see the optimised values of the other inputs at each grid point.\n"
            "Use 'Goal Seek' to set a target output and search for an input recipe that achieves it (all inputs free).\n"
        )

        # plot area (pack canvas + toolbar in inner tk.Frame)
        plot_host = ctk.CTkFrame(root)
        plot_host.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        plot_host.grid_rowconfigure(0, weight=1)
        plot_host.grid_columnconfigure(0, weight=1)

        pack_parent = tk.Frame(plot_host)  # safe place to 'pack' toolbar/canvas
        pack_parent.grid(row=0, column=0, sticky="nsew")

        self._fig = Figure(figsize=(7.6, 6.0), dpi=100)
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._fig, master=pack_parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, pack_parent).update()

    # -------- train surrogate on whole dataset --------
    def _maybe_retrain_on_focus(self, *_):
        df, path = self.registry.get_dataset()
        if path and path != self._last_path:
            self._load_cols_and_train(initial=False)

    def _load_cols_and_train(self, initial=False):
        df, path = self.registry.get_dataset()
        if df is None or df.empty:
            if initial:
                self._msg.insert("end", "No dataset loaded yet (open Data Manager to load a CSV).\n")
            return

        self._last_path = path
        n = df.shape[1]
        if n < 4:
            messagebox.showwarning("Dataset", "Need at least 4 columns (≥1 output among the last 3).")
            return

        # split columns
        self.in_cols  = list(df.columns[:n-3])
        self.out_cols = list(df.columns[n-3:])

        # populate menus
        self._out_menu.configure(values=self.out_cols)
        self._x1_menu.configure(values=self.in_cols)
        self._x2_menu.configure(values=self.in_cols)

        if self._out_val.get() not in self.out_cols:
            self._out_val.set(self.out_cols[0])
        if self._x1_val.get() not in self.in_cols:
            self._x1_val.set(self.in_cols[0])
        if (self._x2_val.get() not in self.in_cols) or (self._x2_val.get() == self._x1_val.get()):
            self._x2_val.set(self.in_cols[1] if len(self.in_cols) > 1 else self.in_cols[0])

        self._out_menu.set(self._out_val.get())
        self._x1_menu.set(self._x1_val.get())
        self._x2_menu.set(self._x2_val.get())

        # train surrogate (MinMax on inputs & outputs)
        X = df[self.in_cols].apply(pd.to_numeric, errors="coerce").values
        Y = df[self.out_cols].apply(pd.to_numeric, errors="coerce").values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[mask]; Y = Y[mask]
        if len(X) < 50:
            messagebox.showwarning("Dataset", "Not enough valid rows after cleaning.")
            return

        self._sx = MinMaxScaler().fit(X)
        self._sy = MinMaxScaler().fit(Y)
        Xs = self._sx.transform(X)
        Ys = self._sy.transform(Y)

        # light but expressive NN
        self._model = Sequential([
            Dense(128, activation="relu", input_shape=(Xs.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(Ys.shape[1], activation="linear")
        ])
        self._model.compile(optimizer="adam", loss="mse")
        self._model.fit(Xs, Ys, epochs=150, batch_size=32, verbose=0)

        self._msg.insert("end", f"Trained surrogate on {len(X)} rows from:\n{path}\n")

    # -------- optimisation on a grid (vectorised) --------
    def _optimise_and_plot(self):
        df, path = self.registry.get_dataset()
        if df is None or df.empty or self._model is None or self._sx is None or self._sy is None:
            messagebox.showwarning("Optimisation", "Load a dataset and click Refresh/Train first.")
            return
        try:
            x1_name = self._x1_val.get()
            x2_name = self._x2_val.get()
            y_name  = self._out_val.get()
            if x1_name == x2_name:
                messagebox.showwarning("Inputs", "Please select two different inputs.")
                return

            n = df.shape[1]
            if x1_name not in df.columns[:n-3] or x2_name not in df.columns[:n-3] or y_name not in df.columns[n-3:]:
                raise KeyError("Inputs must be in first N−3; output in last 3.")

            # numeric cleaned frame
            D = df[self.in_cols + self.out_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if D.empty:
                raise ValueError("No valid rows after numeric coercion.")
            X_all = D[self.in_cols].values

            # bounds per input for projection
            x_min = X_all.min(axis=0)
            x_max = X_all.max(axis=0)

            # index of our axes & output
            i_x1 = self.in_cols.index(x1_name)
            i_x2 = self.in_cols.index(x2_name)
            i_y  = self.out_cols.index(y_name)

            # grid for the two axes in ORIGINAL units
            g1 = np.linspace(x_min[i_x1], x_max[i_x1], 48)
            g2 = np.linspace(x_min[i_x2], x_max[i_x2], 48)
            X1, X2 = np.meshgrid(g1, g2)  # shape (Ny, Nx)
            G = X1.size

            # build initial "others" = medians, vectorised for all grid points
            med = np.median(X_all, axis=0)
            others_idx = [i for i in range(len(self.in_cols)) if i not in (i_x1, i_x2)]
            init_others = np.tile(med[others_idx], (G, 1)).astype(np.float32)

            # TF tensors
            x1_tf = tf.constant(X1.reshape(-1, 1), dtype=tf.float32)
            x2_tf = tf.constant(X2.reshape(-1, 1), dtype=tf.float32)
            others = tf.Variable(init_others)  # (G, n_other)

            # constants for minmax transform
            x_data_min = tf.constant(self._sx.data_min_, dtype=tf.float32)
            x_data_max = tf.constant(self._sx.data_max_, dtype=tf.float32)

            # column composer
            def assemble_full_X(x1col, x2col, othersmat):
                parts = []
                k = 0
                for j in range(len(self.in_cols)):
                    if j == i_x1:
                        parts.append(x1col)
                    elif j == i_x2:
                        parts.append(x2col)
                    else:
                        parts.append(othersmat[:, k:k+1]); k += 1
                return tf.concat(parts, axis=1)  # (G, n_inputs)

            # objective: minimise y for GV/Airblast; maximise y for Fragmentation -> minimise (-y)
            minimise_sign = -1.0 if y_name.lower().startswith("frag") else 1.0

            opt = tf.keras.optimizers.Adam(learning_rate=0.05)
            steps = 70  # inner GD iterations (vectorised over the whole grid)

            # bounds tensors for projection
            lo = tf.constant(x_min[others_idx], dtype=tf.float32)
            hi = tf.constant(x_max[others_idx], dtype=tf.float32)

            for _ in range(steps):
                with tf.GradientTape() as tape:
                    X_full = assemble_full_X(x1_tf, x2_tf, others)
                    Xs_tf  = _minmax_tf(X_full, x_data_min, x_data_max)
                    y_pred_scaled_all = self._model(Xs_tf, training=False)
                    y_pred_scaled = y_pred_scaled_all[:, i_y:i_y+1]
                    loss = tf.reduce_mean(minimise_sign * y_pred_scaled)

                grads = tape.gradient(loss, [others])
                opt.apply_gradients(zip(grads, [others]))
                # project to [min,max] bounds of each "other" column
                others.assign(tf.clip_by_value(others, lo, hi))

            # final predictions per grid point
            X_full = assemble_full_X(x1_tf, x2_tf, others)
            Xs_tf  = _minmax_tf(X_full, x_data_min, x_data_max)
            y_scaled = self._model(Xs_tf, training=False)[:, i_y].numpy().reshape(-1, 1)

            # inverse to original units of the chosen output
            y_min = float(self._sy.data_min_[i_y])
            y_max = float(self._sy.data_max_[i_y])
            Z = _inv_minmax_np(y_scaled, y_min, y_max).reshape(X1.shape)

            # store others (final) for hover: shape (G, n_other)
            others_final = others.numpy()  # (G, n_other)

            # find global optimum on the surface
            if minimise_sign > 0:  # minimise
                idx = np.unravel_index(np.argmin(Z), Z.shape)
            else:                   # maximise
                idx = np.unravel_index(np.argmax(Z), Z.shape)

            best_x1 = X1[idx]; best_x2 = X2[idx]; best_y = Z[idx]
            best_others_vec = others_final[np.ravel_multi_index(idx, X1.shape)]
            best_others = {self.in_cols[j]: val for j, val in zip(others_idx, best_others_vec)}

            # plot the smooth surface
            self._fig.clear()
            ax = self._fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                X1, X2, Z, cmap="viridis", edgecolor="none", linewidth=0,
                antialiased=True, alpha=0.97
            )
            ax.set_xlabel(x1_name); ax.set_ylabel(x2_name); ax.set_zlabel(y_name)
            title_verb = "minimised" if minimise_sign > 0 else "maximised"
            ax.set_title(f"{y_name} vs ({x1_name}, {x2_name}) — {title_verb} over other inputs")
            self._fig.colorbar(surf, ax=ax, shrink=0.6, aspect=16)

            # add a *pickable* scatter for hover picking
            self._hover_points_artist = ax.scatter(
                X1.ravel(), X2.ravel(), Z.ravel(),
                s=9, alpha=0.01, picker=True, pickradius=6, zorder=10
            )
            self._fig.tight_layout(); self._canvas.draw_idle()

            # cache for export & hover
            self._last_surface = {
                "x_name": x1_name,
                "y_name": x2_name,
                "z_name": y_name,
                "X1": X1, "X2": X2, "Z": Z,
                "OTHERS": others_final,           # (G, n_other) array (original units)
                "OTHERS_IDX": others_idx,         # indices in self.in_cols for the "other" inputs
                "X_MIN": x_min, "X_MAX": x_max,   # bounds (original units)
            }

            # setup hover tooltips
            if _HAVE_MPLCURSORS and self._hover_points_artist is not None:
                if self._cursor:
                    try: self._cursor.remove()
                    except Exception: pass
                    self._cursor = None

                self._cursor = mplcursors.cursor(self._hover_points_artist, hover=True)

                @self._cursor.connect("add")
                def _on_add(sel):
                    try:
                        i = int(sel.index)
                        txt = self._format_hover_text(i, X1, X2, Z, others_final, others_idx)
                        sel.annotation.set_text(txt)
                        sel.annotation.get_bbox_patch().set(alpha=0.9)
                    except Exception:
                        sel.annotation.set_text("(hover info unavailable)")
            else:
                self._enable_fallback_hover(ax, X1, X2, Z, others_final, others_idx)
                if not _HAVE_MPLCURSORS:
                    self._msg.insert("end",
                        "Hover tooltips enabled via fallback (install 'mplcursors' for richer hover).\n"
                        "  pip install mplcursors\n"
                    )

            # message with optimised "other" inputs at the global optimum
            best_other_text = ", ".join([f"{k}={best_others[k]:.6g}" for k in best_others.keys()])
            self._msg.delete("1.0", "end")
            self._msg.insert(
                "end",
                f"Optimised surface built from: {self._last_path}\n"
                f"Output: {y_name}  |  Inputs (axes): {x1_name}, {x2_name}\n"
                f"Global optimum on surface @ ({x1_name}={best_x1:.6g}, {x2_name}={best_x2:.6g}) → {y_name}={best_y:.6g}\n"
                f"Optimised values for other inputs ({len(best_others)}):\n  {best_other_text}\n"
                f"Hover on the surface to inspect the 'other' inputs at any grid point.\n"
            )

            # highlight optimum point
            ax.scatter([best_x1], [best_x2], [best_y], s=60, c="k", marker="x")
            self._canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Optimisation", f"Failed: {e}")

    # -------- hover helpers --------
    def _format_hover_text(self, i, X1, X2, Z, others_final, others_idx):
        x = float(X1.ravel()[i]); y = float(X2.ravel()[i]); z = float(Z.ravel()[i])
        lines = [f"{self.in_cols[j]}={float(others_final[i][k]):.6g}"
                 for k, j in enumerate(others_idx)]
        txt = (
            f"{self._last_surface['x_name']}: {x:.6g}\n"
            f"{self._last_surface['y_name']}: {y:.6g}\n"
            f"{self._last_surface['z_name']}: {z:.6g}\n"
            f"Other inputs ({len(lines)}):\n• " + "\n• ".join(lines)
        )
        return txt

    def _enable_fallback_hover(self, ax, X1, X2, Z, others_final, others_idx):
        """Fallback hover when mplcursors is unavailable."""
        xs = X1.ravel(); ys = X2.ravel(); zs = Z.ravel()
        xyz = np.vstack([xs, ys, zs]).T

        # Prepare annotation
        if self._hover_annot and self._hover_annot in ax.texts:
            self._hover_annot.remove()
        self._hover_annot = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->", alpha=0.6), fontsize=9, visible=False
        )

        def _on_move(event):
            if event.inaxes != ax:
                if self._hover_annot.get_visible():
                    self._hover_annot.set_visible(False)
                    self._canvas.draw_idle()
                return
            # Reproject each time in case the view changed
            P = np.array([proj3d.proj_transform(x, y, z, ax.get_proj()) for x, y, z in xyz])
            u, v = P[:, 0], P[:, 1]  # projected "data-like" coords
            if event.xdata is None or event.ydata is None:
                return
            d = np.hypot(u - event.xdata, v - event.ydata)
            i = int(np.argmin(d))
            if d[i] > 0.05:
                if self._hover_annot.get_visible():
                    self._hover_annot.set_visible(False)
                    self._canvas.draw_idle()
                return

            txt = self._format_hover_text(i, X1, X2, Z, others_final, others_idx)
            self._hover_annot.xy = (X1.ravel()[i], X2.ravel()[i])
            self._hover_annot.set_text(txt)
            self._hover_annot.set_visible(True)
            self._canvas.draw_idle()

        if self._hover_cid:
            self._canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = self._canvas.mpl_connect("motion_notify_event", _on_move)

    # -------- inverse design (goal seek) — ALL INPUTS --------
    def _goal_seek(self):
        """
        Find a full input recipe (all inputs free within data bounds) that achieves a target output.
        """
        df, path = self.registry.get_dataset()
        if df is None or df.empty or self._model is None or self._sx is None or self._sy is None:
            messagebox.showwarning("Goal Seek", "Load a dataset and click Refresh/Train first.")
            return

        try:
            target_text = self._target_val.get().strip()
            if not target_text:
                messagebox.showwarning("Goal Seek", "Please enter a target output value.")
                return
            target_val = float(target_text)
            tol = abs(float(self._tol_val.get().strip() or "1e-3"))
        except Exception:
            messagebox.showwarning("Goal Seek", "Invalid target/tolerance value.")
            return

        try:
            y_name = self._out_val.get()
            i_y = self.out_cols.index(y_name)

            # clean numeric data and bounds
            D = df[self.in_cols + self.out_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if D.empty:
                raise ValueError("No valid rows after numeric coercion.")
            X_all = D[self.in_cols].values
            x_min = X_all.min(axis=0)
            x_max = X_all.max(axis=0)

            # variables: full input vector (shape: (1, n_inputs))
            med = np.median(X_all, axis=0).astype(np.float32)
            x_var = tf.Variable(med.reshape(1, -1))  # start from medians

            # constants
            x_data_min = tf.constant(self._sx.data_min_, dtype=tf.float32)
            x_data_max = tf.constant(self._sx.data_max_, dtype=tf.float32)
            y_min = float(self._sy.data_min_[i_y])
            y_max = float(self._sy.data_max_[i_y])

            # target in scaled space
            target_scaled = (target_val - y_min) / max(y_max - y_min, 1e-9)
            target_scaled_tf = tf.constant([[float(target_scaled)]], dtype=tf.float32)

            # optimiser
            opt = tf.keras.optimizers.Adam(learning_rate=0.07)

            steps = 300
            lo = tf.constant(x_min.reshape(1, -1), dtype=tf.float32)
            hi = tf.constant(x_max.reshape(1, -1), dtype=tf.float32)

            last_err = None
            for _ in range(steps):
                with tf.GradientTape() as tape:
                    Xs = _minmax_tf(x_var, x_data_min, x_data_max)
                    y_scaled_all = self._model(Xs, training=False)
                    y_scaled = y_scaled_all[:, i_y:i_y+1]  # (1,1)
                    loss = tf.reduce_mean((y_scaled - target_scaled_tf) ** 2)
                grads = tape.gradient(loss, [x_var])
                opt.apply_gradients(zip(grads, [x_var]))
                # project back to bounds
                x_var.assign(tf.clip_by_value(x_var, lo, hi))
                last_err = float(loss.numpy())
                if last_err <= tol**2:  # early stop if within tolerance (scaled)
                    break

            # final prediction in ORIGINAL units
            Xs_final = _minmax_tf(x_var, x_data_min, x_data_max)
            y_scaled_final_all = self._model(Xs_final, training=False).numpy().reshape(-1)
            y_scaled_final = float(y_scaled_final_all[i_y])
            y_final = _inv_minmax_np(np.array([y_scaled_final]), y_min, y_max)[0]

            x_solution = x_var.numpy().reshape(-1)
            report = [f"{name}={val:.6g}" for name, val in zip(self.in_cols, x_solution)]

            self._msg.insert(
                "end",
                "\nGoal Seek results (all inputs free)\n"
                f"  Target {y_name}: {target_val:.6g} (tol={tol:.3g})\n"
                f"  Achieved {y_name}: {y_final:.6g}  |  scaled SSE ~ {last_err:.3g}\n"
                f"  Inputs (solution):\n    " + ", ".join(report) + "\n"
            )

            # If a surface is displayed and axes match, drop a marker at (x1, x2, y_final)
            if self._last_surface is not None:
                try:
                    ix1 = self.in_cols.index(self._last_surface["x_name"])
                    ix2 = self.in_cols.index(self._last_surface["y_name"])
                    x1_sol = float(x_solution[ix1])
                    x2_sol = float(x_solution[ix2])
                    ax = self._fig.axes[0] if self._fig.axes else None
                    if ax is not None:
                        ax.scatter([x1_sol], [x2_sol], [y_final], s=70, c="red", marker="^", depthshade=False)
                        ax.text(x1_sol, x2_sol, y_final, "  goal-seek", color="red")
                        self._canvas.draw_idle()
                except Exception:
                    pass

        except Exception as e:
            messagebox.showerror("Goal Seek", f"Failed: {e}")

    # -------- export --------
    def _export_surface(self):
        if not self._last_surface:
            messagebox.showinfo("Export", "No surface to export yet. Run optimisation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            title="Save optimisation surface grid"
        )
        if not path:
            return
        try:
            X1 = self._last_surface["X1"]; X2 = self._last_surface["X2"]; Z = self._last_surface["Z"]
            df_out = pd.DataFrame({
                self._last_surface["x_name"]: X1.ravel(),
                self._last_surface["y_name"]: X2.ravel(),
                self._last_surface["z_name"]: Z.ravel(),
            })
            # Include "other" inputs used at each grid point (for reproducibility & analysis)
            OTH = self._last_surface.get("OTHERS", None)
            OIDX = self._last_surface.get("OTHERS_IDX", None)
            if OTH is not None and OIDX is not None:
                for col_idx, arr_col in zip(OIDX, OTH.T):
                    df_out[self.in_cols[col_idx]] = arr_col

            df_out.to_csv(path, index=False)
            messagebox.showinfo("Export", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save:\n{e}")


# ---- Standalone run for quick tests ----
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Parameter Optimisation — Standalone")

    # Minimal mock for registry: expects get_dataset()
    class _MockReg:
        def __init__(self):
            import os
            self._df = None
            self._path = None
            for candidate in ["combinedv2Orapa.csv", "combinedv2Jwaneng.csv"]:
                if os.path.exists(candidate):
                    self._path = candidate
                    try:
                        self._df = pd.read_csv(candidate)
                    except Exception:
                        self._df = None
                    break
        def get_dataset(self):
            return self._df, self._path

    INPUT_LABELS = []
    OUTPUTS = []
    ParamOptimWindow(root, _MockReg(), INPUT_LABELS, OUTPUTS)
    root.mainloop()


# In[ ]:




