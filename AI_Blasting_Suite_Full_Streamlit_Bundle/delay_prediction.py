#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# delay_prediction.py
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel
import tkinter as tk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # ensure Tk backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# -------------------------- helpers --------------------------
REQUIRED_BASE = ["X", "Y"]
OPTIONAL_MAP = {
    "HoleID": ["holeid", "hole id", "id", "hole"],
    "Depth": ["depth", "hole_depth", "hole depth (m)", "hole_depth_m"],
    "Charge": ["charge", "explosive mass", "explosive_mass", "charge_kg"],
    "Z": ["z", "rl", "elev", "elevation"],
    "Delay": ["delay", "delay_ms", "predicted delay (ms)", "predicted_delay_ms", "time_ms"]
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    def _find(name, aliases):
        for k, v in cols.items():
            if k == name.lower():
                return v
        for a in aliases:
            if a.lower() in cols:
                return cols[a.lower()]
        return None

    rename = {}
    # required X, Y must exist
    if "x" not in cols or "y" not in cols:
        raise ValueError("CSV must include X and Y columns.")
    rename[cols["x"]] = "X"
    rename[cols["y"]] = "Y"

    for std, alist in OPTIONAL_MAP.items():
        col = _find(std, alist)
        if col is not None:
            rename[col] = std

    return df.rename(columns=rename)

def _has(df, name):
    return name in df.columns

def _hole_labels(df):
    if _has(df, "HoleID"):
        return df["HoleID"].astype(str).values
    return np.arange(len(df)).astype(str)


# -------------------- model loading/training --------------------
def load_and_train_model(model_csv: str = "Hole_data_v1.csv"):
    # Train on whatever we have (or synthetic y if no Delay present)
    df = pd.read_csv(model_csv)
    df = _standardize_columns(df)
    keep = ["Depth","Charge","X","Y"]
    if _has(df, "Z"):
        keep.append("Z")
    df = df.dropna(subset=[c for c in keep])

    X = df[[c for c in ["Depth","Charge","X","Y","Z"] if _has(df,c)]].values
    if _has(df, "Delay"):
        y = df["Delay"].values
    else:
        # simple synthetic target so Predict Delays works on new CSVs too
        y = np.clip(10 + 0.02*X[:,0] + 0.0005*X[:,2], 5, 250)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    try:
        r2 = model.score(X_train_s, y_train)
        print(f"Delay model (RF) R² train: {r2:.3f}")
    except Exception:
        pass
    return model, scaler


# ------------------------- main app -------------------------
class BlastSimApp:
    def __init__(self, parent=None):
        # window
        self.is_main = parent is None
        if self.is_main:
            ctk.set_appearance_mode('System')
            ctk.set_default_color_theme('dark-blue')
            self.root = ctk.CTk()
            self.root.title("Blast Delay & Visualisation Suite")
            self.root.geometry("1180x820")
        else:
            self.root = parent  # provided Toplevel from main
            try:
                ctk.set_appearance_mode('System')
                ctk.set_default_color_theme('dark-blue')
            except Exception:
                pass
            self.root.title("Blast Delay & Visualisation Suite")
            self.root.geometry("1000x800")

        # data + ML
        self.data: pd.DataFrame | None = None
        # if the training CSV isn't present, we still want the UI; so guard:
        try:
            self.model, self.scaler = load_and_train_model()
        except Exception:
            # fallback dummy model/scaler
            self.model = RandomForestRegressor(n_estimators=1, random_state=42).fit([[0,0,0,0]], [0])
            self.scaler = StandardScaler().fit([[0,0,0,0]])

        # ui
        self._build_main_ui()
        if self.is_main:
            self.root.mainloop()

    # ------------------- UI scaffold -------------------
    def _build_main_ui(self):
        top = ctk.CTkFrame(self.root)
        top.pack(fill="x", padx=10, pady=(10,6))

        ctk.CTkButton(top, text="Upload CSV", command=self._upload_data).pack(side="left", padx=5)
        ctk.CTkButton(top, text="Predict Delays", command=self._predict_delays).pack(side="left", padx=5)
        ctk.CTkButton(top, text="Visualize Data", command=self._visualize_data).pack(side="left", padx=5)
        ctk.CTkButton(top, text="Simulate Blast", command=self._simulate_blast).pack(side="left", padx=5)
        ctk.CTkButton(top, text="Export CSV", command=self._export_data).pack(side="left", padx=5)

        self.status = ctk.CTkLabel(self.root, text="Load a CSV to begin.")
        self.status.pack(fill="x", padx=10, pady=(0,6))

    # ------------------- data ops -------------------
    def _upload_data(self):
        path = filedialog.askopenfilename(filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            df = _standardize_columns(df)
            self.data = df
            self.status.configure(text=f"Loaded {len(df)} holes from: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def _predict_delays(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Please upload data first.")
            return
        req = ["Depth","Charge","X","Y"]
        if _has(self.data,"Z"): req += ["Z"]
        df = self.data.dropna(subset=[c for c in req]).copy()
        if df.empty:
            messagebox.showwarning("No Data", "After cleaning, no rows available.")
            return
        X = df[[c for c in ["Depth","Charge","X","Y","Z"] if _has(df,c)]].values
        Xs = self.scaler.transform(X)
        df["Predicted Delay (ms)"] = self.model.predict(Xs)
        # merge back (preserve any additional cols)
        self.data = df
        messagebox.showinfo("Done", "Predicted delays added.")

    # ---------------- Visualise Data (enhanced) ----------------
    def _visualize_data(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Please upload (and predict) first.")
            return

        # always open a **new** window (prevents overlap/packing issues)
        win = Toplevel(self.root)
        win.title("Plan View")
        win.geometry("1200x850")

        # --- controls row (colour/size) ---
        ctrl = ctk.CTkFrame(win); ctrl.pack(fill="x", padx=10, pady=(10,6))
        numeric_cols = [c for c in ["Predicted Delay (ms)","Delay","Charge","Depth","Z"] if c in self.data.columns]
        if not numeric_cols:
            messagebox.showwarning("No numeric columns", "Need at least one numeric column to colour/size by.")
            win.destroy()
            return
        color_var = tk.StringVar(value=numeric_cols[0])
        size_var  = tk.StringVar(value=( "Charge" if "Charge" in self.data.columns else numeric_cols[-1] ))

        ctk.CTkLabel(ctrl, text="Colour by:").pack(side="left", padx=(0,6))
        ctk.CTkOptionMenu(ctrl, values=numeric_cols, variable=color_var).pack(side="left", padx=(0,12))
        ctk.CTkLabel(ctrl, text="Size by:").pack(side="left", padx=(0,6))
        ctk.CTkOptionMenu(ctrl, values=numeric_cols, variable=size_var).pack(side="left", padx=(0,12))
        ctk.CTkButton(ctrl, text="Refresh", command=lambda: draw()).pack(side="left", padx=6)

        # --- plan view: use constrained_layout + colorbar on ax (no tight_layout/add_axes) ---
        fig = plt.Figure(figsize=(11.6, 8.4), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)

        host = ctk.CTkFrame(win); host.pack(fill="both", expand=True, padx=10, pady=(0,10))
        packer = tk.Frame(host); packer.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=packer)
        canvas.draw(); canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, packer).update()

        def draw():
            df = self.data.copy()
            if df.empty:
                return
            cb, sb = color_var.get(), size_var.get()
            cols = ["X","Y", cb, sb]
            df = df.loc[df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)].copy()
            if df.empty:
                messagebox.showwarning("No Data", "Nothing to plot after filtering.")
                return

            ax.clear()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title("Plan view")

            # point sizes (robust scaling)
            svals = df[sb].astype(float)
            s = 30 * (svals - svals.quantile(0.05)) / (svals.quantile(0.95) - svals.quantile(0.05) + 1e-9)
            s = np.clip(s, 24, 180)

            # colour mapping
            cvals = df[cb].astype(float)
            vmin, vmax = cvals.quantile(0.02), cvals.quantile(0.98)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("turbo")

            sc = ax.scatter(df["X"], df["Y"], c=cvals, s=s,
                            cmap=cmap, norm=norm, edgecolors="#333333", linewidths=0.5, alpha=0.97)

            # extents
            dx = df["X"].max() - df["X"].min()
            dy = df["Y"].max() - df["Y"].min()
            pad = 0.03 * max(dx, dy)
            ax.set_xlim(df["X"].min()-pad, df["X"].max()+pad)
            ax.set_ylim(df["Y"].min()-pad, df["Y"].max()+pad)

            # colorbar attached to ax (no add_axes, so no tight_layout warnings)
            # remove old colorbars if any
            if hasattr(draw, "_last_cbar") and draw._last_cbar:
                draw._last_cbar.remove()
            draw._last_cbar = fig.colorbar(sc, ax=ax, location="right", pad=0.02)
            draw._last_cbar.set_label(cb)

            canvas.draw_idle()

        draw()


    # ---------------- Simulate Blast (enhanced) ----------------
    def _simulate_blast(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Upload data first.")
            return

        time_col = "Predicted Delay (ms)" if "Predicted Delay (ms)" in self.data.columns else ("Delay" if "Delay" in self.data.columns else None)
        if not time_col:
            messagebox.showwarning("No Delays", "Please click 'Predict Delays' (or provide a 'Delay' column) first.")
            return

        df = self.data.loc[self.data[["X","Y", time_col]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)].copy()
        if df.empty:
            messagebox.showwarning("No Data", "Nothing to simulate after filtering.")
            return

        # Sort and use UNIQUE time steps so we see the blast hop hole-to-hole
        df = df.sort_values(time_col)
        unique_t = np.sort(df[time_col].unique())
        nsteps = len(unique_t)
        t_min, t_max = float(unique_t.min()), float(unique_t.max())

        # always open a **new** window (prevents overlap/packing issues in embedded mode)
        win = Toplevel(self.root)
        win.title("Blast Simulation")
        win.geometry("1220x860")

        # ---------- controls (top) ----------
        ctl = ctk.CTkFrame(win); ctl.pack(fill="x", padx=10, pady=(10,6))
        idx_var = tk.IntVar(value=0)  # index into unique_t (discrete)
        ctk.CTkLabel(ctl, text="Step:").pack(side="left", padx=(0,4))
        slider = ctk.CTkSlider(ctl, from_=0, to=nsteps-1, number_of_steps=max(1, nsteps-1), variable=idx_var)
        slider.pack(side="left", fill="x", expand=True, padx=8)

        ctk.CTkLabel(ctl, text="Speed:").pack(side="left", padx=(10,4))
        speed_var = tk.DoubleVar(value=1.0)
        ctk.CTkSlider(ctl, from_=0.25, to=5.0, number_of_steps=19, variable=speed_var, width=140).pack(side="left", padx=(0,10))

        btn_box = ctk.CTkFrame(ctl); btn_box.pack(side="left", padx=6)
        start_btn = ctk.CTkButton(btn_box, text="▶ Start"); start_btn.grid(row=0, column=0, padx=4)
        pause_btn = ctk.CTkButton(btn_box, text="⏸ Pause"); pause_btn.grid(row=0, column=1, padx=4)
        reset_btn = ctk.CTkButton(btn_box, text="⟲ Reset"); reset_btn.grid(row=0, column=2, padx=4)

        ring_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctl, text="Shock front", variable=ring_var).pack(side="left", padx=8)
        ctk.CTkLabel(ctl, text="Front speed (m/ms):").pack(side="left", padx=(8,4))
        wave_var = tk.DoubleVar(value=3.0)
        ctk.CTkSlider(ctl, from_=0.5, to=8.0, number_of_steps=30, variable=wave_var, width=140).pack(side="left", padx=(0,6))

        info_lbl = ctk.CTkLabel(win, text="")
        info_lbl.pack(fill="x", padx=10, pady=(0,6))

        # ---------- big figure (constrained_layout + ax colorbar) ----------
        fig = plt.Figure(figsize=(11.8, 7.8), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)

        host = ctk.CTkFrame(win); host.pack(fill="both", expand=True, padx=10, pady=(0,10))
        packer = tk.Frame(host); packer.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=packer)
        canvas.draw(); canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, packer).update()

        # aspect/bounds
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        pad = 0.03 * max(df["X"].max()-df["X"].min(), df["Y"].max()-df["Y"].min())
        ax.set_xlim(df["X"].min()-pad, df["X"].max()+pad)
        ax.set_ylim(df["Y"].min()-pad, df["Y"].max()+pad)

        # Colormap & colourbar on ax (no add_axes)
        norm = matplotlib.colors.Normalize(vmin=t_min, vmax=t_max)
        cmap = plt.get_cmap("turbo")
        # keep a handle to remove if we redraw (we don't here, but for safety)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right", pad=0.02)
        cbar.set_label(time_col)

        # Artists: waiting (grey), fired (coloured), current (highlight), rings
        scat_wait = ax.scatter([], [], color="#9aa0a6", s=42, edgecolors="#222", linewidths=0.4, alpha=0.95)
        scat_fired = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=60, edgecolors="#222", linewidths=0.4, alpha=0.97)
        scat_now   = ax.scatter([], [], s=200, facecolors="none", edgecolors="white", linewidths=1.8, alpha=0.95)
        rings = []

        # Figure-level time label
        fig_time = fig.text(0.02, 0.98, "", ha="left", va="top", fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", alpha=0.85))

        # pre-calc arrays
        XY = df[["X","Y"]].values
        T  = df[time_col].values.astype(float)

        manual_scrub = [False]

        # Update per frame (step over unique times)
        def update(frame_idx):
            i = int(idx_var.get()) if manual_scrub[0] else frame_idx
            i = max(0, min(nsteps-1, i))
            t = unique_t[i]

            fired_mask   = T <= t
            current_mask = T == t
            wait_mask    = T > t

            # update collections
            scat_fired.set_offsets(XY[fired_mask])
            scat_fired.set_array(T[fired_mask])
            scat_wait.set_offsets(XY[wait_mask])
            scat_now.set_offsets(XY[current_mask])

            # shock-front rings (clear & redraw)
            for rr in rings:
                try:
                    rr.remove()
                except Exception:
                    pass
            rings.clear()
            if ring_var.get():
                # Draw circles for holes that fired at the current step (highlight motion)
                r = (t - T[current_mask]) * wave_var.get()  # zero at instant; we force a minimal radius below
                # also draw faint ring for just-fired in previous step
                prev_t = (unique_t[i-1] if i > 0 else t)
                r2_mask = (T < t) & (T >= prev_t)
                r2 = (t - T[r2_mask]) * wave_var.get()

                theta = np.linspace(0, 2*np.pi, 160)
                for (x,y,rad) in zip(XY[current_mask,0], XY[current_mask,1], np.maximum(r, 0.8)):
                    line, = ax.plot(x + rad*np.cos(theta), y + rad*np.sin(theta),
                                    color=(1,1,1,0.35), linewidth=1.2)
                    rings.append(line)

                theta2 = np.linspace(0, 2*np.pi, 100)
                for (x,y,rad) in zip(XY[r2_mask,0], XY[r2_mask,1], np.maximum(r2, 0.6)):
                    line, = ax.plot(x + rad*np.cos(theta2), y + rad*np.sin(theta2),
                                    color=(1,1,1,0.18), linewidth=0.9)
                    rings.append(line)

            # figure label & info
            pct = 100.0 * fired_mask.sum() / len(df)
            fig_time.set_text(f"t = {t:,.1f} ms   •   {fired_mask.sum():,}/{len(df):,} holes fired ({pct:.1f}%)")

            canvas.draw_idle()

        # animation timeline and interactions
        frames = list(range(nsteps))
        self._anim = animation.FuncAnimation(fig, update, frames=frames, interval=int(300), blit=False)
        self._anim.event_source.stop()

        def _set_speed(*_):
            if hasattr(self._anim, "event_source") and self._anim.event_source is not None:
                self._anim.event_source.interval = int(300/max(0.1, float(speed_var.get())))
        speed_var.trace_add("write", lambda *_: _set_speed())

        def _on_press(_):  manual_scrub[0] = True
        def _on_release(_): manual_scrub[0] = False
        slider.bind("<ButtonPress-1>", _on_press)
        slider.bind("<ButtonRelease-1>", _on_release)

        start_btn.configure(command=lambda: self._anim.event_source.start())
        pause_btn.configure(command=lambda: self._anim.event_source.stop())
        def _reset():
            self._anim.event_source.stop()
            idx_var.set(0); update(0)
        reset_btn.configure(command=_reset)

        # initial frame
        update(0)
        info_lbl.configure(text=f"Sequential animation across {nsteps} firing steps. "
                                f"Colourbar is attached to the axes (no overlapping).")

    # ---------------- export ----------------
    def _export_data(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        self.data.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Exported to {path}")


# If run standalone
if __name__ == "__main__":
    BlastSimApp()

