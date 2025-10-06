#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import customtkinter as ctk
import pandas as pd
import numpy as np

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt


# ---------- CTk compatibility shim ----------
# Some CustomTkinter versions don't include CTkCollapsibleFrame.
if not hasattr(ctk, "CTkCollapsibleFrame"):
    class CTkCollapsibleFrame(ctk.CTkFrame):
        def __init__(self, master=None, text: str = "", **kwargs):
            super().__init__(master, **kwargs)
            # Simple header to mimic section title (no collapse behavior)
            if text:
                ctk.CTkLabel(
                    self,
                    text=text,
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=6, pady=(6, 2))
    # expose as if it were part of CTk
    ctk.CTkCollapsibleFrame = CTkCollapsibleFrame
# -------------------------------------------


# --------------------------
# Defaults / expected schema
# --------------------------
DEFAULT_DATASETS = [
    "combinedv2Orapa.csv",
    "combinedv2Jwaneng.csv",
]

EXPECTED_COLUMNS = [
    "Burden", "Spacing", "Stemming", "Hole diameter", "Hole depth",
    "Linear charge", "Powder factor", "Distance", "Rock density",
    "Explosive mass", "Blast volume", "Number of holes",
    "Ground Vibration", "Airblast", "Fragmentation",
]


def _appearance_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {"bg": "#0f172a", "fg": "#e2e8f0", "grid": "#334155"}
    return {"bg": "#ffffff", "fg": "#0f172a", "grid": "#cbd5e1"}


def _theme_matplotlib():
    colors = _appearance_colors()
    plt.rcParams.update({
        "axes.facecolor": colors["bg"],
        "figure.facecolor": colors["bg"],
        "axes.edgecolor": colors["fg"],
        "axes.labelcolor": colors["fg"],
        "xtick.color": colors["fg"],
        "ytick.color": colors["fg"],
        "text.color": colors["fg"],
        "grid.color": colors["grid"],
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


class DataApp:
    def __init__(self, parent):
        # Window / theme
        self.root = parent
        self.root.title("Data Management")
        self.root.geometry("1120x740")
        self.root.minsize(980, 660)
        ctk.set_default_color_theme("blue")
        _theme_matplotlib()

        # Data state
        self.data = self._load_first_available()
        self.filtered = self.data.copy()

        # UI
        self._build_topbar()
        self._build_tabs()

        # Fill initial views
        self._refresh_table()
        self._update_summary()
        self._draw_correlation()

    # ---------------- Data I/O ----------------

    def _load_first_available(self) -> pd.DataFrame:
        for p in DEFAULT_DATASETS:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    return self._ensure_columns(df)
                except Exception as e:
                    messagebox.showwarning("Load error", f"Failed to load {p}:\n{e}")
        messagebox.showinfo("Select data", "Choose a CSV file with your blast dataset.")
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            df = pd.read_csv(path)
            return self._ensure_columns(df)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        cols = [c for c in EXPECTED_COLUMNS if c in dfc.columns] + [c for c in dfc.columns if c not in EXPECTED_COLUMNS]
        dfc = dfc[cols]
        for c in EXPECTED_COLUMNS:
            if c in dfc.columns:
                dfc[c] = pd.to_numeric(dfc[c], errors="ignore")
        return dfc

    def _save_filtered_csv(self):
        if self.filtered.empty:
            messagebox.showwarning("Export", "No filtered data to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            title="Save filtered data as CSV"
        )
        if not path:
            return
        try:
            self.filtered.to_csv(path, index=False)
            messagebox.showinfo("Export", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _save_filtered_excel(self):
        if self.filtered.empty:
            messagebox.showwarning("Export", "No filtered data to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel","*.xlsx")],
            title="Save filtered data as Excel"
        )
        if not path:
            return
        try:
            self.filtered.to_excel(path, index=False)
            messagebox.showinfo("Export", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # ------------------- UI: topbar -------------------

    def _build_topbar(self):
        bar = ctk.CTkFrame(self.root)
        bar.pack(side="top", fill="x", padx=10, pady=(10, 0))

        ctk.CTkLabel(bar, text="Data Management",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=6)

        ctk.CTkButton(bar, text="Load CSV", command=self._action_load).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Append CSV", command=self._action_append).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Reset Filters", command=self._reset_filters).pack(side="left", padx=6)

        self.status = ctk.CTkLabel(bar, text=f"Rows: {len(self.data)}  |  Columns: {len(self.data.columns)}")
        self.status.pack(side="right", padx=8)

    def _action_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.data = self._ensure_columns(df)
            self.filtered = self.data.copy()
            self._refresh_table()
            self._update_summary()
            self._draw_correlation()
            self._refresh_filters_panel()
            self._set_status()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _action_append(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            df = self._ensure_columns(pd.read_csv(path))
            missing = [c for c in self.data.columns if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            self.data = pd.concat([self.data, df[self.data.columns]], ignore_index=True)
            self.filtered = self.data.copy()
            self._refresh_table()
            self._update_summary()
            self._draw_correlation()
            self._refresh_filters_panel()
            self._set_status()
            messagebox.showinfo("Append", f"Appended {len(df)} rows from:\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Append error", str(e))

    def _set_status(self):
        self.status.configure(text=f"Rows: {len(self.filtered)}  |  Columns: {len(self.filtered.columns)}")

    # ------------------- UI: tabs -------------------

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_table = self.tabs.add("Table")
        self.tab_summary = self.tabs.add("Summary")
        self.tab_visuals = self.tabs.add("Visuals")
        self.tab_corr = self.tabs.add("Correlations")
        self.tab_filters = self.tabs.add("Filters")
        self.tab_calib = self.tabs.add("Calibration")
        self.tab_export = self.tabs.add("Export")

        self._build_table_tab()
        self._build_summary_tab()
        self._build_visuals_tab()
        self._build_corr_tab()
        self._build_filters_tab()
        self._build_calib_tab()
        self._build_export_tab()

    # ---------------- Table tab ----------------

    def _build_table_tab(self):
        wrap = ctk.CTkFrame(self.tab_table)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        top = ctk.CTkFrame(wrap)
        top.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(top, text="Search (contains):").pack(side="left", padx=4)
        self.search_entry = ctk.CTkEntry(top, width=240)
        self.search_entry.pack(side="left", padx=4)
        ctk.CTkButton(top, text="Apply", command=self._search_table).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Clear", command=self._clear_search).pack(side="left", padx=4)

        self.tree_frame = ctk.CTkFrame(wrap)
        self.tree_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self._create_treeview(self.tree_frame)

        add_panel = ctk.CTkCollapsibleFrame(wrap, text="Add Row")
        add_panel.pack(fill="x", padx=6, pady=6)
        self._build_add_row_panel(add_panel)

    def _create_treeview(self, parent):
        style = ttk.Style()
        style.theme_use("clam")
        colors = _appearance_colors()
        style.configure("Treeview",
                        background=colors["bg"], fieldbackground=colors["bg"],
                        foreground=colors["fg"], rowheight=24)
        style.configure("Treeview.Heading",
                        background=colors["bg"], foreground=colors["fg"])
        style.map("Treeview", background=[("selected", "#3b82f6")],
                  foreground=[("selected", "#ffffff")])

        container = ctk.CTkFrame(parent)
        container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(container, columns=list(self.filtered.columns), show="headings")
        for col in self.filtered.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        sy = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        sx = ttk.Scrollbar(container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=sy.set, xscroll=sx.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="we")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    def _refresh_table(self):
        if hasattr(self, "tree"):
            for i in self.tree.get_children():
                self.tree.delete(i)
            if list(self.tree["columns"]) != list(self.filtered.columns):
                self.tree.destroy()
                self._create_treeview(self.tree_frame)
        if hasattr(self, "tree"):
            for _, row in self.filtered.iterrows():
                vals = [row.get(c, "") for c in self.filtered.columns]
                self.tree.insert("", "end", values=vals)
        self._set_status()

    def _build_add_row_panel(self, panel: ctk.CTkCollapsibleFrame):
        grid = ctk.CTkScrollableFrame(panel, height=160)
        grid.pack(fill="x", padx=6, pady=6)

        self.add_entries = {}
        for i, col in enumerate(self.data.columns):
            ctk.CTkLabel(grid, text=col).grid(row=i, column=0, padx=6, pady=4, sticky="e")
            ent = ctk.CTkEntry(grid, width=200)
            ent.grid(row=i, column=1, padx=6, pady=4, sticky="w")
            self.add_entries[col] = ent

        ctk.CTkButton(panel, text="Add Row", command=self._add_row).pack(pady=6, padx=6, anchor="e")

    def _add_row(self):
        new_vals = {}
        for col, ent in self.add_entries.items():
            v = ent.get().strip()
            try:
                new_vals[col] = float(v) if v != "" else np.nan
            except:
                new_vals[col] = v
        self.data = pd.concat([self.data, pd.DataFrame([new_vals])], ignore_index=True)
        self.filtered = self.data.copy()
        self._refresh_table()
        self._update_summary()
        self._draw_correlation()
        messagebox.showinfo("Add Row", "New row added.")
        for ent in self.add_entries.values():
            ent.delete(0, "end")

    def _search_table(self):
        q = self.search_entry.get().strip().lower()
        if q == "":
            self.filtered = self.data.copy()
        else:
            mask = pd.Series(False, index=self.data.index)
            for c in self.data.columns:
                mask = mask | self.data[c].astype(str).str.lower().str.contains(q, na=False)
            self.filtered = self.data[mask].copy()
        self._refresh_table()

    def _clear_search(self):
        self.search_entry.delete(0, "end")
        self.filtered = self.data.copy()
        self._refresh_table()

    # ---------------- Summary tab ----------------

    def _build_summary_tab(self):
        wrap = ctk.CTkFrame(self.tab_summary)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        left = ctk.CTkFrame(wrap, width=420)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        right = ctk.CTkFrame(wrap)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(left, text="Descriptive Statistics",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=8, pady=(8,4))
        self.summary_box = ctk.CTkTextbox(left, height=520)
        self.summary_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.summary_box.configure(state="disabled")

        aud = ctk.CTkCollapsibleFrame(right, text="Quick Audits (computed columns)")
        aud.pack(fill="x", padx=6, pady=6)

        self.ppv_limit_entry = ctk.CTkEntry(aud, placeholder_text="PPV limit (mm/s)", width=140)
        self.ppv_limit_entry.insert(0, "12.5")
        self.air_limit_entry = ctk.CTkEntry(aud, placeholder_text="Air limit (dB)", width=140)
        self.air_limit_entry.insert(0, "134")
        self.hpd_entry = ctk.CTkEntry(aud, placeholder_text="HPD (holes/delay)", width=140)
        self.hpd_entry.insert(0, "1")

        row = ctk.CTkFrame(aud); row.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(row, text="PPV limit").pack(side="left", padx=4)
        self.ppv_limit_entry.pack(side="left", padx=6)
        ctk.CTkLabel(row, text="Air limit").pack(side="left", padx=8)
        self.air_limit_entry.pack(side="left", padx=6)

        row2 = ctk.CTkFrame(aud); row2.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(row2, text="Holes per delay").pack(side="left", padx=4)
        self.hpd_entry.pack(side="left", padx=6)
        ctk.CTkButton(aud, text="Compute KPIs (PF, Q/delay, pass rates)", command=self._audit_kpis).pack(padx=6, pady=6)

        self.audit_box = ctk.CTkTextbox(right, height=420)
        self.audit_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.audit_box.configure(state="disabled")

    def _update_summary(self):
        if self.filtered.empty:
            text = "No data loaded."
        else:
            desc = self.filtered.describe(include="all").to_string()
            miss = self.filtered.isna().sum().rename("Missing").to_string()
            rng_lines = []
            for c in self._numeric_cols():
                s = self.filtered[c].dropna()
                if len(s):
                    rng_lines.append(f"{c}: min={s.min():.3g}, max={s.max():.3g}, median={s.median():.3g}")
            rng = "\n".join(rng_lines)
            text = f"{desc}\n\n--- Missing values ---\n{miss}\n\n--- Ranges ---\n{rng}"
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", text)
        self.summary_box.configure(state="disabled")

    def _audit_kpis(self):
        try:
            ppv_lim = float(self.ppv_limit_entry.get())
            air_lim = float(self.air_limit_entry.get())
            hpd = max(1, int(float(self.hpd_entry.get())))
        except:
            messagebox.showwarning("Input", "Enter numeric PPV/Air/HPD values."); return

        df = self.filtered.copy()
        out = []

        if "Hole depth" in df and "Stemming" in df:
            df["Charge length"] = (df["Hole depth"] - df["Stemming"]).clip(lower=0)
        if "Hole diameter" in df:
            d_m = df["Hole diameter"] / 1000.0
            df["Hole area"] = np.pi * (d_m**2) / 4.0
        if "Linear charge" in df and "Charge length" in df:
            df["Mass/hole (est)"] = df["Linear charge"] * df["Charge length"]
        if "Explosive mass" in df and "Number of holes" in df:
            df["Mass/hole (from total)"] = df["Explosive mass"] / df["Number of holes"]
            if "Mass/hole (est)" in df:
                df["Mass/hole"] = df[["Mass/hole (est)", "Mass/hole (from total)"]].mean(axis=1, skipna=True)
            else:
                df["Mass/hole"] = df["Mass/hole (from total)"]
        elif "Mass/hole (est)" in df:
            df["Mass/hole"] = df["Mass/hole (est)"]

        if "Blast volume" in df and "Explosive mass" in df:
            df["PF (from provided)"] = df["Explosive mass"] / df["Blast volume"]
        elif "Mass/hole" in df and "Burden" in df and "Spacing" in df and "Hole depth" in df and "Number of holes" in df:
            df["Blast volume (auto)"] = df["Burden"] * df["Spacing"] * df["Hole depth"] * df["Number of holes"]
            df["Explosive total (auto)"] = df["Mass/hole"] * df["Number of holes"]
            df["PF (auto)"] = df["Explosive total (auto)"] / df["Blast volume (auto)"]

        if "Mass/hole" in df:
            df["Q/delay"] = hpd * df["Mass/hole"]
        if "Distance" in df and "Q/delay" in df:
            df["SD (PPV)"] = df["Distance"] / np.sqrt(df["Q/delay"].clip(lower=1e-9))
            df["SD_air"] = np.log10((df["Q/delay"].clip(lower=1e-9) ** (1/3)) / df["Distance"].clip(lower=1e-9))

        if "Ground Vibration" in df:
            out.append(f"PPV pass rate (@{ppv_lim} mm/s): {(df['Ground Vibration'] <= ppv_lim).mean()*100:.1f}%")
        if "Airblast" in df:
            out.append(f"Airblast pass rate (@{air_lim} dB): {(df['Airblast'] <= air_lim).mean()*100:.1f}%")
        if "Powder factor" in df:
            s = df["Powder factor"].dropna()
            if len(s): out.append(f"Powder Factor: mean={s.mean():.3g} kg/m³ (min={s.min():.3g}, max={s.max():.3g})")
        if "Fragmentation" in df:
            s = df["Fragmentation"].dropna()
            if len(s): out.append(f"Fragmentation X50: mean={s.mean():.3g} mm (min={s.min():.3g}, max={s.max():.3g})")

        key = [c for c in ["Ground Vibration","Airblast","Powder factor","Fragmentation"] if c in df.columns]
        for c in key:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((df[c] < (q1 - 1.5*iqr)) | (df[c] > (q3 + 1.5*iqr))).sum()
                out.append(f"Outliers {c}: {outliers} rows (IQR rule)")

        txt = "\n".join(out) if out else "No KPIs computed (check columns)."
        self.audit_box.configure(state="normal")
        self.audit_box.delete("1.0", "end")
        self.audit_box.insert("end", txt)
        self.audit_box.configure(state="disabled")

    # ---------------- Visuals tab ----------------

    def _build_visuals_tab(self):
        wrap = ctk.CTkFrame(self.tab_visuals)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        controls = ctk.CTkFrame(wrap, width=320)
        controls.pack(side="left", fill="y", padx=6, pady=6)

        viz = ctk.CTkFrame(wrap)
        viz.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(controls, text="Plot Controls",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=8, pady=(8,4))

        self.plot_type = ctk.CTkOptionMenu(controls, values=[
            "Scatter", "Line", "Bar", "Histogram", "Box",
            "Hexbin", "PPV vs Scaled Distance", "Airblast Scaling",
            "Fragmentation vs PF"
        ])
        self.plot_type.set("Scatter"); self.plot_type.pack(fill="x", padx=8, pady=4)

        cols = list(self.data.columns) or [""]
        self.x_var = ctk.CTkOptionMenu(controls, values=cols); self.x_var.set(cols[0]); self.x_var.pack(fill="x", padx=8, pady=4)
        self.y_var = ctk.CTkOptionMenu(controls, values=cols); self.y_var.set(cols[1] if len(cols)>1 else cols[0]); self.y_var.pack(fill="x", padx=8, pady=4)

        self.logx = ctk.CTkSwitch(controls, text="Log X"); self.logx.deselect(); self.logx.pack(anchor="w", padx=8, pady=2)
        self.logy = ctk.CTkSwitch(controls, text="Log Y"); self.logy.deselect(); self.logy.pack(anchor="w", padx=8, pady=2)

        ctk.CTkButton(controls, text="Draw Plot", command=self._draw_plot).pack(fill="x", padx=8, pady=8)

        self.fig_viz = plt.Figure(figsize=(6.2, 4.4), dpi=100, constrained_layout=True)
        self.ax_viz = self.fig_viz.add_subplot(111)
        self.canvas_viz = FigureCanvasTkAgg(self.fig_viz, master=viz)
        self.canvas_viz.draw()
        self.canvas_viz.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_viz = NavigationToolbar2Tk(self.canvas_viz, viz)
        self.toolbar_viz.update()

    def _draw_plot(self):
        if self.filtered.empty:
            messagebox.showinfo("Plot", "No data to plot."); return

        kind = self.plot_type.get()
        x = self.x_var.get()
        y = self.y_var.get()

        self.ax_viz.clear()

        def _apply_log(ax):
            if self.logx.get(): ax.set_xscale("log")
            if self.logy.get(): ax.set_yscale("log")

        df = self.filtered.copy()

        if kind == "Histogram":
            if y not in df.columns:
                messagebox.showwarning("Plot", "Choose a Y column"); return
            s = pd.to_numeric(df[y], errors="coerce").dropna()
            self.ax_viz.hist(s, bins=30)
            self.ax_viz.set_xlabel(y); self.ax_viz.set_ylabel("Count")
            self.ax_viz.set_title(f"Histogram: {y}")

        elif kind == "Box":
            if y not in df.columns:
                messagebox.showwarning("Plot", "Choose a Y column"); return
            s = pd.to_numeric(df[y], errors="coerce").dropna()
            self.ax_viz.boxplot(s, vert=True, labels=[y])
            self.ax_viz.set_title(f"Box plot: {y}")

        elif kind == "Hexbin":
            if x not in df.columns or y not in df.columns:
                messagebox.showwarning("Plot", "Choose X and Y columns"); return
            xs = pd.to_numeric(df[x], errors="coerce")
            ys = pd.to_numeric(df[y], errors="coerce")
            m = xs.notna() & ys.notna()
            hb = self.ax_viz.hexbin(xs[m], ys[m], gridsize=30, mincnt=1)
            self.fig_viz.colorbar(hb, ax=self.ax_viz, label="Counts")
            self.ax_viz.set_xlabel(x); self.ax_viz.set_ylabel(y)
            self.ax_viz.set_title(f"Hexbin: {y} vs {x}")
            _apply_log(self.ax_viz)

        elif kind == "PPV vs Scaled Distance":
            need = ["Ground Vibration","Distance"]
            for c in need:
                if c not in df.columns:
                    messagebox.showwarning("Plot", f"Missing column: {c}"); return
            Q = self._estimate_mass_per_hole(df)
            if Q is None:
                messagebox.showwarning("Plot", "Cannot estimate per-hole mass. Ensure columns exist."); return
            SD = df["Distance"] / np.sqrt(Q.clip(lower=1e-9))
            PPV = pd.to_numeric(df["Ground Vibration"], errors="coerce")
            m = SD.notna() & PPV.notna() & (SD>0)
            xs = np.log10(SD[m]); ys = np.log10(PPV[m].clip(lower=1e-9))
            if len(xs) < 3:
                messagebox.showwarning("Plot", "Not enough valid points for PPV scaling."); return
            b, a = np.polyfit(xs, ys, 1)  # y = b*x + a
            beta = -b; K = 10**a
            self.ax_viz.scatter(SD[m], PPV[m], alpha=0.6)
            sd_line = np.linspace(SD[m].min(), SD[m].max(), 100)
            self.ax_viz.plot(sd_line, K * (sd_line**(-beta)), linewidth=2)
            self.ax_viz.set_xscale("log"); self.ax_viz.set_yscale("log")
            self.ax_viz.set_xlabel("Scaled distance SD = R / sqrt(Q_hole)")
            self.ax_viz.set_ylabel("PPV (mm/s)")
            self.ax_viz.set_title(f"PPV scaling: K≈{K:.0f}, β≈{beta:.2f}")

        elif kind == "Airblast Scaling":
            need = ["Airblast","Distance"]
            for c in need:
                if c not in df.columns:
                    messagebox.showwarning("Plot", f"Missing column: {c}"); return
            Q = self._estimate_mass_per_hole(df)
            if Q is None:
                messagebox.showwarning("Plot", "Cannot estimate per-hole mass. Ensure columns exist."); return
            t = np.log10((Q.clip(lower=1e-9)**(1/3)) / df["Distance"].clip(lower=1e-9))
            L = pd.to_numeric(df["Airblast"], errors="coerce")
            m = t.notna() & L.notna()
            if m.sum() < 3:
                messagebox.showwarning("Plot", "Not enough valid points for airblast scaling."); return
            b, a = np.polyfit(t[m], L[m], 1)  # L = a + b*t
            self.ax_viz.scatter(t[m], L[m], alpha=0.6)
            tt = np.linspace(t[m].min(), t[m].max(), 100)
            self.ax_viz.plot(tt, a + b*tt, linewidth=2)
            self.ax_viz.set_xlabel(r"log10(Q$^{1/3}$/R)")
            self.ax_viz.set_ylabel("Airblast (dB)")
            self.ax_viz.set_title(f"Airblast scaling: K_air≈{a:.1f}, B_air≈{b:.1f}")

        elif kind == "Fragmentation vs PF":
            if "Powder factor" not in df.columns:
                if "Explosive mass" in df and "Blast volume" in df:
                    df = df.copy()
                    df["Powder factor"] = df["Explosive mass"] / df["Blast volume"]
            if "Powder factor" not in df.columns or "Fragmentation" not in df.columns:
                messagebox.showwarning("Fragmentation not available."); return
            PF = pd.to_numeric(df["Powder factor"], errors="coerce")
            X50 = pd.to_numeric(df["Fragmentation"], errors="coerce")
            m = PF.notna() & X50.notna() & (PF>0)
            if m.sum() < 3:
                messagebox.showwarning("Plot", "Not enough valid PF/X50 points."); return
            self.ax_viz.scatter(1/PF[m], X50[m], alpha=0.6)
            xx = np.log(1/PF[m]); yy = np.log(X50[m])
            p, logA = np.polyfit(xx, yy, 1)
            A = np.exp(logA)
            u = np.linspace((1/PF[m]).min(), (1/PF[m]).max(), 100)
            self.ax_viz.plot(u, A * (u ** p), linewidth=2)
            self.ax_viz.set_xscale("log")
            self.ax_viz.set_xlabel("1 / PF")
            self.ax_viz.set_ylabel("X50 (mm)")
            self.ax_viz.set_title(f"Fragmentation: A≈{A:.1f}, exponent≈{p:.2f}")

        else:
            if kind in ["Scatter","Line","Bar"] and (x not in df.columns or y not in df.columns):
                messagebox.showwarning("Plot", "Choose X and Y columns"); return
            if kind == "Scatter":
                xs = pd.to_numeric(df[x], errors="coerce")
                ys = pd.to_numeric(df[y], errors="coerce")
                m = xs.notna() & ys.notna()
                self.ax_viz.scatter(xs[m], ys[m], alpha=0.7)
                self.ax_viz.set_xlabel(x); self.ax_viz.set_ylabel(y)
                self.ax_viz.set_title(f"{y} vs {x}")
                _apply_log(self.ax_viz)
            elif kind == "Line":
                self.ax_viz.plot(df[x], df[y])
                self.ax_viz.set_xlabel(x); self.ax_viz.set_ylabel(y)
                self.ax_viz.set_title(f"{y} vs {x}")
                _apply_log(self.ax_viz)
            elif kind == "Bar":
                try:
                    grouped = df.groupby(x)[y].mean().sort_index()
                    self.ax_viz.bar(grouped.index.astype(str), grouped.values)
                    self.ax_viz.set_xlabel(x); self.ax_viz.set_ylabel(f"Mean {y}")
                    self.ax_viz.set_title(f"Mean {y} by {x}")
                except Exception:
                    self.ax_viz.bar(df[x].astype(str), pd.to_numeric(df[y], errors="coerce"))
                    self.ax_viz.set_xlabel(x); self.ax_viz.set_ylabel(y)
                    self.ax_viz.set_title(f"{y} by {x}")

        self.canvas_viz.draw_idle()

    # ---------------- Correlations tab ----------------

    def _build_corr_tab(self):
        wrap = ctk.CTkFrame(self.tab_corr)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        ctk.CTkButton(wrap, text="Draw Correlation Heatmap", command=self._draw_correlation).pack(anchor="w", padx=6, pady=6)

        self.fig_corr = plt.Figure(figsize=(6.2, 4.4), dpi=100, constrained_layout=True)
        self.ax_corr = self.fig_corr.add_subplot(111)
        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, master=wrap)
        self.canvas_corr.draw()
        self.canvas_corr.get_tk_widget().pack(fill="both", expand=True)

    def _draw_correlation(self):
        self.ax_corr.clear()
        if self.filtered.empty:
            self.ax_corr.set_title("No data")
            self.canvas_corr.draw_idle(); return
        num = self.filtered[self._numeric_cols()]
        if num.empty:
            self.ax_corr.set_title("No numeric columns")
            self.canvas_corr.draw_idle(); return
        corr = num.corr(numeric_only=True)
        im = self.ax_corr.imshow(corr, cmap="viridis")
        self.fig_corr.colorbar(im, ax=self.ax_corr, fraction=0.046, pad=0.04)
        self.ax_corr.set_xticks(range(len(corr.columns)))
        self.ax_corr.set_yticks(range(len(corr.columns)))
        self.ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        self.ax_corr.set_yticklabels(corr.columns, fontsize=8)
        self.ax_corr.set_title("Correlation heatmap")
        self.canvas_corr.draw_idle()

    # ---------------- Filters tab ----------------

    def _build_filters_tab(self):
        wrap = ctk.CTkFrame(self.tab_filters)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        self.filters_panel = ctk.CTkScrollableFrame(wrap, height=520)
        self.filters_panel.pack(fill="both", expand=True, padx=6, pady=6)

        qb = ctk.CTkCollapsibleFrame(wrap, text="Query (pandas expression)")
        qb.pack(fill="x", padx=6, pady=6)
        self.query_entry = ctk.CTkEntry(qb, placeholder_text='e.g., `Ground Vibration <= 12.5 and Distance >= 400`')
        self.query_entry.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(qb, text="Apply Query", command=self._apply_query).pack(padx=6, pady=6, anchor="e")

        btns = ctk.CTkFrame(wrap); btns.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(btns, text="Apply Filters", command=self._apply_filters).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Clear Filters", command=self._reset_filters).pack(side="left", padx=6)

        self._refresh_filters_panel()

    def _refresh_filters_panel(self):
        for w in self.filters_panel.winfo_children():
            w.destroy()

        self.filter_entries = {}
        for i, c in enumerate(self._numeric_cols()):
            row = ctk.CTkFrame(self.filters_panel)
            row.grid(row=i, column=0, sticky="we", padx=6, pady=2)
            ctk.CTkLabel(row, text=c).pack(side="left", padx=4)
            ent_min = ctk.CTkEntry(row, width=100, placeholder_text="min")
            ent_max = ctk.CTkEntry(row, width=100, placeholder_text="max")
            ent_min.pack(side="left", padx=4); ent_max.pack(side="left", padx=4)
            self.filter_entries[c] = (ent_min, ent_max)

    def _apply_filters(self):
        df = self.data.copy()
        for c, (emin, emax) in self.filter_entries.items():
            vmin = emin.get().strip(); vmax = emax.get().strip()
            if vmin != "":
                try: df = df[df[c] >= float(vmin)]
                except: pass
            if vmax != "":
                try: df = df[df[c] <= float(vmax)]
                except: pass
        self.filtered = df
        self._refresh_table()
        self._update_summary()
        self._draw_correlation()

    def _apply_query(self):
        expr = self.query_entry.get().strip()
        if expr == "":
            return
        try:
            self.filtered = self.filtered.query(expr, engine="python")
            self._refresh_table(); self._update_summary(); self._draw_correlation()
        except Exception as e:
            messagebox.showerror("Query error", f"Invalid query:\n{e}")

    def _reset_filters(self):
        self.filtered = self.data.copy()
        self._refresh_table(); self._update_summary(); self._draw_correlation()
        self.query_entry.delete(0, "end")
        for c, (emin, emax) in getattr(self, "filter_entries", {}).items():
            emin.delete(0, "end"); emax.delete(0, "end")

    # ---------------- Calibration tab ----------------

    def _build_calib_tab(self):
        wrap = ctk.CTkFrame(self.tab_calib)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        left = ctk.CTkFrame(wrap, width=360)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = ctk.CTkFrame(wrap)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(left, text="Site Calibration",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=8, pady=(8,4))

        self.hpd_calib = ctk.CTkEntry(left, placeholder_text="HPD (holes/delay)", width=160)
        self.hpd_calib.insert(0, "1")
        self.hpd_calib.pack(padx=8, pady=4)

        ctk.CTkButton(left, text="Calibrate PPV (K, β)", command=self._calibrate_ppv).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(left, text="Calibrate Airblast (K_air, B_air)", command=self._calibrate_air).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(left, text="Calibrate Fragmentation (A_kuz, exponent)", command=self._calibrate_frag).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(left, text="Save Site Model JSON", command=self._save_site_model).pack(fill="x", padx=8, pady=12)

        self.calib_box = ctk.CTkTextbox(right, height=520)
        self.calib_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.calib_box.configure(state="disabled")

        self.site_model = {}

    def _estimate_mass_per_hole(self, df: pd.DataFrame):
        m1 = None; m2 = None
        if ("Linear charge" in df.columns) and ("Hole depth" in df.columns) and ("Stemming" in df.columns):
            m1 = df["Linear charge"] * (df["Hole depth"] - df["Stemming"]).clip(lower=0)
        if ("Explosive mass" in df.columns) and ("Number of holes" in df.columns):
            with np.errstate(invalid="ignore", divide="ignore"):
                m2 = df["Explosive mass"] / df["Number of holes"].replace(0, np.nan)
        if m1 is None and m2 is None:
            return None
        if m1 is not None and m2 is not None:
            return m1.combine_first(m2)
        return m1 if m1 is not None else m2

    def _calibrate_ppv(self):
        try:
            hpd = max(1, int(float(self.hpd_calib.get())))
        except:
            messagebox.showwarning("Input", "Enter numeric HPD."); return

        df = self.filtered.copy()
        need = ["Ground Vibration","Distance"]
        if any(c not in df.columns for c in need):
            messagebox.showwarning("Columns", "Need Ground Vibration and Distance columns."); return

        m_hole = self._estimate_mass_per_hole(df)
        if m_hole is None:
            messagebox.showwarning("Inputs", "Cannot estimate mass per hole from columns."); return

        Qd = hpd * m_hole
        SD = df["Distance"] / np.sqrt(Qd.clip(lower=1e-9))
        PPV = pd.to_numeric(df["Ground Vibration"], errors="coerce")
        m = SD.notna() & PPV.notna() & (SD>0) & (PPV>0)
        if m.sum() < 3:
            messagebox.showwarning("Data", "Not enough valid PPV/SD points."); return

        xs = np.log10(SD[m]); ys = np.log10(PPV[m])
        b, a = np.polyfit(xs, ys, 1)  # y = b*x + a
        beta = -b; K = 10**a
        yhat = b*xs + a
        r2 = 1 - ((ys - yhat)**2).sum() / ((ys - ys.mean())**2).sum()

        self.site_model["PPV"] = {"K": float(K), "beta": float(beta), "R2": float(r2), "HPD_assumed": int(hpd)}
        self._append_calib(f"PPV calibration:\n  K ≈ {K:.1f}\n  β ≈ {beta:.3f}\n  R² ≈ {r2:.3f}\n  (HPD assumed {hpd})\n")

    def _calibrate_air(self):
        try:
            hpd = max(1, int(float(self.hpd_calib.get())))
        except:
            messagebox.showwarning("Input", "Enter numeric HPD."); return

        df = self.filtered.copy()
        if "Airblast" not in df.columns or "Distance" not in df.columns:
            messagebox.showwarning("Columns", "Need Airblast and Distance columns."); return

        m_hole = self._estimate_mass_per_hole(df)
        if m_hole is None:
            messagebox.showwarning("Inputs", "Cannot estimate mass per hole from columns."); return

        Qd = hpd * m_hole
        t = np.log10((Qd.clip(lower=1e-9)**(1/3)) / df["Distance"].clip(lower=1e-9))
        L = pd.to_numeric(df["Airblast"], errors="coerce")
        m = t.notna() & L.notna()
        if m.sum() < 3:
            messagebox.showwarning("Data", "Not enough valid airblast points."); return

        b, a = np.polyfit(t[m], L[m], 1)  # L = a + b*t
        yhat = a + b*t[m]
        r2 = 1 - ((L[m] - yhat)**2).sum() / ((L[m] - L[m].mean())**2).sum()

        self.site_model["Airblast"] = {"K_air": float(a), "B_air": float(b), "R2": float(r2), "HPD_assumed": int(hpd)}
        self._append_calib(f"Airblast calibration:\n  K_air ≈ {a:.2f}\n  B_air ≈ {b:.2f}\n  R² ≈ {r2:.3f}\n  (HPD assumed {hpd})\n")

    def _calibrate_frag(self):
        df = self.filtered.copy()
        if "Powder factor" not in df.columns:
            if "Explosive mass" in df and "Blast volume" in df:
                df["Powder factor"] = df["Explosive mass"] / df["Blast volume"]
        if "Powder factor" not in df.columns or "Fragmentation" not in df.columns:
            messagebox.showwarning("Columns", "Need Powder factor and Fragmentation."); return

        PF = pd.to_numeric(df["Powder factor"], errors="coerce")
        X50 = pd.to_numeric(df["Fragmentation"], errors="coerce")
        m = PF.notna() & X50.notna() & (PF>0) & (X50>0)
        if m.sum() < 3:
            messagebox.showwarning("Data", "Not enough valid PF/X50 points."); return

        xx = np.log(1/PF[m]); yy = np.log(X50[m])
        p, logA = np.polyfit(xx, yy, 1)
        A = float(np.exp(logA))
        yhat = p*xx + logA
        r2 = 1 - ((yy - yhat)**2).sum() / ((yy - yy.mean())**2).sum()

        self.site_model["Fragmentation"] = {"A_kuz": A, "exponent": float(p), "R2": float(r2)}
        self._append_calib(f"Fragmentation calibration:\n  A_kuz ≈ {A:.1f} mm\n  exponent ≈ {p:.3f}\n  R² ≈ {r2:.3f}\n")

    def _append_calib(self, text):
        self.calib_box.configure(state="normal")
        self.calib_box.insert("end", text + "\n")
        self.calib_box.configure(state="disabled")
        self.calib_box.see("end")

    def _save_site_model(self):
        if not self.site_model:
            messagebox.showinfo("Save", "No calibrated values to save yet."); return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON","*.json")],
                                            title="Save site model JSON")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.site_model, f, indent=2)
            messagebox.showinfo("Saved", f"Site model saved:\n{path}\n\n"
                                 "You can copy these values into the Cost Optimisation module.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ---------------- Export tab ----------------

    def _build_export_tab(self):
        wrap = ctk.CTkFrame(self.tab_export)
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(wrap, text="Export",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=8, pady=(8,4))
        ctk.CTkButton(wrap, text="Export Filtered → CSV", command=self._save_filtered_csv).pack(padx=8, pady=6, anchor="w")
        ctk.CTkButton(wrap, text="Export Filtered → Excel", command=self._save_filtered_excel).pack(padx=8, pady=6, anchor="w")

        tip = (
            "Tip: Use Filters/Query to subset by compliance (e.g., `Ground Vibration <= 12.5`).\n"
            "Use Calibration to fit PPV/Air/Frag site constants and save as JSON."
        )
        ctk.CTkLabel(wrap, text=tip, wraplength=680, justify="left").pack(padx=8, pady=10, anchor="w")

    # ---------------- helpers ----------------

    def _numeric_cols(self):
        return [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]


# Standalone run for quick tests
if __name__ == "__main__":
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Data Management — Standalone")
    DataApp(root)
    root.mainloop()


# In[ ]:




