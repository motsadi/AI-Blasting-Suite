#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# cost_optimization.py — CTk version with Pareto + Charts
# Uses: Kuz–Ram (requested form) for Xm, then RR for X50 & Oversize
# Requires: customtkinter, numpy, scipy, matplotlib

import math
import csv
import numpy as np
from dataclasses import dataclass

import customtkinter as ctk
from tkinter import messagebox, filedialog
from scipy.optimize import minimize

# Matplotlib embedding
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


@dataclass
class SiteModels:
    K_ppv: float = 1000.0  # PPV = K_ppv * SD^-beta
    beta: float = 1.60
    K_air: float = 170.0   # L = K_air + B_air*log10(Qd^(1/3)/R)
    B_air: float = 20.0
    A_kuz: float = 22.0    # Rock factor A (Kuz–Ram)
    n_rr: float = 1.8      # Rosin–Rammler n


def safe_log10(x, eps=1e-12):
    return math.log10(max(x, eps))


class OptimizationApp:
    """
    Cost & quality optimisation for bench blasting, aligned with your CTk shell.
    Adds Pareto exploration and embedded charts.
    """
    def __init__(self, parent, registry=None, input_labels=None, outputs=None):
        self.root = parent
        self.root.title("Cost Optimisation")
        self.registry = registry
        self.input_labels = input_labels or []
        self.outputs = outputs or []

        ctk.set_default_color_theme("blue")

        self.root.geometry("1080x720")
        self.root.minsize(1000, 660)

        # Left: Inputs (scroll); Right: KPIs + Charts (tabs)
        self.left = ctk.CTkScrollableFrame(self.root, width=600)
        self.left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right = ctk.CTkFrame(self.root, width=420)
        self.right.pack(side="right", fill="both", expand=False, padx=(0, 10), pady=10)

        title = ctk.CTkLabel(self.left, text="Blasting Cost Optimisation",
                             font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(4, 10), sticky="w")

        # ---------------------
        # INPUT GROUPS (left)
        # ---------------------
        g = self._card(self.left, "Geometry & Pattern"); self._grid(g, 1)
        self.diameter_mm   = self._num(g, "Diameter (mm)",              102.0)
        self.bench_m       = self._num(g, "Bench height (m)",            10.0)
        self.burden_m      = self._num(g, "Burden (m)",                   3.0)
        self.spacing_m     = self._num(g, "Spacing (m)",                  3.3)
        self.subdrill_m    = self._num(g, "Subdrilling (m)",              2.0)
        self.stemming_m    = self._num(g, "Stemming (m)",                 1.8)
        self.num_holes     = self._num(g, "Number of holes",             30.0, step=1.0)
        self.holes_per_delay = self._num(g, "Holes per delay (HPD)",      1.0, step=1.0)
        self.volume_m3     = self._num(g, "Block Volume (m³, 0=auto)",    0.0)

        e = self._card(self.left, "Explosive & Costs"); self._grid(e, 2)
        self.rho_gcc       = self._num(e, "Explosive density (g/cc)",     1.15)
        self.rws           = self._num(e, "RWS (relative strength)",     115.0)  # default to 115 as per request
        self.rws.set_value(115.0)
        self.rws.get_value = self.rws.get_value  # keep helper
        self.rws_label_note = ctk.CTkLabel(e, text="(ANFO≈100, Emulsion often 110–125)", text_color="#777")
        self.rws_label_note.grid(row=e.grid_size()[1], column=0, columnspan=2, sticky="w", padx=6, pady=(0,6))

        self.rws  # to silence linters
        self.cost_initiation_per_hole = self._num(e, "Initiation cost / hole (BWP)",  10.0)
        self.cost_explosive_per_kg    = self._num(e, "Explosive cost / kg (BWP)",      4.0)
        self.cost_drilling_per_m      = self._num(e, "Drilling cost / m (BWP)",        7.0)

        s = self._card(self.left, "Site Effects & Limits"); self._grid(s, 3)
        self.distance_m    = self._num(s, "Distance R (m)",             500.0)
        self.K_ppv         = self._num(s, "PPV K",                      1000.0)
        self.beta          = self._num(s, "PPV β",                         1.6)
        self.ppv_limit     = self._num(s, "PPV limit (mm/s)",             12.5)
        self.en_ppv        = self._switch(s, "Constrain PPV",             True)
        self.K_air         = self._num(s, "Airblast K_air",               170.0)
        self.B_air         = self._num(s, "Airblast B_air",                20.0)
        self.air_limit     = self._num(s, "Airblast limit (dB)",          134.0)
        self.en_air        = self._switch(s, "Constrain Airblast",         True)

        f = self._card(self.left, "Fragmentation (Kuz–Ram / Rosin–Rammler)"); self._grid(f, 4)
        self.A_kuz         = self._num(f, "Rock factor A (Kuz–Ram)",       22.0)
        self.n_rr          = self._num(f, "Uniformity n (RR)",              1.8)
        self.target_x50    = self._num(f, "Target X50 (mm)",               120.0)
        self.oversize_size = self._num(f, "Oversize threshold (mm)",       500.0)
        self.oversize_max  = self._num(f, "Allow oversize (%)",             10.0)
        self.en_frag       = self._switch(f, "Use fragmentation in objective", True)

        c = self._card(self.left, "Engineering Constraints"); self._grid(c, 5)
        self.min_burden    = self._num(c, "Burden min (m)",                 2.5)
        self.max_burden    = self._num(c, "Burden max (m)",                 4.5)
        self.min_spacing_k = self._num(c, "Spacing/Burden min",             1.05)
        self.max_spacing_k = self._num(c, "Spacing/Burden max",             1.50)
        self.min_stem_k    = self._num(c, "Stemming/Burden min",            0.70)
        self.max_stem_k    = self._num(c, "Stemming/Burden max",            1.00)
        self.min_sub_k     = self._num(c, "Subdrill/Burden min",            0.30)
        self.max_sub_k     = self._num(c, "Subdrill/Burden max",            0.50)
        self.min_stiff     = self._num(c, "Stiffness ratio min (Bench/B)",  2.50)
        self.max_stiff     = self._num(c, "Stiffness ratio max",            4.50)

        o = self._card(self.left, "Optimisation Settings"); self._grid(o, 6)
        self.method = ctk.CTkOptionMenu(o, values=["SLSQP", "trust-constr"]); self.method.set("SLSQP")
        self._row(o, "Method", self.method)
        self.objective = ctk.CTkOptionMenu(o, values=["Min Cost", "Min Cost + Frag", "Min Cost + Frag + PPV/Air"])
        self.objective.set("Min Cost + Frag + PPV/Air")
        self._row(o, "Objective", self.objective)
        self.w_frag = self._num(o, "Weight (Frag)",  1.0)
        self.w_ppv  = self._num(o, "Weight (PPV)",   1.0)
        self.w_air  = self._num(o, "Weight (Air)",   0.7)

        # Buttons
        btns = ctk.CTkFrame(self.left); btns.grid(row=7, column=0, columnspan=2, sticky="we", pady=(6, 10))
        self._btn(btns, "Compute KPIs", self._compute_and_show, 0)
        self._btn(btns, "Optimise", self._optimise, 1)
        self._btn(btns, "Explore Pareto", self._explore_pareto, 2)

        # ---------------------
        # RIGHT: Tabs (KPIs + Charts)
        # ---------------------
        self.tabs = ctk.CTkTabview(self.right)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_kpi = self.tabs.add("KPIs")
        self.tab_charts = self.tabs.add("Charts")

        # KPIs textbox
        self.kpi = ctk.CTkTextbox(self.tab_kpi, height=560)
        self.kpi.pack(fill="both", expand=True, padx=10, pady=10)
        self.kpi.configure(state="disabled")

        # Charts
        self._init_charts()

        # Default site models
        self.site = SiteModels()

        # Initial compute
        self._compute_and_show()

    # ---------- UI helpers ----------
    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent)
        lab = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold"))
        lab.grid(row=0, column=0, columnspan=2, sticky="w", pady=(6, 2))
        return frame

    def _grid(self, frame, row):
        frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=(0, 10), pady=6)

    def _row(self, frame, label, widget):
        r = frame.grid_size()[1]
        ctk.CTkLabel(frame, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        widget.grid(row=r, column=1, sticky="we", padx=6, pady=4)

    def _num(self, frame, label, default, step=0.1):
        ent = ctk.CTkEntry(frame); ent.insert(0, str(default))
        self._row(frame, label, ent)
        def _get():
            try: return float(ent.get())
            except: return default
        ent.get_value = _get
        ent.set_value = lambda x: (ent.delete(0, "end"), ent.insert(0, f"{x:.4g}"))
        return ent

    def _switch(self, frame, label, default=False):
        sw = ctk.CTkSwitch(frame, text=label); sw.select() if default else sw.deselect()
        r = frame.grid_size()[1]; sw.grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        sw.is_on = lambda: bool(sw.get())
        return sw

    def _btn(self, frame, text, cmd, col):
        b = ctk.CTkButton(frame, text=text, command=cmd)
        b.grid(row=0, column=col, padx=6, pady=6, sticky="we")
        return b

    def _append_kpi(self, msg):
        self.kpi.configure(state="normal"); self.kpi.insert("end", msg + "\n"); self.kpi.see("end"); self.kpi.configure(state="disabled")

    def _clear_kpi(self):
        self.kpi.configure(state="normal"); self.kpi.delete("1.0", "end"); self.kpi.configure(state="disabled")

    # ---------- Charts ----------
    def _init_charts(self):
        # Two bar charts: Cost components, Penalty breakdown
        container = ctk.CTkFrame(self.tab_charts); container.pack(fill="both", expand=True, padx=10, pady=10)

        # Cost chart
        self.fig_cost = plt.Figure(figsize=(4.2, 2.8), dpi=100, constrained_layout=True)
        self.ax_cost = self.fig_cost.add_subplot(111)
        self.canvas_cost = FigureCanvasTkAgg(self.fig_cost, master=container)
        self.canvas_cost.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Penalty chart
        self.fig_pen = plt.Figure(figsize=(4.2, 2.8), dpi=100, constrained_layout=True)
        self.ax_pen = self.fig_pen.add_subplot(111)
        self.canvas_pen = FigureCanvasTkAgg(self.fig_pen, master=container)
        self.canvas_pen.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _update_charts(self, res, penalties):
        # Cost components bar
        ci, ce, cd = res["cost_break"]
        self.ax_cost.clear()
        self.ax_cost.bar(["Initiation", "Explosive", "Drilling"], [ci, ce, cd])
        self.ax_cost.set_title("Cost components (BWP)")
        self.ax_cost.set_ylabel("BWP")
        self.canvas_cost.draw_idle()

        # Penalty breakdown bar (in cost-units added to objective)
        self.ax_pen.clear()
        labels, vals = [], []
        if penalties["frag"] > 0: labels.append("Frag"); vals.append(penalties["frag"])
        if penalties["ppv"]  > 0: labels.append("PPV");  vals.append(penalties["ppv"])
        if penalties["air"]  > 0: labels.append("Air");  vals.append(penalties["air"])
        if not labels:
            labels, vals = ["No penalties"], [0.0]
        self.ax_pen.bar(labels, vals)
        self.ax_pen.set_title("Penalty breakdown (BWP eq.)")
        self.ax_pen.set_ylabel("Penalty")
        self.canvas_pen.draw_idle()

    # ---------- Physics / Cost ----------
    def _inputs(self):
        d_mm = self.diameter_mm.get_value()
        bench = self.bench_m.get_value()
        B = self.burden_m.get_value()
        S = self.spacing_m.get_value()
        sub = self.subdrill_m.get_value()
        stem = self.stemming_m.get_value()
        n_holes = max(1, round(self.num_holes.get_value()))
        hpd = max(1, round(self.holes_per_delay.get_value()))
        vol = self.volume_m3.get_value()
        rho_gcc = self.rho_gcc.get_value()
        rws = self.rws.get_value()
        ci = self.cost_initiation_per_hole.get_value()
        ce = self.cost_explosive_per_kg.get_value()
        cd = self.cost_drilling_per_m.get_value()
        R = self.distance_m.get_value()
        Kp = self.K_ppv.get_value()
        beta = self.beta.get_value()
        ppv_lim = self.ppv_limit.get_value()
        Ka = self.K_air.get_value()
        Ba = self.B_air.get_value()
        air_lim = self.air_limit.get_value()
        Ak = self.A_kuz.get_value()
        nrr = self.n_rr.get_value()
        x50_target = self.target_x50.get_value()
        x_ov = self.oversize_size.get_value()
        ov_max = self.oversize_max.get_value() / 100.0
        Bmin = self.min_burden.get_value()
        Bmax = self.max_burden.get_value()
        kS_min = self.min_spacing_k.get_value()
        kS_max = self.max_spacing_k.get_value()
        kStem_min = self.min_stem_k.get_value()
        kStem_max = self.max_stem_k.get_value()
        kSub_min = self.min_sub_k.get_value()
        kSub_max = self.max_sub_k.get_value()
        stiff_min = self.min_stiff.get_value()
        stiff_max = self.max_stiff.get_value()
        return {
            "d_mm": d_mm, "bench": bench, "B": B, "S": S, "sub": sub, "stem": stem,
            "n_holes": n_holes, "hpd": hpd, "vol": vol, "rho_gcc": rho_gcc, "rws": rws,
            "ci": ci, "ce": ce, "cd": cd, "R": R, "Kp": Kp, "beta": beta, "ppv_lim": ppv_lim,
            "Ka": Ka, "Ba": Ba, "air_lim": air_lim, "Ak": Ak, "nrr": nrr,
            "x50_target": x50_target, "x_ov": x_ov, "ov_max": ov_max,
            "Bmin": Bmin, "Bmax": Bmax, "kS_min": kS_min, "kS_max": kS_max,
            "kStem_min": kStem_min, "kStem_max": kStem_max,
            "kSub_min": kSub_min, "kSub_max": kSub_max,
            "stiff_min": stiff_min, "stiff_max": stiff_max,
        }

    def _derived(self, p):
        d_m = p["d_mm"] / 1000.0
        area = math.pi * (d_m**2) / 4.0
        hole_len = max(0.0, p["bench"] + p["sub"])          # drilled length
        charge_len = max(0.0, hole_len - p["stem"])          # charged column
        rho = p["rho_gcc"] * 1000.0                          # kg/m3
        m_per_hole = rho * area * charge_len                 # kg  (Q per hole)
        m_total = m_per_hole * p["n_holes"]
        vol = p["vol"] if p["vol"] > 0 else p["B"] * p["S"] * p["bench"] * p["n_holes"]
        PF = m_total / max(1e-9, vol)
        drill_len_total = p["n_holes"] * hole_len
        Q_delay = p["hpd"] * m_per_hole
        return {
            "d_m": d_m, "area": area, "hole_len": hole_len, "charge_len": charge_len, "rho": rho,
            "m_per_hole": m_per_hole, "m_total": m_total, "PF": PF, "vol": vol,
            "drill_len_total": drill_len_total, "Q_delay": Q_delay
        }

    def _cost(self, p, d):
        initiation_cost = p["n_holes"] * p["ci"]
        explosive_cost  = d["m_total"] * p["ce"]
        drilling_cost   = d["drill_len_total"] * p["cd"]
        total = initiation_cost + explosive_cost + drilling_cost
        return total, initiation_cost, explosive_cost, drilling_cost

    def _vibration(self, p, d):
        SD = p["R"] / max(1e-9, math.sqrt(d["Q_delay"]))
        PPV = p["Kp"] * (SD ** (-p["beta"]))
        return SD, PPV

    def _airblast(self, p, d):
        denom = max(1e-9, d["Q_delay"] ** (1.0/3.0))
        L = p["Ka"] + p["Ba"] * safe_log10(p["R"] / denom)
        return L

    # ---------- NEW: Kuz–Ram Xm (requested) + RR for X50 & oversize ----------
    def _fragmentation(self, p, d):
        """
        Computes:
          Xm = A * K^-0.8 * Q^(1/6) * (115/RWS)^(19/20)
          λ  = Xm / Γ(1+1/n)
          X50 = λ*(ln 2)^(1/n)
          Oversize(x_ov) = exp(-(x_ov/λ)^n)
        Falls back to old V/Q form ONLY if inputs are insufficient.
        Returns (Xm_mm, X50_mm, oversize_fraction).
        """
        A   = max(0.0, p["Ak"])
        RWS = max(1e-9, p["rws"])
        n   = max(0.5, p["nrr"])
        K_pf = d["PF"]
        Q_ph = d["m_per_hole"]  # per-hole charge (kg)

        if A > 0.0 and K_pf > 0.0 and Q_ph > 0.0 and RWS > 0.0:
            Xm = A * (K_pf ** -0.8) * (Q_ph ** (1.0/6.0)) * ((115.0 / RWS) ** (19.0/20.0))
            lam = Xm / math.gamma(1.0 + 1.0/n)
            X50 = lam * (math.log(2.0) ** (1.0/n))
            oversize = math.exp(- (p["x_ov"] / max(1e-9, lam)) ** n)
            return Xm, X50, oversize

        # ---- Fallback (legacy): use previous X50≈A*(V/Q)^0.8 and convert to RR ----
        V_over_Q = d["vol"] / max(1e-9, d["m_total"])
        X50_legacy = p["Ak"] * (V_over_Q ** 0.8)
        lam = X50_legacy / (math.log(2.0) ** (1.0/n))
        Xm_legacy = lam * math.gamma(1.0 + 1.0/n)
        oversize = math.exp(- (p["x_ov"] / max(1e-9, lam)) ** n)
        return Xm_legacy, X50_legacy, oversize

    # ---------- Objective & constraints ----------
    def _objective_factory(self, p, weights, use_ppv, use_air, use_frag):
        def obj(x):
            B, S, sub = x
            trial = p.copy(); trial["B"], trial["S"], trial["sub"] = B, S, sub
            d = self._derived(trial)
            total_cost, _, _, _ = self._cost(trial, d)
            penalty = 0.0

            if use_frag:
                Xm, X50, ov = self._fragmentation(trial, d)
                pen_x50 = max(0.0, (X50 - trial["x50_target"]) / max(1e-9, trial["x50_target"]))
                pen_ov  = max(0.0, ov - trial["ov_max"])
                penalty += weights["frag"] * (10.0 * pen_x50**2 + 20.0 * pen_ov**2)

            if use_ppv:
                _, PPV = self._vibration(trial, d)
                pen_ppv = max(0.0, (PPV - trial["ppv_lim"]) / max(1e-9, trial["ppv_lim"]))
                penalty += weights["ppv"] * (15.0 * pen_ppv**2)

            if use_air:
                L = self._airblast(trial, d)
                pen_air = max(0.0, (L - trial["air_lim"]) / max(1e-9, trial["air_lim"]))
                penalty += weights["air"] * (10.0 * pen_air**2)

            return total_cost + penalty
        return obj

    def _constraints_scipy(self, p):
        cons = []
        def c_spacing_min(x): B, S, sub = x; return S - p["kS_min"] * B
        def c_spacing_max(x): B, S, sub = x; return p["kS_max"] * B - S
        def c_stem_min(x):    B, S, sub = x; return p["stem"] - p["kStem_min"] * B
        def c_stem_max(x):    B, S, sub = x; return p["kStem_max"] * B - p["stem"]
        def c_sub_min(x):     B, S, sub = x; return sub - p["kSub_min"] * B
        def c_sub_max(x):     B, S, sub = x; return p["kSub_max"] * B - sub
        def c_stiff_min(x):   B, S, sub = x; return (p["bench"] / max(1e-9, B)) - p["stiff_min"]
        def c_stiff_max(x):   B, S, sub = x; return p["stiff_max"] - (p["bench"] / max(1e-9, B))
        for fn in [c_spacing_min, c_spacing_max, c_stem_min, c_stem_max,
                   c_sub_min, c_sub_max, c_stiff_min, c_stiff_max]:
            cons.append({'type': 'ineq', 'fun': fn})
        return cons

    def _bounds(self, p):
        B_lo, B_hi = p["Bmin"], p["Bmax"]
        S_lo, S_hi = 0.5, 8.0
        sub_lo, sub_hi = 0.0, max(6.0, p["bench"])
        return [(B_lo, B_hi), (S_lo, S_hi), (sub_lo, sub_hi)]

    # ---------- Actions ----------
    def _compute_metrics(self, p):
        d = self._derived(p)
        cost, ci, ce, cd = self._cost(p, d)
        SD, PPV = self._vibration(p, d)
        L = self._airblast(p, d)
        Xm, X50, ov = self._fragmentation(p, d)
        return {"inputs": p, "derived": d, "cost": cost, "cost_break": (ci, ce, cd),
                "SD": SD, "PPV": PPV, "L": L, "Xm": Xm, "X50": X50, "oversize": ov}

    def _compute_penalties_now(self, p, res):
        # Use current UI weights & toggles to compute penalty components
        use_frag = (self.objective.get() in ["Min Cost + Frag", "Min Cost + Frag + PPV/Air"]) and self.en_frag.is_on()
        use_ppv  = (self.objective.get() == "Min Cost + Frag + PPV/Air") and self.en_ppv.is_on()
        use_air  = (self.objective.get() == "Min Cost + Frag + PPV/Air") and self.en_air.is_on()
        wf, wp, wa = self.w_frag.get_value(), self.w_ppv.get_value(), self.w_air.get_value()

        pen_frag = pen_ppv = pen_air = 0.0

        if use_frag:
            X50, ov = res["X50"], res["oversize"]
            pen_x50 = max(0.0, (X50 - p["x50_target"]) / max(1e-9, p["x50_target"]))
            pen_ov  = max(0.0, ov - p["ov_max"])
            pen_frag = wf * (10.0 * pen_x50**2 + 20.0 * pen_ov**2)

        if use_ppv:
            pen_ppv = wp * (15.0 * max(0.0, (res["PPV"] - p["ppv_lim"]) / max(1e-9, p["ppv_lim"]))**2)

        if use_air:
            pen_air = wa * (10.0 * max(0.0, (res["L"] - p["air_lim"]) / max(1e-9, p["air_lim"]))**2)

        return {"frag": pen_frag, "ppv": pen_ppv, "air": pen_air}

    def _format_report(self, res):
        p = res["inputs"]; d = res["derived"]
        ci, ce, cd = res["cost_break"]
        lines = []
        lines.append("=== Pattern & Charge ===")
        lines.append(f"Diameter: {p['d_mm']:.0f} mm | Bench: {p['bench']:.2f} m")
        lines.append(f"Burden: {p['B']:.2f} m | Spacing: {p['S']:.2f} m | Subdrill: {p['sub']:.2f} m | Stemming: {p['stem']:.2f} m")
        lines.append(f"Holes: {p['n_holes']} | HPD: {p['hpd']} | Charge length: {d['charge_len']:.2f} m | ρ: {p['rho_gcc']:.2f} g/cc")
        lines.append(f"Mass/hole (Q): {d['m_per_hole']:.2f} kg | Total explosive: {d['m_total']:.2f} kg")
        lines.append(f"Block volume: {d['vol']:.1f} m³ | Achieved PF (K): {d['PF']:.3f} kg/m³")
        lines.append("")
        lines.append("=== Cost ===")
        lines.append(f"Total Cost: BWP {res['cost']:.2f}")
        lines.append(f"  • Initiation: BWP {ci:.2f}")
        lines.append(f"  • Explosive:  BWP {ce:.2f}")
        lines.append(f"  • Drilling:   BWP {cd:.2f}")
        lines.append("")
        lines.append("=== Vibration & Airblast ===")
        lines.append(f"Q/delay: {d['Q_delay']:.2f} kg | Distance R: {p['R']:.1f} m | Scaled distance SD: {res['SD']:.3f}")
        lines.append(f"PPV: {res['PPV']:.2f} mm/s  (limit {p['ppv_lim']:.2f})  [{'OK' if res['PPV']<=p['ppv_lim'] else 'EXCEED'}]")
        lines.append(f"Airblast: {res['L']:.1f} dB  (limit {p['air_lim']:.1f})  [{'OK' if res['L']<=p['air_lim'] else 'EXCEED'}]")
        lines.append("")
        lines.append("=== Fragmentation (Kuz–Ram + RR) ===")
        lines.append(f"Xm (Kuz–Ram): {res['Xm']:.1f} mm   |   X50 (RR): {res['X50']:.1f} mm  (target {p['x50_target']:.1f})")
        lines.append(f"Oversize@{p['x_ov']:.0f} mm: {100*res['oversize']:.1f}%  (allowed {100*p['ov_max']:.1f}%)  "
                     f"[{'OK' if res['oversize']<=p['ov_max'] else 'EXCEED'}]")
        lines.append(f"Uniformity n (RR): {p['nrr']:.2f}   |   RWS: {p['rws']:.0f}   |   A: {p['Ak']:.1f}")
        lines.append("")
        lines.append("=== Engineering Constraint Checks ===")
        B = p["B"]; S = p["S"]; sub = p["sub"]; bench = p["bench"]; stem = p["stem"]
        checks = [
            ("Spacing/Burden min", S >= p["kS_min"]*B),
            ("Spacing/Burden max", S <= p["kS_max"]*B),
            ("Stemming/Burden min", stem >= p["kStem_min"]*B),
            ("Stemming/Burden max", stem <= p["kStem_max"]*B),
            ("Subdrill/Burden min", sub >= p["kSub_min"]*B),
            ("Subdrill/Burden max", sub <= p["kSub_max"]*B),
            ("Stiffness ratio min", (bench/B) >= p["stiff_min"]),
            ("Stiffness ratio max", (bench/B) <= p["stiff_max"]),
        ]
        for name, ok in checks: lines.append(f"{name}: {'✓' if ok else '✗'}")
        return "\n".join(lines)

    def _compute_and_show(self):
        self._clear_kpi()
        p = self._inputs()
        self.site.K_ppv, self.site.beta = p["Kp"], p["beta"]
        self.site.K_air, self.site.B_air = p["Ka"], p["Ba"]
        self.site.A_kuz, self.site.n_rr = p["Ak"], p["nrr"]

        res = self._compute_metrics(p)
        self._append_kpi(self._format_report(res))

        # update charts
        penalties = self._compute_penalties_now(p, res)
        self._update_charts(res, penalties)

    def _optimise(self):
        p = self._inputs()
        x0 = np.array([p["B"], p["S"], p["sub"]], dtype=float)

        use_frag = (self.objective.get() in ["Min Cost + Frag", "Min Cost + Frag + PPV/Air"]) and self.en_frag.is_on()
        use_ppv  = (self.objective.get() == "Min Cost + Frag + PPV/Air") and self.en_ppv.is_on()
        use_air  = (self.objective.get() == "Min Cost + Frag + PPV/Air") and self.en_air.is_on()
        weights = {"frag": self.w_frag.get_value(), "ppv": self.w_ppv.get_value(), "air": self.w_air.get_value()}

        obj = self._objective_factory(p, weights, use_ppv, use_air, use_frag)
        cons = self._constraints_scipy(p); bnds = self._bounds(p)
        method = self.method.get()

        try:
            res = minimize(obj, x0, method=("SLSQP" if method=="SLSQP" else "trust-constr"),
                           bounds=bnds, constraints=cons,
                           options=dict(maxiter=400, ftol=1e-7))
        except Exception as e:
            messagebox.showerror("Error", f"Optimiser error: {e}"); return

        if not res.success:
            messagebox.showwarning("Optimisation", f"Finished with status: {res.message}")

        x_opt = res.x
        self.burden_m.set_value(float(x_opt[0])); self.spacing_m.set_value(float(x_opt[1])); self.subdrill_m.set_value(float(x_opt[2]))

        # recompute & show
        self._clear_kpi()
        self._append_kpi("Optimisation completed.\n")
        pres = self._inputs(); rep = self._format_report(self._compute_metrics(pres))
        self._append_kpi(rep)

        # update charts
        res_now = self._compute_metrics(pres)
        penalties = self._compute_penalties_now(pres, res_now)
        self._update_charts(res_now, penalties)

    # ---------- Pareto Exploration ----------
    def _explore_pareto(self):
        # Small grid of weights for quick exploration
        w_list = [0.0, 1.0, 2.0]  # 3^3 = 27 runs
        base_p = self._inputs()
        cons = self._constraints_scipy(base_p); bnds = self._bounds(base_p)
        method = "SLSQP" if self.method.get()=="SLSQP" else "trust-constr"

        # Enable flags according to current toggles
        en_frag = self.en_frag.is_on()
        en_ppv  = self.en_ppv.is_on()
        en_air  = self.en_air.is_on()

        rows = []
        x0 = np.array([base_p["B"], base_p["S"], base_p["sub"]], dtype=float)

        for wf in w_list:
            for wp in w_list:
                for wa in w_list:
                    if wf==0 and wp==0 and wa==0 and len(rows)>0:
                        continue
                    weights = {"frag": wf, "ppv": wp, "air": wa}

                    obj = self._objective_factory(base_p, weights,
                                                  use_ppv=(en_ppv and (wp>0)),
                                                  use_air=(en_air and (wa>0)),
                                                  use_frag=(en_frag and (wf>0)))

                    try:
                        res = minimize(obj, x0, method=method, bounds=bnds, constraints=cons,
                                       options=dict(maxiter=300, ftol=1e-7))
                        x = res.x if res.success else x0
                    except Exception:
                        x = x0  # fallback

                    p_try = base_p.copy(); p_try["B"], p_try["S"], p_try["sub"] = float(x[0]), float(x[1]), float(x[2])
                    met = self._compute_metrics(p_try)
                    rows.append({
                        "wf": wf, "wp": wp, "wa": wa,
                        "B": p_try["B"], "S": p_try["S"], "sub": p_try["sub"],
                        "cost": met["cost"], "PPV": met["PPV"], "Air": met["L"],
                        "Oversize%": 100.0*met["oversize"], "X50": met["X50"], "Xm": met["Xm"],
                        "PF": met["derived"]["PF"], "Qdelay": met["derived"]["Q_delay"], "R": p_try["R"]
                    })
                    x0 = x

        # Pareto filter: minimise (cost, PPV, Oversize%, Air)
        def dominates(a, b):
            keys = ["cost", "PPV", "Oversize%", "Air"]
            not_worse = all(a[k] <= b[k] + 1e-9 for k in keys)
            strictly_better = any(a[k] < b[k] - 1e-9 for k in keys)
            return not_worse and strictly_better

        front = []
        for i, r in enumerate(rows):
            if not any(dominates(o, r) for o in rows):
                front.append(r)

        self._show_pareto_window(front)

    def _show_pareto_window(self, front):
        win = ctk.CTkToplevel(self.root); win.title("Pareto Frontier")
        win.geometry("980x640")
        header = ctk.CTkLabel(win, text=f"Non-dominated solutions: {len(front)}",
                              font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(pady=(10, 6), anchor="w", padx=10)

        table_frame = ctk.CTkScrollableFrame(win, height=260)
        table_frame.pack(fill="x", padx=10, pady=6)

        cols = ["wf","wp","wa","B","S","sub","PF","Qdelay","cost","PPV","Air","Oversize%","X50","Xm","R"]
        nice = ["w_frag","w_ppv","w_air","B","S","Sub","PF","Q/delay","Cost","PPV","Air (dB)","Oversize %","X50 (mm)","Xm (mm)","R (m)"]

        for j, h in enumerate(nice):
            ctk.CTkLabel(table_frame, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=j, padx=6, pady=4, sticky="w")

        for i, r in enumerate(sorted(front, key=lambda z: z["cost"])):  # type: ignore
            vals = [r[c] for c in cols]
            fmt = [
                ".2f",".2f",".2f",  # weights
                ".2f",".2f",".2f",  # B,S,sub
                ".3f",".2f",        # PF, Qdelay
                ".2f",".2f",".1f",".2f",".1f",".1f",".1f",  # cost, PPV, Air, Oversize%, X50, Xm, R
            ]
            for j, v in enumerate(vals):
                txt = f"{v:{fmt[min(j,len(fmt)-1)]}}"
                ctk.CTkLabel(table_frame, text=txt).grid(row=i+1, column=j, padx=6, pady=2, sticky="w")

        bottom = ctk.CTkFrame(win); bottom.pack(fill="both", expand=True, padx=10, pady=(6,10))

        fig = plt.Figure(figsize=(5.6, 3.2), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        cost = [r["cost"] for r in front]
        ppv  = [r["PPV"] for r in front]
        size = [max(20, 5 + r["Oversize%"]) for r in front]
        color= [r["Air"] for r in front]
        sc = ax.scatter(cost, ppv, s=size, c=color)
        ax.set_xlabel("Cost (BWP)"); ax.set_ylabel("PPV (mm/s)")
        ax.set_title("Pareto: Cost vs PPV (marker size = Oversize%, color = Air dB)")
        canvas = FigureCanvasTkAgg(fig, master=bottom); canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

        btns = ctk.CTkFrame(bottom); btns.pack(side="right", fill="y", padx=6, pady=6)
        def save_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], title="Save Pareto CSV")
            if not path: return
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["w_frag","w_ppv","w_air","B","S","Sub","PF","Qdelay","Cost","PPV","Air_dB","Oversize_pct","X50_mm","Xm_mm","R_m"])
                w.writeheader()
                for r in front:
                    w.writerow({
                        "w_frag":r["wf"], "w_ppv":r["wp"], "w_air":r["wa"],
                        "B":r["B"], "S":r["S"], "Sub":r["sub"],
                        "PF":r["PF"], "Qdelay":r["Qdelay"],
                        "Cost":r["cost"], "PPV":r["PPV"], "Air_dB":r["Air"],
                        "Oversize_pct":r["Oversize%"], "X50_mm":r["X50"], "Xm_mm":r["Xm"], "R_m":r["R"]
                    })
            messagebox.showinfo("Saved", f"Pareto CSV saved to:\n{path}")

        ctk.CTkButton(btns, text="Save CSV", command=save_csv).pack(pady=6, fill="x")


# Standalone run
if __name__ == "__main__":
    app = ctk.CTk()
    app.title("Cost Optimisation — Standalone")
    OptimizationApp(app)
    app.mainloop()

