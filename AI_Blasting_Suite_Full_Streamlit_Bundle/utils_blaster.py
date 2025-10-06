
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from math import gamma, log10

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

INPUT_LABELS = [
    "Hole depth (m)", "Hole diameter (mm)", "Burden (m)", "Spacing (m)", "Stemming (m)",
    "Distance (m)", "Powder factor (kg/m³)", "Rock density (t/m³)", "Linear charge (kg/m)",
    "Explosive mass (kg)", "Blast volume (m³)", "# Holes"
]

def slider_ranges_from_df(df: pd.DataFrame, labels: List[str]) -> Dict[str, Tuple[float,float,float]]:
    ranges = {}
    for name in labels:
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce").dropna()
        else:
            s = pd.Series([], dtype=float)
        if s.empty:
            ranges[name] = (0.0, 0.5, 1.0); continue
        lo, hi, md = float(s.quantile(0.02)), float(s.quantile(0.98)), float(s.median())
        if lo == hi:
            lo, hi = float(s.min()), float(s.max())
        if lo == hi:
            lo, hi, md = 0.0, 1.0, 0.5
        ranges[name] = (lo, md, hi)
    return ranges

@dataclass
class EmpiricalParams:
    K_ppv: float = 1000.0
    beta: float  = 1.60
    K_air: float = 170.0
    B_air: float = 20.0
    A_kuz: float = 22.0
    RWS: float   = 115.0

def derived_charge_and_volume(vals: Dict[str, float]):
    depth = vals.get("Hole depth (m)", 0.0)
    stem  = vals.get("Stemming (m)", 0.0)
    Lc    = max(0.0, depth - stem)
    lin   = vals.get("Linear charge (kg/m)", 0.0)
    mass_per_hole_lin = lin * Lc

    mass_per_hole_exp = vals.get("Explosive mass (kg)", 0.0)
    n_holes = int(round(vals.get("# Holes", 1))) or 1
    if n_holes < 1: n_holes = 1

    vol = vals.get("Blast volume (m³)", 0.0)
    if vol <= 0.0:
        B = vals.get("Burden (m)", 0.0); S = vals.get("Spacing (m)", 0.0)
        vol = max(0.0, B * S * depth * n_holes)

    PF_slider = vals.get("Powder factor (kg/m³)", 0.0)

    per_hole = 0.0
    for cand in (mass_per_hole_lin, mass_per_hole_exp):
        if cand and cand > 0: per_hole = cand; break

    total_mass = per_hole * n_holes
    if PF_slider > 0 and vol > 0:
        total_mass = PF_slider * vol
        per_hole = total_mass / n_holes

    HPD = max(1.0, float(vals.get("HPD_override", 1.0)))
    Qd = per_hole * HPD
    return per_hole, total_mass, Qd, vol

def safe_log10(x: float, eps: float = 1e-12) -> float:
    return log10(max(x, eps))

def empirical_predictions(vals: Dict[str, float], p: EmpiricalParams, outputs: List[str]):
    R = vals.get("Distance (m)", 0.0)
    per_hole, total_mass, Qd, V = derived_charge_and_volume(vals)
    out: Dict[str, float] = {}

    if "Ground Vibration" in outputs and R > 0 and Qd > 0:
        SD = R / max(Qd, 1e-9) ** 0.5
        out["Ground Vibration"] = p.K_ppv * (SD ** (-p.beta))

    if "Airblast" in outputs and R > 0 and Qd > 0:
        out["Airblast"] = p.K_air + p.B_air * safe_log10((Qd ** (1/3)) / R)

    if "Fragmentation" in outputs:
        A, RWS = max(0.0, p.A_kuz), max(1e-6, p.RWS)
        K_pf = vals.get("Powder factor (kg/m³)", 0.0)
        if (K_pf <= 0.0) and (V > 0.0):
            K_pf = total_mass / max(V, 1e-9)
        if K_pf > 0.0 and per_hole > 0.0:
            out["Fragmentation"] = A * (K_pf ** -0.8) * (per_hole ** (1/6)) * ((115.0 / RWS) ** (19/20))
    return out

def estimate_n_rr(B: float, d_mm: float):
    if B <= 0 or d_mm <= 0: return None
    d_m = d_mm / 1000.0
    n_hat = 2.2 - 14.0 * (d_m / max(B, 1e-9))
    import numpy as np
    return float(np.clip(n_hat, 0.5, 3.5))

def rr_lambda_from_xm_n(Xm_mm: float, n: float) -> float:
    return float(Xm_mm / max(gamma(1.0 + 1.0/n), 1e-9))

def rr_cdf(x_mm: np.ndarray, lam: float, n: float) -> np.ndarray:
    import numpy as np
    x = np.maximum(1e-9, np.asarray(x_mm, dtype=float))
    return 1.0 - np.exp(- (x/lam)**n)
