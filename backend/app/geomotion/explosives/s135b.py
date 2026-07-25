from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class S135BProfile:
    name: str = "S135B bulk emulsion"
    density_kg_m3: float = 1250.51
    rws_percent: float = 115.0
    nominal_vod_m_s: float = 4500.0
    vod_range_m_s: tuple[float, float] = (3500.0, 5500.0)
    anfo_energy_mj_kg: float = 3.7
    pentolite_booster_kg: float = 0.4
    provenance: str = "site_supplied"

    @property
    def effective_energy_mj_kg(self) -> float:
        return self.anfo_energy_mj_kg * self.rws_percent / 100.0

    def cj_pressure_proxy_pa(self, vod_m_s: float | None = None) -> float:
        """Reduced-order pressure proxy, not a product-certified CJ/JWL value."""
        velocity = vod_m_s or self.nominal_vod_m_s
        return 0.25 * self.density_kg_m3 * velocity * velocity


def linear_charge_kg_m(diameter_mm: float, density_kg_m3: float = 1250.51) -> float:
    radius_m = diameter_mm / 2000.0
    return pi * radius_m * radius_m * density_kg_m3


def build_continuous_deck(
    depth_m: float,
    stemming_m: float,
    charge_kg: float,
    explosive: str = "S135B",
) -> dict:
    bottom = max(depth_m, 0.01)
    top = min(max(stemming_m, 0.0), bottom - 0.01)
    return {
        "explosive": explosive,
        "top_depth_m": top,
        "bottom_depth_m": bottom,
        "mass_kg": max(charge_kg, 0.01),
        "primer": {
            "explosive": "Pentolite",
            "mass_kg": 0.4,
            "position_from_toe_m": 0.5,
            "provenance": "site_supplied",
        },
        "provenance": "synthetic",
    }
