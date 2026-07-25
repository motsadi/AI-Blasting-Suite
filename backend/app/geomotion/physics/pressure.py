from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..explosives import S135BProfile
from .scheduler import BlastEvent


@dataclass(frozen=True)
class EventEnergy:
    vod_m_s: float
    detonation_pressure_pa: float
    chemical_energy_j: float
    movement_energy_j: float
    gas_time_constant_ms: float
    stemming_effectiveness: float


def event_energy(
    event: BlastEvent,
    profile: S135BProfile,
    vod_m_s: float,
    burden_m: float,
    tensile_strength_mpa: float,
) -> EventEnergy:
    detonation_pressure = profile.cj_pressure_proxy_pa(vod_m_s)
    chemical_energy = event.charge_kg * profile.effective_energy_mj_kg * 1_000_000.0
    # Reduced-order partition: most explosive energy is consumed by fracture,
    # heat and vibration. This bounded fraction drives bulk displacement.
    strength_penalty = np.clip(tensile_strength_mpa / 25.0, 0.15, 1.4)
    movement_fraction = float(np.clip(0.0006 / strength_penalty, 0.0002, 0.002))
    charge_length = max(0.1, event.deck_top_z - event.deck_bottom_z)
    stemming_ratio = event.stemming_m / max(event.stemming_m + charge_length, 0.1)
    stemming_effectiveness = float(np.clip(0.35 + 1.8 * stemming_ratio, 0.25, 1.0))
    movement_energy = chemical_energy * movement_fraction * stemming_effectiveness
    gas_tau = float(np.clip(1.5 + 0.55 * burden_m + 0.08 * charge_length, 2.0, 18.0))
    return EventEnergy(
        vod_m_s=vod_m_s,
        detonation_pressure_pa=detonation_pressure,
        chemical_energy_j=chemical_energy,
        movement_energy_j=movement_energy,
        gas_time_constant_ms=gas_tau,
        stemming_effectiveness=stemming_effectiveness,
    )


def pressure_decay(peak_pressure_pa: float, elapsed_ms: float, tau_ms: float) -> float:
    return float(peak_pressure_pa * np.exp(-max(0.0, elapsed_ms) / max(tau_ms, 1e-6)))
