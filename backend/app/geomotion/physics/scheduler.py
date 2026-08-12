from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..explosives import build_continuous_deck
from ..schemas import GeoMotionRequest


@dataclass(frozen=True)
class BlastEvent:
    index: int
    hole_index: int
    hole_id: str
    nominal_time_ms: float
    actual_time_ms: float
    timing_error_ms: float
    x: float
    y: float
    collar_z: float
    toe_z: float
    deck_top_z: float
    deck_bottom_z: float
    charge_kg: float
    primer_kg: float
    diameter_mm: float
    stemming_m: float


def build_event_queue(request: GeoMotionRequest, realization: int = 0) -> list[BlastEvent]:
    a = request.assumptions
    rng = np.random.default_rng(request.seed + realization * 104729)
    minimum_delay = min(hole.delay_ms for hole in request.holes)
    events: list[BlastEvent] = []
    for hole_index, hole in enumerate(request.holes):
        nominal = hole.delay_ms - minimum_delay
        sigma = (
            a.electronic_scatter_base_ms + a.electronic_scatter_per_delay * nominal
            if a.electronic_scatter_enabled
            else 0.0
        )
        timing_error = float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0
        collar_z = hole.z if hole.z is not None else 680.0
        depth = hole.depth if hole.depth is not None else 15.7
        stemming = hole.stemming_m if hole.stemming_m is not None else a.stemming_m
        decks = hole.decks or [
            build_continuous_deck(
                depth,
                stemming,
                hole.charge if hole.charge is not None else 625.0,
            )
        ]
        for deck in decks:
            if hasattr(deck, "model_dump"):
                raw = deck.model_dump()
            else:
                raw = deck
            primer = raw["primer"]
            events.append(
                BlastEvent(
                    index=len(events),
                    hole_index=hole_index,
                    hole_id=hole.id,
                    nominal_time_ms=float(nominal),
                    actual_time_ms=max(0.0, float(nominal + timing_error)),
                    timing_error_ms=timing_error,
                    x=hole.x,
                    y=hole.y,
                    collar_z=collar_z,
                    toe_z=collar_z - depth,
                    deck_top_z=collar_z - float(raw["top_depth_m"]),
                    deck_bottom_z=collar_z - float(raw["bottom_depth_m"]),
                    charge_kg=float(raw["mass_kg"]),
                    primer_kg=float(primer["mass_kg"]),
                    diameter_mm=hole.diameter_mm or a.hole_diameter_mm,
                    stemming_m=stemming,
                )
            )
    ordered = sorted(events, key=lambda event: (event.actual_time_ms, event.hole_id))
    return [
        BlastEvent(**{**event.__dict__, "index": index})
        for index, event in enumerate(ordered)
    ]
