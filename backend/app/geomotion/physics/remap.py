from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RemapResult:
    positions: np.ndarray
    destination_class: np.ndarray
    collision_count: int
    occupied_cells: int
    loader_destination: np.ndarray


def _ore_envelope_grade(
    positions: np.ndarray,
    center: np.ndarray,
    span_x: float,
    span_y: float,
) -> np.ndarray:
    nx = (positions[:, 0] - center[0]) / max(span_x * 0.40, 1.0)
    ny = (positions[:, 1] - center[1] - 0.10 * span_y * np.sin(nx * 1.7)) / max(
        span_y * 0.36, 1.0
    )
    radius = np.hypot(nx, ny)
    return np.where(
        radius < 1.0,
        18.0 + 22.0 * np.maximum(0.0, 1.0 - radius),
        np.where(radius < 1.12, 3.0 * (1.12 - radius) / 0.12, 0.0),
    )


def settle_and_remap(
    raw_destination: np.ndarray,
    center: np.ndarray,
    floor_rl: float,
    cell_size_m: float,
    source_span: tuple[float, float],
    cutoff_grade_cpht: float,
    minimum_mining_unit_m: float,
    grades: np.ndarray,
    tonnes: np.ndarray,
) -> RemapResult:
    cell = cell_size_m
    ix = np.rint((raw_destination[:, 0] - center[0]) / cell).astype(np.int64)
    iy = np.rint((raw_destination[:, 1] - center[1]) / cell).astype(np.int64)
    preferred_z = np.maximum(
        0,
        np.rint((raw_destination[:, 2] - floor_rl - cell * 0.5) / cell).astype(np.int64),
    )
    order = np.lexsort((preferred_z, iy, ix))
    settled_levels = np.empty(len(raw_destination), dtype=np.int64)
    column_next: dict[tuple[int, int], int] = {}
    collisions = 0
    for index in order:
        key = (int(ix[index]), int(iy[index]))
        next_level = column_next.get(key, 0)
        level = max(next_level, int(preferred_z[index]))
        if level != preferred_z[index]:
            collisions += 1
        settled_levels[index] = level
        column_next[key] = level + 1
    positions = np.column_stack(
        [
            center[0] + ix * cell,
            center[1] + iy * cell,
            floor_rl + (settled_levels + 0.5) * cell,
        ]
    )
    span_x, span_y = source_span
    destination_grade = _ore_envelope_grade(positions, center, span_x, span_y)
    destination_class = np.where(destination_grade >= cutoff_grade_cpht, "ORE", "WASTE")

    mmu = max(minimum_mining_unit_m, cell)
    mx = np.floor((positions[:, 0] - np.min(positions[:, 0])) / mmu).astype(np.int64)
    my = np.floor((positions[:, 1] - np.min(positions[:, 1])) / mmu).astype(np.int64)
    loader_destination = np.empty(len(positions), dtype="<U5")
    for key in set(zip(mx.tolist(), my.tolist())):
        mask = (mx == key[0]) & (my == key[1])
        average_grade = float(np.sum(grades[mask] * tonnes[mask]) / max(np.sum(tonnes[mask]), 1e-9))
        loader_destination[mask] = "ORE" if average_grade >= cutoff_grade_cpht else "WASTE"
    return RemapResult(
        positions=positions,
        destination_class=destination_class,
        collision_count=collisions,
        occupied_cells=len(positions),
        loader_destination=loader_destination,
    )
