from __future__ import annotations

from functools import lru_cache
from math import cos, pi, sin
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .physics.remap import settle_and_remap
from .physics.solver import run_event_physics
from .schemas import GeoMotionRequest


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _percent(numerator: float, denominator: float) -> float:
    return _round(100.0 * numerator / denominator, 2) if denominator > 0 else 0.0


def _nearest_distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _validation(request: GeoMotionRequest) -> dict[str, Any]:
    holes = request.holes
    points = np.array([[h.x, h.y] for h in holes], dtype=float)
    nearest = _nearest_distances(points)
    ids: dict[str, int] = {}
    for hole in holes:
        ids[hole.id] = ids.get(hole.id, 0) + 1
    duplicate_ids = sorted(key for key, count in ids.items() if count > 1)
    close_pairs: list[dict[str, Any]] = []
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            distance = float(np.linalg.norm(points[i] - points[j]))
            if distance < max(0.5, request.assumptions.hole_diameter_mm / 1000.0):
                close_pairs.append(
                    {"hole_a": holes[i].id, "hole_b": holes[j].id, "distance_m": _round(distance)}
                )
    floor_values = [
        h.z - h.depth for h in holes if h.z is not None and h.depth is not None
    ]
    warnings: list[str] = []
    if duplicate_ids:
        warnings.append(f"Duplicate Hole ID values detected: {', '.join(duplicate_ids[:8])}.")
    if close_pairs:
        warnings.append(f"{len(close_pairs)} near-overlapping hole pair(s) require review.")
    missing_depth = sum(h.depth is None for h in holes)
    missing_charge = sum(h.charge is None for h in holes)
    missing_z = sum(h.z is None for h in holes)
    if missing_depth or missing_charge or missing_z:
        warnings.append(
            f"Defaults imputed for {missing_depth} depth, {missing_charge} charge, and {missing_z} elevation values."
        )
    if request.block_model:
        warnings.append(
            f"Measured 1 m block model accepted ({len(request.block_model):,} cells); movement calibration remains unvalidated."
        )
    else:
        warnings.append(
            "Synthetic geology and ML calibration are demonstration data, not measured mine evidence."
        )
    return {
        "status": "review" if duplicate_ids or close_pairs else "synthetic",
        "warnings": warnings,
        "duplicate_ids": duplicate_ids,
        "near_overlap_pairs": close_pairs[:20],
        "median_nearest_hole_m": _round(float(np.median(nearest))),
        "floor_rl_m": _round(float(np.median(floor_values))) if floor_values else None,
    }


def _timing(holes: list, points: np.ndarray, assumptions) -> np.ndarray:
    supplied = np.array(
        [h.delay_ms if h.delay_ms is not None else np.nan for h in holes], dtype=float
    )
    if np.isfinite(supplied).all():
        return supplied
    azimuth = assumptions.free_face_azimuth_deg * pi / 180.0
    relief = np.array([sin(azimuth), cos(azimuth)])
    along_face = np.array([relief[1], -relief[0]])
    centered = points - np.mean(points, axis=0)
    rows = centered @ (-relief)
    lateral = centered @ along_face
    row_index = np.rint((rows - np.min(rows)) / max(assumptions.burden_m, 0.1))
    v_rank = np.abs(lateral) / max(assumptions.spacing_m, 0.1)
    generated = row_index * 42.0 + v_rank * 17.0
    generated -= np.min(generated)
    return np.where(np.isfinite(supplied), supplied, generated)


def _make_blocks(request: GeoMotionRequest, floor_rl: float, collar_rl: float) -> dict[str, np.ndarray]:
    holes = request.holes
    a = request.assumptions
    rng = np.random.default_rng(request.seed)
    xs = np.array([h.x for h in holes], dtype=float)
    ys = np.array([h.y for h in holes], dtype=float)
    pad = max(a.spacing_m, a.burden_m) * 0.8
    cell = a.cell_size_m
    gx = np.arange(np.min(xs) - pad, np.max(xs) + pad + cell * 0.5, cell)
    gy = np.arange(np.min(ys) - pad, np.max(ys) + pad + cell * 0.5, cell)
    bench_height = max(6.0, collar_rl - floor_rl - a.subdrill_m)
    levels = max(2, min(6, int(np.ceil(bench_height / cell))))
    gz = floor_rl + (np.arange(levels) + 0.5) * (bench_height / levels)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    collar_points = np.column_stack([xs, ys])
    distance_to_pattern = np.min(
        np.linalg.norm(positions[:, None, :2] - collar_points[None, :, :], axis=2),
        axis=1,
    )
    # Restrict the synthetic bench to the drilled footprint instead of the
    # larger rectangular bounds used to construct the grid.
    positions = positions[distance_to_pattern <= max(a.spacing_m, a.burden_m) * 0.90]

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    span_x = max(float(np.ptp(xs)), cell)
    span_y = max(float(np.ptp(ys)), cell)
    nx = (positions[:, 0] - cx) / (span_x * 0.40)
    ny = (positions[:, 1] - cy - 0.10 * span_y * np.sin(nx * 1.7)) / (span_y * 0.36)
    radius = np.sqrt(nx * nx + ny * ny)
    kimberlite = radius < 1.0
    core = radius < 0.48
    contact = (radius >= 0.82) & (radius < 1.12)
    facies = np.where(core, "VK", np.where(kimberlite, "SVK_M1", np.where(contact, "CONTACT", "WASTE")))
    density = np.where(kimberlite, a.rock_density_t_m3, a.rock_density_t_m3 + 0.30)
    density += rng.normal(0, 0.025, len(positions))
    grade = np.where(
        kimberlite,
        18.0 + 22.0 * np.maximum(0.0, 1.0 - radius) + 5.0 * np.sin(nx * 3.0),
        np.where(contact, 3.0 * np.maximum(0.0, 1.12 - radius) / 0.30, 0.0),
    )
    grade = np.maximum(0.0, grade + rng.normal(0, 1.8, len(positions)))
    tonnes = density * cell * cell * (bench_height / levels)
    source_class = np.where(grade >= a.cutoff_grade_cpht, "ORE", "WASTE")
    return {
        "positions": positions,
        "facies": facies,
        "density": density,
        "grade": grade,
        "tonnes": tonnes,
        "source_class": source_class,
        "center": np.array([cx, cy]),
        "radius": radius,
        "bench_height": np.array([bench_height]),
    }


@lru_cache(maxsize=8)
def _synthetic_residual_model(seed: int) -> RandomForestRegressor:
    """Fit a small surrogate to synthetic heterogeneous rock-response sweeps."""
    rng = np.random.default_rng(seed)
    n = 2400
    x = rng.uniform(0, 1, (n, 8))
    energy, distance, relief, timing, depth, grade, contact, confinement = x.T
    target = np.column_stack(
        [
            0.85 * energy * relief * np.exp(-1.6 * distance)
            + 0.18 * np.sin(grade * pi * 2)
            - 0.25 * confinement,
            0.35 * energy * (timing - 0.5) * np.exp(-distance)
            + 0.16 * np.cos(contact * pi),
            0.55 * energy * (1.0 - depth) * np.exp(-distance)
            - 0.20 * confinement,
        ]
    )
    model = RandomForestRegressor(
        n_estimators=48,
        max_depth=9,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, target)
    return model


def _movement(
    request: GeoMotionRequest,
    block_data: dict[str, np.ndarray],
    hole_points: np.ndarray,
    delays: np.ndarray,
    depths: np.ndarray,
    charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = request.assumptions
    positions = block_data["positions"]
    xy = positions[:, :2]
    horizontal_delta = xy[:, None, :] - hole_points[None, :, :]
    horizontal_distance = np.linalg.norm(horizontal_delta, axis=2)
    influence = np.exp(-horizontal_distance / max(1.0, a.spacing_m * 1.65))
    charge_scale = charges / max(float(np.median(charges)), 1.0)
    influence *= np.clip(charge_scale, 0.45, 1.8)[None, :]
    influence_sum = np.sum(influence, axis=1) + 1e-9

    azimuth = a.free_face_azimuth_deg * pi / 180.0
    face_direction = np.array([sin(azimuth), cos(azimuth)])
    center = block_data["center"]
    radial = xy - center
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1.0)
    weighted_delay = (influence @ delays) / influence_sum
    delay_span = max(float(np.ptp(delays)), 1.0)
    timing_relief = np.clip((weighted_delay - float(np.min(delays))) / delay_span, 0, 1)
    direction = face_direction[None, :] * (0.78 - 0.18 * timing_relief[:, None]) + radial * 0.22
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)

    energy = np.clip(influence_sum / max(float(np.percentile(influence_sum, 65)), 1e-6), 0.25, 1.7)
    charge_energy = np.clip(float(np.mean(charges)) / 625.0, 0.55, 1.55)
    base_throw = (
        a.burden_m
        * (0.55 + 0.55 * energy)
        * a.explosive_relative_energy
        * charge_energy
        / max(a.swell_factor, 1.0)
    )
    z_fraction = np.clip(
        (positions[:, 2] - np.min(positions[:, 2]))
        / max(float(np.ptp(positions[:, 2])), 1.0),
        0,
        1,
    )
    horizontal = direction * base_throw[:, None] * (0.72 + 0.42 * z_fraction[:, None])
    vertical = (
        a.burden_m
        * 0.30
        * energy
        * a.explosive_relative_energy
        * (0.55 + 0.70 * z_fraction)
    )
    vectors = np.column_stack([horizontal, vertical])

    nearest = np.min(horizontal_distance, axis=1)
    feature_matrix = np.column_stack(
        [
            np.clip(energy / 1.7, 0, 1),
            np.clip(nearest / (a.spacing_m * 2.5), 0, 1),
            np.clip((direction @ face_direction + 1) / 2, 0, 1),
            timing_relief,
            z_fraction,
            np.clip(block_data["grade"] / 60.0, 0, 1),
            np.clip(np.abs(block_data["radius"] - 1.0), 0, 1),
            np.clip(1.0 - influence_sum / max(float(np.max(influence_sum)), 1e-6), 0, 1),
        ]
    )
    if request.mode == "hybrid":
        residual = _synthetic_residual_model(request.seed).predict(feature_matrix)
        residual_scale = np.array([a.burden_m * 0.45, a.spacing_m * 0.22, a.burden_m * 0.30])
        vectors += residual * residual_scale

    vector_magnitude = np.linalg.norm(vectors, axis=1)
    maximum = max(a.spacing_m * 2.4, a.burden_m * 2.4)
    vectors *= np.minimum(1.0, maximum / np.maximum(vector_magnitude, 1e-6))[:, None]
    uncertainty = 0.45 + 0.10 * np.linalg.norm(vectors, axis=1) + 0.65 * feature_matrix[:, 7]
    return vectors, uncertainty, weighted_delay


def _destination_class(
    positions: np.ndarray, center: np.ndarray, source_span: tuple[float, float], cutoff: float
) -> np.ndarray:
    span_x, span_y = source_span
    nx = (positions[:, 0] - center[0]) / max(span_x * 0.40, 1.0)
    ny = (positions[:, 1] - center[1] - 0.10 * span_y * np.sin(nx * 1.7)) / max(
        span_y * 0.36, 1.0
    )
    radius = np.sqrt(nx * nx + ny * ny)
    expected_grade = np.where(
        radius < 1.0,
        18.0 + 22.0 * np.maximum(0.0, 1.0 - radius),
        np.where(radius < 1.12, 3.0 * (1.12 - radius) / 0.12, 0.0),
    )
    return np.where(expected_grade >= cutoff, "ORE", "WASTE")


def _surface(destination: np.ndarray, cell_size: float) -> list[dict[str, float]]:
    bins: dict[tuple[int, int], list[float]] = {}
    x0, y0 = float(np.min(destination[:, 0])), float(np.min(destination[:, 1]))
    for x, y, z in destination:
        key = (int(round((x - x0) / cell_size)), int(round((y - y0) / cell_size)))
        bins.setdefault(key, []).append(float(z))
    return [
        {
            "x": _round(x0 + key[0] * cell_size),
            "y": _round(y0 + key[1] * cell_size),
            "z": _round(max(values)),
        }
        for key, values in bins.items()
    ]


def simulate(request: GeoMotionRequest) -> dict[str, Any]:
    holes = request.holes
    a = request.assumptions
    validation = _validation(request)
    physics = run_event_physics(request, realization=1 if request.mode == "hybrid" else 0)
    source = physics.source
    hole_points = np.array([[hole.x, hole.y] for hole in holes], dtype=float)
    span = (float(np.ptp(hole_points[:, 0])), float(np.ptp(hole_points[:, 1])))
    remap = settle_and_remap(
        physics.destination_positions,
        source.center,
        source.floor_rl,
        a.cell_size_m,
        span,
        a.cutoff_grade_cpht,
        a.minimum_mining_unit_m,
        source.grade,
        source.tonnes,
    )
    destination = remap.positions
    vectors = destination - source.positions
    dest_class = remap.destination_class
    source_class = source.source_class
    tonnes = source.tonnes
    grade = source.grade
    carats = tonnes * grade / 100.0
    source_ore = source_class == "ORE"
    destination_ore = dest_class == "ORE"
    ore_tonnes = float(np.sum(tonnes[source_ore]))
    ore_retained = float(np.sum(tonnes[source_ore & destination_ore]))
    ore_lost = float(np.sum(tonnes[source_ore & ~destination_ore]))
    waste_dilution = float(np.sum(tonnes[~source_ore & destination_ore]))
    ore_stream = float(np.sum(tonnes[destination_ore]))
    total_carats = float(np.sum(carats[source_ore]))
    recovered_carats = float(np.sum(carats[source_ore & destination_ore]))
    loader_ore = remap.loader_destination == "ORE"
    loader_ore_retained = float(np.sum(tonnes[source_ore & loader_ore]))
    loader_dilution = float(np.sum(tonnes[~source_ore & loader_ore]))
    loader_stream = float(np.sum(tonnes[loader_ore]))

    mixing_matrix = []
    for source_name in ("ORE", "WASTE"):
        for destination_name in ("ORE", "WASTE"):
            mask = (source_class == source_name) & (dest_class == destination_name)
            mixing_matrix.append(
                {
                    "source": source_name,
                    "destination": destination_name,
                    "tonnes": _round(float(np.sum(tonnes[mask])), 1),
                    "percent_of_total": _percent(float(np.sum(tonnes[mask])), float(np.sum(tonnes))),
                }
            )

    total_blocks = len(source.positions)
    lod_factor = 1
    while total_blocks / (lod_factor**3) > a.max_visual_blocks:
        lod_factor += 1
    origin = np.min(source.positions, axis=0)
    lod_keys = np.floor(
        (source.positions - origin) / (a.cell_size_m * lod_factor) + 1e-9
    ).astype(np.int64)
    _, first_indices, inverse = np.unique(
        lod_keys, axis=0, return_index=True, return_inverse=True
    )
    group_count = len(first_indices)
    counts = np.bincount(inverse, minlength=group_count).astype(float)

    def group_mean(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.bincount(inverse, weights=values, minlength=group_count) / counts
        return np.column_stack(
            [
                np.bincount(inverse, weights=values[:, axis], minlength=group_count)
                / counts
                for axis in range(values.shape[1])
            ]
        )

    visual_source = group_mean(source.positions)
    visual_destination = group_mean(destination)
    visual_vectors = visual_destination - visual_source
    visual_tonnes = np.bincount(inverse, weights=tonnes, minlength=group_count)
    visual_carats = np.bincount(inverse, weights=carats, minlength=group_count)
    visual_grade = visual_carats * 100.0 / np.maximum(visual_tonnes, 1e-9)
    visual_source_ore = (
        np.bincount(inverse, weights=tonnes * source_ore, minlength=group_count)
        / np.maximum(visual_tonnes, 1e-9)
    ) >= 0.5
    visual_destination_ore = (
        np.bincount(inverse, weights=tonnes * destination_ore, minlength=group_count)
        / np.maximum(visual_tonnes, 1e-9)
    ) >= 0.5
    visual_uncertainty = group_mean(physics.uncertainty)
    visual_effective_time = group_mean(physics.effective_time_ms)
    visual_velocity = group_mean(physics.velocities)
    visual_peak_impulse = group_mean(physics.peak_impulse_m_s)
    visual_burden_velocity = group_mean(physics.burden_velocity_m_s)
    block_rows = []
    for group_index, first_index in enumerate(first_indices):
        source_position = visual_source[group_index]
        dest = visual_destination[group_index]
        vector = visual_vectors[group_index]
        block_rows.append(
            {
                "id": group_index + 1,
                "source": [_round(v) for v in source_position],
                "destination": [_round(v) for v in dest],
                "vector": [_round(v) for v in vector],
                "displacement_m": _round(float(np.linalg.norm(vector))),
                "uncertainty_m": _round(visual_uncertainty[group_index]),
                "facies": str(source.facies[first_index]),
                "source_class": "ORE" if visual_source_ore[group_index] else "WASTE",
                "destination_class": "ORE" if visual_destination_ore[group_index] else "WASTE",
                "density_t_m3": _round(visual_tonnes[group_index] / ((a.cell_size_m**3) * counts[group_index])),
                "grade_cpht": _round(visual_grade[group_index]),
                "tonnes": _round(visual_tonnes[group_index]),
                "contained_carats": _round(visual_carats[group_index]),
                "effective_time_ms": _round(visual_effective_time[group_index], 1),
                "velocity": [_round(v) for v in visual_velocity[group_index]],
                "peak_impulse_m_s": _round(visual_peak_impulse[group_index]),
                "burden_velocity_m_s": _round(visual_burden_velocity[group_index]),
                "contributing_event": int(physics.contributing_event[first_index]),
                "size_m": a.cell_size_m * lod_factor,
                "provenance": "measured_block_model" if request.block_model else "synthetic",
            }
        )

    hole_rows = []
    minimum_delay = min(hole.delay_ms for hole in holes)
    for idx, hole in enumerate(holes):
        hole_rows.append(
            {
                "id": hole.id,
                "x": _round(hole.x),
                "y": _round(hole.y),
                "z": _round(hole.z if hole.z is not None else 680.0),
                "depth_m": _round(hole.depth if hole.depth is not None else 15.7),
                "charge_kg": _round(hole.charge if hole.charge is not None else 625.0),
                "original_delay_ms": _round(hole.original_delay_ms if hole.original_delay_ms is not None else hole.delay_ms, 3),
                "delay_ms": _round(hole.delay_ms - minimum_delay, 3),
                "diameter_mm": _round(hole.diameter_mm or a.hole_diameter_mm),
                "inclination_deg": _round(hole.inclination_deg),
                "azimuth_deg": _round(hole.azimuth_deg),
                "decks": len(hole.decks) or 1,
            }
        )

    magnitudes = np.linalg.norm(vectors, axis=1)
    metrics = {
        "cells": total_blocks,
        "visual_cells": len(block_rows),
        "voxel_size_m": a.cell_size_m,
        "total_tonnes": _round(float(np.sum(tonnes)), 1),
        "mass_balance_error_percent": 0.0,
        "contained_carats": _round(total_carats, 1),
        "recovered_carats": _round(recovered_carats, 1),
        "carat_recovery_percent": _percent(recovered_carats, total_carats),
        "ore_recovery_percent": _percent(ore_retained, ore_tonnes),
        "ore_loss_percent": _percent(ore_lost, ore_tonnes),
        "dilution_percent": _percent(waste_dilution, ore_stream),
        "ore_tonnes_in_situ": _round(ore_tonnes, 1),
        "ore_tonnes_recovered": _round(ore_retained, 1),
        "waste_dilution_tonnes": _round(waste_dilution, 1),
        "predicted_feed_grade_cpht": _round(
            float(np.sum(carats[destination_ore]) * 100.0 / max(ore_stream, 1e-9)), 2
        ),
        "loader_recovery_percent": _percent(loader_ore_retained, ore_tonnes),
        "loader_dilution_percent": _percent(loader_dilution, loader_stream),
        "minimum_mining_unit_m": a.minimum_mining_unit_m,
        "mean_displacement_m": _round(float(np.mean(magnitudes)), 2),
        "p95_displacement_m": _round(float(np.percentile(magnitudes, 95)), 2),
        "mean_heave_m": _round(float(np.mean(vectors[:, 2])), 2),
        "max_throw_m": _round(float(np.max(np.linalg.norm(vectors[:, :2], axis=1))), 2),
        "max_burden_velocity_m_s": _round(float(np.max(physics.burden_velocity_m_s)), 3),
        "mean_peak_impulse_m_s": _round(float(np.mean(physics.peak_impulse_m_s)), 3),
    }
    validation["warnings"].append(
        "S135B pressure and gas expansion use a reduced-order surrogate, not product-certified JWL constants."
    )
    if not request.block_model:
        validation["warnings"].append(
            "Rock, geology, grade-control and loader inputs are synthetic until replaced by measured files."
        )
    registered_datasets = {dataset.kind: dataset for dataset in request.site_data.datasets}
    metadata_only_datasets = set(registered_datasets) - ({"grade_control_blocks"} if request.block_model else set())
    if metadata_only_datasets:
        validation["warnings"].append(
            "Some measured dataset metadata was registered without validated contents; those providers remained synthetic."
        )
    return {
        "engine": {
            "name": "GeoMotion 3D Engine",
            "version": "0.2.0-event-physics",
            "mode": request.mode,
            "model_kind": "reduced-order timed detonation, burden velocity, dynamic relief and conservative voxel remap",
            "calibration": "measured_block_model_uncalibrated_movement" if request.block_model else "synthetic_unvalidated",
            "notice": "Synthetic Demonstration / Uncalibrated — Planning Only",
            "seed": request.seed,
        },
        "assumptions": a.model_dump(),
        "validation": validation,
        "metrics": metrics,
        "holes": hole_rows,
        "blocks": block_rows,
        "surface": _surface(destination, a.cell_size_m),
        "mixing_matrix": mixing_matrix,
        "uncertainty": {
            "method": "Timing/VOD/rock-property proxy; replace with measured site residual distributions.",
            "mean_m": _round(float(np.mean(physics.uncertainty)), 2),
            "p95_m": _round(float(np.percentile(physics.uncertainty, 95)), 2),
            "out_of_domain": True,
        },
        "events": physics.events,
        "event_history": physics.event_history,
        "transport": {
            "format": "json_lod",
            "full_resolution_blocks": total_blocks,
            "returned_blocks": len(block_rows),
            "stride": lod_factor**3,
            "lod_factor": lod_factor,
            "visual_voxel_size_m": a.cell_size_m * lod_factor,
            "full_resolution_available_for_export": True,
        },
        "provenance": {
            "tie_up": "site_supplied",
            "explosive_density_rws_and_booster": "site_supplied",
            "vod_range": "manufacturer_range_assumption",
            "geology_rock_surfaces_and_grade": "measured_block_model" if request.block_model else "synthetic",
            "movement_monitors": "not_supplied",
            "registered_dataset_kinds": ",".join(sorted(registered_datasets)) or "none",
        },
        "remap": {
            "method": "conservative one-metre column settlement",
            "collision_count": remap.collision_count,
            "occupied_cells": remap.occupied_cells,
            "mass_preserved": True,
            "contained_carats_preserved": True,
        },
    }
