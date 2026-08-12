from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from scipy.spatial import cKDTree

from ..explosives import S135BProfile
from ..schemas import GeoMotionRequest
from .pressure import event_energy
from .scheduler import BlastEvent, build_event_queue


@dataclass
class VoxelField:
    positions: np.ndarray
    density: np.ndarray
    grade: np.ndarray
    tonnes: np.ndarray
    facies: np.ndarray
    source_class: np.ndarray
    center: np.ndarray
    floor_rl: float
    collar_rl: float


@dataclass
class EventPhysicsResult:
    source: VoxelField
    destination_positions: np.ndarray
    vectors: np.ndarray
    velocities: np.ndarray
    uncertainty: np.ndarray
    effective_time_ms: np.ndarray
    peak_impulse_m_s: np.ndarray
    burden_velocity_m_s: np.ndarray
    contributing_event: np.ndarray
    events: list[dict]
    event_history: list[dict]


def build_synthetic_voxels(request: GeoMotionRequest) -> VoxelField:
    holes = request.holes
    a = request.assumptions
    if request.block_model:
        blocks = request.block_model
        positions = np.array([[block.x, block.y, block.z] for block in blocks], dtype=float)
        density = np.array([block.density_t_m3 for block in blocks], dtype=float)
        grade = np.array([block.grade_cpht for block in blocks], dtype=float)
        tonnes = density.copy()  # Validated 1 m × 1 m × 1 m cells.
        facies = np.array([block.facies for block in blocks], dtype=object)
        source_class = np.where(grade >= a.cutoff_grade_cpht, "ORE", "WASTE")
        return VoxelField(
            positions=positions,
            density=density,
            grade=grade,
            tonnes=tonnes,
            facies=facies,
            source_class=source_class,
            center=np.mean(positions[:, :2], axis=0),
            floor_rl=float(np.min(positions[:, 2]) - 0.5),
            collar_rl=float(np.max(positions[:, 2]) + 0.5),
        )

    rng = np.random.default_rng(request.seed)
    xs = np.asarray([hole.x for hole in holes], dtype=float)
    ys = np.asarray([hole.y for hole in holes], dtype=float)
    elevations = np.asarray([hole.z if hole.z is not None else 680.0 for hole in holes])
    depths = np.asarray([hole.depth if hole.depth is not None else 15.7 for hole in holes])
    floor_values = elevations - depths
    floor_rl = float(np.median(floor_values))
    collar_rl = float(np.median(elevations))
    bench_height = max(6.0, collar_rl - floor_rl - a.subdrill_m)
    cell = a.cell_size_m
    pad = max(a.spacing_m, a.burden_m)
    gx = np.arange(np.min(xs) - pad, np.max(xs) + pad + cell * 0.5, cell)
    gy = np.arange(np.min(ys) - pad, np.max(ys) + pad + cell * 0.5, cell)
    levels = max(1, int(np.ceil(bench_height / cell)))
    gz = floor_rl + (np.arange(levels) + 0.5) * cell
    gz = gz[gz <= floor_rl + bench_height + 1e-6]
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    collar_tree = cKDTree(np.column_stack([xs, ys]))
    nearest, _ = collar_tree.query(positions[:, :2], k=1)
    positions = positions[nearest <= max(a.spacing_m, a.burden_m) * 0.90]

    center = np.array([float(np.mean(xs)), float(np.mean(ys))])
    span_x = max(float(np.ptp(xs)), cell)
    span_y = max(float(np.ptp(ys)), cell)
    nx = (positions[:, 0] - center[0]) / (span_x * 0.40)
    ny = (positions[:, 1] - center[1] - 0.10 * span_y * np.sin(nx * 1.7)) / (span_y * 0.36)
    radius = np.hypot(nx, ny)
    kimberlite = radius < 1.0
    core = radius < 0.48
    contact = (radius >= 0.82) & (radius < 1.12)
    facies = np.where(
        core,
        "VK",
        np.where(kimberlite, "SVK_M1", np.where(contact, "CONTACT", "WASTE")),
    )
    density = np.where(kimberlite, a.rock_density_t_m3, a.rock_density_t_m3 + 0.30)
    density += rng.normal(0, 0.025, len(positions))
    grade = np.where(
        kimberlite,
        18.0 + 22.0 * np.maximum(0.0, 1.0 - radius) + 5.0 * np.sin(nx * 3.0),
        np.where(contact, 3.0 * np.maximum(0.0, 1.12 - radius) / 0.30, 0.0),
    )
    grade = np.maximum(0.0, grade + rng.normal(0, 1.8, len(positions)))
    tonnes = density * cell**3
    source_class = np.where(grade >= a.cutoff_grade_cpht, "ORE", "WASTE")
    return VoxelField(
        positions=positions,
        density=density,
        grade=grade,
        tonnes=tonnes,
        facies=facies,
        source_class=source_class,
        center=center,
        floor_rl=floor_rl,
        collar_rl=collar_rl,
    )


def _event_dict(event: BlastEvent, energy, burden_velocity: float, released_voxels: int) -> dict:
    return {
        "event_index": event.index,
        "hole_id": event.hole_id,
        "nominal_time_ms": round(event.nominal_time_ms, 3),
        "actual_time_ms": round(event.actual_time_ms, 3),
        "timing_error_ms": round(event.timing_error_ms, 4),
        "charge_kg": round(event.charge_kg, 3),
        "vod_m_s": round(energy.vod_m_s, 1),
        "detonation_pressure_gpa_proxy": round(energy.detonation_pressure_pa / 1e9, 3),
        "chemical_energy_mj": round(energy.chemical_energy_j / 1e6, 2),
        "movement_energy_mj": round(energy.movement_energy_j / 1e6, 2),
        "gas_time_constant_ms": round(energy.gas_time_constant_ms, 3),
        "stemming_effectiveness": round(energy.stemming_effectiveness, 3),
        "burden_velocity_m_s": round(burden_velocity, 3),
        "released_voxels": released_voxels,
    }


def run_event_physics(request: GeoMotionRequest, realization: int = 0) -> EventPhysicsResult:
    a = request.assumptions
    source = build_synthetic_voxels(request)
    events = build_event_queue(request, realization)
    profile = S135BProfile(
        density_kg_m3=a.explosive_density_kg_m3,
        rws_percent=a.explosive_rws_percent,
        nominal_vod_m_s=a.vod_m_s,
        vod_range_m_s=(a.vod_m_s - a.vod_uncertainty_m_s, a.vod_m_s + a.vod_uncertainty_m_s),
    )
    rng = np.random.default_rng(request.seed + 7919 * realization)
    n = len(source.positions)
    positions = source.positions.copy()
    velocities = np.zeros((n, 3), dtype=np.float64)
    release = np.zeros(n, dtype=np.float32)
    effective_time = np.zeros(n, dtype=np.float64)
    timing_weight = np.zeros(n, dtype=np.float64)
    peak_impulse = np.zeros(n, dtype=np.float32)
    burden_velocity = np.zeros(n, dtype=np.float32)
    contributing_event = np.full(n, -1, dtype=np.int32)
    tree = cKDTree(source.positions)

    azimuth = a.free_face_azimuth_deg * pi / 180.0
    face = np.array([sin(azimuth), cos(azimuth), 0.0])
    face_projection = (source.positions[:, :2] - source.center) @ face[:2]
    initial_relief = (face_projection - np.min(face_projection)) / max(float(np.ptp(face_projection)), 1.0)
    joint_azimuth = a.joint_dip_direction_deg * pi / 180.0
    joint_direction = np.array([sin(joint_azimuth), cos(joint_azimuth), 0.0])
    joint_alignment = abs(float(np.dot(face, joint_direction)))
    rock_impedance = max(
        1e6,
        source.density.mean() * 1000.0 * np.sqrt(a.youngs_modulus_gpa * 1e9 / (source.density.mean() * 1000.0)),
    )
    previous_time = 0.0
    event_rows: list[dict] = []
    history: list[dict] = []
    snapshot_every = max(1, len(events) // 24)

    for event in events:
        dt = max(0.0, (event.actual_time_ms - previous_time) / 1000.0)
        if dt > 0:
            positions += velocities * dt
            velocities *= np.exp(-a.damping_ratio * 6.0 * dt)
        previous_time = event.actual_time_ms

        vod = float(np.clip(rng.normal(a.vod_m_s, a.vod_uncertainty_m_s / 2.0), 2500, 7000))
        energy = event_energy(event, profile, vod, a.burden_m, a.tensile_strength_mpa)
        deck_center_z = 0.5 * (event.deck_top_z + event.deck_bottom_z)
        event_center = np.array([event.x, event.y, deck_center_z])
        radius = max(a.spacing_m * 3.0, a.burden_m * 3.5)
        indices = np.asarray(tree.query_ball_point(event_center, radius), dtype=np.int64)
        if not len(indices):
            event_rows.append(_event_dict(event, energy, 0.0, 0))
            continue

        delta = source.positions[indices] - event_center
        distance = np.linalg.norm(delta, axis=1)
        attenuation = np.exp(-distance / max(a.spacing_m * 1.45, 0.1))
        local_relief = np.clip(0.18 + 0.48 * initial_relief[indices] + 0.62 * release[indices], 0.1, 1.25)
        confinement = np.clip(1.18 - local_relief, 0.12, 1.0)
        horizontal = delta.copy()
        horizontal[:, 2] = 0.0
        horizontal_norm = np.linalg.norm(horizontal, axis=1, keepdims=True)
        radial = horizontal / np.maximum(horizontal_norm, 1.0)
        direction = face[None, :] * (0.74 + 0.16 * local_relief[:, None]) + radial * 0.24
        direction[:, 2] = 0.18 + 0.44 * (
            (source.positions[indices, 2] - source.floor_rl)
            / max(source.collar_rl - source.floor_rl, 1.0)
        )
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)

        structure_factor = 1.0 + a.joint_persistence * 0.20 * joint_alignment
        strength_factor = np.clip(120.0 / a.ucs_mpa, 0.35, 2.0)
        weights = attenuation * local_relief * structure_factor * strength_factor
        weights /= max(float(np.sum(weights)), 1e-12)
        local_mass_kg = np.maximum(source.tonnes[indices] * 1000.0, 1.0)
        speed = np.sqrt(2.0 * energy.movement_energy_j * weights / local_mass_kg)
        impedance_factor = float(np.clip(14e6 / rock_impedance, 0.22, 1.35))
        speed *= impedance_factor
        impulse = direction * speed[:, None]
        velocities[indices] += impulse

        improved = speed > peak_impulse[indices]
        improved_indices = indices[improved]
        contributing_event[improved_indices] = event.index
        peak_impulse[indices] = np.maximum(peak_impulse[indices], speed).astype(np.float32)
        burden_velocity[indices] = np.maximum(
            burden_velocity[indices], (speed * confinement).astype(np.float32)
        )
        effective_time[indices] += attenuation * event.actual_time_ms
        timing_weight[indices] += attenuation
        release_gain = attenuation * (0.20 + 0.45 * a.fragmentation_index) * (1.0 - 0.35 * confinement)
        release[indices] = np.clip(release[indices] + release_gain, 0.0, 1.0)

        representative_velocity = float(np.percentile(speed, 90)) if len(speed) else 0.0
        event_rows.append(_event_dict(event, energy, representative_velocity, int(np.sum(release_gain > 0.05))))
        if event.index % snapshot_every == 0 or event.index == len(events) - 1:
            displacement = np.linalg.norm(positions - source.positions, axis=1)
            history.append(
                {
                    "event_index": event.index,
                    "time_ms": round(event.actual_time_ms, 3),
                    "fired_events": event.index + 1,
                    "mean_displacement_m": round(float(np.mean(displacement)), 3),
                    "p95_velocity_m_s": round(float(np.percentile(np.linalg.norm(velocities, axis=1), 95)), 3),
                    "released_fraction": round(float(np.mean(release)), 4),
                }
            )

    settle_time = float(np.clip(0.55 + 1.2 * (1.0 - a.damping_ratio), 0.6, 1.8))
    positions += velocities * settle_time
    positions[:, 2] -= 0.5 * 9.81 * settle_time**2 * (0.22 + 0.35 * a.damping_ratio)
    positions[:, 2] += (a.swell_factor - 1.0) * max(source.collar_rl - source.floor_rl, 1.0) * 0.30
    positions[:, 2] = np.maximum(positions[:, 2], source.floor_rl + a.cell_size_m * 0.5)

    vectors = positions - source.positions
    maximum = max(a.spacing_m, a.burden_m) * 2.6
    magnitudes = np.linalg.norm(vectors, axis=1)
    vectors *= np.minimum(1.0, maximum / np.maximum(magnitudes, 1e-9))[:, None]
    positions = source.positions + vectors
    uncertainty = (
        0.25
        + 0.08 * np.linalg.norm(vectors, axis=1)
        + 0.45 * (1.0 - release)
        + 0.002 * a.vod_uncertainty_m_s
    )
    effective_time = np.divide(
        effective_time,
        np.maximum(timing_weight, 1e-9),
        out=np.zeros_like(effective_time),
        where=timing_weight > 0,
    )
    return EventPhysicsResult(
        source=source,
        destination_positions=positions,
        vectors=vectors,
        velocities=velocities,
        uncertainty=uncertainty,
        effective_time_ms=effective_time,
        peak_impulse_m_s=peak_impulse,
        burden_velocity_m_s=burden_velocity,
        contributing_event=contributing_event,
        events=event_rows,
        event_history=history,
    )
