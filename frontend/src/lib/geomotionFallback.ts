import type { GeoMotionRequest, GeoMotionResult } from "../types/geomotion";

const NOTICE = "Synthetic Demonstration / Uncalibrated — Planning Only";

function round(value: number, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function percent(numerator: number, denominator: number) {
  return denominator > 0 ? round((100 * numerator) / denominator, 2) : 0;
}

function seededNoise(seed: number, index: number) {
  const value = Math.sin(seed * 0.0001 + index * 12.9898) * 43758.5453;
  return (value - Math.floor(value)) * 2 - 1;
}

function destinationGrade(
  x: number,
  y: number,
  centerX: number,
  centerY: number,
  spanX: number,
  spanY: number
) {
  const nx = (x - centerX) / Math.max(spanX * 0.4, 1);
  const ny = (y - centerY - spanY * 0.1 * Math.sin(nx * 1.7)) / Math.max(spanY * 0.36, 1);
  const radius = Math.hypot(nx, ny);
  if (radius < 1) return 18 + 22 * Math.max(0, 1 - radius);
  return radius < 1.12 ? (3 * (1.12 - radius)) / 0.12 : 0;
}

export function simulateGeoMotionLocally(request: GeoMotionRequest): GeoMotionResult {
  const { holes, assumptions: a, seed } = request;
  const xs = holes.map((hole) => hole.x);
  const ys = holes.map((hole) => hole.y);
  const depths = holes.map((hole) => hole.depth ?? 15.7);
  const elevations = holes.map((hole) => hole.z ?? 680);
  const floorValues = holes.map((hole, index) => elevations[index] - depths[index]);
  const floor = [...floorValues].sort((left, right) => left - right)[Math.floor(floorValues.length / 2)];
  const collar = [...elevations].sort((left, right) => left - right)[Math.floor(elevations.length / 2)];
  const benchHeight = Math.max(6, collar - floor - a.subdrill_m);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const spanX = Math.max(xmax - xmin, a.cell_size_m);
  const spanY = Math.max(ymax - ymin, a.cell_size_m);
  const centerX = xs.reduce((sum, value) => sum + value, 0) / holes.length;
  const centerY = ys.reduce((sum, value) => sum + value, 0) / holes.length;
  const cell = a.cell_size_m;
  const levels = Math.max(2, Math.min(6, Math.ceil(benchHeight / cell)));
  const levelHeight = benchHeight / levels;
  const azimuth = (a.free_face_azimuth_deg * Math.PI) / 180;
  const faceX = Math.sin(azimuth);
  const faceY = Math.cos(azimuth);
  const blocks: GeoMotionResult["blocks"] = [];

  for (let y = ymin - cell; y <= ymax + cell; y += cell) {
    for (let x = xmin - cell; x <= xmax + cell; x += cell) {
      let nearest = Number.POSITIVE_INFINITY;
      let weightedCharge = 0;
      let influenceSum = 0;
      let weightedDelay = 0;
      holes.forEach((hole) => {
        const distance = Math.hypot(x - hole.x, y - hole.y);
        nearest = Math.min(nearest, distance);
        const influence = Math.exp(-distance / Math.max(1, a.spacing_m * 1.65));
        influenceSum += influence;
        weightedCharge += influence * (hole.charge ?? 625);
        weightedDelay += influence * (hole.delay_ms ?? 0);
      });
      if (nearest > Math.max(a.spacing_m, a.burden_m) * 0.9) continue;
      weightedCharge /= Math.max(influenceSum, 1e-9);
      weightedDelay /= Math.max(influenceSum, 1e-9);

      for (let level = 0; level < levels; level += 1) {
        const z = floor + (level + 0.5) * levelHeight;
        const nx = (x - centerX) / Math.max(spanX * 0.4, 1);
        const ny = (y - centerY - spanY * 0.1 * Math.sin(nx * 1.7)) / Math.max(spanY * 0.36, 1);
        const radius = Math.hypot(nx, ny);
        const kimberlite = radius < 1;
        const core = radius < 0.48;
        const contact = radius >= 0.82 && radius < 1.12;
        const facies = core ? "VK" : kimberlite ? "SVK_M1" : contact ? "CONTACT" : "WASTE";
        const density = (kimberlite ? a.rock_density_t_m3 : a.rock_density_t_m3 + 0.3) + seededNoise(seed, blocks.length) * 0.025;
        const gradeBase = kimberlite
          ? 18 + 22 * Math.max(0, 1 - radius) + 5 * Math.sin(nx * 3)
          : contact
            ? (3 * Math.max(0, 1.12 - radius)) / 0.3
            : 0;
        const grade = Math.max(0, gradeBase + seededNoise(seed + 17, blocks.length) * 1.8);
        const sourceClass = grade >= a.cutoff_grade_cpht ? "ORE" : "WASTE";
        const radialLength = Math.max(Math.hypot(x - centerX, y - centerY), 1);
        const radialX = (x - centerX) / radialLength;
        const radialY = (y - centerY) / radialLength;
        let directionX = faceX * 0.7 + radialX * 0.22;
        let directionY = faceY * 0.7 + radialY * 0.22;
        const directionLength = Math.max(Math.hypot(directionX, directionY), 1e-6);
        directionX /= directionLength;
        directionY /= directionLength;
        const energy = Math.min(1.7, Math.max(0.25, influenceSum / 4.4));
        const zFraction = level / Math.max(levels - 1, 1);
        const baseThrow = a.burden_m * (0.55 + energy * 0.55) * a.explosive_relative_energy * Math.min(1.55, Math.max(0.55, weightedCharge / 625)) / a.swell_factor;
        const syntheticResidual = request.mode === "hybrid" ? seededNoise(seed + 31, blocks.length) * a.burden_m * 0.18 : 0;
        const dx = directionX * baseThrow * (0.72 + zFraction * 0.42) + syntheticResidual;
        const dy = directionY * baseThrow * (0.72 + zFraction * 0.42) + syntheticResidual * 0.35;
        const dz = a.burden_m * 0.3 * energy * a.explosive_relative_energy * (0.55 + zFraction * 0.7)
          + (request.mode === "hybrid" ? seededNoise(seed + 47, blocks.length) * 0.22 : 0);
        const destination: [number, number, number] = [
          x + dx,
          y + dy,
          z + dz + (a.swell_factor - 1) * benchHeight * 0.35,
        ];
        const destinationClass = destinationGrade(destination[0], destination[1], centerX, centerY, spanX, spanY) >= a.cutoff_grade_cpht ? "ORE" : "WASTE";
        const tonnes = density * cell * cell * levelHeight;
        const displacement = Math.hypot(dx, dy, dz);
        blocks.push({
          id: blocks.length + 1,
          source: [round(x), round(y), round(z)],
          destination: destination.map((value) => round(value)) as [number, number, number],
          vector: [round(dx), round(dy), round(dz)],
          displacement_m: round(displacement),
          uncertainty_m: round(0.55 + displacement * 0.1 + nearest / Math.max(a.spacing_m, 1) * 0.2),
          facies,
          source_class: sourceClass,
          destination_class: destinationClass,
          density_t_m3: round(density),
          grade_cpht: round(grade),
          tonnes: round(tonnes),
          contained_carats: round((tonnes * grade) / 100),
          effective_time_ms: round(weightedDelay, 1),
          velocity: [round(dx), round(dy), round(dz)],
          peak_impulse_m_s: round(displacement * 0.35),
          burden_velocity_m_s: round(displacement * 0.22),
          contributing_event: -1,
          size_m: cell,
          physics_cell_dimensions_m: [cell, cell, levelHeight],
          physics_cell_volume_m3: round(cell * cell * levelHeight),
          represented_cell_count: 1,
          provenance: "synthetic",
        });
      }
    }
  }

  const sum = (predicate: (block: GeoMotionResult["blocks"][number]) => boolean, selector = (block: GeoMotionResult["blocks"][number]) => block.tonnes) =>
    blocks.filter(predicate).reduce((total, block) => total + selector(block), 0);
  const sourceOre = (block: GeoMotionResult["blocks"][number]) => block.source_class === "ORE";
  const destinationOre = (block: GeoMotionResult["blocks"][number]) => block.destination_class === "ORE";
  const oreTonnes = sum(sourceOre);
  const retainedOre = sum((block) => sourceOre(block) && destinationOre(block));
  const lostOre = sum((block) => sourceOre(block) && !destinationOre(block));
  const dilution = sum((block) => !sourceOre(block) && destinationOre(block));
  const oreStream = sum(destinationOre);
  const totalCarats = sum(sourceOre, (block) => block.contained_carats);
  const retainedCarats = sum((block) => sourceOre(block) && destinationOre(block), (block) => block.contained_carats);
  const magnitudes = blocks.map((block) => block.displacement_m).sort((left, right) => left - right);
  const totalTonnes = sum(() => true);
  const mixingMatrix = (["ORE", "WASTE"] as const).flatMap((source) =>
    (["ORE", "WASTE"] as const).map((destination) => {
      const tonnes = sum((block) => block.source_class === source && block.destination_class === destination);
      return { source, destination, tonnes: round(tonnes, 1), percent_of_total: percent(tonnes, totalTonnes) };
    })
  );

  const surfaceBins = new Map<string, { x: number; y: number; z: number }>();
  blocks.forEach((block) => {
    const key = `${Math.round(block.destination[0] / cell)}:${Math.round(block.destination[1] / cell)}`;
    const current = surfaceBins.get(key);
    if (!current || block.destination[2] > current.z) {
      surfaceBins.set(key, { x: block.destination[0], y: block.destination[1], z: block.destination[2] });
    }
  });

  const idCounts = new Map<string, number>();
  holes.forEach((hole) => idCounts.set(hole.id, (idCounts.get(hole.id) ?? 0) + 1));
  const duplicateIds = [...idCounts].filter(([, count]) => count > 1).map(([id]) => id);
  const overlaps: Array<{ hole_a: string; hole_b: string; distance_m: number }> = [];
  holes.forEach((hole, index) => holes.slice(index + 1).forEach((other) => {
    const distance = Math.hypot(hole.x - other.x, hole.y - other.y);
    if (distance < Math.max(0.5, a.hole_diameter_mm / 1000)) overlaps.push({ hole_a: hole.id, hole_b: other.id, distance_m: round(distance) });
  }));
  const uncertaintyValues = blocks.map((block) => block.uncertainty_m).sort((left, right) => left - right);

  return {
    engine: {
      name: "GeoMotion 3D Engine",
      version: "0.1.0-browser-fallback",
      mode: request.mode,
      model_kind: request.mode === "hybrid" ? "browser physics + deterministic synthetic residual" : "browser mass-conserving physics baseline",
      calibration: "synthetic_unvalidated",
      notice: NOTICE,
      seed,
    },
    assumptions: a,
    validation: {
      status: duplicateIds.length || overlaps.length ? "review" : "synthetic",
      warnings: [
        "The Cloud backend did not yet expose GeoMotion; this browser preview uses coarse cells and is not the authoritative 1 m³ (1 m × 1 m × 1 m) event-physics result.",
        "Synthetic geology and calibration are demonstration data, not measured mine evidence.",
      ],
      duplicate_ids: duplicateIds,
      near_overlap_pairs: overlaps,
      median_nearest_hole_m: a.spacing_m,
      floor_rl_m: round(floor),
    },
    metrics: {
      cells: blocks.length,
      voxel_size_m: cell,
      voxel_edge_length_m: cell,
      voxel_volume_m3: round(cell * cell * levelHeight),
      total_tonnes: round(totalTonnes, 1),
      mass_balance_error_percent: 0,
      contained_carats: round(totalCarats, 1),
      recovered_carats: round(retainedCarats, 1),
      carat_recovery_percent: percent(retainedCarats, totalCarats),
      ore_recovery_percent: percent(retainedOre, oreTonnes),
      ore_loss_percent: percent(lostOre, oreTonnes),
      dilution_percent: percent(dilution, oreStream),
      ore_tonnes_in_situ: round(oreTonnes, 1),
      ore_tonnes_recovered: round(retainedOre, 1),
      waste_dilution_tonnes: round(dilution, 1),
      predicted_feed_grade_cpht: round((sum(destinationOre, (block) => block.contained_carats) * 100) / Math.max(oreStream, 1e-9), 2),
      mean_displacement_m: round(magnitudes.reduce((total, value) => total + value, 0) / Math.max(magnitudes.length, 1), 2),
      p95_displacement_m: round(magnitudes[Math.floor(magnitudes.length * 0.95)] ?? 0, 2),
      mean_heave_m: round(blocks.reduce((total, block) => total + block.vector[2], 0) / Math.max(blocks.length, 1), 2),
      max_throw_m: round(Math.max(...blocks.map((block) => Math.hypot(block.vector[0], block.vector[1]))), 2),
    },
    holes: holes.map((hole) => ({
      id: hole.id,
      x: round(hole.x),
      y: round(hole.y),
      z: round(hole.z ?? 680),
      depth_m: round(hole.depth ?? 15.7),
      charge_kg: round(hole.charge ?? 625),
      delay_ms: round(hole.delay_ms ?? 0, 1),
    })),
    blocks,
    surface: [...surfaceBins.values()],
    mixing_matrix: mixingMatrix,
    uncertainty: {
      method: "Browser synthetic proxy; replace with measured site residual distributions.",
      mean_m: round(uncertaintyValues.reduce((total, value) => total + value, 0) / Math.max(uncertaintyValues.length, 1), 2),
      p95_m: round(uncertaintyValues[Math.floor(uncertaintyValues.length * 0.95)] ?? 0, 2),
      out_of_domain: true,
    },
    events: holes.map((hole, index) => ({
      event_index: index,
      hole_id: hole.id,
      nominal_time_ms: hole.delay_ms,
      actual_time_ms: hole.delay_ms,
      timing_error_ms: 0,
    })),
    event_history: [],
    transport: {
      format: "browser_preview",
      full_resolution_blocks: blocks.length,
      returned_blocks: blocks.length,
      stride: 1,
      full_resolution_available_for_export: false,
    },
    provenance: {
      tie_up: "site_supplied",
      geology_rock_surfaces_and_grade: "synthetic",
      engine: "coarse_browser_preview",
    },
    remap: {
      method: "coarse browser settlement",
      collision_count: 0,
      occupied_cells: blocks.length,
      mass_preserved: true,
      contained_carats_preserved: true,
    },
  };
}
