import type { BlastHole } from "./blast";

export type GeoMotionMode = "physics" | "hybrid";
export type GeoMotionView = "source" | "destination" | "movement";
export type GeoMotionColor = "classification" | "facies" | "grade" | "displacement" | "uncertainty";

export interface GeoMotionAssumptions {
  burden_m: number;
  spacing_m: number;
  hole_diameter_mm: number;
  stemming_m: number;
  subdrill_m: number;
  rock_density_t_m3: number;
  powder_factor_kg_m3: number;
  swell_factor: number;
  cutoff_grade_cpht: number;
  free_face_azimuth_deg: number;
  cell_size_m: number;
  explosive_relative_energy: number;
}

export interface GeoMotionBlock {
  id: number;
  source: [number, number, number];
  destination: [number, number, number];
  vector: [number, number, number];
  displacement_m: number;
  uncertainty_m: number;
  facies: "VK" | "SVK_M1" | "CONTACT" | "WASTE";
  source_class: "ORE" | "WASTE";
  destination_class: "ORE" | "WASTE";
  density_t_m3: number;
  grade_cpht: number;
  tonnes: number;
  contained_carats: number;
  effective_time_ms: number;
}

export interface GeoMotionResult {
  engine: {
    name: string;
    version: string;
    mode: GeoMotionMode;
    model_kind: string;
    calibration: string;
    notice: string;
    seed: number;
  };
  assumptions: GeoMotionAssumptions;
  validation: {
    status: string;
    warnings: string[];
    duplicate_ids: string[];
    near_overlap_pairs: Array<{ hole_a: string; hole_b: string; distance_m: number }>;
    median_nearest_hole_m: number;
    floor_rl_m: number | null;
  };
  metrics: Record<string, number>;
  holes: Array<{
    id: string;
    x: number;
    y: number;
    z: number;
    depth_m: number;
    charge_kg: number;
    delay_ms: number;
  }>;
  blocks: GeoMotionBlock[];
  surface: Array<{ x: number; y: number; z: number }>;
  mixing_matrix: Array<{ source: string; destination: string; tonnes: number; percent_of_total: number }>;
  uncertainty: { method: string; mean_m: number; p95_m: number; out_of_domain: boolean };
}

export interface GeoMotionRequest {
  project_name: string;
  seed: number;
  mode: GeoMotionMode;
  holes: Array<{
    id: string;
    x: number;
    y: number;
    z?: number;
    depth?: number;
    charge?: number;
    delay_ms?: number;
  }>;
  assumptions: GeoMotionAssumptions;
}

export const DIAMOND_DEMO_ASSUMPTIONS: GeoMotionAssumptions = {
  burden_m: 6,
  spacing_m: 7,
  hole_diameter_mm: 250,
  stemming_m: 5.02,
  subdrill_m: 1,
  rock_density_t_m3: 2.35,
  powder_factor_kg_m3: 0.92,
  swell_factor: 1.25,
  cutoff_grade_cpht: 12,
  free_face_azimuth_deg: 180,
  cell_size_m: 5,
  explosive_relative_energy: 1,
};

export function toGeoMotionHoles(holes: BlastHole[]): GeoMotionRequest["holes"] {
  return holes.map((hole) => ({
    id: hole.id,
    x: hole.x,
    y: hole.y,
    z: hole.z,
    depth: hole.depth,
    charge: hole.charge,
    delay_ms: hole.delayMs,
  }));
}
