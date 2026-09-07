import type { BlastHole } from "./blast";

export type GeoMotionMode = "physics" | "hybrid";
export type GeoMotionView = "source" | "destination" | "movement";
export type GeoMotionColor = "classification" | "facies" | "grade" | "displacement" | "uncertainty" | "burdenVelocity" | "impulse";

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
  explosive_density_kg_m3: number;
  explosive_rws_percent: number;
  vod_m_s: number;
  vod_uncertainty_m_s: number;
  electronic_scatter_enabled: boolean;
  electronic_scatter_base_ms: number;
  electronic_scatter_per_delay: number;
  ucs_mpa: number;
  tensile_strength_mpa: number;
  youngs_modulus_gpa: number;
  poisson_ratio: number;
  damping_ratio: number;
  fragmentation_index: number;
  joint_dip_deg: number;
  joint_dip_direction_deg: number;
  joint_spacing_m: number;
  joint_persistence: number;
  loader_bucket_t: number;
  minimum_mining_unit_m: number;
  max_visual_blocks: number;
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
  velocity: [number, number, number];
  peak_impulse_m_s: number;
  burden_velocity_m_s: number;
  contributing_event: number;
  size_m: number;
  physics_cell_dimensions_m?: [number, number, number];
  physics_cell_volume_m3?: number;
  represented_cell_count?: number;
  provenance: string;
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
    original_delay_ms?: number;
    diameter_mm?: number;
    inclination_deg?: number;
    azimuth_deg?: number;
    decks?: number;
  }>;
  blocks: GeoMotionBlock[];
  surface: Array<{ x: number; y: number; z: number }>;
  mixing_matrix: Array<{ source: string; destination: string; tonnes: number; percent_of_total: number }>;
  uncertainty: { method: string; mean_m: number; p95_m: number; out_of_domain: boolean };
  events: Array<Record<string, number | string>>;
  event_history: Array<Record<string, number>>;
  transport: {
    format: string;
    full_resolution_blocks: number;
    returned_blocks: number;
    stride: number;
    full_resolution_available_for_export: boolean;
  };
  provenance: Record<string, string>;
  remap: Record<string, number | string | boolean>;
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
    delay_ms: number;
    original_delay_ms?: number;
    diameter_mm?: number;
    inclination_deg?: number;
    azimuth_deg?: number;
    stemming_m?: number;
    decks?: Array<{
      explosive: string;
      top_depth_m: number;
      bottom_depth_m: number;
      mass_kg: number;
      primer: {
        explosive: string;
        mass_kg: number;
        position_from_toe_m: number;
        provenance: string;
      };
      provenance: string;
    }>;
  }>;
  assumptions: GeoMotionAssumptions;
  site_data: {
    datasets: Array<{ kind: string; provenance: string; filename?: string; records: number; metadata?: Record<string, unknown> }>;
    synthetic_defaults_enabled: boolean;
  };
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
  cell_size_m: 1,
  explosive_relative_energy: 1,
  explosive_density_kg_m3: 1250.51,
  explosive_rws_percent: 115,
  vod_m_s: 4500,
  vod_uncertainty_m_s: 1000,
  electronic_scatter_enabled: true,
  electronic_scatter_base_ms: 0.094,
  electronic_scatter_per_delay: 0.000345,
  ucs_mpa: 120,
  tensile_strength_mpa: 10,
  youngs_modulus_gpa: 55,
  poisson_ratio: 0.24,
  damping_ratio: 0.28,
  fragmentation_index: 0.55,
  joint_dip_deg: 70,
  joint_dip_direction_deg: 90,
  joint_spacing_m: 2.5,
  joint_persistence: 0.6,
  loader_bucket_t: 100,
  minimum_mining_unit_m: 5,
  max_visual_blocks: 35000,
};

export function toGeoMotionHoles(holes: BlastHole[]): GeoMotionRequest["holes"] {
  return holes.map((hole) => ({
    id: hole.id,
    x: hole.x,
    y: hole.y,
    z: hole.z,
    depth: hole.depth,
    charge: hole.charge,
    delay_ms: hole.delayMs as number,
    original_delay_ms: hole.originalDelayMs ?? hole.delayMs,
  }));
}
