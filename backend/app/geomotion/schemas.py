from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Provenance = Literal["synthetic", "site_supplied", "manufacturer_supplied", "measured"]


class GeoMotionPrimer(BaseModel):
    explosive: str = "Pentolite"
    mass_kg: float = Field(default=0.4, gt=0, le=10)
    position_from_toe_m: float = Field(default=0.5, ge=0, le=100)
    provenance: Provenance = "site_supplied"


class GeoMotionDeck(BaseModel):
    explosive: str = "S135B"
    top_depth_m: float = Field(ge=0)
    bottom_depth_m: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    primer: GeoMotionPrimer = Field(default_factory=GeoMotionPrimer)
    provenance: Provenance = "synthetic"

    @field_validator("bottom_depth_m")
    @classmethod
    def deck_bottom_must_exceed_top(cls, value: float, info) -> float:
        top = info.data.get("top_depth_m", 0.0)
        if value <= top:
            raise ValueError("Deck bottom depth must exceed deck top depth")
        return value


class GeoMotionHole(BaseModel):
    id: str
    x: float
    y: float
    z: float | None = None
    depth: float | None = Field(default=None, gt=0)
    charge: float | None = Field(default=None, ge=0)
    delay_ms: float = Field(ge=0)
    original_delay_ms: float | None = Field(default=None, ge=0)
    diameter_mm: float | None = Field(default=None, gt=25, le=500)
    inclination_deg: float = Field(default=0.0, ge=0, le=90)
    azimuth_deg: float = Field(default=0.0, ge=0, lt=360)
    stemming_m: float | None = Field(default=None, ge=0, le=50)
    decks: list[GeoMotionDeck] = Field(default_factory=list, max_length=12)


class GeoMotionAssumptions(BaseModel):
    burden_m: float = Field(default=6.0, gt=0, le=30)
    spacing_m: float = Field(default=7.0, gt=0, le=40)
    hole_diameter_mm: float = Field(default=250.0, gt=25, le=500)
    stemming_m: float = Field(default=5.02, ge=0, le=30)
    subdrill_m: float = Field(default=1.0, ge=0, le=10)
    rock_density_t_m3: float = Field(default=2.35, gt=0.5, le=5)
    powder_factor_kg_m3: float = Field(default=0.92, gt=0.01, le=10)
    swell_factor: float = Field(default=1.25, ge=1.0, le=2.0)
    cutoff_grade_cpht: float = Field(default=12.0, ge=0, le=1000)
    free_face_azimuth_deg: float = Field(default=180.0, ge=0, lt=360)
    cell_size_m: float = Field(default=1.0, ge=1.0, le=5.0)
    explosive_relative_energy: float = Field(default=1.0, gt=0.2, le=2.5)
    explosive_density_kg_m3: float = Field(default=1250.51, ge=500, le=2000)
    explosive_rws_percent: float = Field(default=115.0, ge=25, le=250)
    vod_m_s: float = Field(default=4500.0, ge=1000, le=9000)
    vod_uncertainty_m_s: float = Field(default=1000.0, ge=0, le=4000)
    electronic_scatter_enabled: bool = True
    electronic_scatter_base_ms: float = Field(default=0.094, ge=0, le=10)
    electronic_scatter_per_delay: float = Field(default=0.000345, ge=0, le=0.01)
    ucs_mpa: float = Field(default=120.0, gt=1, le=500)
    tensile_strength_mpa: float = Field(default=10.0, gt=0.1, le=100)
    youngs_modulus_gpa: float = Field(default=55.0, gt=0.1, le=200)
    poisson_ratio: float = Field(default=0.24, gt=0, lt=0.5)
    damping_ratio: float = Field(default=0.28, ge=0.01, le=0.95)
    fragmentation_index: float = Field(default=0.55, ge=0, le=1)
    joint_dip_deg: float = Field(default=70.0, ge=0, le=90)
    joint_dip_direction_deg: float = Field(default=90.0, ge=0, lt=360)
    joint_spacing_m: float = Field(default=2.5, gt=0.05, le=100)
    joint_persistence: float = Field(default=0.6, ge=0, le=1)
    loader_bucket_t: float = Field(default=100.0, gt=1, le=1000)
    minimum_mining_unit_m: float = Field(default=5.0, ge=1, le=50)
    max_visual_blocks: int = Field(default=35000, ge=1000, le=500000)


class GeoMotionDatasetRef(BaseModel):
    kind: Literal[
        "grade_control_blocks",
        "geological_structures",
        "preblast_surface",
        "postblast_surface",
        "movement_monitors",
        "dig_limits",
        "loader_geometry",
    ]
    provenance: Provenance = "synthetic"
    filename: str | None = None
    records: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeoMotionBlockInput(BaseModel):
    id: str
    x: float
    y: float
    z: float
    size_x_m: float = Field(default=1.0, ge=0.99, le=1.01)
    size_y_m: float = Field(default=1.0, ge=0.99, le=1.01)
    size_z_m: float = Field(default=1.0, ge=0.99, le=1.01)
    density_t_m3: float = Field(gt=0.5, le=6.0)
    grade_cpht: float = Field(default=0.0, ge=0)
    facies: str = "UNKNOWN"
    provenance: Provenance = "measured"


class GeoMotionSiteData(BaseModel):
    datasets: list[GeoMotionDatasetRef] = Field(default_factory=list)
    synthetic_defaults_enabled: bool = True


class GeoMotionRequest(BaseModel):
    project_name: str = Field(default="Diamond mine synthetic demonstration", max_length=160)
    seed: int = Field(default=66532, ge=0, le=2_147_483_647)
    mode: Literal["physics", "hybrid"] = "hybrid"
    holes: list[GeoMotionHole] = Field(min_length=3, max_length=1000)
    assumptions: GeoMotionAssumptions = Field(default_factory=GeoMotionAssumptions)
    site_data: GeoMotionSiteData = Field(default_factory=GeoMotionSiteData)
    block_model: list[GeoMotionBlockInput] = Field(default_factory=list, max_length=500_000)

    @field_validator("holes")
    @classmethod
    def coordinates_must_span_an_area(cls, holes: list[GeoMotionHole]) -> list[GeoMotionHole]:
        if max(h.x for h in holes) - min(h.x for h in holes) < 0.1:
            raise ValueError("Hole X coordinates do not span a blast area")
        if max(h.y for h in holes) - min(h.y for h in holes) < 0.1:
            raise ValueError("Hole Y coordinates do not span a blast area")
        delays = [h.delay_ms for h in holes]
        if len(set(delays)) != len(delays):
            raise ValueError("GeoMotion requires a unique cumulative firing time for every hole")
        return holes


class GeoMotionResponse(BaseModel):
    engine: dict
    assumptions: dict
    validation: dict
    metrics: dict
    holes: list[dict]
    blocks: list[dict]
    surface: list[dict]
    mixing_matrix: list[dict]
    uncertainty: dict
    events: list[dict] = Field(default_factory=list)
    event_history: list[dict] = Field(default_factory=list)
    transport: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    remap: dict = Field(default_factory=dict)
