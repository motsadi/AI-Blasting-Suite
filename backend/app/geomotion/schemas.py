from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GeoMotionHole(BaseModel):
    id: str
    x: float
    y: float
    z: float | None = None
    depth: float | None = Field(default=None, gt=0)
    charge: float | None = Field(default=None, ge=0)
    delay_ms: float | None = Field(default=None, ge=0)


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
    cell_size_m: float = Field(default=5.0, ge=2.0, le=12.0)
    explosive_relative_energy: float = Field(default=1.0, gt=0.2, le=2.5)


class GeoMotionRequest(BaseModel):
    project_name: str = Field(default="Diamond mine synthetic demonstration", max_length=160)
    seed: int = Field(default=66532, ge=0, le=2_147_483_647)
    mode: Literal["physics", "hybrid"] = "hybrid"
    holes: list[GeoMotionHole] = Field(min_length=3, max_length=1000)
    assumptions: GeoMotionAssumptions = Field(default_factory=GeoMotionAssumptions)

    @field_validator("holes")
    @classmethod
    def coordinates_must_span_an_area(cls, holes: list[GeoMotionHole]) -> list[GeoMotionHole]:
        if max(h.x for h in holes) - min(h.x for h in holes) < 0.1:
            raise ValueError("Hole X coordinates do not span a blast area")
        if max(h.y for h in holes) - min(h.y for h in holes) < 0.1:
            raise ValueError("Hole Y coordinates do not span a blast area")
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
