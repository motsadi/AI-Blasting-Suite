from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class EmpiricalParamsIn(BaseModel):
    K_ppv: float = 1000.0
    beta: float = 1.60
    K_air: float = 170.0
    B_air: float = 20.0
    A_kuz: float = 22.0
    RWS: float = 115.0


class PredictRequest(BaseModel):
    """
    Mirrors the current Streamlit Predict tab:
    - inputs: dict keyed by utils_blaster.INPUT_LABELS
    - empirical parameters
    - optional HPD_override to compute Q/delay
    """

    inputs: dict[str, float] = Field(default_factory=dict)
    hpd_override: float = 1.0
    empirical: EmpiricalParamsIn = Field(default_factory=EmpiricalParamsIn)
    want_ml: bool = True


class PredictResponse(BaseModel):
    empirical: dict[str, float]
    ml: Optional[Dict[str, float]] = None
    assets_loaded: dict[str, bool]


class AssetsStatus(BaseModel):
    scaler: bool
    model_fragmentation: bool
    model_ground_vibration: bool
    model_airblast: bool

