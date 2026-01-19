from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib


@dataclass
class LoadedAssets:
    scaler: Optional[Any]
    mdl_frag: Optional[Any]
    mdl_ppv: Optional[Any]
    mdl_air: Optional[Any]


def _safe_load_joblib(p: Path):
    try:
        return joblib.load(p)
    except Exception:
        return None


def load_local_assets(core_bundle_path: Path) -> LoadedAssets:
    """
    Load joblib assets from the core bundle folder (local dev) or a copied folder (Cloud Run image).
    """
    scaler = _safe_load_joblib(core_bundle_path / "scaler1.joblib")
    mdl_frag = _safe_load_joblib(core_bundle_path / "random_forest_model_Fragmentation.joblib")
    mdl_ppv = _safe_load_joblib(core_bundle_path / "random_forest_model_Ground Vibration.joblib")
    mdl_air = _safe_load_joblib(core_bundle_path / "random_forest_model_Airblast.joblib")
    return LoadedAssets(scaler=scaler, mdl_frag=mdl_frag, mdl_ppv=mdl_ppv, mdl_air=mdl_air)


def assets_status(a: LoadedAssets) -> dict[str, bool]:
    return {
        "scaler": a.scaler is not None,
        "model_fragmentation": a.mdl_frag is not None,
        "model_ground_vibration": a.mdl_ppv is not None,
        "model_airblast": a.mdl_air is not None,
    }

