from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from google.cloud import storage


REQUIRED_ASSET_FILES = [
    "scaler1.joblib",
    "random_forest_model_Fragmentation.joblib",
    "random_forest_model_Ground Vibration.joblib",
    "random_forest_model_Airblast.joblib",
]

REQUIRED_DATASET_FILES = [
    "combinedv2Orapa.csv",
    "combinedv2Jwaneng.xlsx",
    "Backbreak.csv",
    "flyrock_synth.csv",
    "slope data.csv",
    "Hole_data_v1.csv",
    "Hole_data_v2.csv",
]


@dataclass
class GcsSyncResult:
    downloaded: List[str]
    skipped: List[str]
    missing: List[str]


def _client() -> storage.Client:
    return storage.Client()


def sync_assets_from_gcs(
    *,
    bucket: str,
    prefix: str,
    dest_dir: Path,
    required_files: Optional[Iterable[str]] = None,
) -> GcsSyncResult:
    """
    Download required model assets from GCS into dest_dir.

    - Uses Application Default Credentials (recommended on Cloud Run).
    - `prefix` should be something like "assets/".
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    required = list(required_files or REQUIRED_ASSET_FILES)

    cli = _client()
    b = cli.bucket(bucket)

    downloaded: List[str] = []
    skipped: List[str] = []
    missing: List[str] = []

    for name in required:
        blob_name = f"{prefix}{name}" if prefix else name
        blob = b.blob(blob_name)
        if not blob.exists(client=cli):
            missing.append(blob_name)
            continue

        out_path = dest_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped.append(name)
            continue

        blob.download_to_filename(str(out_path))
        downloaded.append(name)

    return GcsSyncResult(downloaded=downloaded, skipped=skipped, missing=missing)

