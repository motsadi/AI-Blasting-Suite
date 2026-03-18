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
    "combinedv2Jwaneng.csv",
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

    prefix_candidates: list[str] = []
    for candidate in [prefix, "assets/", "datasets/", ""]:
        normalized = candidate or ""
        if normalized and not normalized.endswith("/"):
            normalized = f"{normalized}/"
        if normalized not in prefix_candidates:
            prefix_candidates.append(normalized)

    for name in required:
        out_path = dest_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped.append(name)
            continue

        found = False
        attempted: list[str] = []
        for pref in prefix_candidates:
            blob_name = f"{pref}{name}" if pref else name
            attempted.append(blob_name)
            blob = b.blob(blob_name)
            if not blob.exists(client=cli):
                continue
            blob.download_to_filename(str(out_path))
            downloaded.append(name)
            found = True
            break

        if not found:
            missing.extend(attempted)

    return GcsSyncResult(downloaded=downloaded, skipped=skipped, missing=missing)

