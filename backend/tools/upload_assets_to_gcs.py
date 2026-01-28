from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from google.cloud import storage

ASSET_FILES = [
    "scaler1.joblib",
    "random_forest_model_Fragmentation.joblib",
    "random_forest_model_Ground Vibration.joblib",
    "random_forest_model_Airblast.joblib",
]

DATASET_FILES = [
    "combinedv2Orapa.csv",
    "Backbreak.csv",
    "flyrock_synth.csv",
    "slope data.csv",
    "Hole_data_v1.csv",
    "Hole_data_v2.csv",
]


def _env_path(key: str) -> Path | None:
    val = os.getenv(key)
    return Path(val).expanduser().resolve() if val else None


def _iter_existing(root: Path | None, names: Iterable[str]):
    if not root:
        return
    for name in names:
        p = root / name
        if p.exists():
            yield p


def main() -> None:
    bucket = os.getenv("BLAST_GCS_BUCKET", "").strip()
    prefix = os.getenv("BLAST_GCS_PREFIX", "assets/").strip()
    if not bucket:
        raise SystemExit("Missing BLAST_GCS_BUCKET.")

    assets_dir = _env_path("BLAST_LOCAL_ASSETS_DIR")
    data_dir = _env_path("BLAST_LOCAL_DATA_DIR")

    candidates = list(_iter_existing(assets_dir, ASSET_FILES)) + list(
        _iter_existing(data_dir, DATASET_FILES)
    )
    if not candidates:
        raise SystemExit(
            "No files found. Set BLAST_LOCAL_ASSETS_DIR and BLAST_LOCAL_DATA_DIR."
        )

    client = storage.Client()
    b = client.bucket(bucket)

    for p in candidates:
        name = p.name
        blob_name = f"{prefix}{name}" if prefix else name
        blob = b.blob(blob_name)
        blob.upload_from_filename(str(p))
        print(f"Uploaded {name} -> gs://{bucket}/{blob_name}")


if __name__ == "__main__":
    main()
