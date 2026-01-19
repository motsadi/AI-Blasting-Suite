from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BLAST_", extra="ignore")

    cors_origins: str = "*"  # comma-separated, overridden in prod
    cors_origin_regex: Optional[str] = None  # e.g. https://.*\\.vercel\\.app

    # Optional: GCS bucket for model/data assets
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "assets/"

    # InstantDB (placeholders; wire once token verification mechanism is confirmed)
    instantdb_api_uri: str = "https://api.instantdb.com"
    instantdb_app_id: Optional[str] = None  # required for verifying refresh_token
    instantdb_admin_key: Optional[str] = None


settings = Settings()

