from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_auth
from app.assets import assets_status, load_local_assets
from app.core_imports import add_core_bundle_to_path
from app.gcs import sync_assets_from_gcs
from app.schemas import AssetsStatus, PredictRequest, PredictResponse
from app.settings import settings


core_bundle_path = add_core_bundle_to_path()

# Import read-only core functions (do not modify them)
from utils_blaster import INPUT_LABELS, EmpiricalParams, empirical_predictions  # type: ignore

app = FastAPI(title="AI Blasting Suite API", version="0.1.0")

origins = [o.strip() for o in (settings.cors_origins or "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    # We authenticate via Authorization header (refresh_token), not cookies.
    # Keeping allow_credentials=False makes wildcard origins safe when needed.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=settings.cors_origin_regex,
)

_assets = load_local_assets(core_bundle_path)


@app.on_event("startup")
def _startup_sync_assets():
    """
    If BLAST_GCS_BUCKET is set, download model/scaler assets from GCS into the core bundle folder
    (or fallback to a writable cache directory).
    """
    if not settings.gcs_bucket:
        return

    bucket = settings.gcs_bucket.strip()
    prefix = (settings.gcs_prefix or "").strip()
    if not bucket:
        return

    dest = Path(core_bundle_path)
    # If the destination isn't writable (common in some container setups), use /tmp
    try:
        dest.mkdir(parents=True, exist_ok=True)
        test = dest / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        dest = Path("/tmp/ai-blasting-assets")

    try:
        sync_assets_from_gcs(
            bucket=bucket,
            prefix=prefix,
            dest_dir=dest,
        )
    except Exception:
        # Never prevent the API from starting due to asset sync issues.
        # (Assets can still be loaded from the baked image, or re-synced later.)
        return

    # Reload assets after sync
    global _assets
    _assets = load_local_assets(dest)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/assets/status", response_model=AssetsStatus)
def get_assets_status():
    st = assets_status(_assets)
    return AssetsStatus(**st)

@app.get("/v1/meta")
def get_meta():
    """
    Frontend bootstrap metadata so the UI can render forms without hardcoding.
    """
    d = EmpiricalParams()
    return {
        "input_labels": list(INPUT_LABELS),
        "outputs": ["Ground Vibration", "Airblast", "Fragmentation"],
        "empirical_defaults": {
            "K_ppv": d.K_ppv,
            "beta": d.beta,
            "K_air": d.K_air,
            "B_air": d.B_air,
            "A_kuz": d.A_kuz,
            "RWS": d.RWS,
        },
        "defaults": {"hpd_override": 1.0},
    }


@app.post("/v1/assets/sync", response_model=AssetsStatus)
def sync_assets(_token: str = Depends(require_auth)):
    if not settings.gcs_bucket:
        return AssetsStatus(**assets_status(_assets))
    _startup_sync_assets()
    return AssetsStatus(**assets_status(_assets))


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _token: str = Depends(require_auth)):
    # Map request inputs onto the exact labels expected by the core bundle
    vals = {k: float(v) for k, v in req.inputs.items()}
    vals["HPD_override"] = float(req.hpd_override)

    outputs = ["Ground Vibration", "Airblast", "Fragmentation"]
    p = EmpiricalParams(**req.empirical.model_dump())
    emp = empirical_predictions(vals, p, outputs)

    ml = None
    if req.want_ml and _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        import numpy as np

        X = np.array([[vals.get(n, 0.0) for n in INPUT_LABELS]], dtype=float)
        Xs = _assets.scaler.transform(X)
        ml = {k: float("nan") for k in outputs}
        if _assets.mdl_ppv:
            ml["Ground Vibration"] = float(_assets.mdl_ppv.predict(Xs)[0])
        if _assets.mdl_air:
            ml["Airblast"] = float(_assets.mdl_air.predict(Xs)[0])
        if _assets.mdl_frag:
            ml["Fragmentation"] = float(_assets.mdl_frag.predict(Xs)[0])

    return PredictResponse(empirical=emp, ml=ml, assets_loaded=assets_status(_assets))

