from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_auth, require_user
from app.assets import LoadedAssets, assets_status, load_local_assets
from app.core_imports import add_core_bundle_to_path
from app.gcs import REQUIRED_DATASET_FILES, sync_assets_from_gcs
from app.schemas import AssetsStatus, PredictRequest, PredictResponse
from app.settings import settings


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

core_bundle_path = add_core_bundle_to_path()
if settings.local_assets_dir:
    local_assets_path = Path(settings.local_assets_dir).resolve()
else:
    # Vercel / local default: allow committing model assets into repo_root/assets/
    default_assets = (REPO_ROOT / "assets").resolve()
    local_assets_path = default_assets if default_assets.exists() else None
if settings.local_data_dir:
    local_data_path = Path(settings.local_data_dir).resolve()
else:
    # Vercel / local default: allow committing datasets into repo_root/datasets/
    default_datasets = (REPO_ROOT / "datasets").resolve()
    local_data_path = default_datasets if default_datasets.exists() else None

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

_assets = load_local_assets(*(p for p in [local_assets_path, core_bundle_path] if p))
_ml_rollout_state: dict[str, dict] = {}
_ml_residual_models: dict[str, object] = {}
_ml_model_choice: dict[str, str] = {}

DATASETS = {
    "combined": "combinedv2Orapa.csv",
    "backbreak": "Backbreak.csv",
    "flyrock": "flyrock_synth.csv",
    "slope": "slope data.csv",
    "delay_v1": "Hole_data_v1.csv",
    "delay_v2": "Hole_data_v2.csv",
}

COMBINED_DATASET_CHOICES = [
    "combinedv2Orapa.csv",
    "combinedv2Jwaneng.csv",
    "combinedv2Jwaneng.xlsx",
]

APP_ACTIVITY_PATH = Path(tempfile.gettempdir()) / "ai-blasting-suite-activity.json"


def _dataset_search_candidates(name: str) -> list[Path]:
    return [
        *([local_data_path / name] if local_data_path else []),
        REPO_ROOT / "datasets" / name,
        WORKSPACE_ROOT / name,
        WORKSPACE_ROOT / "datasets" / name,
        Path(core_bundle_path) / name,
        Path("/tmp/ai-blasting-assets") / name,
        Path("/tmp/ai-blasting-assets/datasets") / name,
    ]


def _paired_dataset_name(name: str) -> str | None:
    low = name.lower()
    if low.endswith(".csv"):
        return name[:-4] + ".xlsx"
    if low.endswith(".xlsx"):
        return name[:-5] + ".csv"
    if low.endswith(".xls"):
        return name[:-4] + ".csv"
    return None


def _resolve_dataset_name(name: str, allowed: list[str] | None = None) -> str:
    allowed_set = set(allowed or [])
    if not allowed_set or name in allowed_set:
        return name
    alt = _paired_dataset_name(name)
    if alt and alt in allowed_set:
        return alt
    return name


def _combined_dataset_choices() -> list[str]:
    existing = [name for name in COMBINED_DATASET_CHOICES if any(path.exists() for path in _dataset_search_candidates(name))]
    if existing:
        return existing
    # If no direct hits, keep static choices so UI still renders but runtime
    # resolution can map csv<->xlsx where needed.
    return list(COMBINED_DATASET_CHOICES)


def _asset_search_paths() -> list[Path]:
    return [
        *([local_assets_path] if local_assets_path else []),
        REPO_ROOT / "assets",
        WORKSPACE_ROOT / "assets",
        Path(core_bundle_path),
        Path("/tmp/ai-blasting-assets"),
    ]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_activity_state() -> dict:
    default_state = {"last_activity": None, "recent": []}
    try:
        if not APP_ACTIVITY_PATH.exists():
            return default_state
        raw = json.loads(APP_ACTIVITY_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_state
        recent = raw.get("recent")
        return {
            "last_activity": raw.get("last_activity"),
            "recent": recent[:12] if isinstance(recent, list) else [],
        }
    except Exception:
        return default_state


def _write_activity_state(state: dict) -> dict:
    safe_state = {
        "last_activity": state.get("last_activity"),
        "recent": (state.get("recent") or [])[:12],
    }
    try:
        APP_ACTIVITY_PATH.write_text(json.dumps(safe_state, indent=2), encoding="utf-8")
    except Exception:
        pass
    return safe_state


def _record_app_activity(email: str, module_key: str, module_title: str) -> dict:
    state = _read_activity_state()
    entry = {
        "email": email or "Unknown",
        "module_key": module_key or "home",
        "module_title": module_title or module_key or "Home",
        "timestamp": _utcnow_iso(),
    }
    recent = [entry]
    for item in state.get("recent", []):
        if not isinstance(item, dict):
            continue
        if item.get("email") == entry["email"] and item.get("module_key") == entry["module_key"]:
            continue
        recent.append(item)
    return _write_activity_state({"last_activity": entry, "recent": recent})


def _reload_assets_from_disk() -> LoadedAssets:
    global _assets
    search_paths = [p for p in _asset_search_paths() if p and p.exists()]
    _assets = load_local_assets(*search_paths)
    return _assets


def _ensure_prediction_assets_ready() -> None:
    global _assets
    if _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        return

    _reload_assets_from_disk()
    if _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        return

    if not settings.gcs_bucket:
        return

    dest = Path("/tmp/ai-blasting-assets")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        sync_assets_from_gcs(
            bucket=settings.gcs_bucket.strip(),
            prefix=(settings.gcs_prefix or "").strip(),
            dest_dir=dest,
        )
    except Exception:
        return

    _reload_assets_from_disk()


@app.on_event("startup")
def _startup_sync_assets():
    """
    If BLAST_GCS_BUCKET is set, download model/scaler assets and datasets from GCS into the
    core bundle folder (or fallback to a writable cache directory).
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
        sync_assets_from_gcs(
            bucket=bucket,
            # Datasets are stored under the "datasets/" folder in the bucket.
            prefix="datasets/",
            dest_dir=dest,
            required_files=REQUIRED_DATASET_FILES,
        )
    except Exception:
        # Never prevent the API from starting due to asset sync issues.
        # (Assets can still be loaded from the baked image, or re-synced later.)
        return

    # Reload assets after sync
    global _assets
    _assets = load_local_assets(*(p for p in [local_assets_path, dest] if p))


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/activity")
def get_activity(user: dict = Depends(require_user)):
    state = _read_activity_state()
    return {
        "current_user": {"email": user.get("email", "Unknown")},
        "last_activity": state.get("last_activity"),
        "recent": state.get("recent", []),
        "server_time": _utcnow_iso(),
    }


@app.post("/v1/activity")
def record_activity(payload: dict = Body(default={}), user: dict = Depends(require_user)):
    module_key = str(payload.get("module_key") or "home").strip() or "home"
    module_title = str(payload.get("module_title") or module_key).strip() or module_key
    state = _record_app_activity(str(user.get("email") or "Unknown"), module_key, module_title)
    return {
        "current_user": {"email": user.get("email", "Unknown")},
        "last_activity": state.get("last_activity"),
        "recent": state.get("recent", []),
        "server_time": _utcnow_iso(),
    }


@app.get("/v1/assets/status", response_model=AssetsStatus)
def get_assets_status():
    _ensure_prediction_assets_ready()
    st = assets_status(_assets)
    return AssetsStatus(**st)

@app.get("/v1/meta")
def get_meta():
    """
    Frontend bootstrap metadata so the UI can render forms without hardcoding.
    """
    combined_choices = _combined_dataset_choices()
    active_combined = DATASETS.get("combined")
    if active_combined not in combined_choices:
        active_combined = combined_choices[0] if combined_choices else DATASETS.get("combined")

    d = EmpiricalParams()
    input_stats = {}
    try:
        import pandas as pd

        df = _read_upload_df(None, active_combined)
        cols = {str(c).lower().strip(): c for c in df.columns}
        norm = {_normalize_col(c): c for c in df.columns}
        for label in INPUT_LABELS:
            key = str(label).lower().strip()
            col = cols.get(key) or norm.get(_normalize_col(label))
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            input_stats[label] = {
                "min": float(s.quantile(0.02)),
                "max": float(s.quantile(0.98)),
                "median": float(s.median()),
            }
    except Exception:
        input_stats = {}

    return {
        "input_labels": list(INPUT_LABELS),
        "outputs": ["Ground Vibration", "Airblast", "Fragmentation"],
        "default_dataset": active_combined,
        "combined_dataset": active_combined,
        "combined_dataset_choices": combined_choices,
        "empirical_defaults": {
            "K_ppv": d.K_ppv,
            "beta": d.beta,
            "K_air": d.K_air,
            "B_air": d.B_air,
            "A_kuz": d.A_kuz,
            "RWS": d.RWS,
        },
        "defaults": {"hpd_override": 1.0},
        "input_stats": input_stats,
    }


@app.get("/v1/datasets/combined")
def get_combined_dataset(_token: str = Depends(require_auth)):
    choices = _combined_dataset_choices()
    active = DATASETS.get("combined")
    if active not in choices:
        active = choices[0] if choices else active
    return {"active": active, "choices": choices}


@app.post("/v1/datasets/combined")
def set_combined_dataset(payload: dict = Body(...), _token: str = Depends(require_auth)):
    """
    Set the active "combined" dataset used by Data Manager default preview, Prediction,
    Feature Importance, and Parameter Optimisation.
    """
    name = str(payload.get("name") or "").strip()
    choices = _combined_dataset_choices()
    name_resolved = _resolve_dataset_name(name, allowed=choices)
    if name_resolved not in choices:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset. Choose one of: {choices}")

    DATASETS["combined"] = name_resolved

    # Reset ML/optimisation caches so the next request retrains on the selected dataset.
    global _ml_cache_key, _ml_rollout_cache_key, _assets, _param_cache_key, _param_cache_bundle
    global _ml_rollout_state, _ml_residual_models, _ml_model_choice
    _ml_cache_key = None
    _ml_rollout_cache_key = None
    _param_cache_key = None
    _param_cache_bundle = None
    _ml_rollout_state = {}
    _ml_residual_models = {}
    _ml_model_choice = {}
    _assets = LoadedAssets(scaler=None, mdl_frag=None, mdl_ppv=None, mdl_air=None)

    return {"ok": True, "active": name_resolved}


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

    # RR curve (from local prediction_module logic)
    rr = None
    try:
        import math
        import numpy as np

        def _safe_log10(x: float, eps: float = 1e-12) -> float:
            return math.log10(max(x, eps))

        def _derived_charge_volume() -> tuple[float, float, float, float]:
            depth = float(vals.get("Hole depth (m)", 0.0))
            stem = float(vals.get("Stemming (m)", 0.0))
            Lc = max(0.0, depth - stem)
            lin = float(vals.get("Linear charge (kg/m)", 0.0))
            mass_per_hole_lin = lin * Lc
            mass_per_hole_exp = float(vals.get("Explosive mass (kg)", 0.0))
            n_holes = max(1, int(round(float(vals.get("# Holes", 1)))))
            vol = float(vals.get("Blast volume (m³)", 0.0))
            if vol <= 0.0:
                B = float(vals.get("Burden (m)", 0.0))
                S = float(vals.get("Spacing (m)", 0.0))
                vol = max(0.0, B * S * depth * n_holes)
            PF_slider = float(vals.get("Powder factor (kg/m³)", 0.0))

            per_hole = 0.0
            for cand in (mass_per_hole_lin, mass_per_hole_exp):
                if cand > 0:
                    per_hole = cand
                    break
            total_mass = per_hole * n_holes
            if PF_slider > 0.0 and vol > 0.0:
                total_mass = PF_slider * vol
                if n_holes > 0:
                    per_hole = total_mass / n_holes
            hpd = max(1.0, float(req.hpd_override))
            qd = per_hole * hpd
            return per_hole, total_mass, qd, vol

        def _estimate_n_rr() -> float | None:
            B = float(vals.get("Burden (m)", 0.0))
            d_mm = float(vals.get("Hole diameter (mm)", 0.0))
            if B <= 0 or d_mm <= 0:
                return None
            d_m = d_mm / 1000.0
            n_hat = 2.2 - 14.0 * (d_m / max(B, 1e-9))
            return float(np.clip(n_hat, 0.5, 3.5))

        n = req.rr_n
        if (req.rr_mode or "").startswith("estimate") or n is None:
            n = _estimate_n_rr() or n
        if n is None:
            n = 1.8

        per_hole, total_mass, qd, vol = _derived_charge_volume()
        A = max(0.0, float(req.empirical.A_kuz))
        RWS = max(1e-6, float(req.empirical.RWS))
        K_pf = float(vals.get("Powder factor (kg/m³)", 0.0))
        if (K_pf <= 0.0) and (vol > 0.0):
            K_pf = total_mass / vol
        xm = None
        if K_pf > 0.0 and per_hole > 0.0:
            xm = A * (K_pf ** -0.8) * (per_hole ** (1.0 / 6.0)) * ((115.0 / RWS) ** (19.0 / 20.0))
        if xm is not None and n > 0:
            lam = xm / math.gamma(1.0 + 1.0 / n)
            x50 = lam * (math.log(2.0) ** (1.0 / n))
            x_ov = float(req.rr_x_ov or 500.0)
            xmax = max(6.0 * xm, 1.5 * x_ov, 10.0)
            xs = np.logspace(math.log10(max(0.1, xm / 20.0)), math.log10(xmax), 200)
            P = 100.0 * (1.0 - np.exp(-((xs / lam) ** n)))
            oversize = 100.0 * float(np.exp(-((x_ov / lam) ** n)))
            rr = {
                "n": float(n),
                "xm": float(xm),
                "x50": float(x50),
                "x_ov": float(x_ov),
                "oversize_pct": float(oversize),
                "xs": xs.tolist(),
                "cdf": P.tolist(),
            }
    except Exception:
        rr = None

    ml = None
    if req.want_ml:
        _ensure_prediction_assets_ready()
    if req.want_ml and not (_assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air)):
        _maybe_train_models_from_default_dataset()
    if req.want_ml and _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        _maybe_prepare_physics_rollout_from_default_dataset()
    if req.want_ml and _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        ml = {k: float("nan") for k in outputs}
        gv = _predict_ml_output("Ground Vibration", vals)
        ab = _predict_ml_output("Airblast", vals)
        fr = _predict_ml_output("Fragmentation", vals)
        if gv is not None:
            ml["Ground Vibration"] = float(gv)
        if ab is not None:
            ml["Airblast"] = float(ab)
        if fr is not None:
            ml["Fragmentation"] = float(fr)

    return PredictResponse(empirical=emp, ml=ml, assets_loaded=assets_status(_assets), rr=rr)


@app.post("/v1/predict/upload", response_model=PredictResponse)
def predict_upload(
    file: UploadFile = File(...),
    inputs_json: str = Form(...),
    hpd_override: float = Form(default=1.0),
    empirical_json: str = Form(default="{}"),
    want_ml: bool = Form(default=True),
    rr_n: float | None = Form(default=None),
    rr_mode: str | None = Form(default=None),
    rr_x_ov: float | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    """
    Predict using an uploaded dataset (fallback ML training when joblib assets are missing).
    """
    import json

    try:
        inputs = json.loads(inputs_json)
    except Exception:
        inputs = {}
    try:
        empirical = json.loads(empirical_json)
    except Exception:
        empirical = {}

    req = PredictRequest(
        inputs=inputs,
        hpd_override=hpd_override,
        empirical=empirical,
        want_ml=want_ml,
        rr_n=rr_n,
        rr_mode=rr_mode,
        rr_x_ov=rr_x_ov,
    )

    if want_ml:
        try:
            df = _read_upload_df(file)
            _maybe_train_models_from_df(df)
        except Exception:
            pass

    return predict(req, _token=_token)


def _ensure_dataset(name: str) -> Path:
    """
    Ensure a dataset exists locally; otherwise attempt download from GCS.
    """
    local_candidates = _dataset_search_candidates(name)
    for p in local_candidates:
        if p.exists():
            return p

    alt_name = _paired_dataset_name(name)
    if alt_name:
        alt_candidates = _dataset_search_candidates(alt_name)
        for p in alt_candidates:
            if p.exists():
                return p

    if not settings.gcs_bucket:
        raise FileNotFoundError(f"Dataset not found locally and BLAST_GCS_BUCKET is not set: {name}")

    from google.cloud import storage

    prefixes = ["datasets/", "assets/", ""]
    dest_dir = Path("/tmp/ai-blasting-assets/datasets")
    dest_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket)
    for pref in prefixes:
        blob = bucket.blob(f"{pref}{name}")
        try:
            if blob.exists():
                dest = dest_dir / name
                blob.download_to_filename(dest)
                return dest
        except Exception:
            continue
    if alt_name:
        for pref in prefixes:
            blob = bucket.blob(f"{pref}{alt_name}")
            try:
                if blob.exists():
                    dest = dest_dir / alt_name
                    blob.download_to_filename(dest)
                    return dest
            except Exception:
                continue
    raise FileNotFoundError(f"Dataset not found in GCS bucket {settings.gcs_bucket}: {name}")


def _read_upload_df(up: UploadFile | None = None, dataset_name: str | None = None):
    import pandas as pd
    import io

    def _read_csv_robust(path_or_buf):
        """Try encodings to match local slope_stability.py (utf-8-sig, latin-1 for Unicode headers)."""
        last_err = None
        for enc in (None, "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path_or_buf, encoding=enc) if enc else pd.read_csv(path_or_buf)
            except (UnicodeDecodeError, Exception) as e:
                last_err = e
                if hasattr(path_or_buf, "seek"):
                    path_or_buf.seek(0)
        raise last_err

    if up is None:
        if not dataset_name:
            raise ValueError("No file or dataset_name provided")
        path = _ensure_dataset(dataset_name)
        if str(path).lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return _read_csv_robust(path)

    data = up.file.read()
    up.file.seek(0)
    if up.filename and up.filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return _read_csv_robust(io.BytesIO(data))


def _preview_df(df, n=20):
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sample": df.head(n).to_dict(orient="records"),
    }


def _normalize_col(s: str) -> str:
    keep = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s))
    return " ".join(keep.split())


def _score_r2(y_true, y_pred) -> float | None:
    try:
        from sklearn.metrics import r2_score

        return float(r2_score(y_true, y_pred))
    except Exception:
        return None


def _flyrock_norm(s: str) -> str:
    s = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s))
    s = " ".join(s.split())
    s = (
        s.replace(" kg/m3", "")
        .replace(" kg m3", "")
        .replace(" kg m^3", "")
        .replace(" kn/m3", "")
        .replace(" kn m3", "")
        .replace("(m)", "")
        .replace(" m", "")
        .replace(" mm", "")
        .replace(" kpa", "")
    )
    return " ".join(s.split())


def _flyrock_resolve_one(df_cols, candidates):
    cols = list(df_cols)
    norm_map = {_flyrock_norm(c): c for c in cols}
    for cand in candidates:
        nc = _flyrock_norm(cand)
        if nc in norm_map:
            return norm_map[nc]
    for c in cols:
        cn = _flyrock_norm(c)
        if any(_flyrock_norm(cand) in cn for cand in candidates):
            return c
    return None


def _flyrock_infer_target(df):
    import numpy as np
    import pandas as pd

    name = _flyrock_resolve_one(df.columns.tolist(), ["flyrock", "fly rock", "flyrock distance", "throw", "max throw"])
    if name is not None:
        return name
    num = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    if num.shape[1] >= 1:
        last = num.columns[-1]
        if np.isfinite(num[last]).sum() >= 10 and float(num[last].std()) > 1e-9:
            return last
    return None


def _prepare_flyrock_data(df):
    import numpy as np
    import pandas as pd

    target = _flyrock_infer_target(df)
    if target is None:
        raise ValueError(
            "Could not infer flyrock target column. Expected a column like Flyrock, Flyrock (m), Throw, or Max Throw."
        )

    num = df.apply(pd.to_numeric, errors="coerce")
    y = num[target]
    X = num.drop(columns=[target]).copy()

    drop_ml = []
    for c in X.columns:
        cn = _flyrock_norm(c)
        if any(k in cn for k in ["lundborg", "sdob", "spacing burden ratio", "spacing to burden", "s/b"]):
            drop_ml.append(c)
    if drop_ml:
        X = X.drop(columns=drop_ml, errors="ignore")

    valid_cols = [c for c in X.columns if np.isfinite(X[c]).sum() >= max(20, int(0.3 * len(X)))]
    X = X[valid_cols]

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask].copy()
    y = y[mask].copy()
    if X.shape[0] < 30 or X.shape[1] < 2:
        raise ValueError("Need >=30 clean rows and >=2 usable numeric features after cleaning.")
    return X, y, target


def _slope_norm(name: str) -> str:
    import re

    s = str(name).strip().lower()
    greek = {"γ": "gamma", "𝛾": "gamma", "𝜸": "gamma", "𝝲": "gamma", "φ": "phi", "ϕ": "phi", "𝜑": "phi", "𝝋": "phi", "β": "beta", "𝛽": "beta", "𝜷": "beta"}
    for g, rep in greek.items():
        s = s.replace(g, rep)
    s = (
        s.replace("kn/m3", "")
        .replace("kpa", "")
        .replace("(m)", "")
        .replace("°", "")
        .replace("/", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("_", " ")
    )
    return re.sub(r"\s+", " ", s).strip()


def _prepare_slope_df(df):
    import pandas as pd

    canon = {
        "gamma_kN_m3": ["gamma", "unit weight", "gamma kn m3", "gamma knm3"],
        "c_kPa": ["c", "cohesion"],
        "phi_deg": ["phi", "friction angle", "phi deg"],
        "beta_deg": ["beta", "slope angle", "beta deg"],
        "H_m": ["h", "height", "h m"],
        "ru": ["ru", "r u", "pore pressure ratio", "pore pressure ratio ru"],
        "status": ["status", "class", "label"],
    }
    rename = {}
    nmap = {c: _slope_norm(c) for c in df.columns}
    for target, options in canon.items():
        for col, normed in nmap.items():
            if normed in options:
                rename[col] = target
                break
    out = df.rename(columns=rename).copy()
    for c in list(out.columns):
        if _slope_norm(c) in {"no", "#", "index", "id", "sr"}:
            out = out.drop(columns=[c])

    needed = ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]
    if not all(c in out.columns for c in needed):
        maybe = df.copy()
        if _slope_norm(maybe.columns[0]) in {"no", "#", "index", "id", "sr"} and maybe.shape[1] >= 8:
            maybe = maybe.drop(columns=[maybe.columns[0]])
        if maybe.shape[1] == 7:
            maybe.columns = needed
            out = maybe
    if not all(c in out.columns for c in needed):
        raise ValueError("Missing required slope columns: gamma, c, phi, beta, H, ru, status.")

    for c in ["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["status"] = (
        out["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"stable": "stable", "failure": "failure", "failed": "failure", "unstable": "failure"})
    )
    out = out.dropna(subset=["gamma_kN_m3", "c_kPa", "phi_deg", "beta_deg", "H_m", "ru", "status"]).copy()
    out["y"] = (out["status"] == "stable").astype(int)
    if out["y"].nunique() < 2:
        raise ValueError("Dataset contains only one class (all Stable or all Failure).")
    return out


def _infer_backbreak_target(df):
    import pandas as pd

    names = list(df.columns)
    norm_map = {_normalize_col(c): c for c in names}
    candidates = [
        "backbreak",
        "back break",
        "bb",
        "backbreak m",
        "backbreak mm",
        "back break m",
        "back break mm",
    ]
    for key in candidates:
        if key in norm_map:
            return norm_map[key]
    last = names[-1]
    try:
        x = pd.to_numeric(df[last], errors="coerce")
        if x.notna().sum() >= 10 and float(x.std()) > 1e-9:
            return last
    except Exception:
        pass
    return None


def _prepare_backbreak_bundle(df):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (features + target).")

    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass

    tgt = _infer_backbreak_target(df)
    if tgt is None:
        raise ValueError("Could not infer Back Break target column.")

    num = df.select_dtypes(include=[np.number]).copy()
    if tgt not in num.columns:
        ytmp = pd.to_numeric(df[tgt], errors="coerce")
        num[tgt] = ytmp

    X = num.drop(columns=[tgt], errors="ignore").copy()
    y = pd.to_numeric(df[tgt], errors="coerce")

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    if X.shape[0] < 30 or X.shape[1] < 2:
        raise ValueError("Not enough clean rows or features to train Random Forest (need >=30 rows, >=2 features).")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(Xtr, ytr)
    train_r2 = _score_r2(ytr, model.predict(Xtr))
    test_r2 = _score_r2(yte, model.predict(Xte))

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    feat_names = list(X.columns)
    keep = [feat_names[i] for i in order[: min(6, len(order))]]
    feat_importance = [{"feature": feat_names[i], "importance": float(imp[i])} for i in order]

    stats = {}
    for name in keep:
        col = pd.to_numeric(X[name], errors="coerce").dropna()
        vmin = float(col.quantile(0.02)) if col.size else 0.0
        vmax = float(col.quantile(0.98)) if col.size else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = (float(col.min()), float(col.max())) if col.size else (0.0, 1.0)
            if vmin == vmax:
                vmax = vmin + 1.0
        stats[name] = {"min": vmin, "max": vmax, "median": float(col.median()) if col.size else 0.0}

    return {
        "model": model,
        "keep": keep,
        "stats": stats,
        "feat_importance": feat_importance,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


@app.post("/v1/data/preview")
def data_preview(file: UploadFile = File(...), _token: str = Depends(require_auth)):
    df = _read_upload_df(file)
    return _preview_df(df)


@app.get("/v1/data/default")
def data_default(_token: str = Depends(require_auth)):
    df = _read_upload_df(None, DATASETS["combined"])
    return _preview_df(df)


@app.post("/v1/data/upload")
def data_upload(
    file: UploadFile = File(...),
    _token: str = Depends(require_auth),
):
    if not settings.gcs_bucket:
        return {"error": "GCS bucket not configured"}
    import datetime
    from google.cloud import storage

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = file.filename or "dataset.csv"
    obj = f"datasets/{ts}_{name}"
    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket)
    blob = bucket.blob(obj)
    blob.upload_from_file(file.file, content_type=file.content_type)
    return {"gs_uri": f"gs://{settings.gcs_bucket}/{obj}"}


@app.get("/v1/feature-importance")
def feature_importance(_token: str = Depends(require_auth)):
    """
    Return model feature importances for the preloaded RF models.
    """
    out = {}
    for name, mdl in [
        ("Ground Vibration", _assets.mdl_ppv),
        ("Airblast", _assets.mdl_air),
        ("Fragmentation", _assets.mdl_frag),
    ]:
        if mdl is None or not hasattr(mdl, "feature_importances_"):
            continue
        imp = list(getattr(mdl, "feature_importances_", []))
        out[name] = [
            {"feature": f, "importance": float(v)}
            for f, v in zip(list(INPUT_LABELS), imp)
        ]
    return {"feature_importance": out}


@app.get("/v1/models/rollout")
def model_rollout_status(_token: str = Depends(require_auth)):
    return {
        "active_models": dict(_ml_model_choice),
        "rollout": dict(_ml_rollout_state),
    }


def _feature_map_synonyms():
    return {
        "inputs": {
            "hole depth (m)": ["hole depth", "depth", "holedepth"],
            "hole diameter (mm)": ["hole diameter", "diameter", "hole dia", "hole_diameter", "holediameter"],
            "burden (m)": ["burden"],
            "spacing (m)": ["spacing"],
            "stemming (m)": ["stemming"],
            "distance (m)": ["distance", "monitor distance", "distance m", "monitoring distance"],
            "powder factor (kg/m³)": ["powder factor", "powderfactor", "pf", "powder factor (kg/m3)"],
            "rock density (t/m³)": ["rock density", "density", "rock density (t/m3)"],
            "linear charge (kg/m)": ["linear charge", "linearcharge", "kg/m", "charge per metre", "charge/m"],
            "explosive mass (kg)": ["explosive mass", "charge mass", "explosivemass"],
            "blast volume (m³)": ["blast volume", "volume", "blast volume (m3)"],
            "# holes": ["number of holes", "no. holes", "holes", "holes count", "holes #"],
        },
        "outputs": {
            "fragmentation": ["mean fragmentation", "fragmentation", "p80", "frag", "fragmentation (mm)"],
            "ground vibration": ["ground vibration", "ppv", "peak particle velocity", "ppv (mm/s)", "ppv mms"],
            "airblast": ["airblast", "air blast", "air overpressure", "overpressure", "db", "air blast (db)"],
        },
    }


_ml_cache_key: str | None = None
_ml_rollout_cache_key: str | None = None
_param_cache_key: str | None = None
_param_cache_bundle: dict | None = None

PHYSICAL_OUTPUT_FLOORS = {
    "Ground Vibration": 1e-3,
    "Airblast": 1e-3,
    "Fragmentation": 1e-3,
}
FRAGMENTATION_TARGET_MM = 100.0
FRAGMENTATION_TOLERANCE_MM = 10.0


def _clamp_physical_output(name: str, value: float) -> float:
    low = str(name).strip().lower()
    for label, floor in PHYSICAL_OUTPUT_FLOORS.items():
        label_low = label.lower()
        if low == label_low or label_low in low or low in label_low:
            return float(max(floor, value))
    return float(value)


def _empirical_value_for_output(vals: dict[str, float], output_name: str) -> float:
    p = EmpiricalParams()
    out = empirical_predictions(vals, p, [output_name])
    raw = float(out.get(output_name, 0.0))
    return _clamp_physical_output(output_name, raw)


def _empirical_series_for_frame(frame, output_name: str):
    import numpy as np

    ys = []
    for _, row in frame.iterrows():
        vals = {name: float(row.get(name, 0.0)) for name in INPUT_LABELS}
        vals["HPD_override"] = 1.0
        try:
            ys.append(_empirical_value_for_output(vals, output_name))
        except Exception:
            ys.append(float("nan"))
    return np.array(ys, dtype=float)


def _check_physics_consistency(
    output_name: str,
    scaler,
    baseline_model,
    residual_model,
    feature_frame,
):
    import numpy as np
    import pandas as pd

    if feature_frame is None or feature_frame.empty:
        return {"passed": False, "checks": {"error": "No feature rows for consistency checks"}}

    ref = {}
    for c in INPUT_LABELS:
        s = pd.to_numeric(feature_frame[c], errors="coerce")
        finite = s[np.isfinite(s.values)]
        ref[c] = float(finite.median()) if len(finite) else 0.0

    def predict_mode(mode: str, vec: dict[str, float]) -> float:
        X = np.array([[float(vec.get(c, 0.0)) for c in INPUT_LABELS]], dtype=float)
        Xs = scaler.transform(X)
        if mode == "baseline":
            raw = float(baseline_model.predict(Xs)[0])
            return _clamp_physical_output(output_name, raw)
        emp = _empirical_value_for_output({**vec, "HPD_override": 1.0}, output_name)
        raw = float(emp + residual_model.predict(Xs)[0])
        return _clamp_physical_output(output_name, raw)

    checks = {}

    distance_name = "Distance (m)"
    if distance_name in feature_frame.columns:
        s = pd.to_numeric(feature_frame[distance_name], errors="coerce").dropna()
        if len(s):
            lo = float(s.quantile(0.2))
            hi = float(s.quantile(0.8))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                lo_vec = dict(ref)
                hi_vec = dict(ref)
                lo_vec[distance_name] = lo
                hi_vec[distance_name] = hi
                low_pred = predict_mode("candidate", lo_vec)
                high_pred = predict_mode("candidate", hi_vec)
                checks["distance_decay"] = bool(low_pred >= high_pred)

    mass_name = "Explosive mass (kg)"
    if mass_name in feature_frame.columns:
        s = pd.to_numeric(feature_frame[mass_name], errors="coerce").dropna()
        if len(s):
            lo = float(s.quantile(0.2))
            hi = float(s.quantile(0.8))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                lo_vec = dict(ref)
                hi_vec = dict(ref)
                lo_vec[mass_name] = lo
                hi_vec[mass_name] = hi
                low_pred = predict_mode("candidate", lo_vec)
                high_pred = predict_mode("candidate", hi_vec)
                checks["mass_monotonic"] = bool(high_pred >= low_pred)

    if output_name == "Fragmentation":
        pf_name = "Powder factor (kg/m³)"
        if pf_name in feature_frame.columns:
            s = pd.to_numeric(feature_frame[pf_name], errors="coerce").dropna()
            if len(s):
                lo = float(s.quantile(0.2))
                hi = float(s.quantile(0.8))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    lo_vec = dict(ref)
                    hi_vec = dict(ref)
                    lo_vec[pf_name] = lo
                    hi_vec[pf_name] = hi
                    low_pred = predict_mode("candidate", lo_vec)
                    high_pred = predict_mode("candidate", hi_vec)
                    checks["powder_factor_reduces_size"] = bool(high_pred <= low_pred)

    if not checks:
        checks["basic_non_negative"] = True
    passed = all(bool(v) for v in checks.values())
    return {"passed": bool(passed), "checks": checks}


def _predict_ml_output(output_name: str, vals: dict[str, float]) -> float | None:
    import numpy as np

    model_map = {
        "Ground Vibration": _assets.mdl_ppv,
        "Airblast": _assets.mdl_air,
        "Fragmentation": _assets.mdl_frag,
    }
    mdl = model_map.get(output_name)
    if mdl is None or _assets.scaler is None:
        return None

    X = np.array([[vals.get(n, 0.0) for n in INPUT_LABELS]], dtype=float)
    Xs = _assets.scaler.transform(X)
    base_pred = _clamp_physical_output(output_name, float(mdl.predict(Xs)[0]))

    if _ml_model_choice.get(output_name) != "physics_hybrid":
        return base_pred
    residual = _ml_residual_models.get(output_name)
    if residual is None:
        return base_pred

    try:
        emp = _empirical_value_for_output(vals, output_name)
        pred = _clamp_physical_output(output_name, float(emp + residual.predict(Xs)[0]))
        return pred
    except Exception:
        return base_pred


def _maybe_train_models_from_default_dataset() -> None:
    try:
        df = _read_upload_df(None, DATASETS["combined"])
        _maybe_train_models_from_df(df)
    except Exception:
        return


def _maybe_prepare_physics_rollout_from_default_dataset() -> None:
    try:
        df = _read_upload_df(None, DATASETS["combined"])
        _maybe_prepare_physics_rollout_from_df(df)
    except Exception:
        return


def _maybe_prepare_physics_rollout_from_df(df) -> None:
    global _ml_rollout_state, _ml_residual_models, _ml_model_choice, _ml_rollout_cache_key
    if not (_assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air)):
        return

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    cols = list(df.columns)
    key = f"{len(df)}|{','.join(map(str, cols))}"
    if _ml_rollout_cache_key == key and _ml_rollout_state:
        return

    syn = _feature_map_synonyms()
    in_res = _resolve_map(df.columns, INPUT_LABELS, syn["inputs"])
    out_res = _resolve_map(df.columns, ["Fragmentation", "Ground Vibration", "Airblast"], syn["outputs"])
    name_mode_ok = (not any(v is None for v in in_res)) and (not any(v is None for v in out_res))
    if not name_mode_ok:
        return

    Xraw = df[in_res].copy()
    Yraw = df[out_res].copy()
    Xraw.columns = list(INPUT_LABELS)
    Yraw.columns = ["Fragmentation", "Ground Vibration", "Airblast"]
    Xnum = Xraw.apply(pd.to_numeric, errors="coerce")
    Ynum = Yraw.apply(pd.to_numeric, errors="coerce")
    work = Xnum.join(Ynum, how="inner").dropna()
    if work.empty:
        return

    Xdf = work.iloc[:, : Xraw.shape[1]].copy()
    Ydf = work.iloc[:, Xraw.shape[1] :].copy()
    X = Xdf.values

    model_map = {
        "Ground Vibration": _assets.mdl_ppv,
        "Airblast": _assets.mdl_air,
        "Fragmentation": _assets.mdl_frag,
    }
    rollout = {}
    residual_models = {}
    model_choice = {}

    for out_name in ["Fragmentation", "Ground Vibration", "Airblast"]:
        base = model_map.get(out_name)
        if base is None or out_name not in Ydf.columns:
            continue
        y = pd.to_numeric(Ydf[out_name], errors="coerce").values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if int(mask.sum()) < 10:
            continue
        Xc = X[mask]
        Xc_df = Xdf.loc[mask].reset_index(drop=True)
        yc = y[mask]
        if len(yc) >= 25:
            Xtr, Xte, ytr, yte, Xtr_df, Xte_df = train_test_split(
                Xc,
                yc,
                Xc_df,
                test_size=0.2,
                random_state=42,
            )
        else:
            Xtr, Xte, ytr, yte, Xtr_df, Xte_df = Xc, Xc, yc, yc, Xc_df, Xc_df

        Xte_s = _assets.scaler.transform(Xte)
        base_mae = float(mean_absolute_error(yte, base.predict(Xte_s)))
        rollout_item = {
            "active": "baseline",
            "baseline_mae": float(base_mae),
            "candidate_mae": None,
            "physics_checks": None,
            "candidate_ready": False,
            "reason": "Physics candidate unavailable or did not outperform baseline",
        }
        model_choice[out_name] = "baseline"

        emp_tr = _empirical_series_for_frame(Xtr_df, out_name)
        emp_te = _empirical_series_for_frame(Xte_df, out_name)
        ok_mask_tr = np.isfinite(emp_tr) & np.isfinite(ytr)
        ok_mask_te = np.isfinite(emp_te) & np.isfinite(yte)
        if int(ok_mask_tr.sum()) >= 10 and int(ok_mask_te.sum()) >= 3:
            res_model = RandomForestRegressor(n_estimators=400, random_state=42)
            Xtr_h = Xtr[ok_mask_tr]
            Xte_h = Xte[ok_mask_te]
            ytr_h = ytr[ok_mask_tr]
            yte_h = yte[ok_mask_te]
            emp_tr_h = emp_tr[ok_mask_tr]
            emp_te_h = emp_te[ok_mask_te]
            Xtr_hs = _assets.scaler.transform(Xtr_h)
            Xte_hs = _assets.scaler.transform(Xte_h)
            res_model.fit(Xtr_hs, ytr_h - emp_tr_h)
            cand_pred = emp_te_h + res_model.predict(Xte_hs)
            cand_pred = np.array([_clamp_physical_output(out_name, float(v)) for v in cand_pred], dtype=float)
            cand_mae = float(mean_absolute_error(yte_h, cand_pred))
            consistency = _check_physics_consistency(out_name, _assets.scaler, base, res_model, Xtr_df)
            improved = cand_mae <= (base_mae * 0.995)
            if improved and bool(consistency.get("passed")):
                model_choice[out_name] = "physics_hybrid"
                residual_models[out_name] = res_model
                rollout_item = {
                    "active": "physics_hybrid",
                    "baseline_mae": float(base_mae),
                    "candidate_mae": float(cand_mae),
                    "physics_checks": consistency.get("checks"),
                    "candidate_ready": True,
                    "reason": "Promoted: lower holdout MAE and passed physics consistency checks",
                }
            else:
                rollout_item = {
                    "active": "baseline",
                    "baseline_mae": float(base_mae),
                    "candidate_mae": float(cand_mae),
                    "physics_checks": consistency.get("checks"),
                    "candidate_ready": True,
                    "reason": "Not promoted: candidate did not beat baseline and/or failed checks",
                }
        rollout[out_name] = rollout_item

    _ml_rollout_state = rollout
    _ml_residual_models = residual_models
    _ml_model_choice = model_choice
    _ml_rollout_cache_key = key


def _maybe_train_models_from_df(df) -> None:
    """
    Train fallback ML models from a dataset if joblib assets are missing.
    """
    global _assets, _ml_cache_key, _ml_rollout_cache_key, _ml_rollout_state, _ml_residual_models, _ml_model_choice
    if _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        return

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    cols = list(df.columns)
    key = f"{len(df)}|{','.join(map(str, cols))}"
    if _ml_cache_key == key:
        return

    syn = _feature_map_synonyms()
    in_res = _resolve_map(df.columns, INPUT_LABELS, syn["inputs"])
    out_res = _resolve_map(df.columns, ["Fragmentation", "Ground Vibration", "Airblast"], syn["outputs"])
    name_mode_ok = (not any(v is None for v in in_res)) and (not any(v is None for v in out_res))

    if name_mode_ok:
        Xraw = df[in_res].copy()
        Yraw = df[out_res].copy()
        Xraw.columns = list(INPUT_LABELS)
        Yraw.columns = ["Fragmentation", "Ground Vibration", "Airblast"]
    else:
        num = df.select_dtypes(include=[np.number]).copy()
        if num.shape[1] < 4:
            return
        Xraw = num.iloc[:, : num.shape[1] - 3].copy()
        Yraw = num.iloc[:, num.shape[1] - 3 :].copy()

    Xnum = Xraw.apply(pd.to_numeric, errors="coerce")
    Ynum = Yraw.apply(pd.to_numeric, errors="coerce")
    work = Xnum.join(Ynum, how="inner").dropna()
    if work.empty:
        return

    Xdf = work.iloc[:, : Xraw.shape[1]].copy()
    Ydf = work.iloc[:, Xraw.shape[1] :].copy()
    X = Xdf.values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    mdl_frag = mdl_ppv = mdl_air = None
    rollout = {}
    residual_models = {}
    model_choice = {}
    for out_name in ["Fragmentation", "Ground Vibration", "Airblast"]:
        if out_name not in Ydf.columns:
            continue
        y = pd.to_numeric(Ydf[out_name], errors="coerce").values
        # Allow training on smaller datasets (preview/sample datasets can be small).
        if len(y) < 10:
            continue

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if int(mask.sum()) < 10:
            continue
        Xc = X[mask]
        Xc_df = Xdf.loc[mask].reset_index(drop=True)
        yc = y[mask]
        if len(yc) >= 25:
            Xtr, Xte, ytr, yte, Xtr_df, Xte_df = train_test_split(
                Xc,
                yc,
                Xc_df,
                test_size=0.2,
                random_state=42,
            )
        else:
            Xtr, Xte, ytr, yte, Xtr_df, Xte_df = Xc, Xc, yc, yc, Xc_df, Xc_df

        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        base = RandomForestRegressor(n_estimators=400, random_state=42)
        base.fit(Xtr_s, ytr)
        base_pred = base.predict(Xte_s)
        base_mae = float(mean_absolute_error(yte, base_pred))

        rollout_item = {
            "active": "baseline",
            "baseline_mae": float(base_mae),
            "candidate_mae": None,
            "physics_checks": None,
            "candidate_ready": False,
            "reason": "Physics candidate unavailable or did not outperform baseline",
        }
        active_model = base
        chosen = "baseline"
        residual_model = None

        if name_mode_ok:
            emp_tr = _empirical_series_for_frame(Xtr_df, out_name)
            emp_te = _empirical_series_for_frame(Xte_df, out_name)
            ok_mask_tr = np.isfinite(emp_tr) & np.isfinite(ytr)
            ok_mask_te = np.isfinite(emp_te) & np.isfinite(yte)
            if int(ok_mask_tr.sum()) >= 10 and int(ok_mask_te.sum()) >= 3:
                res_model = RandomForestRegressor(n_estimators=400, random_state=42)
                Xtr_h = Xtr[ok_mask_tr]
                Xte_h = Xte[ok_mask_te]
                ytr_h = ytr[ok_mask_tr]
                yte_h = yte[ok_mask_te]
                emp_tr_h = emp_tr[ok_mask_tr]
                emp_te_h = emp_te[ok_mask_te]
                Xtr_hs = scaler.transform(Xtr_h)
                Xte_hs = scaler.transform(Xte_h)
                res_model.fit(Xtr_hs, ytr_h - emp_tr_h)
                cand_pred = emp_te_h + res_model.predict(Xte_hs)
                cand_pred = np.array([_clamp_physical_output(out_name, float(v)) for v in cand_pred], dtype=float)
                cand_mae = float(mean_absolute_error(yte_h, cand_pred))
                consistency = _check_physics_consistency(out_name, scaler, base, res_model, Xtr_df)
                improved = cand_mae <= (base_mae * 0.995)
                if improved and bool(consistency.get("passed")):
                    active_model = base
                    chosen = "physics_hybrid"
                    residual_model = res_model
                    rollout_item = {
                        "active": "physics_hybrid",
                        "baseline_mae": float(base_mae),
                        "candidate_mae": float(cand_mae),
                        "physics_checks": consistency.get("checks"),
                        "candidate_ready": True,
                        "reason": "Promoted: lower holdout MAE and passed physics consistency checks",
                    }
                else:
                    rollout_item = {
                        "active": "baseline",
                        "baseline_mae": float(base_mae),
                        "candidate_mae": float(cand_mae),
                        "physics_checks": consistency.get("checks"),
                        "candidate_ready": True,
                        "reason": "Not promoted: candidate did not beat baseline and/or failed checks",
                    }

        mdl = active_model
        if residual_model is not None:
            residual_models[out_name] = residual_model
        model_choice[out_name] = chosen
        rollout[out_name] = rollout_item
        if out_name == "Fragmentation":
            mdl_frag = mdl
        elif out_name == "Ground Vibration":
            mdl_ppv = mdl
        elif out_name == "Airblast":
            mdl_air = mdl

    # Only "lock in" the cache key if we successfully trained at least one model.
    # (Otherwise we'd permanently skip retraining for this dataset shape.)
    if mdl_frag or mdl_ppv or mdl_air:
        _assets = LoadedAssets(scaler=scaler, mdl_frag=mdl_frag, mdl_ppv=mdl_ppv, mdl_air=mdl_air)
        _ml_rollout_state = rollout
        _ml_residual_models = residual_models
        _ml_model_choice = model_choice
        _ml_cache_key = key
        _ml_rollout_cache_key = key


def _feature_importance_df(df, top_k: int):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    syn = _feature_map_synonyms()
    in_res = _resolve_map(df.columns, INPUT_LABELS, syn["inputs"])
    out_res = _resolve_map(df.columns, ["Fragmentation", "Ground Vibration", "Airblast"], syn["outputs"])

    name_mode_ok = (not any(v is None for v in in_res)) and (not any(v is None for v in out_res))
    if name_mode_ok:
        Xraw = df[in_res].copy()
        Yraw = df[out_res].copy()
        note = "Mapped by column names/synonyms."
        mapping_mode = "names"
    else:
        if df.shape[1] < 4:
            return {"error": "Dataset needs at least 4 columns."}
        Xraw = df.iloc[:, : df.shape[1] - 3].copy()
        Yraw = df.iloc[:, df.shape[1] - 3 :].copy()
        note = "Name mapping failed — used positional split (inputs = first N−3, outputs = last 3)."
        mapping_mode = "position"

    Xnum = Xraw.copy()
    for c in Xnum.columns:
        Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce")
    Ynum = Yraw.copy()
    for c in Ynum.columns:
        Ynum[c] = pd.to_numeric(Ynum[c], errors="coerce")

    work = Xnum.join(Ynum, how="inner")
    if work.empty:
        return {"error": "No numeric rows available."}

    X = work.iloc[:, : Xraw.shape[1]].values
    Xdf = work.iloc[:, : Xraw.shape[1]].copy()
    ydf = work.iloc[:, Xraw.shape[1] :]
    out = {}
    perm_out = {}
    consensus_out = {}
    explainability = {}
    for out_name in ydf.columns:
        y = ydf[out_name].values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        Xc = X[mask]
        Xc_df = Xdf.loc[mask].reset_index(drop=True)
        yc = y[mask]
        if len(yc) >= 30:
            Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=42)
        else:
            Xtr, Xte, ytr, yte = Xc, Xc, yc, yc
        mdl = RandomForestRegressor(n_estimators=400, random_state=42)
        mdl.fit(Xtr, ytr)
        imp = list(mdl.feature_importances_)
        items = [
            {"feature": f, "importance": float(v)}
            for f, v in zip(list(Xraw.columns), imp)
        ]
        items.sort(key=lambda r: r["importance"], reverse=True)
        out[out_name] = items[: max(3, int(top_k))]
        pitems = []
        try:
            perm = permutation_importance(mdl, Xte, yte, n_repeats=8, random_state=42)
            pitems = [
                {
                    "feature": f,
                    "importance": float(v),
                    "std": float(s),
                }
                for f, v, s in zip(list(Xraw.columns), perm.importances_mean, perm.importances_std)
            ]
            pitems.sort(key=lambda r: r["importance"], reverse=True)
            perm_out[out_name] = pitems[: max(3, int(top_k))]
        except Exception:
            perm_out[out_name] = []

        rf_map = {it["feature"]: float(it["importance"]) for it in items}
        perm_map = {it["feature"]: float(abs(it["importance"])) for it in pitems}
        score_features = list(dict.fromkeys([*(it["feature"] for it in items), *(it["feature"] for it in pitems)]))
        rf_max = max([rf_map.get(f, 0.0) for f in score_features] or [1.0])
        perm_max = max([perm_map.get(f, 0.0) for f in score_features] or [1.0])
        consensus = []
        for feat in score_features:
            rf_score = rf_map.get(feat, 0.0) / (rf_max or 1.0)
            perm_score = perm_map.get(feat, 0.0) / (perm_max or 1.0)
            consensus.append(
                {
                    "feature": feat,
                    "score": float(0.55 * rf_score + 0.45 * perm_score),
                    "rf_importance": float(rf_map.get(feat, 0.0)),
                    "permutation_importance": float(perm_map.get(feat, 0.0)),
                }
            )
        consensus.sort(key=lambda r: r["score"], reverse=True)
        consensus_out[out_name] = consensus[: max(3, int(top_k))]

        pdp_source = perm_out[out_name] if perm_out[out_name] else out[out_name]
        top_features = [it["feature"] for it in pdp_source[: min(4, len(pdp_source))]]
        baseline = pd.DataFrame(Xte if len(Xte) else Xtr, columns=Xraw.columns)
        if len(baseline) > 200:
            baseline = baseline.sample(200, random_state=42)
        pdp = []
        for feat in top_features:
            s = pd.to_numeric(Xraw[feat], errors="coerce").dropna()
            if s.empty:
                continue
            lo = float(s.quantile(0.05))
            hi = float(s.quantile(0.95))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                continue
            xs = np.linspace(lo, hi, 24)
            ys = []
            for xv in xs:
                probe = baseline.copy()
                probe[feat] = float(xv)
                ys.append(float(np.mean(mdl.predict(probe.values))))
            pdp.append({"feature": feat, "xs": xs.tolist(), "ys": ys})

        local_effects = []
        baseline_pred = representative_pred = None
        representative_values = {}
        if len(Xc_df):
            ref = Xc_df.median(numeric_only=True).reindex(Xraw.columns).astype(float)
            scale = Xc_df.std(ddof=0).replace(0, 1).fillna(1.0)
            distances = (((Xc_df - ref) / scale) ** 2).sum(axis=1)
            rep_idx = int(distances.idxmin())
            representative = Xc_df.iloc[rep_idx].astype(float)
            representative_values = {col: float(representative[col]) for col in Xraw.columns}
            baseline_pred = float(mdl.predict(pd.DataFrame([ref], columns=Xraw.columns).values)[0])
            representative_pred = float(mdl.predict(pd.DataFrame([representative], columns=Xraw.columns).values)[0])
            for feat in top_features[: min(6, len(top_features))]:
                probe = ref.copy()
                probe[feat] = float(representative[feat])
                effect = float(mdl.predict(pd.DataFrame([probe], columns=Xraw.columns).values)[0] - baseline_pred)
                local_effects.append(
                    {
                        "feature": feat,
                        "effect": effect,
                        "abs_effect": abs(effect),
                        "value": float(representative[feat]),
                        "baseline_value": float(ref[feat]),
                    }
                )
            local_effects.sort(key=lambda r: abs(r["effect"]), reverse=True)

        explainability[out_name] = {
            "train_r2": _score_r2(ytr, mdl.predict(Xtr)),
            "test_r2": _score_r2(yte, mdl.predict(Xte)),
            "partial_dependence": pdp,
            "local_explanation": {
                "method": "SHAP-style local effect from a representative blast versus the dataset median baseline.",
                "baseline_prediction": baseline_pred,
                "representative_prediction": representative_pred,
                "representative_values": representative_values,
                "feature_impacts": local_effects,
            },
        }

    matrix_outputs = [name for name in ydf.columns if name in consensus_out]
    matrix_features = []
    for out_name in matrix_outputs:
        for item in consensus_out.get(out_name, [])[: min(5, len(consensus_out.get(out_name, [])))]:
            if item["feature"] not in matrix_features:
                matrix_features.append(item["feature"])
    matrix_features = matrix_features[:8]
    importance_matrix = None
    if matrix_outputs and matrix_features:
        importance_matrix = {
            "features": matrix_features,
            "outputs": matrix_outputs,
            "values": [
                [
                    float(
                        next(
                            (
                                item["score"]
                                for item in consensus_out.get(out_name, [])
                                if item["feature"] == feat
                            ),
                            0.0,
                        )
                    )
                    for out_name in matrix_outputs
                ]
                for feat in matrix_features
            ],
        }

    correlation_matrix = None
    if len(matrix_features) >= 2:
        corr = Xdf[matrix_features].corr().fillna(0.0)
        correlation_matrix = {
            "features": matrix_features,
            "values": [[float(corr.loc[row, col]) for col in matrix_features] for row in matrix_features],
        }

    return {
        "feature_importance": out,
        "permutation_importance": perm_out,
        "consensus_importance": consensus_out,
        "importance_matrix": importance_matrix,
        "correlation_matrix": correlation_matrix,
        "explainability": explainability,
        "note": note,
        "diagnostics": {
            "mapping_mode": mapping_mode,
            "rows_total": int(len(df)),
            "rows_used": int(work.shape[0]),
            "rows_dropped": int(len(df) - work.shape[0]),
        },
        "rows_used": int(work.shape[0]),
        "inputs": list(Xraw.columns),
        "outputs": list(ydf.columns),
        "top_k": max(3, int(top_k)),
    }


def _feature_pca_df(df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    syn = _feature_map_synonyms()
    in_res = _resolve_map(df.columns, INPUT_LABELS, syn["inputs"])
    name_mode_ok = not any(v is None for v in in_res)
    if name_mode_ok:
        Xraw = df[in_res].copy()
        note = "Mapped by column names/synonyms."
    else:
        if df.shape[1] < 4:
            return {"error": "Dataset needs at least 4 columns."}
        Xraw = df.iloc[:, : df.shape[1] - 3].copy()
        note = "Name mapping failed — used positional split (inputs = first N−3)."

    Xnum = Xraw.copy()
    for c in Xnum.columns:
        Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce")
    Xnum = Xnum.dropna()
    if len(Xnum) < 10:
        return {"error": "Not enough rows for PCA."}

    Xs = StandardScaler().fit_transform(Xnum.values)
    pca = PCA(n_components=min(5, Xs.shape[1]))
    comps = pca.fit_transform(Xs)
    loadings = pca.components_
    out = []
    for i in range(loadings.shape[0]):
        pairs = list(zip(Xraw.columns, loadings[i]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        out.append([{"feature": k, "loading": float(v)} for k, v in pairs[: min(10, len(pairs))]])

    points = comps[:, :2]
    if len(points) > 800:
        points = points[np.random.choice(len(points), 800, replace=False)]
    return {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_explained_variance": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
        "top_loadings": out,
        "points": [{"pc1": float(p[0]), "pc2": float(p[1])} for p in points],
        "note": note,
        "rows_used": int(len(Xnum)),
        "inputs": list(Xraw.columns),
    }

def _resolve_map(df_cols, expected, synonyms_map):
    import re

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    cols = list(df_cols)
    lower = {c.lower(): c for c in cols}
    norm = {_norm(c): c for c in cols}
    result = []
    for name in expected:
        if name in cols:
            result.append(name)
            continue
        if name.lower() in lower:
            result.append(lower[name.lower()])
            continue
        chosen = None
        syns = synonyms_map.get(name.lower(), [])
        for s in [name] + syns:
            if s in cols:
                chosen = s
                break
            if s.lower() in lower:
                chosen = lower[s.lower()]
                break
            if _norm(s) in norm:
                chosen = norm[_norm(s)]
                break
        result.append(chosen)
    return result


@app.get("/v1/feature/importance")
def feature_importance_dataset(top_k: int = 12, _token: str = Depends(require_auth)):
    df = _combined_df()
    out = _feature_importance_df(df, top_k)
    out["dataset"] = DATASETS["combined"]
    return out


@app.post("/v1/feature/importance")
def feature_importance_dataset_upload(
    file: UploadFile | None = File(default=None),
    top_k: int = Form(default=12),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    out = _feature_importance_df(df, top_k)
    out["dataset"] = file.filename if file is not None else DATASETS["combined"]
    return out


@app.get("/v1/feature/pca")
def feature_pca(_token: str = Depends(require_auth)):
    df = _combined_df()
    out = _feature_pca_df(df)
    out["dataset"] = DATASETS["combined"]
    return out


@app.post("/v1/feature/pca")
def feature_pca_upload(
    file: UploadFile | None = File(default=None),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    if file is None:
        return feature_pca(_token=_token)
    out = _feature_pca_df(df)
    out["dataset"] = file.filename
    return out


@app.post("/v1/flyrock/predict")
def flyrock_predict(
    file: UploadFile | None = File(default=None),
    inputs_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file, DATASETS["flyrock"])
    try:
        X, y, target_name = _prepare_flyrock_data(df)
    except Exception as e:
        return {"error": str(e)}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)
    score = float(rf.score(Xtr, ytr))
    test_score = float(rf.score(Xte, yte))

    stats = {}
    for c in X.columns:
        s = X[c].dropna()
        stats[c] = {
            "min": float(s.quantile(0.02)),
            "max": float(s.quantile(0.98)),
            "median": float(s.median()),
        }

    inputs = None
    if inputs_json:
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in X.columns}

    xstar = np.array([[float(inputs.get(c, stats[c]["median"])) for c in X.columns]], dtype=float)
    yhat = float(rf.predict(xstar)[0])

    # Empirical estimates (from local flyrock module)
    def _lookup(vals, syns, default=float("nan")):
        keys = list(vals.keys())
        for s in syns:
            ns = _flyrock_norm(s)
            if ns in vals:
                return float(vals[ns])
        for k in keys:
            if any(_flyrock_norm(s) in k for s in syns):
                return float(vals[k])
        return float(default)

    BURDEN_SYNS = ["burden", "b"]
    SPACING_SYNS = ["spacing", "s"]
    STEMMING_SYNS = ["stemming", "stem"]
    DIAM_SYNS = ["hole diameter", "diameter", "hole dia", "bh dia", "bit diameter", "drill diameter", "d"]
    BENCH_SYNS = ["bench height", "bench", "height", "h"]
    CHARGE_SYNS = ["charge per delay", "charge", "explosive mass", "charge per hole"]
    PF_SYNS = ["powder factor", "specific charge", "pf", "q"]
    ROCKD_SYNS = ["rock density", "density", "t/m3", "tm3", "rho"]
    SDOB_SYNS = ["sdob", "s/b", "spacing to burden", "spacing burden ratio"]
    LUNDBORG_SYNS = ["lundborg", "lundborg distance", "lundborg m"]

    def _derive_sdob(vals):
        sd = _lookup(vals, SDOB_SYNS, default=float("nan"))
        if np.isfinite(sd):
            return float(sd)
        stem = _lookup(vals, STEMMING_SYNS, default=float("nan"))
        qd = _lookup(vals, CHARGE_SYNS, default=float("nan"))
        if np.isfinite(stem) and np.isfinite(qd) and qd > 0:
            return float(stem / (qd ** (1.0 / 3.0)))
        return None

    def emp_lundborg_1981(vals):
        dmm = _lookup(vals, DIAM_SYNS, default=float("nan"))
        q = _lookup(vals, PF_SYNS, default=float("nan"))
        if not np.isfinite(dmm) or not np.isfinite(q):
            return float("nan")
        d_in = dmm / 25.4
        return float(max(0.0, 143.0 * d_in * (q - 0.2)))

    def emp_mckenzie_sdob(vals):
        dmm = _lookup(vals, DIAM_SYNS, default=float("nan"))
        rho = _lookup(vals, ROCKD_SYNS, default=2.6)
        sd = _derive_sdob(vals)
        if not np.isfinite(dmm) or sd is None or sd <= 0:
            return float("nan")
        return float(10.0 * (max(dmm, 0.0) ** 0.667) * (sd ** -2.167) * (rho / 2.6))

    def emp_lundborg_legacy(vals):
        dmm = _lookup(vals, DIAM_SYNS, default=float("nan"))
        if not np.isfinite(dmm):
            return float("nan")
        return float(30.745 * (max(dmm, 0.0) ** 0.66))

    vals_norm = {_flyrock_norm(k): v for k, v in inputs.items()}
    empirical = {
        "Lundborg_1981": emp_lundborg_1981(vals_norm),
        "McKenzie_SDoB": emp_mckenzie_sdob(vals_norm),
        "Lundborg_Legacy": emp_lundborg_legacy(vals_norm),
    }
    empirical_auto = None
    empirical_method = None
    for key, method in [
        ("Lundborg_1981", "Lundborg (1981): 143*d_in*(q-0.2)"),
        ("McKenzie_SDoB", "McKenzie/SDoB: 10*d_mm^0.667*SDoB^-2.167*(ρ/2.6)"),
        ("Lundborg_Legacy", "Legacy d-only: 30.745*d_mm^0.66"),
    ]:
        val = empirical.get(key)
        if val is not None and np.isfinite(val):
            empirical_auto = float(val)
            empirical_method = method
            break

    imp = list(rf.feature_importances_)
    feature_importance = [
        {"feature": f, "importance": float(v)}
        for f, v in sorted(zip(list(X.columns), imp), key=lambda p: p[1], reverse=True)
    ]

    return {
        "prediction": yhat,
        "train_r2": score,
        "test_r2": test_score,
        "features": list(X.columns),
        "feature_stats": stats,
        "feature_importance": feature_importance,
        "target_name": target_name,
        "empirical": empirical,
        "empirical_auto": empirical_auto,
        "empirical_method": empirical_method,
    }


@app.post("/v1/flyrock/surface")
def flyrock_surface(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    x_name = payload.get("x_name")
    y_name = payload.get("y_name")
    grid = int(payload.get("grid", 40))
    inputs_json = payload.get("inputs_json")

    df = _read_upload_df(None, DATASETS["flyrock"])
    try:
        X, y, _ = _prepare_flyrock_data(df)
    except Exception as e:
        return {"error": str(e)}

    if x_name not in X.columns or y_name not in X.columns or x_name == y_name:
        return {"error": "Invalid x_name/y_name."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)

    stats = {}
    for c in X.columns:
        s = X[c].dropna()
        stats[c] = {
            "min": float(s.quantile(0.02)),
            "max": float(s.quantile(0.98)),
            "median": float(s.median()),
        }

    inputs = None
    if isinstance(inputs_json, str):
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    elif isinstance(inputs_json, dict):
        inputs = inputs_json
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in X.columns}

    x_min, x_max = stats[x_name]["min"], stats[x_name]["max"]
    y_min, y_max = stats[y_name]["min"], stats[y_name]["max"]
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    XX, YY = np.meshgrid(xs, ys)
    base = np.array([[float(inputs.get(c, stats[c]["median"])) for c in X.columns]], dtype=float)
    G = XX.size
    DM = np.repeat(base, G, axis=0)
    ix = list(X.columns).index(x_name)
    iy = list(X.columns).index(y_name)
    DM[:, ix] = XX.ravel()
    DM[:, iy] = YY.ravel()
    Z = rf.predict(DM).reshape(XX.shape)

    return {
        "x_name": x_name,
        "y_name": y_name,
        "grid_x": xs.tolist(),
        "grid_y": ys.tolist(),
        "Z": Z.tolist(),
    }


@app.post("/v1/flyrock/surface/upload")
def flyrock_surface_upload(
    file: UploadFile | None = File(default=None),
    payload_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    payload = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
    x_name = payload.get("x_name")
    y_name = payload.get("y_name")
    grid = int(payload.get("grid", 40))
    inputs_json = payload.get("inputs_json")

    df = _read_upload_df(file, DATASETS["flyrock"])
    try:
        X, y, _ = _prepare_flyrock_data(df)
    except Exception as e:
        return {"error": str(e)}

    if x_name not in X.columns or y_name not in X.columns or x_name == y_name:
        return {"error": "Invalid x_name/y_name."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)

    stats = {}
    for c in X.columns:
        s = X[c].dropna()
        stats[c] = {
            "min": float(s.quantile(0.02)),
            "max": float(s.quantile(0.98)),
            "median": float(s.median()),
        }

    inputs = None
    if isinstance(inputs_json, str):
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    elif isinstance(inputs_json, dict):
        inputs = inputs_json
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in X.columns}

    x_min, x_max = stats[x_name]["min"], stats[x_name]["max"]
    y_min, y_max = stats[y_name]["min"], stats[y_name]["max"]
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    XX, YY = np.meshgrid(xs, ys)
    base = np.array([[float(inputs.get(c, stats[c]["median"])) for c in X.columns]], dtype=float)
    G = XX.size
    DM = np.repeat(base, G, axis=0)
    ix = list(X.columns).index(x_name)
    iy = list(X.columns).index(y_name)
    DM[:, ix] = XX.ravel()
    DM[:, iy] = YY.ravel()
    Z = rf.predict(DM).reshape(XX.shape)
    return {"x_name": x_name, "y_name": y_name, "grid_x": xs.tolist(), "grid_y": ys.tolist(), "Z": Z.tolist()}


@app.post("/v1/backbreak/predict")
def backbreak_predict(
    file: UploadFile | None = File(default=None),
    inputs_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np

    df = _read_upload_df(file, DATASETS["backbreak"])
    try:
        bundle = _prepare_backbreak_bundle(df)
    except Exception as e:
        return {"error": str(e)}
    model = bundle["model"]
    keep = bundle["keep"]
    stats = bundle["stats"]
    feat_importance = bundle["feat_importance"]
    train_r2 = bundle["train_r2"]
    test_r2 = bundle["test_r2"]

    inputs = None
    if inputs_json:
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    if not isinstance(inputs, dict):
        inputs = {k: stats[k]["median"] for k in keep}

    xstar = np.array([[float(inputs.get(c, stats[c]["median"])) for c in keep]], dtype=float)
    yhat = float(model.predict(xstar)[0])

    return {
        "prediction": yhat,
        "features": keep,
        "feature_stats": stats,
        "feature_importance": feat_importance,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


@app.post("/v1/backbreak/surface")
def backbreak_surface(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    import json
    import numpy as np

    x_name = payload.get("x_name")
    y_name = payload.get("y_name")
    grid = int(payload.get("grid", 40))
    inputs_json = payload.get("inputs_json")

    df = _read_upload_df(None, DATASETS["backbreak"])
    try:
        bundle = _prepare_backbreak_bundle(df)
    except Exception as e:
        return {"error": str(e)}

    model = bundle["model"]
    keep = bundle["keep"]
    stats = bundle["stats"]

    if not x_name:
        x_name = keep[0]
    if not y_name:
        y_name = keep[1] if len(keep) > 1 else keep[0]
    if x_name not in keep or y_name not in keep or x_name == y_name:
        return {"error": "Invalid x_name/y_name."}

    inputs = None
    if isinstance(inputs_json, str):
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    elif isinstance(inputs_json, dict):
        inputs = inputs_json
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in keep}

    x_min, x_max = stats[x_name]["min"], stats[x_name]["max"]
    y_min, y_max = stats[y_name]["min"], stats[y_name]["max"]
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    XX, YY = np.meshgrid(xs, ys)
    base = np.array([[float(inputs.get(c, stats[c]["median"])) for c in keep]], dtype=float)
    G = XX.size
    DM = np.repeat(base, G, axis=0)
    ix = keep.index(x_name)
    iy = keep.index(y_name)
    DM[:, ix] = XX.ravel()
    DM[:, iy] = YY.ravel()
    Z = model.predict(DM).reshape(XX.shape)
    return {"x_name": x_name, "y_name": y_name, "grid_x": xs.tolist(), "grid_y": ys.tolist(), "Z": Z.tolist()}


@app.post("/v1/backbreak/surface/upload")
def backbreak_surface_upload(
    file: UploadFile | None = File(default=None),
    payload_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np

    payload = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
    x_name = payload.get("x_name")
    y_name = payload.get("y_name")
    grid = int(payload.get("grid", 40))
    inputs_json = payload.get("inputs_json")

    df = _read_upload_df(file, DATASETS["backbreak"])
    try:
        bundle = _prepare_backbreak_bundle(df)
    except Exception as e:
        return {"error": str(e)}

    model = bundle["model"]
    keep = bundle["keep"]
    stats = bundle["stats"]

    if not x_name:
        x_name = keep[0]
    if not y_name:
        y_name = keep[1] if len(keep) > 1 else keep[0]
    if x_name not in keep or y_name not in keep or x_name == y_name:
        return {"error": "Invalid x_name/y_name."}

    inputs = None
    if isinstance(inputs_json, str):
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    elif isinstance(inputs_json, dict):
        inputs = inputs_json
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in keep}

    x_min, x_max = stats[x_name]["min"], stats[x_name]["max"]
    y_min, y_max = stats[y_name]["min"], stats[y_name]["max"]
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    XX, YY = np.meshgrid(xs, ys)
    base = np.array([[float(inputs.get(c, stats[c]["median"])) for c in keep]], dtype=float)
    G = XX.size
    DM = np.repeat(base, G, axis=0)
    ix = keep.index(x_name)
    iy = keep.index(y_name)
    DM[:, ix] = XX.ravel()
    DM[:, iy] = YY.ravel()
    Z = model.predict(DM).reshape(XX.shape)
    return {"x_name": x_name, "y_name": y_name, "grid_x": xs.tolist(), "grid_y": ys.tolist(), "Z": Z.tolist()}


@app.post("/v1/slope/predict")
def slope_predict(
    file: UploadFile | None = File(default=None),
    inputs_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file, DATASETS["slope"])
    try:
        prepared = _prepare_slope_df(df)
    except Exception as e:
        return {"error": str(e)}

    X = prepared[["H_m", "beta_deg", "c_kPa", "phi_deg", "gamma_kN_m3", "ru"]]
    y = prepared["y"]
    if y.nunique() != 2 or len(y) < 30:
        return {"error": "Dataset must contain both classes and >=30 valid rows."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)),
        ]
    ).fit(Xtr, ytr)

    stats = {}
    for c in X.columns:
        s = X[c].dropna()
        stats[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())}

    inputs = None
    if inputs_json:
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            inputs = None
    if not isinstance(inputs, dict):
        inputs = {c: stats[c]["median"] for c in X.columns}

    xstar = np.array([[float(inputs.get(c, stats[c]["median"])) for c in X.columns]])
    prob = float(np.clip(pipe.predict_proba(xstar)[0, 1], 0.0, 1.0))
    train_acc = float(pipe.score(Xtr, ytr))
    test_acc = float(pipe.score(Xte, yte))
    predicted_class = "stable" if prob >= 0.5 else "failure"
    class_balance = {
        "stable": int((prepared["status"] == "stable").sum()),
        "failure": int((prepared["status"] == "failure").sum()),
    }

    return {
        "prob_stable": prob,
        "prediction": prob,
        "predicted_class": predicted_class,
        "feature_stats": stats,
        "features": list(X.columns),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "class_balance": class_balance,
    }


@app.post("/v1/delay/predict")
def delay_predict(
    file: UploadFile | None = File(default=None),
    _token: str = Depends(require_auth),
):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    def _standardize_delay_df(df_in):
        cols = {c.lower().strip(): c for c in df_in.columns}

        def _find(name, aliases):
            if name.lower() in cols:
                return cols[name.lower()]
            for alias in aliases:
                if alias.lower() in cols:
                    return cols[alias.lower()]
            return None

        rename = {}
        if "x" not in cols or "y" not in cols:
            raise ValueError("CSV must include X and Y columns.")
        rename[cols["x"]] = "X"
        rename[cols["y"]] = "Y"
        opt = {
            "HoleID": ["holeid", "hole id", "id", "hole", "hole_no", "hole number", "hole_number"],
            "Depth": ["depth", "hole_depth", "hole depth (m)", "hole_depth_m"],
            "Charge": ["charge", "explosive mass", "explosive_mass", "charge_kg"],
            "Z": ["z", "rl", "elev", "elevation"],
            "Delay": ["delay", "delay_ms", "predicted delay (ms)", "predicted_delay_ms", "time_ms"],
        }
        for std, aliases in opt.items():
            col = _find(std, aliases)
            if col is not None:
                rename[col] = std
        out = df_in.rename(columns=rename)
        if "HoleID" not in out.columns and len(df_in.columns):
            first_col = df_in.columns[0]
            first_low = str(first_col).strip().lower()
            if first_col not in rename:
                s = df_in[first_col]
                numeric_ratio = float(pd.to_numeric(s, errors="coerce").notna().mean()) if len(s) else 0.0
                unique_ratio = float(s.astype(str).nunique(dropna=True) / max(1, len(s)))
                if "hole" in first_low or "id" in first_low or numeric_ratio < 0.95 or unique_ratio > 0.94:
                    out = out.rename(columns={first_col: "HoleID"})
        return out

    def _rank01(vals):
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        if n <= 1:
            return np.zeros(n, dtype=float)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, n)
        return ranks

    def _nearest_spacing(dist_matrix):
        if len(dist_matrix) <= 1:
            return np.zeros(len(dist_matrix), dtype=float)
        return np.min(dist_matrix, axis=1)

    def _infer_initiation_pattern(df_in, span_x, span_y):
        import re

        for col in ["InitiationPattern", "Pattern", "pattern", "initiation_pattern"]:
            if col in df_in.columns and len(df_in[col].dropna()):
                raw = str(df_in[col].dropna().iloc[0]).strip().lower()
                if "diamond" in raw:
                    return "diamond"
                if "chevron" in raw or "v-cut" in raw or "v cut" in raw:
                    return "chevron"
                if "line" in raw or "row" in raw:
                    return "line"
        if "HoleID" in df_in.columns:
            vals = df_in["HoleID"].astype(str).tolist()
            letters = set()
            numbers = set()
            parsed = 0
            for v in vals:
                m = re.match(r"^\s*([A-Za-z]+)\s*[-_/]?\s*(\d+)\s*$", v)
                if not m:
                    continue
                parsed += 1
                letters.add(m.group(1).upper())
                numbers.add(int(m.group(2)))
            if parsed >= max(8, int(0.6 * len(vals))) and len(letters) >= 4 and len(numbers) >= 4:
                return "line"
        # Elongated / strip benches: echelon "line" (column by column) matches field isochron figures;
        # reserve diamond for nearly square patterns, not chevron-by-default.
        aspect = min(span_x, span_y) / max(1e-9, max(span_x, span_y))
        if aspect >= 0.78:
            return "diamond"
        return "line"

    def _build_physics_sequence(df_in, ml_delay, observed_step_ms):
        x = pd.to_numeric(df_in["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df_in["Y"], errors="coerce").to_numpy(dtype=float)
        n = len(df_in)

        pts = np.column_stack([x, y])
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_matrix = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_spacing = _nearest_spacing(dist_matrix)
        nearest_spacing = np.where(np.isfinite(nearest_spacing), nearest_spacing, np.nanmedian(nearest_spacing[np.isfinite(nearest_spacing)]) if np.isfinite(nearest_spacing).any() else 1.0)

        span_x = float(np.nanmax(x) - np.nanmin(x)) if n else 1.0
        span_y = float(np.nanmax(y) - np.nanmin(y)) if n else 1.0
        use_x = span_x >= span_y
        primary = x if use_x else y
        secondary = y if use_x else x

        pmin = float(np.nanmin(primary))
        pmax = float(np.nanmax(primary))
        q_lo = float(np.nanquantile(primary, 0.15))
        q_hi = float(np.nanquantile(primary, 0.85))
        lo_mask = primary <= q_lo
        hi_mask = primary >= q_hi
        lo_open = float(np.nanmean(nearest_spacing[lo_mask])) if np.any(lo_mask) else 0.0
        hi_open = float(np.nanmean(nearest_spacing[hi_mask])) if np.any(hi_mask) else 0.0
        face_at_min = lo_open >= hi_open
        face_distance = primary - pmin if face_at_min else pmax - primary
        face_rank = _rank01(face_distance)

        pattern = _infer_initiation_pattern(df_in, span_x, span_y)
        sec_center = np.nanmedian(secondary)
        sec_spread = max(1e-9, float(np.nanstd(secondary)))
        lateral = np.abs((secondary - sec_center) / sec_spread)
        radial = np.hypot((x - np.nanmedian(x)) / max(1e-9, span_x), (y - np.nanmedian(y)) / max(1e-9, span_y))
        if pattern == "diamond":
            pattern_rank = _rank01(radial)
        elif pattern == "line":
            pattern_rank = np.zeros(n, dtype=float)
        else:
            pattern_rank = _rank01(lateral)

        charge = pd.to_numeric(df_in.get("Charge", pd.Series(np.nan, index=df_in.index)), errors="coerce").to_numpy(dtype=float)
        depth = pd.to_numeric(df_in.get("Depth", pd.Series(np.nan, index=df_in.index)), errors="coerce").to_numpy(dtype=float)
        charge = np.where(np.isfinite(charge), charge, np.nanmedian(charge[np.isfinite(charge)]) if np.isfinite(charge).any() else 0.0)
        depth = np.where(np.isfinite(depth), depth, np.nanmedian(depth[np.isfinite(depth)]) if np.isfinite(depth).any() else 0.0)

        ml_rank = _rank01(ml_delay)
        burden_step = float(np.nanmedian(nearest_spacing)) if np.isfinite(np.nanmedian(nearest_spacing)) else 1.0
        burden_step = float(np.clip(burden_step, 0.8, 24.0))
        row_index = np.floor(face_distance / max(0.8, burden_step * 0.9)).astype(int)
        row_index = row_index - int(np.min(row_index))

        def _chevron_row_order(idxs):
            if len(idxs) <= 2:
                return list(idxs)
            sorted_by_center = sorted(idxs.tolist(), key=lambda ii: (abs(float(secondary[ii] - sec_center)), float(ml_rank[ii])))
            left, right, center = [], [], []
            for ii in sorted_by_center:
                delta = float(secondary[ii] - sec_center)
                if abs(delta) < 1e-9:
                    center.append(ii)
                elif delta < 0:
                    left.append(ii)
                else:
                    right.append(ii)
            out = list(center)
            for i in range(max(len(left), len(right))):
                if i < len(right):
                    out.append(right[i])
                if i < len(left):
                    out.append(left[i])
            return out

        def _line_order_echelon_by_geometry():
            """Firing order matching echelon isochron plots: left column then next to the right; within
            each column, top to bottom (decreasing Y when Y increases with northing 'up' on the plan)."""
            xmin = float(np.nanmin(x))
            col_pitch = max(1.0, float(np.nanmedian(nearest_spacing[np.isfinite(nearest_spacing)])) * 0.85)
            col_index = np.round((x - xmin) / col_pitch).astype(int)
            # Left to right: smaller X (more negative) first, matching a bench advancing along strike.
            ordered_cols = sorted(np.unique(col_index).tolist())
            out = []
            for ci in ordered_cols:
                idxs = np.where(col_index == ci)[0]
                # Top -> bottom in the column: highest Y first (plan 'top' / northing if Y is north).
                row_order = sorted(
                    idxs.tolist(),
                    key=lambda ii: (-float(y[ii]), float(secondary[ii]), float(ml_rank[ii])),
                )
                out.extend(row_order)
            if not out:
                return list(range(n))
            return out

        order_list = []
        if pattern == "line":
            order_list = _line_order_echelon_by_geometry()
        else:
            for r in np.sort(np.unique(row_index)):
                idxs = np.where(row_index == r)[0]
                if pattern == "chevron":
                    row_order = _chevron_row_order(idxs)
                elif pattern == "diamond":
                    row_order = sorted(
                        idxs.tolist(),
                        key=lambda ii: (
                            abs(float(face_distance[ii] - np.nanmedian(face_distance)))
                            + abs(float(secondary[ii] - sec_center)),
                            float(ml_rank[ii]),
                        ),
                    )
                else:
                    row_order = sorted(idxs.tolist(), key=lambda ii: (float(secondary[ii]), float(ml_rank[ii])))
                order_list.extend(row_order)
        order = np.asarray(order_list, dtype=int)
        sequence_rank = np.empty(n, dtype=int)
        sequence_rank[order] = np.arange(1, n + 1)

        # BME timing guideline anchors:
        # Tmin rock-response guidance typically ranges ~4-8 ms per metre burden.
        # Use a conservative in-range default and scale by local burden proxy.
        burden_proxy = np.maximum(nearest_spacing, 2.0)
        tmin_ms_per_m = 6.0
        tmin_local = burden_proxy * tmin_ms_per_m

        step_base = float(observed_step_ms) if np.isfinite(observed_step_ms) and observed_step_ms > 0 else float(np.nanmedian(tmin_local) * 2.0)
        step_base = float(np.clip(step_base, 12.0, 67.0))
        wave_velocity_m_per_ms = 3.8  # ~3800 m/s shock-wave proxy in competent rock
        delays = np.zeros(n, dtype=float)
        propagation_gap_ms = np.zeros(n, dtype=float)
        interaction_index = np.zeros(n, dtype=float)
        ml_min = float(np.nanmin(ml_delay)) if len(ml_delay) and np.isfinite(np.nanmin(ml_delay)) else 0.0
        start_delay = max(0.0, ml_min)

        if n:
            first_idx = int(order[0])
            delays[first_idx] = start_delay
            propagation_gap_ms[first_idx] = step_base
            interaction_index[first_idx] = 0.0

        for pos in range(1, n):
            idx = int(order[pos])
            prev = order[:pos]
            near = float(np.min(dist_matrix[idx, prev])) if len(prev) else 0.0
            interaction_floor = 5.0 + (near / wave_velocity_m_per_ms)
            # Keep delay >= Tmin-local and >= interaction floor for burden relief.
            local_floor = max(float(tmin_local[idx]), interaction_floor)
            gap = max(step_base, local_floor)
            prev_idx = int(order[pos - 1])
            delays[idx] = delays[prev_idx] + gap
            propagation_gap_ms[idx] = gap
            interaction_index[idx] = interaction_floor / max(gap, 1e-6)

        # Keep strict uniqueness with integer-ms timing.
        sorted_delay = np.round(delays[order]).astype(float)
        for i in range(1, len(sorted_delay)):
            if sorted_delay[i] <= sorted_delay[i - 1]:
                sorted_delay[i] = sorted_delay[i - 1] + 1.0
        delays[order] = sorted_delay

        # BME guide: scaled burden Bs = B / sqrt(Mc), with risk when Bs < 0.7.
        # Approximate B with nearest spacing and Mc with charge-per-metre column.
        powder_factor = charge / np.maximum(depth, 0.5)
        charge_per_m = np.maximum(charge / np.maximum(depth, 0.5), 1e-6)
        scaled_burden = burden_proxy / np.sqrt(charge_per_m)
        scaled_burden_risk = np.maximum(0.0, (0.7 - scaled_burden) / 0.7)
        confinement = 1.0 / np.maximum(nearest_spacing, 2.0)
        c_med = float(np.nanmedian(charge))
        c_iqr = max(1e-6, float(np.nanquantile(charge, 0.75) - np.nanquantile(charge, 0.25)))
        charge_norm = np.clip((charge - c_med) / c_iqr, -2.0, 3.0)
        flyrock_distance = 110.0 + 55.0 * scaled_burden_risk + 18.0 * confinement + 12.0 * np.maximum(charge_norm, 0.0)
        flyrock_distance = np.clip(flyrock_distance, 20.0, 700.0)
        front_face = face_distance <= float(np.nanquantile(face_distance, 0.35))
        high_pf = powder_factor >= float(np.nanquantile(powder_factor, 0.9))
        flyrock_risk = ((scaled_burden < 0.7) & front_face) | ((scaled_burden < 0.8) & high_pf) | (flyrock_distance >= 185.0)

        return {
            "delay": delays,
            "sequence_rank": sequence_rank,
            "row_index": row_index,
            "propagation_gap_ms": propagation_gap_ms,
            "interaction_index": interaction_index,
            "flyrock_distance": flyrock_distance,
            "flyrock_risk": flyrock_risk,
            "scaled_burden": scaled_burden,
            "pattern": pattern,
            "face_axis": "X" if use_x else "Y",
            "face_side": "min" if face_at_min else "max",
            "base_step_ms": step_base,
            "tmin_ms_per_m": tmin_ms_per_m,
            "nearest_spacing": nearest_spacing,
        }

    def _estimate_delay_outputs_with_combined_benchmark(dfv_in, physics_in):
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        try:
            cdf = _read_upload_df(None, DATASETS["combined"])
            syn = _feature_map_synonyms()
            in_res = _resolve_map(cdf.columns, INPUT_LABELS, syn["inputs"])
            out_names = ["Fragmentation", "Ground Vibration", "Airblast"]
            out_res = _resolve_map(cdf.columns, out_names, syn["outputs"])
            if any(v is None for v in in_res) or any(v is None for v in out_res):
                return None, {"available": False, "reason": "combined mapping unavailable"}

            Xraw = cdf[in_res].copy()
            Yraw = cdf[out_res].copy()
            Xraw.columns = list(INPUT_LABELS)
            Yraw.columns = out_names
            Xnum = Xraw.apply(pd.to_numeric, errors="coerce")
            Ynum = Yraw.apply(pd.to_numeric, errors="coerce")
            work = Xnum.join(Ynum, how="inner").dropna()
            if len(work) < 50:
                return None, {"available": False, "reason": "combined dataset too small"}

            xcols = list(INPUT_LABELS)
            X = work[xcols]
            Y = work[out_names]
            med = {c: float(X[c].median()) for c in xcols}
            bounds = {c: (float(X[c].quantile(0.02)), float(X[c].quantile(0.98))) for c in xcols}

            models = {}
            scores = {}
            for out_name in out_names:
                xtr, xte, ytr, yte = train_test_split(X, Y[out_name], test_size=0.2, random_state=42)
                mdl = RandomForestRegressor(n_estimators=300, random_state=42).fit(xtr.values, ytr.values)
                models[out_name] = mdl
                scores[out_name] = float(mdl.score(xte.values, yte.values))

            n = len(dfv_in)
            depth = pd.to_numeric(dfv_in.get("Depth", pd.Series(np.nan, index=dfv_in.index)), errors="coerce").fillna(med.get("Hole depth (m)", 10.0)).to_numpy(float)
            charge = pd.to_numeric(dfv_in.get("Charge", pd.Series(np.nan, index=dfv_in.index)), errors="coerce").fillna(med.get("Explosive mass (kg)", 50.0)).to_numpy(float)
            spacing = np.maximum(np.asarray(physics_in.get("nearest_spacing", np.ones(n)), dtype=float), 0.6)
            burden = np.maximum(spacing * 0.9, 0.5)
            stem = np.maximum(0.7 * depth, 0.2)
            linear_charge = charge / np.maximum(depth, 0.5)
            blast_volume = np.maximum(depth * burden * spacing, 0.5)
            powder_factor = charge / np.maximum(blast_volume, 0.5)
            distance = np.full(n, med.get("Distance (m)", 300.0), dtype=float)
            rock_density = np.full(n, med.get("Rock density (t/m³)", 2.6), dtype=float)
            hole_dia = np.full(n, med.get("Hole diameter (mm)", 165.0), dtype=float)

            rows = []
            for i in range(n):
                row = {
                    "Hole depth (m)": float(depth[i]),
                    "Hole diameter (mm)": float(hole_dia[i]),
                    "Burden (m)": float(burden[i]),
                    "Spacing (m)": float(spacing[i]),
                    "Stemming (m)": float(stem[i]),
                    "Distance (m)": float(distance[i]),
                    "Powder factor (kg/m³)": float(powder_factor[i]),
                    "Rock density (t/m³)": float(rock_density[i]),
                    "Linear charge (kg/m)": float(linear_charge[i]),
                    "Explosive mass (kg)": float(charge[i]),
                    "Blast volume (m³)": float(blast_volume[i]),
                    "# holes": 1.0,
                }
                for c in xcols:
                    lo, hi = bounds[c]
                    row[c] = float(np.clip(row.get(c, med[c]), lo, hi))
                rows.append(row)

            Xdelay = pd.DataFrame(rows)[xcols].values
            frag_ml = models["Fragmentation"].predict(Xdelay)
            gv_ml = models["Ground Vibration"].predict(Xdelay)
            ab_ml = models["Airblast"].predict(Xdelay)

            frag_emp = []
            gv_emp = []
            ab_emp = []
            for row in rows:
                vals = {k: float(v) for k, v in row.items()}
                vals["HPD_override"] = 1.0
                frag_emp.append(_empirical_value_for_output(vals, "Fragmentation"))
                gv_emp.append(_empirical_value_for_output(vals, "Ground Vibration"))
                ab_emp.append(_empirical_value_for_output(vals, "Airblast"))

            frag = 0.55 * np.asarray(frag_ml, dtype=float) + 0.45 * np.asarray(frag_emp, dtype=float)
            gv = 0.55 * np.asarray(gv_ml, dtype=float) + 0.45 * np.asarray(gv_emp, dtype=float)
            ab = 0.55 * np.asarray(ab_ml, dtype=float) + 0.45 * np.asarray(ab_emp, dtype=float)

            # BME guideline PPV estimate: C = 1143 * (R/sqrt(W))^-1.65
            w = np.maximum(charge, 1e-3)
            r = np.maximum(distance, 1.0)
            ppv_bme = 1143.0 * np.power(r / np.sqrt(w), -1.65)

            out_df = dfv_in.copy()
            out_df["Fragmentation"] = np.asarray([_clamp_physical_output("Fragmentation", float(v)) for v in frag], dtype=float)
            out_df["GroundVibration"] = np.asarray([_clamp_physical_output("Ground Vibration", float(v)) for v in gv], dtype=float)
            out_df["Airblast"] = np.asarray([_clamp_physical_output("Airblast", float(v)) for v in ab], dtype=float)
            out_df["PPV_BME"] = np.asarray([max(0.001, float(v)) for v in ppv_bme], dtype=float)

            summary = {
                "available": True,
                "combined_rows_used": int(len(work)),
                "ml_r2": {k: float(v) for k, v in scores.items()},
                "means": {
                    "fragmentation": float(np.nanmean(out_df["Fragmentation"])),
                    "ground_vibration": float(np.nanmean(out_df["GroundVibration"])),
                    "airblast": float(np.nanmean(out_df["Airblast"])),
                    "ppv_bme": float(np.nanmean(out_df["PPV_BME"])),
                },
            }
            return out_df, summary
        except Exception as e:
            return None, {"available": False, "reason": str(e)}

    try:
        dataset_used = DATASETS["delay_v1"]
        train_df = _standardize_delay_df(_read_upload_df(None, DATASETS["delay_v1"]))
    except Exception:
        dataset_used = DATASETS["delay_v2"]
        train_df = _standardize_delay_df(_read_upload_df(None, DATASETS["delay_v2"]))
    infer_df = _standardize_delay_df(_read_upload_df(file, DATASETS["delay_v1"])) if file is not None else train_df.copy()

    train_keep = ["Depth", "Charge", "X", "Y"] + (["Z"] if "Z" in train_df.columns else [])
    train_clean = train_df.dropna(subset=train_keep).copy()
    feature_cols = [c for c in ["Depth", "Charge", "X", "Y", "Z"] if c in train_clean.columns]
    X_train = train_clean[feature_cols].apply(pd.to_numeric, errors="coerce").values
    if "Delay" in train_clean.columns:
        y_train_full = pd.to_numeric(train_clean["Delay"], errors="coerce").values
        target_source = "observed_delay"
    else:
        y_train_full = np.clip(10 + 0.02 * X_train[:, 0] + 0.0005 * X_train[:, 2], 5, 250)
        target_source = "synthetic_fallback"

    Xtr, Xte, ytr, yte = train_test_split(X_train, y_train_full, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mdl = RandomForestRegressor(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)

    infer_keep = ["Depth", "Charge", "X", "Y"] + (["Z"] if "Z" in infer_df.columns and "Z" in train_clean.columns else [])
    infer_clean = infer_df.dropna(subset=infer_keep).copy()
    if len(infer_clean) > 2000:
        infer_clean = infer_clean.sample(2000, random_state=42).copy()
    X_infer = infer_clean[[c for c in ["Depth", "Charge", "X", "Y", "Z"] if c in infer_clean.columns and c in train_clean.columns]].apply(pd.to_numeric, errors="coerce").values
    yhat = mdl.predict(sc.transform(X_infer))
    observed_delay = pd.to_numeric(train_clean["Delay"], errors="coerce") if "Delay" in train_clean.columns else pd.Series(dtype=float)
    observed_unique = np.sort(observed_delay.dropna().unique()) if len(observed_delay) else np.array([])
    observed_step = float(np.median(np.diff(observed_unique))) if len(observed_unique) > 1 else float("nan")
    physics = _build_physics_sequence(infer_clean, yhat, observed_step)
    dfv = pd.DataFrame(
        {
            "X": pd.to_numeric(infer_clean["X"], errors="coerce"),
            "Y": pd.to_numeric(infer_clean["Y"], errors="coerce"),
            "Delay": physics["delay"],
        }
    )
    if "Delay" in infer_clean.columns:
        dfv["ActualDelay"] = pd.to_numeric(infer_clean["Delay"], errors="coerce")
    if "Depth" in infer_clean.columns:
        dfv["Depth"] = pd.to_numeric(infer_clean["Depth"], errors="coerce")
    if "Charge" in infer_clean.columns:
        dfv["Charge"] = pd.to_numeric(infer_clean["Charge"], errors="coerce")
    if "Z" in infer_clean.columns:
        dfv["Z"] = pd.to_numeric(infer_clean["Z"], errors="coerce")
    if "HoleID" in infer_clean.columns:
        hole_ids = infer_clean["HoleID"].astype(str).str.strip()
        invalid = hole_ids.eq("") | hole_ids.str.lower().isin(["nan", "none", "null"])
        if invalid.any():
            fallback = pd.Series([f"H{idx + 1}" for idx in range(len(hole_ids))], index=hole_ids.index)
            hole_ids = hole_ids.where(~invalid, fallback)
        dfv["HoleID"] = hole_ids.values
    else:
        dfv["HoleID"] = [f"H{idx + 1}" for idx in range(len(dfv))]
    dfv["SequenceRank"] = physics["sequence_rank"].astype(int)
    dfv["RowIndex"] = physics["row_index"].astype(int)
    dfv["PropagationGapMs"] = physics["propagation_gap_ms"]
    dfv["InteractionIndex"] = physics["interaction_index"]
    dfv["NearestSpacing"] = physics["nearest_spacing"]
    dfv["ScaledBurden"] = physics["scaled_burden"]
    dfv["FlyrockDistance"] = physics["flyrock_distance"]
    dfv["FlyrockRisk"] = physics["flyrock_risk"]
    dfv["InitiationPattern"] = physics["pattern"]

    enriched_dfv, blast_quality = _estimate_delay_outputs_with_combined_benchmark(dfv, physics)
    if enriched_dfv is not None:
        dfv = enriched_dfv

    dfv = dfv.dropna(subset=["X", "Y", "Delay"])

    # limit to 2000 points for payload size
    if len(dfv) > 2000:
        dfv = dfv.sample(2000, random_state=42)

    uniq_delays = pd.Series(dfv["Delay"]).nunique(dropna=True)
    uniq_ratio = float(uniq_delays / max(1, len(dfv)))

    return {
        "points": dfv.to_dict(orient="records"),
        "train_r2": _score_r2(ytr, mdl.predict(sc.transform(Xtr))),
        "test_r2": _score_r2(yte, mdl.predict(sc.transform(Xte))),
        "training_rows": int(len(train_clean)),
        "predicted_rows": int(len(dfv)),
        "dataset_used": dataset_used,
        "features_used": feature_cols,
        "target_source": target_source,
        "actual_delay_available": bool("ActualDelay" in dfv.columns),
        "initiation_pattern": physics["pattern"],
        "blast_face_axis": physics["face_axis"],
        "blast_face_side": physics["face_side"],
        "base_step_ms": physics["base_step_ms"],
        "tmin_ms_per_m": physics["tmin_ms_per_m"],
        "physics_benchmark": {
            "delay_uniqueness_ratio": uniq_ratio,
            "flyrock_risk_holes": int(np.sum(np.asarray(dfv["FlyrockRisk"], dtype=bool))),
            "flyrock_max_distance": float(np.max(dfv["FlyrockDistance"])) if len(dfv) else 0.0,
        },
        "blast_quality": blast_quality,
    }


def _combined_df():
    return _read_upload_df(None, DATASETS["combined"])


def _split_inputs_outputs(df):
    import numpy as np
    import pandas as pd

    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 4:
        raise ValueError("Dataset must have at least 4 numeric columns.")
    outputs = list(num.columns[-3:])
    inputs = [c for c in num.columns if c not in outputs]
    return num, inputs, outputs


def _fit_param_surrogate(df, cache_key: str | None = None):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler

    global _param_cache_key, _param_cache_bundle
    if cache_key and _param_cache_key == cache_key and _param_cache_bundle is not None:
        return _param_cache_bundle

    num, inputs, outputs = _split_inputs_outputs(df)
    Xdf = num[inputs].apply(pd.to_numeric, errors="coerce")
    Ydf = num[outputs].apply(pd.to_numeric, errors="coerce")
    work = Xdf.join(Ydf, how="inner").dropna()
    if len(work) < 50:
        return {"error": "Not enough clean rows in dataset."}

    Xdf = work[inputs].copy()
    Ydf = work[outputs].copy()
    X = Xdf.values.astype(float)
    Y = Ydf.values.astype(float)

    syn = _feature_map_synonyms()
    input_res = _resolve_map(Xdf.columns, INPUT_LABELS, syn["inputs"])
    canonical_outputs = ["Fragmentation", "Ground Vibration", "Airblast"]
    output_res = _resolve_map(Ydf.columns, canonical_outputs, syn["outputs"])
    empirical_input_map = {
        actual: canonical
        for canonical, actual in zip(INPUT_LABELS, input_res)
        if actual is not None and actual in Xdf.columns
    }
    empirical_output_map = {
        actual: canonical
        for canonical, actual in zip(canonical_outputs, output_res)
        if actual is not None and actual in Ydf.columns
    }
    physics_ready = len(empirical_input_map) == len(INPUT_LABELS) and len(empirical_output_map) >= 1

    output_stats = {}
    for c in outputs:
        s = Ydf[c]
        lo = float(s.quantile(0.02))
        hi = float(s.quantile(0.98))
        md = float(s.median())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(s.min())
            hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = md - 0.5
            hi = md + 0.5
        output_stats[c] = {"min": float(lo), "max": float(hi), "median": float(md)}

    bounds = {}
    medians = {}
    for c in inputs:
        s = Xdf[c]
        lo = float(s.quantile(0.02))
        hi = float(s.quantile(0.98))
        md = float(s.median())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(s.min())
            hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = md - 0.5
            hi = md + 0.5
        bounds[c] = (float(lo), float(hi))
        medians[c] = md

    empirical_base = np.zeros_like(Y, dtype=float)
    if physics_ready:
        empirical_frame = pd.DataFrame(index=work.index)
        for actual_name, canonical_name in empirical_input_map.items():
            empirical_frame[canonical_name] = pd.to_numeric(work[actual_name], errors="coerce")
        empirical_frame = empirical_frame[INPUT_LABELS].copy()
        for i, out_name in enumerate(outputs):
            canonical_name = empirical_output_map.get(out_name)
            if canonical_name is None:
                continue
            empirical_base[:, i] = _empirical_series_for_frame(empirical_frame, canonical_name)

    target_matrix = Y - empirical_base if physics_ready else Y

    sx = MinMaxScaler().fit(X)
    sy = MinMaxScaler().fit(target_matrix)
    Xs = sx.transform(X)
    Ys = sy.transform(target_matrix)

    if len(Xs) >= 80:
        Xtr, Xte, ytr, yte, ytr_raw, yte_raw, base_tr, base_te = train_test_split(
            Xs,
            Ys,
            Y,
            empirical_base,
            test_size=0.2,
            random_state=42,
        )
    else:
        Xtr, Xte, ytr, yte, ytr_raw, yte_raw, base_tr, base_te = Xs, Xs, Ys, Ys, Y, Y, empirical_base, empirical_base

    mdl = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="adam",
        learning_rate_init=0.005,
        max_iter=140,
        random_state=42,
        early_stopping=len(Xtr) >= 120,
        validation_fraction=0.15,
        n_iter_no_change=10,
        alpha=1e-4,
    )
    mdl.fit(Xtr, ytr)

    ytr_hat_target = sy.inverse_transform(mdl.predict(Xtr))
    yte_hat_target = sy.inverse_transform(mdl.predict(Xte))
    ytr_hat = ytr_hat_target + base_tr if physics_ready else ytr_hat_target
    yte_hat = yte_hat_target + base_te if physics_ready else yte_hat_target
    train_r2 = {}
    test_r2 = {}
    for i, out_name in enumerate(outputs):
        canonical_name = empirical_output_map.get(out_name, out_name)
        ytr_col = np.array([_clamp_physical_output(canonical_name, float(v)) for v in ytr_hat[:, i]], dtype=float)
        yte_col = np.array([_clamp_physical_output(canonical_name, float(v)) for v in yte_hat[:, i]], dtype=float)
        train_r2[out_name] = _score_r2(ytr_raw[:, i], ytr_col)
        test_r2[out_name] = _score_r2(yte_raw[:, i], yte_col)

    bundle = {
        "num": work,
        "inputs": inputs,
        "outputs": outputs,
        "sx": sx,
        "sy": sy,
        "model": mdl,
        "bounds": bounds,
        "medians": medians,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "rows_used": int(len(work)),
        "output_stats": output_stats,
        "physics_ready": bool(physics_ready),
        "empirical_input_map": empirical_input_map,
        "empirical_output_map": empirical_output_map,
        "surrogate_label": "physics-informed residual MLP" if physics_ready else "MLP",
    }
    if cache_key:
        _param_cache_key = cache_key
        _param_cache_bundle = bundle
    return bundle


def _param_predict_outputs(model_bundle, vec: dict[str, float]) -> dict[str, float]:
    import numpy as np

    inputs = model_bundle["inputs"]
    outputs = model_bundle["outputs"]
    sx = model_bundle["sx"]
    sy = model_bundle["sy"]
    mdl = model_bundle["model"]
    empirical_input_map = model_bundle.get("empirical_input_map", {})
    empirical_output_map = model_bundle.get("empirical_output_map", {})

    X = np.array([[float(vec[c]) for c in inputs]], dtype=float)
    Ys = mdl.predict(sx.transform(X))
    Y = sy.inverse_transform(Ys)[0]

    empirical_outputs = {name: 0.0 for name in outputs}
    if model_bundle.get("physics_ready") and empirical_input_map:
        empirical_vec = {}
        for actual_name, canonical_name in empirical_input_map.items():
            empirical_vec[canonical_name] = float(vec.get(actual_name, 0.0))
        empirical_vec["HPD_override"] = 1.0
        for out_name in outputs:
            canonical_name = empirical_output_map.get(out_name)
            if canonical_name is None:
                continue
            try:
                empirical_outputs[out_name] = _empirical_value_for_output(empirical_vec, canonical_name)
            except Exception:
                empirical_outputs[out_name] = 0.0

    out = {}
    for i, out_name in enumerate(outputs):
        canonical_name = empirical_output_map.get(out_name, out_name)
        raw = float(empirical_outputs.get(out_name, 0.0) + Y[i]) if model_bundle.get("physics_ready") else float(Y[i])
        out[out_name] = _clamp_physical_output(canonical_name, raw)
    return out


def _param_predict_output(model_bundle, vec: dict[str, float], output_name: str) -> float:
    preds = _param_predict_outputs(model_bundle, vec)
    return float(preds.get(output_name, _clamp_physical_output(output_name, 0.0)))


def _param_normalize_output(bundle, output_name: str, value: float) -> float:
    import math

    stats = (bundle.get("output_stats") or {}).get(output_name) or {}
    lo = float(stats.get("min", 0.0))
    hi = float(stats.get("max", lo + 1.0))
    span = hi - lo
    if not math.isfinite(span) or span <= 1e-9:
        span = max(1.0, abs(float(stats.get("median", 1.0))))
    return float((float(value) - lo) / span)


def _param_fragmentation_errors(preds: dict[str, float]) -> tuple[float, float]:
    frag = float(preds.get("Fragmentation", FRAGMENTATION_TARGET_MM))
    dev = abs(frag - FRAGMENTATION_TARGET_MM)
    band_error = max(0.0, dev - FRAGMENTATION_TOLERANCE_MM)
    return float(dev), float(band_error)


def _param_positive_error(preds: dict[str, float]) -> float:
    total = 0.0
    for name, floor in PHYSICAL_OUTPUT_FLOORS.items():
        total += max(0.0, float(floor) - float(preds.get(name, 0.0)))
    return float(total)


def _param_make_candidate(bundle, vec: dict[str, float], output: str, objective: str) -> dict:
    preds = _param_predict_outputs(bundle, vec)
    frag_dev, frag_band_error = _param_fragmentation_errors(preds)
    positive_error = _param_positive_error(preds)
    gv_norm = _param_normalize_output(bundle, "Ground Vibration", preds.get("Ground Vibration", 0.0))
    air_norm = _param_normalize_output(bundle, "Airblast", preds.get("Airblast", 0.0))
    if output == "Fragmentation":
        objective_value = float(preds.get(output, FRAGMENTATION_TARGET_MM))
        objective_norm = float(frag_dev / max(FRAGMENTATION_TOLERANCE_MM, 1.0))
    else:
        objective_value = float(preds.get(output, 0.0))
        base_norm = _param_normalize_output(bundle, output, objective_value)
        objective_norm = float(base_norm if objective == "min" else -base_norm)
    feasible = frag_band_error <= 1e-9 and positive_error <= 1e-9
    return {
        "inputs": {k: float(v) for k, v in vec.items()},
        "outputs": {k: float(v) for k, v in preds.items()},
        "objective_value": objective_value,
        "objective_norm": float(objective_norm),
        "fragmentation_target_error": float(frag_dev),
        "fragmentation_band_error": float(frag_band_error),
        "positive_error": float(positive_error),
        "positive_error_norm": float(positive_error / max(1e-6, sum(PHYSICAL_OUTPUT_FLOORS.values()))),
        "gv_norm": float(gv_norm),
        "air_norm": float(air_norm),
        "frag_norm": float(frag_dev / max(FRAGMENTATION_TOLERANCE_MM, 1.0)),
        "feasible": bool(feasible),
    }


def _param_scalarized_score(row: dict, weights: dict[str, float]) -> float:
    return float(
        row["objective_norm"]
        + float(weights.get("gv", 0.35)) * row["gv_norm"]
        + float(weights.get("air", 0.35)) * row["air_norm"]
        + float(weights.get("frag", 0.45)) * row["frag_norm"]
        + 4.0 * row["fragmentation_band_error"] / max(FRAGMENTATION_TOLERANCE_MM, 1.0)
        + 8.0 * row["positive_error_norm"]
    )


def _param_objective_vector(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["fragmentation_band_error"] / max(FRAGMENTATION_TOLERANCE_MM, 1.0)),
        float(row["positive_error_norm"]),
        float(row["objective_norm"]),
        float(row["gv_norm"]),
        float(row["air_norm"]),
        float(row["frag_norm"]),
    )


def _param_dominates(left: dict, right: dict, eps: float = 1e-9) -> bool:
    lv = _param_objective_vector(left)
    rv = _param_objective_vector(right)
    return all(a <= b + eps for a, b in zip(lv, rv)) and any(a < b - eps for a, b in zip(lv, rv))


def _param_assign_pareto_ranks(rows: list[dict]) -> list[dict]:
    remaining = set(range(len(rows)))
    rank = 0
    while remaining:
        front = []
        for idx in list(remaining):
            dominated = False
            for other_idx in remaining:
                if other_idx == idx:
                    continue
                if _param_dominates(rows[other_idx], rows[idx]):
                    dominated = True
                    break
            if not dominated:
                front.append(idx)
        if not front:
            break
        for idx in front:
            rows[idx]["pareto_rank"] = int(rank)
            rows[idx]["is_frontier"] = bool(rank == 0)
        remaining.difference_update(front)
        rank += 1
    for idx in remaining:
        rows[idx]["pareto_rank"] = int(rank)
        rows[idx]["is_frontier"] = False
    return rows


def _param_surface_from_rows(rows: list[dict], bounds: dict, x1: str, x2: str, output: str, objective: str, grid: int, best_inputs: dict | None):
    import numpy as np

    if x1 not in bounds or x2 not in bounds or x1 == x2:
        return {"error": "Invalid output/x1/x2 selection."}
    if not rows:
        return {"error": "No optimisation rows available."}

    x1_min, x1_max = bounds[x1]
    x2_min, x2_max = bounds[x2]
    gx = np.linspace(x1_min, x1_max, grid)
    gy = np.linspace(x2_min, x2_max, grid)
    x_span = max(1e-9, x1_max - x1_min)
    y_span = max(1e-9, x2_max - x2_min)
    usable_rows = [
        row for row in rows
        if isinstance(row.get("inputs"), dict) and isinstance(row.get("outputs"), dict)
    ]
    if not usable_rows:
        return {"error": "No optimisation rows available."}

    input_names = list(usable_rows[0].get("inputs", {}).keys())
    output_names = list(usable_rows[0].get("outputs", {}).keys())
    out_values = [float(row.get("outputs", {}).get(output, 0.0)) for row in usable_rows]
    out_min = min(out_values) if out_values else 0.0
    out_span = max(1e-9, (max(out_values) - out_min) if out_values else 1.0)

    def _view_score(row: dict) -> float:
        outputs = row.get("outputs", {})
        value = float(outputs.get(output, 0.0))
        if output == "Fragmentation":
            primary = abs(value - FRAGMENTATION_TARGET_MM) / max(FRAGMENTATION_TOLERANCE_MM, 1.0)
        else:
            norm = (value - out_min) / out_span
            primary = norm if objective == "min" else -norm
        return float(
            primary
            + 4.0 * float(row.get("fragmentation_band_error", 0.0)) / max(FRAGMENTATION_TOLERANCE_MM, 1.0)
            + 8.0 * float(row.get("positive_error_norm", 0.0))
            + (0.0 if row.get("feasible") else 1.0)
            + 0.08 * float(row.get("pareto_rank", 0.0))
        )

    ranked_rows = sorted(usable_rows, key=_view_score)

    def _blend_at_point(xv: float, yv: float) -> tuple[float, dict[str, float], dict[str, float]]:
        local = []
        for row in ranked_rows:
            cand_inputs = row.get("inputs", {})
            dx = (float(cand_inputs.get(x1, xv)) - float(xv)) / x_span
            dy = (float(cand_inputs.get(x2, yv)) - float(yv)) / y_span
            dist = dx * dx + dy * dy
            local.append((dist, row))
        local.sort(key=lambda item: item[0])
        neighbours = local[: min(8, len(local))]
        weights = []
        for dist, row in neighbours:
            rank_bias = 1.0 / (1.0 + 0.15 * float(row.get("pareto_rank", 0.0)))
            weights.append(rank_bias / max(1e-6, dist + 0.015))
        total = max(1e-9, sum(weights))
        blend_inputs = {}
        blend_outputs = {}
        for name in input_names:
            blend_inputs[name] = float(
                sum(weight * float(row.get("inputs", {}).get(name, 0.0)) for weight, (_, row) in zip(weights, neighbours)) / total
            )
        for name in output_names:
            blend_outputs[name] = float(
                sum(weight * float(row.get("outputs", {}).get(name, 0.0)) for weight, (_, row) in zip(weights, neighbours)) / total
            )
        return float(blend_outputs.get(output, 0.0)), blend_inputs, blend_outputs

    Z = []
    other_grid = []
    outputs_grid = []
    for xv in gx:
        row_vals = []
        row_inputs = []
        row_outputs = []
        for yv in gy:
            z_val, blend_inputs, blend_outputs = _blend_at_point(float(xv), float(yv))
            row_vals.append(z_val)
            row_inputs.append(blend_inputs)
            row_outputs.append(blend_outputs)
        Z.append(row_vals)
        other_grid.append(row_inputs)
        outputs_grid.append(row_outputs)

    best_row = ranked_rows[0]
    best_point = {
        "x1": float(best_row.get("inputs", {}).get(x1, best_inputs.get(x1, gx[0]) if best_inputs else gx[0])),
        "x2": float(best_row.get("inputs", {}).get(x2, best_inputs.get(x2, gy[0]) if best_inputs else gy[0])),
    }

    return {
        "x1": x1,
        "x2": x2,
        "output": output,
        "objective": objective,
        "grid_x": gx.tolist(),
        "grid_y": gy.tolist(),
        "Z": Z,
        "other_inputs_grid": other_grid,
        "outputs_grid": outputs_grid,
        "best_point": best_point,
    }


def _param_surface_df(df, payload):
    import numpy as np
    from scipy.optimize import minimize

    cache_key = payload.get("dataset") if not payload.get("_uploaded") else None
    bundle = _fit_param_surrogate(df, cache_key=cache_key)
    if bundle.get("error"):
        return {"error": bundle["error"]}

    inputs = bundle["inputs"]
    outputs = bundle["outputs"]
    bounds = bundle["bounds"]
    medians = bundle["medians"]
    output = payload.get("output", outputs[0])
    x1 = payload.get("x1", inputs[0])
    x2 = payload.get("x2", inputs[1] if len(inputs) > 1 else inputs[0])
    objective = payload.get("objective", "max")
    grid = max(8, min(24, int(payload.get("grid", 10))))
    samples = max(1, min(4, int(payload.get("samples", 2))))
    max_iter = max(8, min(40, int(payload.get("max_iter", 20))))
    fast_mode = bool(payload.get("fast_mode", False))
    if fast_mode:
        grid = min(grid, 10)
        samples = 1
        max_iter = min(max_iter, 8)

    if output not in outputs or x1 not in inputs or x2 not in inputs or x1 == x2:
        return {"error": "Invalid output/x1/x2 selection."}

    med_vec = np.array([medians[c] for c in inputs], dtype=float)
    var_bounds = [bounds[c] for c in inputs]
    rng = np.random.default_rng(42)
    random_count = 64 if fast_mode else 120
    scalar_weights = [
        {"gv": 0.20, "air": 0.20, "frag": 0.55},
        {"gv": 0.40, "air": 0.25, "frag": 0.45},
        {"gv": 0.25, "air": 0.40, "frag": 0.45},
        {"gv": 0.35, "air": 0.35, "frag": 0.35},
    ]
    for _ in range(2 if fast_mode else 3):
        w = rng.dirichlet(np.array([1.0, 1.0, 1.0], dtype=float))
        scalar_weights.append({"gv": float(w[0]), "air": float(w[1]), "frag": float(w[2])})

    rows = []
    seen = set()

    def _vec_to_dict(arr: np.ndarray) -> dict[str, float]:
        return {name: float(arr[i]) for i, name in enumerate(inputs)}

    def _add_candidate(arr: np.ndarray):
        clipped = np.array(
            [np.clip(float(arr[i]), float(var_bounds[i][0]), float(var_bounds[i][1])) for i in range(len(inputs))],
            dtype=float,
        )
        key = tuple(round(float(v), 6) for v in clipped.tolist())
        if key in seen:
            return None
        seen.add(key)
        row = _param_make_candidate(bundle, _vec_to_dict(clipped), output, objective)
        rows.append(row)
        return row

    _add_candidate(med_vec)
    for _ in range(max(0, random_count - 1)):
        guess = np.array([rng.uniform(lo, hi) for lo, hi in var_bounds], dtype=float)
        _add_candidate(guess)

    seed_rows = list(rows)
    local_iter = max(8, min(20, int(max_iter * 2)))

    def _scalar_objective(xvec: np.ndarray, weights: dict[str, float]) -> float:
        row = _param_make_candidate(bundle, _vec_to_dict(xvec), output, objective)
        return _param_scalarized_score(row, weights)

    seeds_per_weight = 1
    for weights in scalar_weights:
        ranked = sorted(seed_rows, key=lambda row: _param_scalarized_score(row, weights))
        for seed in ranked[:seeds_per_weight]:
            x0 = np.array([float(seed["inputs"][name]) for name in inputs], dtype=float)
            try:
                res = minimize(
                    lambda x, w=weights: _scalar_objective(x, w),
                    x0,
                    method="L-BFGS-B",
                    bounds=var_bounds,
                    options={"maxiter": local_iter},
                )
                cand = res.x if getattr(res, "x", None) is not None else x0
            except Exception:
                cand = x0
            _add_candidate(np.array(cand, dtype=float))

    rows = _param_assign_pareto_ranks(rows)
    default_weights = {"gv": 0.35, "air": 0.35, "frag": 0.45}
    rows.sort(
        key=lambda row: (
            not bool(row.get("feasible")),
            int(row.get("pareto_rank", 999)),
            _param_scalarized_score(row, default_weights),
            float(row.get("fragmentation_target_error", 0.0)),
        )
    )
    rows = rows[: (72 if fast_mode else 120)]

    best_row = rows[0] if rows else None
    surface = _param_surface_from_rows(
        rows,
        bounds,
        x1,
        x2,
        output,
        objective,
        grid,
        best_row.get("inputs") if best_row else None,
    )
    if surface.get("error"):
        return surface

    return {
        "dataset": payload.get("dataset") or DATASETS["combined"],
        "x1": surface["x1"],
        "x2": surface["x2"],
        "output": surface["output"],
        "objective": surface["objective"],
        "grid_x": surface["grid_x"],
        "grid_y": surface["grid_y"],
        "Z": surface["Z"],
        "other_inputs_grid": surface["other_inputs_grid"],
        "outputs_grid": surface.get("outputs_grid"),
        "best": {
            "value": float(best_row.get("objective_value")) if best_row else None,
            "point": surface.get("best_point"),
            "inputs": best_row.get("inputs") if best_row else None,
            "outputs": best_row.get("outputs") if best_row else None,
        },
        "train_r2": bundle["train_r2"].get(output),
        "test_r2": bundle["test_r2"].get(output),
        "rows_used": bundle["rows_used"],
        "bounds": {k: {"min": float(v[0]), "max": float(v[1]), "median": float(medians[k])} for k, v in bounds.items()},
        "rows": rows,
        "surrogate": bundle.get("surrogate_label"),
        "fragmentation_target": FRAGMENTATION_TARGET_MM,
        "fragmentation_tolerance": FRAGMENTATION_TOLERANCE_MM,
        "note": "Physics-informed surrogate MLP with a production-safe Pareto candidate search. Saved optimisation rows can be reprojected across different input axes without rerunning the solver.",
    }


def _param_goal_seek_df(df, payload):
    import numpy as np
    from scipy.optimize import minimize

    cache_key = payload.get("dataset") if not payload.get("_uploaded") else None
    bundle = _fit_param_surrogate(df, cache_key=cache_key)
    if bundle.get("error"):
        return {"error": bundle["error"]}

    inputs = bundle["inputs"]
    outputs = bundle["outputs"]
    bounds = bundle["bounds"]
    medians = bundle["medians"]
    output = payload.get("output", outputs[0])
    target = float(payload.get("target", 0.0))
    tolerance = float(payload.get("tolerance", 1e-3))
    samples = max(4, min(24, int(payload.get("samples", 8))))
    if output not in outputs:
        return {"error": "Invalid output selection."}

    var_bounds = [bounds[c] for c in inputs]
    median_start = np.array([medians[c] for c in inputs], dtype=float)

    def _goal_objective(xvec):
        probe = {c: float(v) for c, v in zip(inputs, xvec)}
        preds = _param_predict_outputs(bundle, probe)
        pred = float(preds.get(output, 0.0))
        frag_dev, frag_band_error = _param_fragmentation_errors(preds)
        positive_error = _param_positive_error(preds)
        base_err = float((pred - target) ** 2)
        return float(
            base_err
            + 25.0 * (frag_band_error ** 2)
            + 10.0 * (positive_error ** 2)
            + (0.05 * frag_dev if output != "Fragmentation" else 0.0)
        )

    starts = [median_start]
    for _ in range(max(0, samples - 1)):
        starts.append(np.array([np.random.uniform(lo, hi) for lo, hi in var_bounds], dtype=float))

    best = None
    best_in = None
    for guess in starts:
        res = minimize(
            _goal_objective,
            guess,
            method="L-BFGS-B",
            bounds=var_bounds,
            options={"maxiter": 120},
        )
        cand = np.clip(res.x if getattr(res, "x", None) is not None else guess, [b[0] for b in var_bounds], [b[1] for b in var_bounds])
        probe = {c: float(v) for c, v in zip(inputs, cand)}
        preds = _param_predict_outputs(bundle, probe)
        pred = float(preds.get(output, 0.0))
        err = abs(pred - target)
        if best is None or err < best:
            best = err
            best_in = {"predicted": pred, "inputs": probe, "outputs": preds}

    return {
        "dataset": payload.get("dataset") or DATASETS["combined"],
        "output": output,
        "target": target,
        "tolerance": tolerance,
        "best": best_in,
        "abs_error": float(best) if best is not None else None,
        "within_tolerance": bool(best is not None and best <= tolerance),
        "train_r2": bundle["train_r2"].get(output),
        "test_r2": bundle["test_r2"].get(output),
        "rows_used": bundle["rows_used"],
        "bounds": {k: {"min": float(v[0]), "max": float(v[1]), "median": float(medians[k])} for k, v in bounds.items()},
        "surrogate": bundle.get("surrogate_label"),
        "fragmentation_target": FRAGMENTATION_TARGET_MM,
        "fragmentation_tolerance": FRAGMENTATION_TOLERANCE_MM,
        "note": "Goal seek uses the same physics-informed surrogate as the Pareto optimisation search and keeps fragmentation inside the 90-110 mm band where possible.",
    }


@app.get("/v1/param/meta")
def param_meta(_token: str = Depends(require_auth)):
    df = _combined_df()
    _, inputs, outputs = _split_inputs_outputs(df)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "dataset": DATASETS["combined"],
        "default_output": outputs[0] if outputs else None,
        "default_x1": inputs[0] if inputs else None,
        "default_x2": inputs[1] if len(inputs) > 1 else (inputs[0] if inputs else None),
    }


@app.post("/v1/param/meta")
def param_meta_upload(
    file: UploadFile | None = File(default=None),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    _, inputs, outputs = _split_inputs_outputs(df)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "dataset": file.filename if file is not None else DATASETS["combined"],
        "default_output": outputs[0] if outputs else None,
        "default_x1": inputs[0] if inputs else None,
        "default_x2": inputs[1] if len(inputs) > 1 else (inputs[0] if inputs else None),
    }


@app.post("/v1/param/surface")
def param_surface(payload: dict = Body(...), _token: str = Depends(require_auth)):
    df = _combined_df()
    return _param_surface_df(df, payload)


@app.post("/v1/param/surface/upload")
def param_surface_upload(
    file: UploadFile | None = File(default=None),
    payload_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json

    payload = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
    payload["_uploaded"] = True
    df = _read_upload_df(file, DATASETS["combined"])
    return _param_surface_df(df, payload)


@app.post("/v1/param/goal-seek")
def param_goal_seek(payload: dict = Body(...), _token: str = Depends(require_auth)):
    df = _combined_df()
    return _param_goal_seek_df(df, payload)


@app.post("/v1/param/goal-seek/upload")
def param_goal_seek_upload(
    file: UploadFile | None = File(default=None),
    payload_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json

    payload = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
    payload["_uploaded"] = True
    df = _read_upload_df(file, DATASETS["combined"])
    return _param_goal_seek_df(df, payload)


def _cost_defaults():
    return {
        "d_mm": 102.0,
        "bench": 10.0,
        "B": 3.0,
        "S": 3.3,
        "sub": 2.0,
        "stem": 1.8,
        "n_holes": 30,
        "hpd": 1,
        "vol": 0.0,
        "rho_gcc": 1.15,
        "rws": 115.0,
        "ci": 10.0,
        "ce": 4.0,
        "cd": 7.0,
        "R": 500.0,
        "Kp": 1000.0,
        "beta": 1.6,
        "ppv_lim": 12.5,
        "Ka": 170.0,
        "Ba": 20.0,
        "air_lim": 134.0,
        "Ak": 22.0,
        "nrr": 1.8,
        "x50_target": 120.0,
        "x_ov": 500.0,
        "ov_max": 0.10,
        "Bmin": 2.5,
        "Bmax": 4.5,
        "kS_min": 1.05,
        "kS_max": 1.50,
        "kStem_min": 0.70,
        "kStem_max": 1.00,
        "kSub_min": 0.30,
        "kSub_max": 0.50,
        "stiff_min": 2.50,
        "stiff_max": 4.50,
    }


def _cost_derived(p):
    import math

    d_m = p["d_mm"] / 1000.0
    area = math.pi * (d_m**2) / 4.0
    hole_len = max(0.0, p["bench"] + p["sub"])
    charge_len = max(0.0, hole_len - p["stem"])
    rho = p["rho_gcc"] * 1000.0
    m_per_hole = rho * area * charge_len
    m_total = m_per_hole * p["n_holes"]
    vol = p["vol"] if p["vol"] > 0 else p["B"] * p["S"] * p["bench"] * p["n_holes"]
    PF = m_total / max(1e-9, vol)
    drill_len_total = p["n_holes"] * hole_len
    Q_delay = p["hpd"] * m_per_hole
    return {
        "d_m": d_m,
        "area": area,
        "hole_len": hole_len,
        "charge_len": charge_len,
        "rho": rho,
        "m_per_hole": m_per_hole,
        "m_total": m_total,
        "PF": PF,
        "vol": vol,
        "drill_len_total": drill_len_total,
        "Q_delay": Q_delay,
    }


def _cost_cost(p, d):
    initiation_cost = p["n_holes"] * p["ci"]
    explosive_cost = d["m_total"] * p["ce"]
    drilling_cost = d["drill_len_total"] * p["cd"]
    total = initiation_cost + explosive_cost + drilling_cost
    return total, initiation_cost, explosive_cost, drilling_cost


def _cost_vibration(p, d):
    import math

    SD = p["R"] / max(1e-9, math.sqrt(d["Q_delay"]))
    PPV = p["Kp"] * (SD ** (-p["beta"]))
    return SD, PPV


def _cost_airblast(p, d):
    import math

    denom = max(1e-9, d["Q_delay"] ** (1.0 / 3.0))
    L = p["Ka"] + p["Ba"] * math.log10(max(1e-12, p["R"] / denom))
    return L


def _cost_fragmentation(p, d):
    import math

    A = max(0.0, p["Ak"])
    RWS = max(1e-9, p["rws"])
    n = max(0.5, p["nrr"])
    K_pf = d["PF"]
    Q_ph = d["m_per_hole"]
    if A > 0.0 and K_pf > 0.0 and Q_ph > 0.0 and RWS > 0.0:
        Xm = A * (K_pf ** -0.8) * (Q_ph ** (1.0 / 6.0)) * ((115.0 / RWS) ** (19.0 / 20.0))
        lam = Xm / math.gamma(1.0 + 1.0 / n)
        X50 = lam * (math.log(2.0) ** (1.0 / n))
        oversize = math.exp(-((p["x_ov"] / max(1e-9, lam)) ** n))
        return Xm, X50, oversize
    V_over_Q = d["vol"] / max(1e-9, d["m_total"])
    X50_legacy = p["Ak"] * (V_over_Q ** 0.8)
    lam = X50_legacy / (math.log(2.0) ** (1.0 / n))
    Xm_legacy = lam * math.gamma(1.0 + 1.0 / n)
    oversize = math.exp(-((p["x_ov"] / max(1e-9, lam)) ** n))
    return Xm_legacy, X50_legacy, oversize


def _cost_metrics(p):
    d = _cost_derived(p)
    cost, ci, ce, cd = _cost_cost(p, d)
    SD, PPV = _cost_vibration(p, d)
    L = _cost_airblast(p, d)
    Xm, X50, ov = _cost_fragmentation(p, d)
    return {
        "inputs": p,
        "derived": d,
        "cost": cost,
        "cost_break": (ci, ce, cd),
        "SD": SD,
        "PPV": PPV,
        "L": L,
        "Xm": Xm,
        "X50": X50,
        "oversize": ov,
    }


def _cost_penalties(p, res, weights, use_ppv, use_air, use_frag):
    pen_frag = pen_ppv = pen_air = 0.0
    if use_frag:
        X50, ov = res["X50"], res["oversize"]
        pen_x50 = max(0.0, (X50 - p["x50_target"]) / max(1e-9, p["x50_target"]))
        pen_ov = max(0.0, ov - p["ov_max"])
        pen_frag = weights["frag"] * (10.0 * pen_x50**2 + 20.0 * pen_ov**2)
    if use_ppv:
        pen_ppv = weights["ppv"] * (15.0 * max(0.0, (res["PPV"] - p["ppv_lim"]) / max(1e-9, p["ppv_lim"]))**2)
    if use_air:
        pen_air = weights["air"] * (10.0 * max(0.0, (res["L"] - p["air_lim"]) / max(1e-9, p["air_lim"]))**2)
    return {"frag": pen_frag, "ppv": pen_ppv, "air": pen_air}


def _cost_constraint_summary(p, res):
    spacing_ratio = p["S"] / max(1e-9, p["B"])
    stem_ratio = p["stem"] / max(1e-9, p["B"])
    sub_ratio = p["sub"] / max(1e-9, p["B"])
    stiffness = p["bench"] / max(1e-9, p["B"])
    return {
        "ppv_within_limit": bool(res["PPV"] <= p["ppv_lim"]),
        "air_within_limit": bool(res["L"] <= p["air_lim"]),
        "oversize_within_limit": bool(res["oversize"] <= p["ov_max"]),
        "burden_within_limit": bool(p["Bmin"] <= p["B"] <= p["Bmax"]),
        "spacing_ratio_within_limit": bool(p["kS_min"] <= spacing_ratio <= p["kS_max"]),
        "stemming_ratio_within_limit": bool(p["kStem_min"] <= stem_ratio <= p["kStem_max"]),
        "subdrill_ratio_within_limit": bool(p["kSub_min"] <= sub_ratio <= p["kSub_max"]),
        "stiffness_within_limit": bool(p["stiff_min"] <= stiffness <= p["stiff_max"]),
        "spacing_ratio": spacing_ratio,
        "stemming_ratio": stem_ratio,
        "subdrill_ratio": sub_ratio,
        "stiffness_ratio": stiffness,
    }


def _cost_constraints(p):
    cons = []
    def c_spacing_min(x): B, S, sub = x; return S - p["kS_min"] * B
    def c_spacing_max(x): B, S, sub = x; return p["kS_max"] * B - S
    def c_stem_min(x): B, S, sub = x; return p["stem"] - p["kStem_min"] * B
    def c_stem_max(x): B, S, sub = x; return p["kStem_max"] * B - p["stem"]
    def c_sub_min(x): B, S, sub = x; return sub - p["kSub_min"] * B
    def c_sub_max(x): B, S, sub = x; return p["kSub_max"] * B - sub
    def c_stiff_min(x): B, S, sub = x; return (p["bench"] / max(1e-9, B)) - p["stiff_min"]
    def c_stiff_max(x): B, S, sub = x; return p["stiff_max"] - (p["bench"] / max(1e-9, B))
    for fn in [c_spacing_min, c_spacing_max, c_stem_min, c_stem_max, c_sub_min, c_sub_max, c_stiff_min, c_stiff_max]:
        cons.append({"type": "ineq", "fun": fn})
    return cons


def _cost_bounds(p):
    B_lo, B_hi = p["Bmin"], p["Bmax"]
    S_lo, S_hi = 0.5, 8.0
    sub_lo, sub_hi = 0.0, max(6.0, p["bench"])
    return [(B_lo, B_hi), (S_lo, S_hi), (sub_lo, sub_hi)]


def _cost_minimize_options(method: str, pareto: bool = False):
    if method == "trust-constr":
        return {
            "maxiter": 120 if not pareto else 80,
            "gtol": 1e-6,
            "xtol": 1e-6,
            "barrier_tol": 1e-6,
        }
    return {
        "maxiter": 400 if not pareto else 200,
        "ftol": 1e-7,
    }


def _cost_run_optimizer(p, weights, use_frag, use_ppv, use_air, method: str, x0, pareto: bool = False):
    from scipy.optimize import minimize

    def obj(x):
        trial = p.copy()
        trial["B"], trial["S"], trial["sub"] = float(x[0]), float(x[1]), float(x[2])
        res = _cost_metrics(trial)
        pen = _cost_penalties(trial, res, weights, use_ppv, use_air, use_frag)
        return res["cost"] + pen["frag"] + pen["ppv"] + pen["air"]

    solver = "SLSQP" if method == "SLSQP" else "trust-constr"
    try:
        res = minimize(
            obj,
            x0,
            method=solver,
            bounds=_cost_bounds(p),
            constraints=_cost_constraints(p),
            options=_cost_minimize_options(solver, pareto=pareto),
        )
        if solver == "trust-constr" and not bool(res.success):
            fallback = minimize(
                obj,
                res.x if getattr(res, "x", None) is not None else x0,
                method="SLSQP",
                bounds=_cost_bounds(p),
                constraints=_cost_constraints(p),
                options=_cost_minimize_options("SLSQP", pareto=pareto),
            )
            if bool(fallback.success) or not bool(res.success):
                return fallback, "SLSQP", f"Fallback from trust-constr: {res.message}"
        return res, solver, ""
    except Exception as exc:
        if solver != "trust-constr":
            raise
        fallback = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=_cost_bounds(p),
            constraints=_cost_constraints(p),
            options=_cost_minimize_options("SLSQP", pareto=pareto),
        )
        return fallback, "SLSQP", f"Fallback from trust-constr exception: {exc}"


@app.get("/v1/cost/defaults")
def cost_defaults(_token: str = Depends(require_auth)):
    return _cost_defaults()


@app.post("/v1/cost/compute")
def cost_compute(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    p = _cost_defaults()
    p.update(payload or {})
    res = _cost_metrics(p)
    weights = payload.get("weights", {"frag": 1.0, "ppv": 1.0, "air": 0.7})
    use_frag = bool(payload.get("use_frag", True))
    use_ppv = bool(payload.get("use_ppv", True))
    use_air = bool(payload.get("use_air", True))
    res["penalties"] = _cost_penalties(p, res, weights, use_ppv, use_air, use_frag)
    res["constraint_checks"] = _cost_constraint_summary(p, res)
    return res


@app.post("/v1/cost/optimize")
def cost_optimize(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    import numpy as np

    p = _cost_defaults()
    p.update(payload or {})
    weights = payload.get("weights", {"frag": 1.0, "ppv": 1.0, "air": 0.7})
    use_frag = bool(payload.get("use_frag", True))
    use_ppv = bool(payload.get("use_ppv", True))
    use_air = bool(payload.get("use_air", True))
    method = payload.get("method", "SLSQP")

    x0 = np.array([p["B"], p["S"], p["sub"]], dtype=float)
    res, solver_used, fallback_note = _cost_run_optimizer(p, weights, use_frag, use_ppv, use_air, method, x0, pareto=False)
    best = p.copy()
    best["B"], best["S"], best["sub"] = float(res.x[0]), float(res.x[1]), float(res.x[2])
    result = _cost_metrics(best)
    result["penalties"] = _cost_penalties(best, result, weights, use_ppv, use_air, use_frag)
    result["constraint_checks"] = _cost_constraint_summary(best, result)
    message = str(res.message)
    if fallback_note:
        message = f"{fallback_note}. {message}"
    return {"success": bool(res.success), "message": message, "solver_used": solver_used, "result": result}


@app.post("/v1/cost/pareto")
def cost_pareto(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    import numpy as np

    p = _cost_defaults()
    p.update(payload or {})
    w_list = [0.0, 1.0, 2.0]
    rows = []
    x0 = np.array([p["B"], p["S"], p["sub"]], dtype=float)
    method = payload.get("method", "SLSQP")
    use_frag = bool(payload.get("use_frag", True))
    use_ppv = bool(payload.get("use_ppv", True))
    use_air = bool(payload.get("use_air", True))

    for wf in w_list:
        for wp in w_list:
            for wa in w_list:
                if wf == 0.0 and wp == 0.0 and wa == 0.0 and rows:
                    continue
                weights = {"frag": wf, "ppv": wp, "air": wa}
                solver_success = False
                solver_message = ""
                solver_used = method

                try:
                    res, solver_used, fallback_note = _cost_run_optimizer(
                        p,
                        weights,
                        use_frag and wf > 0,
                        use_ppv and wp > 0,
                        use_air and wa > 0,
                        method,
                        x0,
                        pareto=True,
                    )
                    x = res.x if res.success else x0
                    solver_success = bool(res.success)
                    solver_message = f"{fallback_note}. {res.message}" if fallback_note else str(res.message)
                except Exception:
                    x = x0
                trial = p.copy()
                trial["B"], trial["S"], trial["sub"] = float(x[0]), float(x[1]), float(x[2])
                met = _cost_metrics(trial)
                rows.append(
                    {
                        "wf": wf,
                        "wp": wp,
                        "wa": wa,
                        "B": trial["B"],
                        "S": trial["S"],
                        "sub": trial["sub"],
                        "cost": met["cost"],
                        "PPV": met["PPV"],
                        "Air": met["L"],
                        "Oversize%": 100.0 * met["oversize"],
                        "X50": met["X50"],
                        "Xm": met["Xm"],
                        "PF": met["derived"]["PF"],
                        "Qdelay": met["derived"]["Q_delay"],
                        "R": trial["R"],
                        "solver_success": solver_success,
                        "solver_message": solver_message,
                        "solver_used": solver_used,
                    }
                )
                x0 = x
    frontier = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                other["cost"] <= row["cost"]
                and other["Oversize%"] <= row["Oversize%"]
                and other["PPV"] <= row["PPV"]
                and other["Air"] <= row["Air"]
            )
            strictly_better = (
                other["cost"] < row["cost"]
                or other["Oversize%"] < row["Oversize%"]
                or other["PPV"] < row["PPV"]
                or other["Air"] < row["Air"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (row["cost"], row["PPV"], row["Air"], row["Oversize%"]))
    return {"rows": rows, "frontier": frontier}

