from __future__ import annotations

from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_auth
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


def _combined_dataset_choices() -> list[str]:
    existing = [name for name in COMBINED_DATASET_CHOICES if any(path.exists() for path in _dataset_search_candidates(name))]
    return existing or list(COMBINED_DATASET_CHOICES)


def _asset_search_paths() -> list[Path]:
    return [
        *([local_assets_path] if local_assets_path else []),
        REPO_ROOT / "assets",
        WORKSPACE_ROOT / "assets",
        Path(core_bundle_path),
        Path("/tmp/ai-blasting-assets"),
    ]


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
    if name not in choices:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset. Choose one of: {choices}")

    DATASETS["combined"] = name

    # Reset ML cache/models so the next ML request trains on the selected dataset.
    global _ml_cache_key, _assets
    _ml_cache_key = None
    _assets = LoadedAssets(scaler=None, mdl_frag=None, mdl_ppv=None, mdl_air=None)

    return {"ok": True, "active": name}


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
    raise FileNotFoundError(f"Dataset not found in GCS bucket {settings.gcs_bucket}: {name}")


def _read_upload_df(up: UploadFile | None = None, dataset_name: str | None = None):
    import pandas as pd
    import io

    if up is None:
        if not dataset_name:
            raise ValueError("No file or dataset_name provided")
        path = _ensure_dataset(dataset_name)
        if str(path).lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)

    data = up.file.read()
    up.file.seek(0)
    if up.filename and up.filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


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


def _maybe_train_models_from_default_dataset() -> None:
    try:
        df = _read_upload_df(None, DATASETS["combined"])
        _maybe_train_models_from_df(df)
    except Exception:
        return


def _maybe_train_models_from_df(df) -> None:
    """
    Train fallback ML models from a dataset if joblib assets are missing.
    """
    global _assets, _ml_cache_key
    if _assets.scaler and (_assets.mdl_frag or _assets.mdl_ppv or _assets.mdl_air):
        return

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
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

    X = work.iloc[:, : Xraw.shape[1]].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    mdl_frag = mdl_ppv = mdl_air = None
    for out_name in ["Fragmentation", "Ground Vibration", "Airblast"]:
        if out_name not in Yraw.columns:
            continue
        y = pd.to_numeric(work[out_name], errors="coerce").values
        # Allow training on smaller datasets (preview/sample datasets can be small).
        if len(y) < 10:
            continue
        mdl = RandomForestRegressor(n_estimators=400, random_state=42)
        mdl.fit(Xs, y)
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
        _ml_cache_key = key


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
    ydf = work.iloc[:, Xraw.shape[1] :]
    out = {}
    perm_out = {}
    explainability = {}
    for out_name in ydf.columns:
        y = ydf[out_name].values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        Xc = X[mask]
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

        pdp_source = perm_out[out_name] if perm_out[out_name] else out[out_name]
        top_features = [it["feature"] for it in pdp_source[: min(3, len(pdp_source))]]
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

        explainability[out_name] = {
            "train_r2": _score_r2(ytr, mdl.predict(Xtr)),
            "test_r2": _score_r2(yte, mdl.predict(Xte)),
            "partial_dependence": pdp,
        }
    return {
        "feature_importance": out,
        "permutation_importance": perm_out,
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
    return _feature_importance_df(df, top_k)


@app.post("/v1/feature/importance")
def feature_importance_dataset_upload(
    file: UploadFile | None = File(default=None),
    top_k: int = Form(default=12),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    return _feature_importance_df(df, top_k)


@app.get("/v1/feature/pca")
def feature_pca(_token: str = Depends(require_auth)):
    df = _combined_df()
    return _feature_pca_df(df)


@app.post("/v1/feature/pca")
def feature_pca_upload(
    file: UploadFile | None = File(default=None),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    return feature_pca(_token=_token) if file is None else _feature_pca_df(df)


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
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file, DATASETS["backbreak"])
    if df.shape[1] < 2:
        return {"error": "CSV must have at least 2 columns (features + target)."}

    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass

    tgt = _infer_backbreak_target(df)
    if tgt is None:
        return {"error": "Could not infer Back Break target column."}

    num = df.select_dtypes(include=[np.number]).copy()
    if tgt not in num.columns:
        y = pd.to_numeric(df[tgt], errors="coerce")
        num[tgt] = y

    X = num.drop(columns=[tgt], errors="ignore").copy()
    y = pd.to_numeric(df[tgt], errors="coerce")

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if X.shape[0] < 30 or X.shape[1] < 2:
        return {"error": "Not enough clean rows or features to train Random Forest (need >=30 rows, >=2 features)."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(Xtr, ytr)
    train_r2 = _score_r2(ytr, model.predict(Xtr))
    test_r2 = _score_r2(yte, model.predict(Xte))

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    feat_names = list(X.columns)
    keep = [feat_names[i] for i in order[: min(6, len(order))]]
    feat_importance = [
        {"feature": feat_names[i], "importance": float(imp[i])}
        for i in order
    ]

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
    prob = float(pipe.predict_proba(xstar)[0, 1])
    train_acc = float(pipe.score(Xtr, ytr))
    test_acc = float(pipe.score(Xte, yte))
    class_balance = {
        "stable": int((prepared["status"] == "stable").sum()),
        "failure": int((prepared["status"] == "failure").sum()),
    }

    return {
        "prob_stable": prob,
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
            "HoleID": ["holeid", "hole id", "id", "hole"],
            "Depth": ["depth", "hole_depth", "hole depth (m)", "hole_depth_m"],
            "Charge": ["charge", "explosive mass", "explosive_mass", "charge_kg"],
            "Z": ["z", "rl", "elev", "elevation"],
            "Delay": ["delay", "delay_ms", "predicted delay (ms)", "predicted_delay_ms", "time_ms"],
        }
        for std, aliases in opt.items():
            col = _find(std, aliases)
            if col is not None:
                rename[col] = std
        return df_in.rename(columns=rename)

    try:
        train_df = _standardize_delay_df(_read_upload_df(None, DATASETS["delay_v1"]))
    except Exception:
        train_df = _standardize_delay_df(_read_upload_df(None, DATASETS["delay_v2"]))
    infer_df = _standardize_delay_df(_read_upload_df(file, DATASETS["delay_v1"])) if file is not None else train_df.copy()

    train_keep = ["Depth", "Charge", "X", "Y"] + (["Z"] if "Z" in train_df.columns else [])
    train_clean = train_df.dropna(subset=train_keep).copy()
    X_train = train_clean[[c for c in ["Depth", "Charge", "X", "Y", "Z"] if c in train_clean.columns]].apply(pd.to_numeric, errors="coerce").values
    if "Delay" in train_clean.columns:
        y_train_full = pd.to_numeric(train_clean["Delay"], errors="coerce").values
    else:
        y_train_full = np.clip(10 + 0.02 * X_train[:, 0] + 0.0005 * X_train[:, 2], 5, 250)

    Xtr, Xte, ytr, yte = train_test_split(X_train, y_train_full, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mdl = RandomForestRegressor(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)

    infer_keep = ["Depth", "Charge", "X", "Y"] + (["Z"] if "Z" in infer_df.columns and "Z" in train_clean.columns else [])
    infer_clean = infer_df.dropna(subset=infer_keep).copy()
    X_infer = infer_clean[[c for c in ["Depth", "Charge", "X", "Y", "Z"] if c in infer_clean.columns and c in train_clean.columns]].apply(pd.to_numeric, errors="coerce").values
    yhat = mdl.predict(sc.transform(X_infer))
    dfv = pd.DataFrame(
        {
            "X": pd.to_numeric(infer_clean["X"], errors="coerce"),
            "Y": pd.to_numeric(infer_clean["Y"], errors="coerce"),
            "Delay": yhat,
        }
    )
    if "Depth" in infer_clean.columns:
        dfv["Depth"] = pd.to_numeric(infer_clean["Depth"], errors="coerce")
    if "Charge" in infer_clean.columns:
        dfv["Charge"] = pd.to_numeric(infer_clean["Charge"], errors="coerce")
    if "Z" in infer_clean.columns:
        dfv["Z"] = pd.to_numeric(infer_clean["Z"], errors="coerce")
    if "HoleID" in infer_clean.columns:
        dfv["HoleID"] = infer_clean["HoleID"].astype(str)

    dfv = dfv.dropna(subset=["X", "Y", "Delay"])

    # limit to 2000 points for payload size
    if len(dfv) > 2000:
        dfv = dfv.sample(2000, random_state=42)

    return {
        "points": dfv.to_dict(orient="records"),
        "train_r2": _score_r2(ytr, mdl.predict(sc.transform(Xtr))),
        "test_r2": _score_r2(yte, mdl.predict(sc.transform(Xte))),
        "training_rows": int(len(train_clean)),
        "predicted_rows": int(len(dfv)),
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


def _param_surface_df(df, payload):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    num, inputs, outputs = _split_inputs_outputs(df)
    output = payload.get("output", outputs[0])
    x1 = payload.get("x1", inputs[0])
    x2 = payload.get("x2", inputs[1] if len(inputs) > 1 else inputs[0])
    objective = payload.get("objective", "max")
    grid = int(payload.get("grid", 25))
    samples = int(payload.get("samples", 40))

    if output not in outputs or x1 not in inputs or x2 not in inputs or x1 == x2:
        return {"error": "Invalid output/x1/x2 selection."}

    X = num[inputs].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(num[output], errors="coerce")
    m = X.notna().all(axis=1) & y.notna()
    X = X[m]
    y = y[m]
    if len(y) < 30:
        return {"error": "Not enough clean rows in dataset."}

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    bounds = {}
    for c in inputs:
        s = X[c]
        bounds[c] = (float(s.quantile(0.02)), float(s.quantile(0.98)))

    x1_min, x1_max = bounds[x1]
    x2_min, x2_max = bounds[x2]
    gx = np.linspace(x1_min, x1_max, grid)
    gy = np.linspace(x2_min, x2_max, grid)

    Z = []
    best_overall = None
    best_inputs = None

    for xv in gx:
        row = []
        for yv in gy:
            best = None
            best_in = None
            for _ in range(samples):
                vec = {}
                for c in inputs:
                    lo, hi = bounds[c]
                    vec[c] = float(np.random.uniform(lo, hi))
                vec[x1] = float(xv)
                vec[x2] = float(yv)
                x_arr = np.array([[vec[c] for c in inputs]], dtype=float)
                pred = float(rf.predict(x_arr)[0])
                if best is None or (objective == "min" and pred < best) or (objective != "min" and pred > best):
                    best = pred
                    best_in = vec
            row.append(best)
            if best_overall is None or (objective == "min" and best < best_overall) or (objective != "min" and best > best_overall):
                best_overall = best
                best_inputs = best_in
        Z.append(row)

    return {
        "x1": x1,
        "x2": x2,
        "output": output,
        "grid_x": gx.tolist(),
        "grid_y": gy.tolist(),
        "Z": Z,
        "best": {"value": best_overall, "inputs": best_inputs},
    }


def _param_goal_seek_df(df, payload):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    num, inputs, outputs = _split_inputs_outputs(df)
    output = payload.get("output", outputs[0])
    target = float(payload.get("target", 0.0))
    tolerance = float(payload.get("tolerance", 1e-3))
    samples = int(payload.get("samples", 1500))
    if output not in outputs:
        return {"error": "Invalid output selection."}

    X = num[inputs].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(num[output], errors="coerce")
    m = X.notna().all(axis=1) & y.notna()
    X = X[m]
    y = y[m]
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    bounds = {}
    for c in inputs:
        s = X[c]
        bounds[c] = (float(s.quantile(0.02)), float(s.quantile(0.98)))

    best = None
    best_in = None
    for _ in range(samples):
        vec = {}
        for c in inputs:
            lo, hi = bounds[c]
            vec[c] = float(np.random.uniform(lo, hi))
        x_arr = np.array([[vec[c] for c in inputs]], dtype=float)
        pred = float(rf.predict(x_arr)[0])
        err = abs(pred - target)
        if best is None or err < best:
            best = err
            best_in = {"predicted": pred, "inputs": vec}

    return {
        "target": target,
        "tolerance": tolerance,
        "best": best_in,
        "abs_error": float(best) if best is not None else None,
        "within_tolerance": bool(best is not None and best <= tolerance),
    }


@app.get("/v1/param/meta")
def param_meta(_token: str = Depends(require_auth)):
    df = _combined_df()
    _, inputs, outputs = _split_inputs_outputs(df)
    return {"inputs": inputs, "outputs": outputs}


@app.post("/v1/param/meta")
def param_meta_upload(
    file: UploadFile | None = File(default=None),
    _token: str = Depends(require_auth),
):
    df = _read_upload_df(file, DATASETS["combined"])
    _, inputs, outputs = _split_inputs_outputs(df)
    return {"inputs": inputs, "outputs": outputs}


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
    from scipy.optimize import minimize

    p = _cost_defaults()
    p.update(payload or {})
    weights = payload.get("weights", {"frag": 1.0, "ppv": 1.0, "air": 0.7})
    use_frag = bool(payload.get("use_frag", True))
    use_ppv = bool(payload.get("use_ppv", True))
    use_air = bool(payload.get("use_air", True))
    method = payload.get("method", "SLSQP")

    def obj(x):
        trial = p.copy()
        trial["B"], trial["S"], trial["sub"] = float(x[0]), float(x[1]), float(x[2])
        res = _cost_metrics(trial)
        pen = _cost_penalties(trial, res, weights, use_ppv, use_air, use_frag)
        return res["cost"] + pen["frag"] + pen["ppv"] + pen["air"]

    x0 = np.array([p["B"], p["S"], p["sub"]], dtype=float)
    res = minimize(
        obj,
        x0,
        method=("SLSQP" if method == "SLSQP" else "trust-constr"),
        bounds=_cost_bounds(p),
        constraints=_cost_constraints(p),
        options=dict(maxiter=300, ftol=1e-7),
    )
    best = p.copy()
    best["B"], best["S"], best["sub"] = float(res.x[0]), float(res.x[1]), float(res.x[2])
    result = _cost_metrics(best)
    result["penalties"] = _cost_penalties(best, result, weights, use_ppv, use_air, use_frag)
    result["constraint_checks"] = _cost_constraint_summary(best, result)
    return {"success": bool(res.success), "message": str(res.message), "result": result}


@app.post("/v1/cost/pareto")
def cost_pareto(payload: dict = Body(default={}), _token: str = Depends(require_auth)):
    import numpy as np
    from scipy.optimize import minimize

    p = _cost_defaults()
    p.update(payload or {})
    w_list = [0.0, 1.0, 2.0]
    rows = []
    x0 = np.array([p["B"], p["S"], p["sub"]], dtype=float)
    method = payload.get("method", "SLSQP")

    for wf in w_list:
        for wp in w_list:
            for wa in w_list:
                weights = {"frag": wf, "ppv": wp, "air": wa}
                def obj(x):
                    trial = p.copy()
                    trial["B"], trial["S"], trial["sub"] = float(x[0]), float(x[1]), float(x[2])
                    res = _cost_metrics(trial)
                    pen = _cost_penalties(trial, res, weights, wp > 0, wa > 0, wf > 0)
                    return res["cost"] + pen["frag"] + pen["ppv"] + pen["air"]
                try:
                    res = minimize(
                        obj,
                        x0,
                        method=("SLSQP" if method == "SLSQP" else "trust-constr"),
                        bounds=_cost_bounds(p),
                        constraints=_cost_constraints(p),
                        options=dict(maxiter=200, ftol=1e-7),
                    )
                    x = res.x if res.success else x0
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
    return {"rows": rows, "frontier": frontier}

