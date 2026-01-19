from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile
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


def _read_upload_df(up: UploadFile):
    import pandas as pd
    import io

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


@app.post("/v1/data/preview")
def data_preview(file: UploadFile = File(...), _token: str = Depends(require_auth)):
    df = _read_upload_df(file)
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


@app.post("/v1/flyrock/predict")
def flyrock_predict(
    file: UploadFile = File(...),
    inputs_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file)
    num = df.apply(pd.to_numeric, errors="coerce")
    y = num.iloc[:, -1]
    X = num.iloc[:, :-1]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    if X.shape[0] < 30 or X.shape[1] < 2:
        return {"error": "Need >=30 rows and >=2 numeric features after cleaning."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)
    score = float(rf.score(Xtr, ytr))

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

    return {"prediction": yhat, "train_r2": score, "features": list(X.columns), "feature_stats": stats}


@app.post("/v1/slope/predict")
def slope_predict(
    file: UploadFile = File(...),
    inputs_json: str | None = Form(default=None),
    _token: str = Depends(require_auth),
):
    import json
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    req = dict(
        gamma=pick("gamma", "unit weight"),
        c=pick("c", "cohesion"),
        phi=pick("phi", "friction angle"),
        beta=pick("beta", "slope angle"),
        H=pick("h", "height"),
        ru=pick("ru", "pore pressure ratio"),
        status=pick("status", "label", "class"),
    )
    if None in req.values():
        return {"error": "Missing required columns: gamma, c, phi, beta, H, ru, status"}

    num = df[[req["H"], req["beta"], req["c"], req["phi"], req["gamma"], req["ru"]]].apply(
        pd.to_numeric, errors="coerce"
    )
    ylab = (
        df[req["status"]]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"stable": 1, "failure": 0, "failed": 0, "unstable": 0})
    )
    m = num.notna().all(axis=1) & ylab.notna()
    X = num[m]
    y = ylab[m]
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

    return {"prob_stable": prob, "feature_stats": stats, "features": list(X.columns)}


@app.post("/v1/delay/predict")
def delay_predict(
    file: UploadFile = File(...),
    _token: str = Depends(require_auth),
):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = _read_upload_df(file)
    cols = {c.lower().strip(): c for c in df.columns}

    def getc(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    Xc = getc("x")
    Yc = getc("y")
    Dc = getc("depth", "hole depth (m)", "hole_depth")
    Cc = getc("charge", "explosive mass", "charge_kg")
    Zc = getc("z", "elev", "elevation", "rl")
    Tc = getc("delay", "predicted delay (ms)")

    if None in (Xc, Yc, Dc, Cc):
        return {"error": "CSV must include at least X, Y, Depth, Charge columns."}

    keep = [c for c in [Dc, Cc, Xc, Yc, Zc] if c]
    num = df[keep].apply(pd.to_numeric, errors="coerce").dropna()
    X = num.values
    if Tc and Tc in df.columns:
        y = pd.to_numeric(df[Tc], errors="coerce").dropna()
        X = X[: len(y)]
    else:
        y = np.clip(10 + 0.02 * X[:, 0] + 0.0005 * X[:, 2], 5, 250)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mdl = RandomForestRegressor(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)

    Xs = sc.transform(X)
    yhat = mdl.predict(Xs)
    dfv = pd.DataFrame(
        {
            "X": pd.to_numeric(df[Xc], errors="coerce"),
            "Y": pd.to_numeric(df[Yc], errors="coerce"),
            "Delay": yhat,
        }
    ).dropna()

    # limit to 2000 points for payload size
    if len(dfv) > 2000:
        dfv = dfv.sample(2000, random_state=42)

    return {"points": dfv.to_dict(orient="records")}

