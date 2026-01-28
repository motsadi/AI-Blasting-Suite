## Backend (FastAPI → Cloud Run)

### Local dev

From repo root:

```bash
cd backend
python -m venv .venv
. .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Environment (optional)

- `BLAST_REQUIRE_AUTH=false` to disable auth locally
- `BLAST_LOCAL_ASSETS_DIR` path to joblib models/scaler
- `BLAST_LOCAL_DATA_DIR` path to datasets (CSV/XLSX)
- `BLAST_GCS_BUCKET` and `BLAST_GCS_PREFIX` to sync assets/datasets on startup

### Upload assets to GCS (one-time)

```bash
export BLAST_GCS_BUCKET="your-bucket"
export BLAST_GCS_PREFIX="assets/"
export BLAST_LOCAL_ASSETS_DIR="C:\path\to\joblib"
export BLAST_LOCAL_DATA_DIR="C:\path\to\datasets"
python backend/tools/upload_assets_to_gcs.py
```

### Endpoints (initial)

- `GET /health`
- `GET /v1/assets/status`
- `POST /v1/predict` (empirical + optional ML if model assets present)

### Core logic import

The backend imports the existing modules from `../AI_Blasting_Suite_Full_Streamlit_Bundle` without modifying them.

