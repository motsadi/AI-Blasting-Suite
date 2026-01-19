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

### Endpoints (initial)

- `GET /health`
- `GET /v1/assets/status`
- `POST /v1/predict` (empirical + optional ML if model assets present)

### Core logic import

The backend imports the existing modules from `../AI_Blasting_Suite_Full_Streamlit_Bundle` without modifying them.

