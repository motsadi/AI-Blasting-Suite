from __future__ import annotations

import sys
from pathlib import Path

# Make `backend/app` importable as top-level `app.*`
ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Re-export the existing FastAPI app
from app.main import app  # noqa: E402

