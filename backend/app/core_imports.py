from __future__ import annotations

import os
import sys
from pathlib import Path


def add_core_bundle_to_path() -> Path:
    """
    Add the read-only core bundle folder (existing Python modules) to sys.path.
    We avoid modifying those modules; the backend wraps them.
    """
    env_path = os.getenv("CORE_BUNDLE_PATH")
    if env_path:
        p = Path(env_path).resolve()
    else:
        # backend/app/ -> backend/ -> repo_root/AI_Blasting_Suite_Full_Streamlit_Bundle
        p = (Path(__file__).resolve().parents[2] / "AI_Blasting_Suite_Full_Streamlit_Bundle").resolve()

    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p

