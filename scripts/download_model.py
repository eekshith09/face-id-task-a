"""One-time setup: download InsightFace's buffalo_l model into ./models (needs internet once).

Usage:  .venv/bin/python scripts/download_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.face_engine import FaceEngine  # noqa: E402

if __name__ == "__main__":
    FaceEngine(ROOT / "models").load()  # InsightFace downloads the model if it is missing
    print("Model ready in", ROOT / "models" / "models" / "buffalo_l")
