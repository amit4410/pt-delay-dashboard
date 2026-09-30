
"""Auto-bootstrap: generate data + train model on first import."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA_RAW = _ROOT / "data" / "raw"
_MODELS_DIR = _ROOT / "models"

_DATA_RAW.mkdir(parents=True, exist_ok=True)
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not (_DATA_RAW / "trips.csv").exists():
    from src.generate_data import generate_all
    generate_all()

if not (_MODELS_DIR / "delay_rf.joblib").exists():
    from src.model import train
    train()