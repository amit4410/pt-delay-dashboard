"""Central configuration for the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CITY_NAME = "Demo City"
NUM_ROUTES = 12
NUM_VEHICLES = 120
DAYS_OF_DATA = 7
TRIPS_PER_DAY = 2000

OTP_EARLY = -1.0
OTP_LATE = 5.0

BUS_CAPACITY = 80
