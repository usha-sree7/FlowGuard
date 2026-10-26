"""Project paths and provisional, documented defaults.

Everything here that is a modelling choice (weights, band edges) is provisional and
is reported alongside every experiment result so that numbers stay reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FLOODGUARD_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # original files exactly as downloaded (never modified)
PROCESSED_DIR = DATA_DIR / "processed"  # derived, standardised copies
RESULTS_DIR = ROOT / "results"
# Hosted demo (read-only filesystem, no background jobs): imports and new experiments are disabled.
READ_ONLY = os.environ.get("FLOODGUARD_READONLY") == "1"

# Uncertainty index U = 100 * (wQ*Q + wD*D + wF*F). Provisional, not calibrated.
U_WEIGHTS = {"Q": 0.40, "D": 0.30, "F": 0.30}
U_BANDS = [(25.0, "Low"), (50.0, "Moderate"), (75.0, "High"), (float("inf"), "Very High")]
# U below this (and no essential input missing) -> "Confident".
CONFIDENT_MAX_U = 25.0

# Five alert levels. Proxy labels use the train-split level thresholds mean + k*sd, k = LEVEL_K (calibration.py);
# fuzzy / neuro-fuzzy scores use fixed band edges on the 0-100 output (not tuned on test data).
RISK_CLASSES = ["Normal", "Low", "Moderate", "High", "Critical"]
RISK_EDGES = [20.0, 40.0, 60.0, 80.0]
N_CLASSES = len(RISK_CLASSES)
# Positive class of the binary alert metrics: High or Critical (level above mean + 2 sd = the paper's flood threshold).
ALERT_CLASS = 3

# Chronological split: first TRAIN_FRACTION of rows used for every data-derived parameter.
TRAIN_FRACTION = 0.70
DEFAULT_SEED = 20261025


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
