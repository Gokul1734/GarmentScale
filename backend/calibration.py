"""
Tape-reference calibration: learn per-field scale from your four photos + known Akka values.
After training, new sessions multiply raw model inches by saved multipliers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Full-body tape reference (inches). POST /api/train learns multipliers = these ÷ raw model inches.
REFERENCE_FULL_BODY_INCHES: dict[str, float] = {
    "Bust": 29.9,
    "Underbust": 25.1,
    "Waist": 26.5,
    "High Hip": 33.2,
    "Full Hip": 22.1,
    "Shoulder Width": 11.4,
    "Neck": 14.7,
    "Arm Length": 20.3,
    "Bicep": 10.4,
    "Wrist": 5.3,
    "Thigh": 18.9,
    "Knee": 16.3,
    "Calf": 14.5,
    "Ankle": 8.8,
    "Inseam": 30.0,
    "Outseam": 39.7,
    "Torso Length": 13.8,
    "Crotch Depth": 12.0,
    "Total Height": 66.0,  # 5'6"
}


def _calibration_path() -> Path:
    p = os.environ.get("BODY_MEASURE_CALIB_PATH", "").strip()
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent / "data" / "tape_calibration.json"


def calibration_file_path() -> str:
    return str(_calibration_path())


def load_calibration_mults() -> dict[str, float] | None:
    path = _calibration_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("mults") or data
        return {str(k): float(v) for k, v in m.items()}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_calibration_mults(mults: dict[str, float]) -> Path:
    path = _calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mults": mults,
        "reference": "tape_profile_v2",
        "note": "ref_inch / raw_inch from training images",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def default_mults_until_trained() -> dict[str, float]:
    """
    Before POST /api/train: optional bootstrap toward REFERENCE using a nominal raw run.
    Set nominal ≈ typical uncalibrated model output for your setup; here aligned so mult≈1 when raw≈ref.
    Disable with BODY_MEASURE_DEFAULT_CALIB=0 — raw geometry only until you train.
    """
    if os.environ.get("BODY_MEASURE_DEFAULT_CALIB", "1").strip().lower() in ("0", "false", "no"):
        return {}
    # Slightly above reference so first-time users get a gentle pull if raw is high (tune if needed).
    nominal = {
        "Bust": 31.5,
        "Underbust": 26.5,
        "Waist": 28.0,
        "High Hip": 35.0,
        "Full Hip": 23.5,
        "Shoulder Width": 12.0,
        "Neck": 15.5,
        "Arm Length": 21.5,
        "Bicep": 11.0,
        "Wrist": 5.6,
        "Thigh": 20.0,
        "Knee": 17.0,
        "Calf": 15.2,
        "Ankle": 9.2,
        "Inseam": 31.5,
        "Outseam": 42.0,
        "Torso Length": 14.5,
        "Crotch Depth": 12.5,
        "Total Height": 66.0,
    }
    out: dict[str, float] = {}
    for k, ref in REFERENCE_FULL_BODY_INCHES.items():
        if k not in nominal or nominal[k] <= 0:
            continue
        out[k] = ref / nominal[k]
    return out


def effective_mults() -> dict[str, float]:
    learned = load_calibration_mults()
    if learned:
        return learned
    return default_mults_until_trained()
