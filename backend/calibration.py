"""
Tape-reference calibration: learn per-field scale from your four photos + known Akka values.
After training, new sessions multiply raw model inches by saved multipliers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Akka full-body reference (inches). Used only to compute multipliers during /api/train.
REFERENCE_FULL_BODY_INCHES: dict[str, float] = {
    "Bust": 32.0,
    "Underbust": 28.0,
    "Waist": 30.0,
    "High Hip": 35.0,
    "Full Hip": 38.0,
    "Shoulder Width": 14.5,
    "Neck": 13.0,
    "Arm Length": 22.0,
    "Bicep": 12.0,
    "Wrist": 6.5,
    "Thigh": 21.0,
    "Knee": 16.0,
    "Calf": 14.0,
    "Ankle": 9.0,
    "Inseam": 31.0,
    "Outseam": 42.0,
    "Torso Length": 16.0,
    "Crotch Depth": 13.0,
    "Total Height": 66.0,
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
    payload = {"mults": mults, "reference": "akka_tape", "note": "ref_inch / raw_inch from training images"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def default_mults_until_trained() -> dict[str, float]:
    """
    If you have not run POST /api/train yet, optional bootstrap ratios (Akka / one nominal model run).
    Disable with BODY_MEASURE_DEFAULT_CALIB=0 — then only raw geometry until you train.
    """
    if os.environ.get("BODY_MEASURE_DEFAULT_CALIB", "1").strip().lower() in ("0", "false", "no"):
        return {}
    nominal = {
        "Bust": 37.9,
        "Underbust": 33.3,
        "Waist": 35.6,
        "High Hip": 41.5,
        "Full Hip": 45.1,
        "Shoulder Width": 17.2,
        "Neck": 15.4,
        "Arm Length": 26.1,
        "Bicep": 14.2,
        "Wrist": 7.8,
        "Thigh": 24.9,
        "Knee": 18.9,
        "Calf": 16.6,
        "Ankle": 10.7,
        "Inseam": 36.7,
        "Outseam": 49.8,
        "Torso Length": 18.9,
        "Crotch Depth": 15.3,
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
