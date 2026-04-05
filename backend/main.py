from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.calibration import (
    REFERENCE_FULL_BODY_INCHES,
    calibration_file_path,
    save_calibration_mults,
)
from backend.measure import build_jean_style, compute_raw_inches, run_all

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Body Measure MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = ROOT / "frontend"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "frontend missing"})


@app.get("/api/calibration")
def calibration_status():
    from backend.calibration import load_calibration_mults

    path = calibration_file_path()
    learned = load_calibration_mults()
    return {
        "calibration_file": str(path),
        "learned": learned is not None,
        "field_count": len(learned) if learned else 0,
    }


@app.post("/api/train")
async def train_tape_calibration(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
):
    """
    Upload the four reference photos + use Akka tape values to learn multipliers
    (reference_inch / raw_inch). Saves data/tape_calibration.json (or BODY_MEASURE_CALIB_PATH).
    """
    try:
        imgs = {
            "front": await front.read(),
            "back": await back.read(),
            "left": await left.read(),
            "right": await right.read(),
        }
        raw_in, errors = compute_raw_inches(imgs)
        mults: dict[str, float] = {}
        for k, ref_in in REFERENCE_FULL_BODY_INCHES.items():
            if k not in raw_in or raw_in[k] <= 0.05:
                continue
            mults[k] = ref_in / raw_in[k]
        path = save_calibration_mults(mults)
        return {
            "ok": True,
            "saved_to": str(path),
            "mults": mults,
            "raw_inches": {k: round(v, 2) for k, v in raw_in.items()},
            "warnings": errors,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "error": type(e).__name__},
        )


@app.post("/api/measure")
async def measure(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
):
    try:
        imgs = {
            "front": await front.read(),
            "back": await back.read(),
            "left": await left.read(),
            "right": await right.read(),
        }
        display, errors = run_all(imgs)
        jean = build_jean_style(display)
        return {
            "measurements": display,
            "jean_style": jean,
            "warnings": errors,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "error": type(e).__name__},
        )
