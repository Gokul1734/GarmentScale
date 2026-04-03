from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.measure import build_jean_style, run_all

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
