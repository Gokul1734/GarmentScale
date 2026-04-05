# Body Measure MVP — deploy as a Web Service (Render, Fly.io, Railway, etc.)
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxcb-xfixes0 \
    libxcb-render0 \
    libxcb-shm0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

ENV PYTHONUNBUFFERED=1
# Wider default FOV + multiplier lift pinhole scale toward real tape (tune per camera).
ENV BODY_MEASURE_HFOV=72
ENV BODY_MEASURE_CM_CALIB=1.0
ENV BODY_MEASURE_DEFAULT_CALIB=1

# Render (and others) set PORT; default 8000 for local Docker
CMD sh -c 'python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}'
