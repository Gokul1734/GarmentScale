# Body Measure MVP — deploy as a Web Service (Render, Fly.io, Railway, etc.)
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

ENV PYTHONUNBUFFERED=1

# Render (and others) set PORT; default 8000 for local Docker
CMD sh -c 'python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}'
