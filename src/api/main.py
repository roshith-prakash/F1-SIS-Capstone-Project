"""
src/api/main.py
===============
FastAPI entrypoint for F1-SIS Strategy Intelligence Dashboard backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import race, sandbox, ws

app = FastAPI(
    title="F1-SIS Pit Wall API",
    description="Real-time race strategy decision support API powered by Monte Carlo simulation.",
    version="1.0.0",
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(race.router)
app.include_router(sandbox.router)
app.include_router(ws.router)


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "F1-SIS Pit Wall Strategy Engine",
        "version": "1.0.0",
    }
