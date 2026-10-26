"""
src/api/routers/sandbox.py
==========================
REST endpoint for the Strategy Sandbox: hypotheticals and what-if evaluations.
"""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

from src.api.schemas import SandboxRequest, SandboxResult
from src.api.session import get_session
from src.api.engine import evaluate_sandbox

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


@router.post("/simulate", response_model=SandboxResult)
async def run_sandbox_simulation(req: SandboxRequest) -> dict[str, Any]:
    """
    Evaluates a user-forced hypothetical strategy (e.g. pit this lap for HARDs)
    against the current race state and compares it with the AI baseline recommendation.
    """
    session = get_session()
    if not session.is_initialised():
        raise HTTPException(status_code=400, detail="Race session is not initialised")

    try:
        result = await evaluate_sandbox(
            session=session,
            action=req.action,
            compound=req.compound,
            stay_out_laps=req.stay_out_laps,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sandbox simulation failed: {e}")
