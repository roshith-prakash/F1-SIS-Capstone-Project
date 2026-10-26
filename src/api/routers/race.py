"""
src/api/routers/race.py
=======================
REST endpoints for race lifecycle: available races, setup, advance, history, and reset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import SetupRequest, SetupResponse, HistoryLap, OverrideRequest
from src.api.session import get_session, reset_session
from src.api.engine import evaluate_lap

router = APIRouter(prefix="/api/race", tags=["race"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data_fastf1_v1" / "laps"
_DRY_RACES_CACHE: list[dict[str, Any]] | None = None
WET_COMPOUNDS = {"INTERMEDIATE", "WET", "WETS", "INTERMEDIATES"}


def _is_wet_race(csv_path: Path) -> bool:
    """Check whether a Grand Prix CSV contains wet tyre compounds or sustained rainfall."""
    try:
        df_sample = pd.read_csv(csv_path, usecols=lambda c: c in ["Compound", "Rainfall"])
        if "Compound" in df_sample.columns:
            compounds = set(df_sample["Compound"].dropna().astype(str).str.upper().unique())
            if any(c in compounds for c in WET_COMPOUNDS):
                return True
        if "Rainfall" in df_sample.columns:
            if (df_sample["Rainfall"] == True).sum() >= 2:
                return True
        return False
    except Exception:
        return False


@router.get("/available")
def get_available_races() -> list[dict[str, Any]]:
    """
    Returns available dry race CSV files from data_fastf1_v1/laps,
    with year, grand prix name, file path, and participating drivers.
    Wet races (with intermediate/wet tyres or rainfall) are filtered out.
    """
    global _DRY_RACES_CACHE
    if _DRY_RACES_CACHE is not None:
        return _DRY_RACES_CACHE

    if not DATA_DIR.exists():
        return []

    races = []
    # Scan years in descending order (e.g. 2025, 2024, 2023, ...)
    for year_dir in sorted(DATA_DIR.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for csv_file in sorted(year_dir.glob("*.csv")):
            # Filter out wet races
            if _is_wet_race(csv_file):
                continue

            gp_name = csv_file.stem.replace("_", " ")
            rel_path = f"{year_dir.name}/{csv_file.name}"

            # Extract all drivers participating in this specific Grand Prix
            try:
                driver_df = pd.read_csv(csv_file, usecols=["Driver"])
                drivers = sorted([
                    str(d).strip().upper()
                    for d in driver_df["Driver"].dropna().unique()
                    if str(d).strip() != "" and str(d).strip().lower() != "nan"
                ])
            except Exception:
                drivers = []

            races.append({
                "year": year,
                "name": f"{year} {gp_name}",
                "file": rel_path,
                "drivers": drivers,
            })

    _DRY_RACES_CACHE = races
    return races


@router.post("/setup", response_model=SetupResponse)
def setup_race(req: SetupRequest) -> SetupResponse:
    """
    Initialises a race session with the selected CSV, driver, profile, and rollouts.
    """
    session = get_session()
    csv_path = DATA_DIR / req.race_file
    if not csv_path.exists():
        # Try finding anywhere under DATA_DIR if path was passed simply
        matches = list(DATA_DIR.rglob(Path(req.race_file).name))
        if matches:
            csv_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Race file '{req.race_file}' not found")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    session.race_file = req.race_file
    session.driver = req.driver
    session.profile = req.profile
    session.rollouts = req.rollouts
    session.horizon_laps = req.horizon_laps
    session.df_laps = df
    session.total_laps = int(df["LapNumber"].max()) if "LapNumber" in df.columns else 50
    session.current_lap = 1
    session.lap_cache = {}
    session.is_complete = False
    session.is_running = True

    gp_name = csv_path.stem.replace("_", " ")
    return SetupResponse(
        ok=True,
        driver=req.driver,
        total_laps=session.total_laps,
        race_name=gp_name,
    )


@router.post("/advance")
async def advance_lap() -> dict[str, Any]:
    """
    Advances race simulation by one lap. Pushes lap_start, progress, and lap_complete events.
    """
    session = get_session()
    if not session.is_initialised():
        raise HTTPException(status_code=400, detail="Race session is not initialised")

    if session.current_lap > session.total_laps:
        session.is_complete = True
        await session.send({
            "event": "race_complete",
            "total_laps": session.total_laps,
            "message": "Race completed successfully",
        })
        return {"status": "complete", "lap": session.total_laps}

    lap = session.current_lap
    await session.send({"event": "lap_start", "lap": lap, "total": session.total_laps})

    payload = await evaluate_lap(session, lap)
    session.lap_cache[lap] = payload

    await session.send({
        "event": "lap_complete",
        "data": payload,
    })

    # Prepare for next advance
    session.current_lap += 1
    if session.current_lap > session.total_laps:
        session.is_complete = True
        await session.send({
            "event": "race_complete",
            "total_laps": session.total_laps,
            "message": "Race completed successfully",
        })

    return payload


@router.get("/history")
def get_history() -> list[dict[str, Any]]:
    """
    Returns history of evaluated laps for replay and scrubber.
    """
    session = get_session()
    history = []
    for lap_num in sorted(session.lap_cache.keys()):
        lap_data = session.lap_cache[lap_num]
        ego = lap_data.get("participants", {}).get(session.driver, {})
        rec = lap_data.get("recommendation", {})
        history.append({
            "lap": lap_num,
            "recommendation_action": rec.get("action", "STAY_OUT"),
            "recommendation_score": rec.get("score", 0.0),
            "actual_compound": ego.get("compound", "MEDIUM"),
            "actual_tyre_life": ego.get("tyre_life", 1.0),
            "actual_position": ego.get("position", 10),
            "actual_lap_time": ego.get("last_lap_time"),
            "full_data": lap_data,
        })
    return history


@router.post("/override")
async def set_override(req: OverrideRequest) -> dict[str, Any]:
    """
    Arms an alternate strategic or tactical user override (e.g. pit next lap for Mediums,
    or adopt a specific candidate strategy ID).
    """
    session = get_session()
    if not session.is_initialised():
        raise HTTPException(status_code=400, detail="Race session is not initialised")

    if req.action == "clear":
        session.active_override = None
        session.adopted_strategy_id = None
        await session.send({
            "event": "override_updated",
            "active_override": None,
            "adopted_strategy_id": None,
        })
        return {"status": "cleared", "active_override": None, "adopted_strategy_id": None}

    if req.action == "adopt_strategy":
        session.adopted_strategy_id = req.strategy_id
        await session.send({
            "event": "override_updated",
            "active_override": session.active_override,
            "adopted_strategy_id": req.strategy_id,
        })
        return {"status": "adopted", "strategy_id": req.strategy_id}

    target_lap = session.current_lap if req.action == "pit_this_lap" else session.current_lap + 1
    override_data = {
        "action": req.action,
        "compound": (req.compound or "MEDIUM").upper(),
        "target_lap": target_lap,
    }
    session.active_override = override_data
    await session.send({
        "event": "override_updated",
        "active_override": override_data,
        "adopted_strategy_id": session.adopted_strategy_id,
    })
    return {"status": "armed", "active_override": override_data, "adopted_strategy_id": session.adopted_strategy_id}


@router.post("/reset")
def reset() -> dict[str, bool]:
    """Resets the current race session."""
    reset_session()
    return {"ok": True}
