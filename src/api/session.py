"""
src/api/session.py
==================
In-memory race session state. One session per server lifetime (single-user demo).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# ── lazy imports keep startup fast ────────────────────────────────────────────
def _load_strategy_engine():
    import sys
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from strategy_engine.engine import StrategyEngine
    from strategy_engine.types import StrategyEngineConfig
    from decision_engine.rl.dqn import RLDecisionPolicy
    from race_state.manager import RaceStateManager
    from lap_time.adapter import LapTimeAdapter
    from tyre_deg.adapter import TyreDegAdapter
    from pitstop.adapter import PitstopAdapter
    from sc_risk.adapter import SCRiskAdapter

    return (StrategyEngine, StrategyEngineConfig, RLDecisionPolicy,
            RaceStateManager, LapTimeAdapter, TyreDegAdapter,
            PitstopAdapter, SCRiskAdapter)


@dataclass
class RaceSession:
    # Config (locked at setup)
    race_file: str = ""
    driver: str = "VER"
    profile: str = "balanced"
    rollouts: int = 15          # keep low for demo speed; configurable at setup
    horizon_laps: int = 8

    # Runtime state
    current_lap: int = 0
    total_laps: int = 0
    is_running: bool = False
    is_complete: bool = False

    # Data
    df_laps: Optional[pd.DataFrame] = None
    lap_cache: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Pipeline objects (initialised on setup)
    strategy_engine: Any = None
    decision_policy: Any = None
    state_manager: Any = None

    # User Strategic & Tactical Overrides
    active_override: Optional[dict[str, Any]] = None
    adopted_strategy_id: Optional[str] = None
    ego_driver_overrides: Optional[dict[str, Any]] = None

    # WebSockets — active connections
    websockets: set[Any] = field(default_factory=set)

    # Progress callback (set per lap_advance call)
    _progress_cb: Optional[Callable] = None

    def is_initialised(self) -> bool:
        return self.df_laps is not None

    async def send(self, payload: dict):
        """Broadcast a JSON message over all active WebSockets."""
        dead = []
        for ws in list(self.websockets):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.websockets.discard(ws)

    async def progress(self, message: str, pct: int):
        await self.send({"event": "progress", "message": message, "pct": pct})


# Singleton session shared across the FastAPI app
_session = RaceSession()


def get_session() -> RaceSession:
    return _session


def reset_session() -> RaceSession:
    global _session
    _session = RaceSession()
    return _session
