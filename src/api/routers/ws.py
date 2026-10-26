"""
src/api/routers/ws.py
=====================
WebSocket endpoint at /ws/race for streaming lap progress and real-time events.
"""
from __future__ import annotations

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.session import get_session
from src.api.engine import evaluate_lap

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/race")
async def race_websocket(websocket: WebSocket):
    await websocket.accept()
    session = get_session()
    session.websockets.add(websocket)

    # Send current status on connect
    await websocket.send_json({
        "event": "connected",
        "current_lap": session.current_lap,
        "total_laps": session.total_laps,
        "is_complete": session.is_complete,
        "driver": session.driver,
        "is_initialised": session.is_initialised(),
    })

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
            except Exception:
                continue

            event_type = msg.get("event")

            if event_type == "ping":
                await websocket.send_json({"event": "pong"})

            elif event_type == "set_override":
                act = msg.get("action")
                if act == "clear":
                    session.active_override = None
                    session.adopted_strategy_id = None
                elif act == "adopt_strategy":
                    session.adopted_strategy_id = msg.get("strategy_id")
                else:
                    target_lap = session.current_lap if act == "pit_this_lap" else session.current_lap + 1
                    session.active_override = {
                        "action": act,
                        "compound": (msg.get("compound") or "MEDIUM").upper(),
                        "target_lap": target_lap,
                    }
                await session.send({
                    "event": "override_updated",
                    "active_override": session.active_override,
                    "adopted_strategy_id": session.adopted_strategy_id,
                })

            elif event_type == "advance":
                if not session.is_initialised():
                    await websocket.send_json({
                        "event": "error",
                        "message": "Race session is not initialised. Please configure a race first.",
                    })
                    continue

                if session.current_lap > session.total_laps:
                    session.is_complete = True
                    await session.send({
                        "event": "race_complete",
                        "total_laps": session.total_laps,
                        "message": "Race complete!",
                    })
                    continue

                lap = session.current_lap
                await session.send({
                    "event": "lap_start",
                    "lap": lap,
                    "total": session.total_laps,
                })

                try:
                    payload = await evaluate_lap(session, lap)
                    session.lap_cache[lap] = payload

                    await session.send({
                        "event": "lap_complete",
                        "data": payload,
                    })

                    session.current_lap += 1
                    if session.current_lap > session.total_laps:
                        session.is_complete = True
                        await session.send({
                            "event": "race_complete",
                            "total_laps": session.total_laps,
                            "message": "Race complete!",
                        })
                except Exception as err:
                    await session.send({
                        "event": "error",
                        "message": f"Calculation error on Lap {lap}: {err}",
                    })

    except WebSocketDisconnect:
        session.websockets.discard(websocket)
    except Exception:
        session.websockets.discard(websocket)
    finally:
        session.websockets.discard(websocket)
