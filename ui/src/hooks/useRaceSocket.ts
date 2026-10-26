"use client";

import { useEffect, useCallback } from "react";
import { useRaceStore } from "@/store/raceStore";
import { WS_BASE_URL } from "@/lib/constants";
import { advanceRaceLap, setRaceOverride } from "@/lib/api";

let socketInstance: WebSocket | null = null;
let reconnectTimer: NodeJS.Timeout | null = null;

export function useRaceSocket() {
  const {
    startLapCalculation,
    updateProgress,
    applyLapComplete,
    setRaceComplete,
    setSocketStatus,
    setSessionInfo,
    setIsConfigured,
    setError,
    setActiveOverride,
    setAdoptedStrategyId,
    isCalculating,
    isRaceComplete,
    currentLap,
    totalLaps,
  } = useRaceStore();

  const connect = useCallback(() => {
    if (typeof window === "undefined") return;
    if (
      socketInstance &&
      (socketInstance.readyState === WebSocket.OPEN ||
        socketInstance.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setSocketStatus("connecting");

    try {
      const ws = new WebSocket(WS_BASE_URL);
      socketInstance = ws;

      ws.onopen = () => {
        setSocketStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.event) {
            case "connected":
              if (msg.total_laps && msg.is_initialised) {
                setSessionInfo(msg.current_lap || 1, msg.total_laps, msg.driver);
                setIsConfigured(true);
              }
              break;

            case "lap_start":
              startLapCalculation(msg.lap, msg.total);
              break;

            case "progress":
              updateProgress(msg.message, msg.pct);
              break;

            case "lap_complete":
              applyLapComplete(msg.data);
              break;

            case "override_updated":
              if ("active_override" in msg) {
                setActiveOverride(msg.active_override ?? null);
              }
              if ("adopted_strategy_id" in msg) {
                setAdoptedStrategyId(msg.adopted_strategy_id ?? null);
              }
              break;

            case "race_complete":
              setRaceComplete(msg.total_laps);
              break;

            case "error":
              setError(msg.message || "An unknown error occurred during calculation");
              break;

            default:
              break;
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onerror = () => {
        setSocketStatus("error");
      };

      ws.onclose = () => {
        setSocketStatus("disconnected");
        socketInstance = null;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
          connect();
        }, 3000);
      };
    } catch {
      setSocketStatus("error");
    }
  }, [
    startLapCalculation,
    updateProgress,
    applyLapComplete,
    setRaceComplete,
    setSocketStatus,
    setSessionInfo,
    setError,
    setActiveOverride,
    setAdoptedStrategyId,
  ]);

  useEffect(() => {
    connect();
  }, [connect]);

  const armOverride = useCallback(
    async (action: string, compound?: string, strategy_id?: string) => {
      if (socketInstance && socketInstance.readyState === WebSocket.OPEN) {
        socketInstance.send(
          JSON.stringify({ event: "set_override", action, compound, strategy_id })
        );
      } else {
        try {
          const res = await setRaceOverride(action, compound, strategy_id);
          setActiveOverride(res.active_override ?? null);
          setAdoptedStrategyId(res.adopted_strategy_id ?? null);
        } catch (err: any) {
          setError(err.message || "Failed to set strategy override");
        }
      }
    },
    [setActiveOverride, setAdoptedStrategyId, setError]
  );

  const advance = useCallback(async () => {
    if (isCalculating || isRaceComplete) return;

    if (socketInstance && socketInstance.readyState === WebSocket.OPEN) {
      socketInstance.send(JSON.stringify({ event: "advance" }));
    } else {
      // Fallback to HTTP REST endpoint
      try {
        startLapCalculation(currentLap || 1, totalLaps);
        const data = await advanceRaceLap();
        if (data && data.lap) {
          applyLapComplete(data);
        }
      } catch (err: any) {
        setError(err.message || "Failed to advance lap via REST");
      }
    }
  }, [
    isCalculating,
    isRaceComplete,
    currentLap,
    totalLaps,
    startLapCalculation,
    applyLapComplete,
    setError,
  ]);

  return {
    advance,
    armOverride,
    isConnected: socketInstance?.readyState === WebSocket.OPEN,
  };
}
