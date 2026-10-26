"use client";

import React from "react";
import Link from "next/link";
import { useRaceControl } from "@/hooks/useRaceControl";
import { useRaceStore } from "@/store/raceStore";

export function PitwallHeader() {
  const { advance, isCalculating, isRaceComplete, currentLap, totalLaps, isConnected } = useRaceControl();
  const { config, lapData, error, setError } = useRaceStore();

  const cond = lapData?.conditions;
  let trackStatus = { label: "GREEN TRACK", color: "var(--neon-green)", bg: "rgba(0, 230, 118, 0.15)" };
  if (cond?.has_red_flag) {
    trackStatus = { label: "RED FLAG", color: "#FF3B30", bg: "rgba(255, 59, 48, 0.2)" };
  } else if (cond?.has_safety_car) {
    trackStatus = { label: "SAFETY CAR", color: "var(--racing-amber)", bg: "rgba(255, 140, 0, 0.2)" };
  } else if (cond?.has_vsc) {
    trackStatus = { label: "VIRTUAL SC", color: "#FFD60A", bg: "rgba(255, 214, 10, 0.2)" };
  }

  const lapProgress = totalLaps > 0 ? (currentLap / totalLaps) * 100 : 0;

  return (
    <header className="pitwall-header">
      {/* Left: Branding & Race Info */}
      <div style={{ display: "flex", alignItems: "center", gap: "18px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              background: "var(--f1-red)",
              color: "#fff",
              fontWeight: 900,
              fontSize: "0.85rem",
              padding: "4px 8px",
              borderRadius: "4px",
              letterSpacing: "0.08em",
            }}
          >
            F1-SIS
          </div>
          <div>
            <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "#fff", letterSpacing: "-0.01em" }}>
              {config.raceName || "Grand Prix"}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <span>Driver: <strong style={{ color: "#fff" }}>{config.driver}</strong></span>
              <span>•</span>
              <span style={{ textTransform: "capitalize", color: "var(--electric-blue)" }}>{config.profile}</span>
            </div>
          </div>
        </div>

        {/* Track Condition Badge */}
        <div
          className="font-mono"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            padding: "4px 10px",
            borderRadius: "9999px",
            background: trackStatus.bg,
            color: trackStatus.color,
            border: `1px solid ${trackStatus.color}44`,
            fontSize: "0.75rem",
            fontWeight: 800,
            letterSpacing: "0.05em",
          }}
        >
          <span
            style={{
              width: "6px",
              height: "6px",
              borderRadius: "50%",
              backgroundColor: trackStatus.color,
              boxShadow: `0 0 8px ${trackStatus.color}`,
            }}
          />
          {trackStatus.label}
        </div>
      </div>

      {/* Center: Lap Controller */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "6px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button
            type="button"
            onClick={advance}
            disabled={isCalculating || isRaceComplete}
            className="btn-next-lap"
            title="Press Space or Right Arrow to advance"
          >
            {isCalculating ? (
              <>
                <span style={{ animation: "spin 1.5s linear infinite", display: "inline-block" }}>⚙</span>
                <span>Calculating Lap {currentLap}...</span>
              </>
            ) : isRaceComplete ? (
              <>
                <span>🏁</span>
                <span>Race Completed</span>
              </>
            ) : (
              <>
                <span>▶</span>
                <span>NEXT LAP</span>
                <span
                  style={{
                    background: "rgba(0, 0, 0, 0.35)",
                    fontSize: "0.7rem",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontWeight: 700,
                    marginLeft: "4px",
                  }}
                >
                  SPACE
                </span>
              </>
            )}
          </button>
        </div>

        {/* Lap Progress bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", width: "240px" }}>
          <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Lap {currentLap}
          </span>
          <div
            style={{
              flex: 1,
              height: "5px",
              background: "rgba(255, 255, 255, 0.1)",
              borderRadius: "3px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${lapProgress}%`,
                height: "100%",
                background: "var(--f1-red)",
                transition: "width 0.3s ease",
              }}
            />
          </div>
          <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            {totalLaps}
          </span>
        </div>
      </div>

      {/* Right: Socket Status & Navigation */}
      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        <div
          title={isConnected ? "WebSocket streaming live" : "Connecting..."}
          style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem", color: "var(--text-dim)" }}
        >
          <span
            style={{
              width: "7px",
              height: "7px",
              borderRadius: "50%",
              background: isConnected ? "var(--neon-green)" : "var(--racing-amber)",
              boxShadow: isConnected ? "0 0 8px var(--neon-green-glow)" : "none",
            }}
          />
          <span className="font-mono">{isConnected ? "ONLINE" : "RECONNECTING"}</span>
        </div>

        <Link
          href="/race/replay"
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border-subtle)",
            color: "var(--text-main)",
            padding: "8px 14px",
            borderRadius: "var(--radius-md)",
            fontSize: "0.8rem",
            fontWeight: 700,
            textDecoration: "none",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            transition: "all 0.15s ease",
          }}
        >
          <span>📊</span>
          <span>Replay / Scrubber</span>
        </Link>
      </div>

      {error && (
        <div
          style={{
            position: "absolute",
            bottom: "-38px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#2A080C",
            border: "1px solid var(--f1-red)",
            color: "#fff",
            padding: "5px 16px",
            borderRadius: "var(--radius-sm)",
            fontSize: "0.75rem",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            zIndex: 50,
            boxShadow: "0 4px 16px rgba(0, 0, 0, 0.6)",
          }}
        >
          <span>⚠ {error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontWeight: 800,
              fontSize: "0.8rem",
            }}
          >
            ✕
          </button>
        </div>
      )}
    </header>
  );
}
