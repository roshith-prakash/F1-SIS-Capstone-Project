"use client";

import React from "react";
import Link from "next/link";
import { useRaceControl } from "@/hooks/useRaceControl";
import { useRaceStore } from "@/store/raceStore";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function Header() {
  const { advance, isCalculating, isRaceComplete, currentLap, totalLaps, isConnected } = useRaceControl();
  const { config, lapData, error, setError, calculationStartTime, lastEvaluationTimeMs } = useRaceStore();

  const [elapsedSec, setElapsedSec] = React.useState(0);

  React.useEffect(() => {
    if (!isCalculating || !calculationStartTime) {
      setElapsedSec(0);
      return;
    }
    const timer = setInterval(() => {
      setElapsedSec((Date.now() - calculationStartTime) / 1000);
    }, 80);
    return () => clearInterval(timer);
  }, [isCalculating, calculationStartTime]);

  const cond = lapData?.conditions;
  let trackType: "track_green" | "track_sc" | "track_vsc" | "track_red" = "track_green";
  if (cond?.has_red_flag) trackType = "track_red";
  else if (cond?.has_safety_car) trackType = "track_sc";
  else if (cond?.has_vsc) trackType = "track_vsc";

  const lapProgress = totalLaps > 0 ? (currentLap / totalLaps) * 100 : 0;

  return (
    <header className="app-header">
      {/* Left: Branding & Session Context */}
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              background: "var(--text-primary)",
              color: "#FFFFFF",
              fontWeight: 900,
              fontSize: "0.82rem",
              padding: "3px 8px",
              borderRadius: "var(--radius-xs)",
              letterSpacing: "0.08em",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>F1-SIS</span>
            <span
              style={{
                width: "5px",
                height: "5px",
                borderRadius: "50%",
                background: "var(--f1-red)",
              }}
            />
          </div>

          <div>
            <div style={{ fontSize: "0.92rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
              {config.raceName || "Grand Prix Session"}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.72rem", color: "var(--text-secondary)" }}>
              <span>Driver: <strong style={{ color: "var(--text-primary)" }}>{config.driver}</strong></span>
              <span>•</span>
              <span style={{ textTransform: "capitalize", color: "var(--color-info)" }}>{config.profile} Strategy</span>
            </div>
          </div>
        </div>

        <StatusBadge type={trackType} size="sm" />
      </div>

      {/* Center: Lap Progress & Next Lap Trigger */}
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        {/* Lap Progress bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", width: "220px" }}>
          <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-secondary)", fontWeight: 600 }}>
            L{currentLap}
          </span>
          <div
            style={{
              flex: 1,
              height: "4px",
              background: "var(--border)",
              borderRadius: "var(--radius-pill)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${lapProgress}%`,
                height: "100%",
                background: "var(--f1-red)",
                transition: "width 0.25s ease",
              }}
            />
          </div>
          <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            {totalLaps || "--"}
          </span>
        </div>

        {/* Primary Action Button */}
        <button
          type="button"
          onClick={advance}
          disabled={isCalculating || isRaceComplete}
          className="btn-next-lap"
          title="Press Space to advance lap"
        >
          {isCalculating ? (
            <>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⚙</span>
              <span>CALCULATING LAP {currentLap}...</span>
              <span
                className="font-mono"
                style={{
                  background: "rgba(0, 0, 0, 0.28)",
                  padding: "1px 6px",
                  borderRadius: "var(--radius-xs)",
                  fontSize: "0.7rem",
                  fontWeight: 800,
                  marginLeft: "4px",
                  letterSpacing: "0.02em",
                }}
              >
                ⏱ {elapsedSec.toFixed(1)}s
              </span>
            </>
          ) : isRaceComplete ? (
            <>
              <span>🏁</span>
              <span>RACE COMPLETED</span>
            </>
          ) : (
            <>
              <span>▶</span>
              <span>NEXT LAP</span>
              <span
                style={{
                  background: "rgba(255, 255, 255, 0.22)",
                  fontSize: "0.68rem",
                  padding: "1px 5px",
                  borderRadius: "var(--radius-xs)",
                  fontWeight: 800,
                  marginLeft: "2px",
                }}
              >
                SPACE
              </span>
            </>
          )}
        </button>

        {/* Evaluation Latency Badge */}
        {lastEvaluationTimeMs !== null && !isCalculating && (
          <div
            className="font-mono"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "4px",
              fontSize: "0.72rem",
              fontWeight: 700,
              background: "var(--surface-secondary)",
              border: "1px solid var(--border)",
              padding: "3px 8px",
              borderRadius: "var(--radius-pill)",
              color: "var(--color-info)",
              whiteSpace: "nowrap",
            }}
            title={`Last Strategy Engine Pipeline Runtime: ${lastEvaluationTimeMs}ms`}
          >
            <span>⚡</span>
            <span>
              {lastEvaluationTimeMs < 1000
                ? `${Math.round(lastEvaluationTimeMs)}ms`
                : `${(lastEvaluationTimeMs / 1000).toFixed(2)}s`}
            </span>
          </div>
        )}
      </div>

      {/* Right: Data Stream Status & Quick Links */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <StatusBadge type={isConnected ? "live" : "historical"} label={isConnected ? "STREAMING" : "OFFLINE"} size="sm" />

        <Link
          href="/race/replay"
          className="btn-secondary"
          style={{ textDecoration: "none", fontSize: "0.78rem", padding: "5px 10px" }}
        >
          <span>⏪</span>
          <span>Scrubber & Replay</span>
        </Link>

        <Link
          href="/"
          className="btn-secondary"
          style={{ textDecoration: "none", fontSize: "0.78rem", padding: "5px 10px" }}
          title="Return to session configuration"
        >
          <span>⚙</span>
          <span>Setup</span>
        </Link>
      </div>

      {/* Error alert toast */}
      {error && (
        <div
          style={{
            position: "absolute",
            bottom: "-36px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--color-danger-bg)",
            border: "1px solid var(--color-danger-border)",
            color: "var(--color-danger)",
            padding: "4px 14px",
            borderRadius: "var(--radius-sm)",
            fontSize: "0.75rem",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            zIndex: 50,
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <span>⚠ {error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: "none",
              border: "none",
              color: "var(--color-danger)",
              cursor: "pointer",
              fontWeight: 800,
            }}
          >
            ✕
          </button>
        </div>
      )}
    </header>
  );
}
