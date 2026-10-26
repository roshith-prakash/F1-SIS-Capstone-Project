"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";

export function CalculatingOverlay() {
  const {
    isCalculating,
    calculatingLap,
    calculationStartTime,
    totalLaps,
    progressMessage,
    progressPct,
    config,
  } = useRaceStore();

  const [elapsedSec, setElapsedSec] = React.useState(0);

  React.useEffect(() => {
    if (!isCalculating || !calculationStartTime) {
      setElapsedSec(0);
      return;
    }
    const timer = setInterval(() => {
      setElapsedSec((Date.now() - calculationStartTime) / 1000);
    }, 50);
    return () => clearInterval(timer);
  }, [isCalculating, calculationStartTime]);

  if (!isCalculating) return null;

  return (
    <div className="calculating-overlay">
      <div className="calculating-box fade-in">
        {/* Subtle Brand Tag */}
        <div
          style={{
            fontSize: "0.72rem",
            fontWeight: 800,
            letterSpacing: "0.12em",
            color: "var(--f1-red)",
            textTransform: "uppercase",
            marginBottom: "6px",
          }}
        >
          F1-SIS STRATEGY ENGINE ACTIVE
        </div>

        <h2
          style={{
            fontSize: "1.45rem",
            fontWeight: 800,
            letterSpacing: "-0.02em",
            color: "var(--text-primary)",
          }}
        >
          Evaluating Lap {calculatingLap} of {totalLaps || "--"}
        </h2>

        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          Ego Driver: <strong style={{ color: "var(--text-primary)" }}>{config.driver}</strong> · Tactical Profile:{" "}
          <strong style={{ color: "var(--color-info)", textTransform: "capitalize" }}>
            {config.profile}
          </strong>
        </p>

        {/* Progress Bar */}
        <div
          style={{
            background: "var(--surface-secondary)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-pill)",
            height: "8px",
            overflow: "hidden",
            margin: "24px 0 10px 0",
            position: "relative",
          }}
        >
          <div
            style={{
              width: `${Math.max(5, Math.min(100, progressPct))}%`,
              height: "100%",
              background: "var(--f1-red)",
              borderRadius: "var(--radius-pill)",
              transition: "width 0.25s ease-out",
            }}
          />
        </div>

        {/* Step Message & Percentage */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.78rem",
            marginBottom: "20px",
          }}
        >
          <span
            className="font-mono"
            style={{
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: "var(--f1-red)",
                display: "inline-block",
              }}
            />
            {progressMessage || "Running Monte Carlo rollout evaluations..."}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span
              className="font-mono"
              style={{
                fontSize: "0.72rem",
                fontWeight: 700,
                color: "var(--f1-red)",
                background: "var(--f1-red-subtle)",
                padding: "2px 6px",
                borderRadius: "var(--radius-xs)",
                border: "1px solid var(--f1-red-border)",
              }}
            >
              ⏱ {elapsedSec.toFixed(1)}s
            </span>
            <span className="font-mono" style={{ color: "var(--text-primary)", fontWeight: 800 }}>
              {progressPct}%
            </span>
          </div>
        </div>

        {/* Diagnostic Metadata Footer */}
        <div
          style={{
            background: "var(--surface-secondary)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            padding: "10px 14px",
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "10px",
            textAlign: "center",
            fontSize: "0.75rem",
          }}
        >
          <div>
            <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
              Elapsed Time
            </div>
            <div className="font-mono" style={{ fontWeight: 800, color: "var(--f1-red)", marginTop: "2px" }}>
              ⏱ {elapsedSec.toFixed(1)}s
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
              Lookahead Horizon
            </div>
            <div className="font-mono" style={{ fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
              {config.horizon_laps} Laps
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
              Monte Carlo Rollouts
            </div>
            <div className="font-mono" style={{ fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
              {config.rollouts} / strategy
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
              Execution
            </div>
            <div className="font-mono" style={{ fontWeight: 800, color: "var(--color-success)", marginTop: "2px" }}>
              Fast Stochastic
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
