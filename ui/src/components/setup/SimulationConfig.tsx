"use client";

import React from "react";

interface SimulationConfigProps {
  rollouts: number;
  horizonLaps: number;
  onChangeRollouts: (val: number) => void;
  onChangeHorizon: (val: number) => void;
}

export function SimulationConfig({
  rollouts,
  horizonLaps,
  onChangeRollouts,
  onChangeHorizon,
}: SimulationConfigProps) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        padding: "16px",
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "20px",
      }}
    >
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
          <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Monte Carlo Rollouts
          </label>
          <span className="font-mono" style={{ fontWeight: 800, color: "var(--text-primary)" }}>
            {rollouts} runs
          </span>
        </div>
        <input
          type="range"
          min={5}
          max={100}
          step={5}
          value={rollouts}
          onChange={(e) => onChangeRollouts(parseInt(e.target.value, 10))}
          style={{ width: "100%", accentColor: "var(--f1-red)", cursor: "pointer" }}
        />
        <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          Higher values generate tighter variance bands at nominal computation time.
        </div>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
          <label style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Lookahead Horizon
          </label>
          <span className="font-mono" style={{ fontWeight: 800, color: "var(--color-info)" }}>
            {horizonLaps} Laps
          </span>
        </div>
        <input
          type="range"
          min={3}
          max={20}
          step={1}
          value={horizonLaps}
          onChange={(e) => onChangeHorizon(parseInt(e.target.value, 10))}
          style={{ width: "100%", accentColor: "var(--color-info)", cursor: "pointer" }}
        />
        <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          Forward lap projection depth for degradation cliffs and safety car windows.
        </div>
      </div>
    </div>
  );
}
