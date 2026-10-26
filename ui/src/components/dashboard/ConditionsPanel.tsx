"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";

export function ConditionsPanel() {
  const { lapData } = useRaceStore();
  const cond = lapData?.conditions;

  const trackTemp = cond?.track_temp ? `${Math.round(cond.track_temp)}°C` : "42°C";
  const airTemp = cond?.air_temp ? `${Math.round(cond.air_temp)}°C` : "28°C";
  const isRaining = cond?.rainfall ?? false;

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "14px 16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "10px" }}>
        <div className="sis-card-title">
          <span>🌤</span>
          <span>Track & Environmental Telemetry</span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "8px",
          textAlign: "center",
        }}
      >
        {/* Track Temp */}
        <div
          style={{
            background: "var(--surface-secondary)",
            padding: "8px 6px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Track Temp
          </div>
          <div className="font-mono" style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "1px" }}>
            {trackTemp}
          </div>
        </div>

        {/* Air Temp */}
        <div
          style={{
            background: "var(--surface-secondary)",
            padding: "8px 6px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Air Temp
          </div>
          <div className="font-mono" style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "1px" }}>
            {airTemp}
          </div>
        </div>

        {/* Rain Status */}
        <div
          style={{
            background: "var(--surface-secondary)",
            padding: "8px 6px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Surface
          </div>
          <div
            className="font-mono"
            style={{
              fontSize: "0.92rem",
              fontWeight: 800,
              color: isRaining ? "var(--color-info)" : "var(--color-success)",
              marginTop: "2px",
            }}
          >
            {isRaining ? "WET" : "DRY"}
          </div>
        </div>
      </div>
    </div>
  );
}
