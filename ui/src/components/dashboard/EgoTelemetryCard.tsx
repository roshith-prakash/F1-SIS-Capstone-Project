"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { formatLapTime, formatInterval } from "@/lib/formatters";

export function EgoTelemetryCard() {
  const { lapData, config } = useRaceStore();
  const ego = lapData?.participants ? lapData.participants[config.driver] : null;

  const compound = ego?.compound || "MEDIUM";
  const tyreLife = ego?.tyre_life || 1;
  const position = ego?.position || 1;
  const lastLap = ego?.last_lap_time || null;
  const interval = ego?.interval_ahead || null;
  const stint = (ego?.pit_count || 0) + 1;

  // Approximate cliff limit per compound
  const cliffLimits: Record<string, number> = {
    SOFT: 22,
    MEDIUM: 32,
    HARD: 45,
    INTERMEDIATE: 30,
    WET: 35,
  };
  const maxLaps = cliffLimits[compound.toUpperCase()] || 32;
  const tyreWearPct = Math.min(100, (tyreLife / maxLaps) * 100);

  let wearColor = "var(--color-success)";
  if (tyreWearPct > 75) wearColor = "var(--color-danger)";
  else if (tyreWearPct > 50) wearColor = "var(--color-warning)";

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "12px" }}>
        <div className="sis-card-title">
          <span>🏎</span>
          <span>Ego Car Telemetry · {config.driver}</span>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <CompoundBadge compound={compound} size="lg" showLabel />
          <div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Stint #{stint}
            </div>
            <div className="font-mono" style={{ fontSize: "0.88rem", fontWeight: 700, color: "var(--text-primary)" }}>
              {Math.round(tyreLife)} Laps on Set
            </div>
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Track Position
          </div>
          <div className="font-mono" style={{ fontSize: "1.45rem", fontWeight: 900, color: "var(--f1-red)" }}>
            P{position}
          </div>
        </div>
      </div>

      {/* Tyre Degradation & Cliff Wear Indicator */}
      <div style={{ marginBottom: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", marginBottom: "4px" }}>
          <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Estimated Tyre Wear & Cliff Proximity</span>
          <span className="font-mono" style={{ color: wearColor, fontWeight: 700 }}>
            {Math.round(tyreWearPct)}%
          </span>
        </div>
        <div
          style={{
            height: "5px",
            background: "var(--surface-secondary)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-pill)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${tyreWearPct}%`,
              height: "100%",
              background: wearColor,
              borderRadius: "var(--radius-pill)",
              transition: "width 0.3s ease, background 0.3s ease",
            }}
          />
        </div>
      </div>

      {/* Lap Time & Interval */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
          background: "var(--surface-secondary)",
          padding: "10px 12px",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div>
          <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Last Lap Pace
          </div>
          <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "2px" }}>
            {formatLapTime(lastLap)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
            Interval Ahead
          </div>
          <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--color-info)", marginTop: "2px" }}>
            {formatInterval(interval)}
          </div>
        </div>
      </div>
    </div>
  );
}
