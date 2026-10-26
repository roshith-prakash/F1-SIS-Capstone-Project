"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { formatLapTime } from "@/lib/formatters";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function LapTimeHistory() {
  const { lapHistory, config } = useRaceStore();

  if (lapHistory.length < 2) return null;

  const validLaps = lapHistory.filter((h) => h.actual_lap_time && h.actual_lap_time > 0);
  if (validLaps.length === 0) return null;

  const minTime = Math.min(...validLaps.map((l) => l.actual_lap_time!));
  const maxTime = Math.max(...validLaps.map((l) => l.actual_lap_time!));
  const range = maxTime - minTime || 1;

  return (
    <div className="sis-card" style={{ marginTop: "16px", padding: "18px 20px" }}>
      <div className="sis-card-header" style={{ marginBottom: "8px" }}>
        <div className="sis-card-title">
          <span>📈</span>
          <span>Ego Lap Pace Telemetry ({config.driver})</span>
        </div>
        <StatusBadge type="live" size="sm" />
      </div>

      <div style={{ height: "90px", display: "flex", alignItems: "flex-end", gap: "4px", padding: "8px 0" }}>
        {validLaps.map((item) => {
          const norm = (item.actual_lap_time! - minTime) / range;
          // invert height so faster laps are higher
          const barHeight = Math.max(15, (1 - norm) * 85 + 15);
          const isPit = item.recommendation_action.toUpperCase().includes("PIT");

          return (
            <div
              key={item.lap}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                height: "100%",
                justifyContent: "flex-end",
              }}
              title={`Lap ${item.lap}: ${formatLapTime(item.actual_lap_time)} (${item.actual_compound})`}
            >
              <div
                style={{
                  width: "100%",
                  height: `${barHeight}%`,
                  background: isPit ? "var(--color-warning)" : "var(--color-info)",
                  borderRadius: "2px",
                  transition: "height 0.2s ease",
                }}
              />
              <span className="font-mono" style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: "3px" }}>
                {item.lap}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "6px", borderTop: "1px solid var(--border-subtle)", paddingTop: "6px" }}>
        <span>Fastest: <strong className="font-mono">{formatLapTime(minTime)}</strong></span>
        <span>Slowest: <strong className="font-mono">{formatLapTime(maxTime)}</strong></span>
      </div>
    </div>
  );
}
