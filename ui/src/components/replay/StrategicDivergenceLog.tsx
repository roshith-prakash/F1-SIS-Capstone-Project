"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { formatLapTime } from "@/lib/formatters";

export function StrategicDivergenceLog() {
  const { lapHistory } = useRaceStore();

  return (
    <div className="sis-card" style={{ flex: 1, padding: "18px 20px" }}>
      <div className="sis-card-header" style={{ marginBottom: "10px" }}>
        <div className="sis-card-title">
          <span>📋</span>
          <span>Strategic Decision Log · Lap by Lap</span>
        </div>
        <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
          {lapHistory.length} ENTRIES
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "50px 110px 80px 80px 80px 1fr",
          gap: "8px",
          padding: "6px 8px",
          borderBottom: "1.5px solid var(--border)",
          fontSize: "0.68rem",
          fontWeight: 700,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        <span>Lap</span>
        <span>AI Directive</span>
        <span>Position</span>
        <span>Compound</span>
        <span>Tyre Age</span>
        <span>Pace</span>
      </div>

      <div style={{ overflowY: "auto", maxHeight: "420px", display: "flex", flexDirection: "column", gap: "1px" }}>
        {lapHistory.length === 0 ? (
          <div style={{ padding: "28px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
            No lap history recorded yet.
          </div>
        ) : (
          lapHistory.map((h) => {
            const isPit = h.recommendation_action.toUpperCase().includes("PIT");
            return (
              <div
                key={h.lap}
                style={{
                  display: "grid",
                  gridTemplateColumns: "50px 110px 80px 80px 80px 1fr",
                  alignItems: "center",
                  gap: "8px",
                  padding: "6px 8px",
                  borderRadius: "var(--radius-xs)",
                  background: isPit ? "var(--f1-red-subtle)" : "transparent",
                  borderBottom: "1px solid var(--border-subtle)",
                  fontSize: "0.78rem",
                }}
              >
                <span className="font-mono" style={{ fontWeight: 800, color: "var(--text-primary)" }}>
                  L{h.lap}
                </span>

                <span
                  className="font-mono"
                  style={{
                    fontWeight: 800,
                    color: isPit ? "var(--f1-red)" : "var(--color-success)",
                  }}
                >
                  {h.recommendation_action}
                </span>

                <span className="font-mono" style={{ color: "var(--text-secondary)", fontWeight: 700 }}>
                  P{h.actual_position}
                </span>

                <div style={{ display: "flex", alignItems: "center" }}>
                  <CompoundBadge compound={h.actual_compound} size="sm" />
                </div>

                <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
                  {Math.round(h.actual_tyre_life)}L
                </span>

                <span className="font-mono" style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                  {formatLapTime(h.actual_lap_time)}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
