"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { TEAM_COLORS } from "@/lib/constants";
import { formatGap, formatInterval } from "@/lib/formatters";

export function RaceTower() {
  const { lapData, config } = useRaceStore();
  const participants = lapData?.participants ? Object.values(lapData.participants) : [];

  // Sort by position
  const sorted = [...participants].sort((a, b) => a.position - b.position);

  return (
    <div className="sis-card" style={{ flex: 1, minHeight: "420px", padding: "16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "10px" }}>
        <div className="sis-card-title">
          <span>⏱</span>
          <span>Race Classification Tower</span>
        </div>
        <span className="font-mono" style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
          {sorted.length} DRIVERS
        </span>
      </div>

      {/* Table Column Headers */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "24px 44px 28px 1fr 44px",
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
        <span>Pos</span>
        <span>Driver</span>
        <span>Tyre</span>
        <span>Gap / Interval</span>
        <span style={{ textAlign: "right" }}>Stops</span>
      </div>

      <div
        style={{
          overflowY: "auto",
          maxHeight: "560px",
          display: "flex",
          flexDirection: "column",
          gap: "1px",
          marginTop: "2px",
        }}
      >
        {sorted.length === 0 ? (
          <div style={{ padding: "32px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
            Awaiting race telemetry session start...
          </div>
        ) : (
          sorted.map((p) => {
            const isEgo = p.driver === config.driver;
            const teamColor = TEAM_COLORS[p.team] || "var(--border-strong)";

            return (
              <div
                key={p.driver}
                style={{
                  display: "grid",
                  gridTemplateColumns: "24px 44px 28px 1fr 44px",
                  alignItems: "center",
                  gap: "8px",
                  padding: "5px 8px",
                  borderRadius: "var(--radius-xs)",
                  background: isEgo ? "var(--f1-red-subtle)" : "transparent",
                  borderLeft: isEgo ? "3px solid var(--f1-red)" : `3px solid ${teamColor}`,
                  borderBottom: "1px solid var(--border-subtle)",
                  fontSize: "0.8rem",
                  transition: "background 0.12s ease",
                }}
              >
                {/* Position */}
                <span
                  className="font-mono"
                  style={{
                    fontWeight: 800,
                    color: isEgo ? "var(--f1-red)" : "var(--text-secondary)",
                    fontSize: "0.78rem",
                  }}
                >
                  {p.position}
                </span>

                {/* Driver */}
                <span
                  className="font-mono"
                  style={{
                    fontWeight: isEgo ? 900 : 700,
                    color: isEgo ? "var(--f1-red)" : "var(--text-primary)",
                    letterSpacing: "0.04em",
                  }}
                >
                  {p.driver}
                </span>

                {/* Compound & tyre age */}
                <div style={{ display: "flex", alignItems: "center" }}>
                  <CompoundBadge compound={p.compound} size="sm" />
                </div>

                {/* Gap & Interval */}
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span
                    className="font-mono"
                    style={{
                      fontSize: "0.74rem",
                      color: isEgo ? "var(--f1-red)" : "var(--text-primary)",
                      fontWeight: 600,
                    }}
                  >
                    {formatGap(p.gap_to_leader)}
                  </span>
                  <span className="font-mono" style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                    L{Math.round(p.tyre_life)} · {formatInterval(p.interval_ahead)}
                  </span>
                </div>

                {/* Pit Stops */}
                <div className="font-mono" style={{ textAlign: "right", color: "var(--text-secondary)", fontSize: "0.74rem" }}>
                  {p.pit_count} {p.pit_count === 1 ? "stop" : "stops"}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
