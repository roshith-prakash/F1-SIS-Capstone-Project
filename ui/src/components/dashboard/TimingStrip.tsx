"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { formatLapTime, formatGap } from "@/lib/formatters";

export function TimingStrip() {
  const { lapData, config } = useRaceStore();
  const participants = lapData?.participants ? Object.values(lapData.participants) : [];
  const sorted = [...participants].sort((a, b) => a.position - b.position);

  if (sorted.length === 0) return null;

  return (
    <footer className="timing-strip">
      <div
        className="font-mono"
        style={{
          fontSize: "0.68rem",
          fontWeight: 800,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          paddingRight: "10px",
          borderRight: "1px solid var(--border)",
          whiteSpace: "nowrap",
          display: "flex",
          alignItems: "center",
          gap: "4px",
        }}
      >
        <span>TIMING</span>
      </div>

      <div style={{ display: "flex", gap: "6px", overflowX: "auto", flex: 1, padding: "0 4px" }}>
        {sorted.map((p) => {
          const isEgo = p.driver === config.driver;
          return (
            <div
              key={p.driver}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                background: isEgo ? "var(--f1-red-subtle)" : "var(--surface)",
                border: `1px solid ${isEgo ? "var(--f1-red-border)" : "var(--border)"}`,
                borderRadius: "var(--radius-sm)",
                padding: "3px 8px",
                flexShrink: 0,
                fontSize: "0.75rem",
                boxShadow: "var(--shadow-xs)",
              }}
            >
              <span
                className="font-mono"
                style={{
                  fontWeight: 800,
                  color: isEgo ? "var(--f1-red)" : "var(--text-muted)",
                  fontSize: "0.72rem",
                }}
              >
                P{p.position}
              </span>

              <span
                className="font-mono"
                style={{
                  fontWeight: isEgo ? 900 : 700,
                  color: isEgo ? "var(--f1-red)" : "var(--text-primary)",
                  letterSpacing: "0.02em",
                }}
              >
                {p.driver}
              </span>

              <CompoundBadge compound={p.compound} size="sm" />

              <span className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>
                {formatGap(p.gap_to_leader)}
              </span>

              <span className="font-mono" style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                {formatLapTime(p.last_lap_time)}
              </span>
            </div>
          );
        })}
      </div>
    </footer>
  );
}
