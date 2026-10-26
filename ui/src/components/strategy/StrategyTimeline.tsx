"use client";

import React from "react";
import type { StrategyItem } from "@/types/race";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { TYRE_COMPOUNDS } from "@/lib/constants";

interface StrategyTimelineProps {
  strategy: StrategyItem;
  totalLaps: number;
  currentLap: number;
  compact?: boolean;
}

export function StrategyTimeline({
  strategy,
  totalLaps = 70,
  currentLap = 1,
  compact = false,
}: StrategyTimelineProps) {
  const compounds = strategy.compounds || ["MEDIUM", "HARD"];
  const pitLaps = strategy.pit_laps || [25];
  const maxLaps = totalLaps > 0 ? totalLaps : 70;

  // Compute stint lap intervals
  const stints: { compound: string; startLap: number; endLap: number; length: number }[] = [];
  let prevLap = 1;

  compounds.forEach((cmp, idx) => {
    const endLap = idx < pitLaps.length ? pitLaps[idx] : maxLaps;
    stints.push({
      compound: cmp,
      startLap: prevLap,
      endLap: endLap,
      length: Math.max(1, endLap - prevLap),
    });
    prevLap = endLap;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {/* Lap Progress Indicator Header */}
      {!compact && (
        <div
          className="font-mono"
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.68rem",
            color: "var(--text-muted)",
          }}
        >
          <span>L1</span>
          <span>L{Math.round(maxLaps * 0.25)}</span>
          <span>L{Math.round(maxLaps * 0.5)}</span>
          <span>L{Math.round(maxLaps * 0.75)}</span>
          <span>L{maxLaps}</span>
        </div>
      )}

      {/* Stint Bars Container */}
      <div
        style={{
          position: "relative",
          height: compact ? "24px" : "32px",
          background: "var(--surface-secondary)",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border)",
          overflow: "hidden",
          display: "flex",
        }}
      >
        {stints.map((stint, idx) => {
          const widthPct = (stint.length / maxLaps) * 100;
          const info = TYRE_COMPOUNDS[stint.compound.toUpperCase()] || TYRE_COMPOUNDS.UNKNOWN;

          return (
            <div
              key={idx}
              style={{
                width: `${widthPct}%`,
                height: "100%",
                background: info.bg,
                borderRight: idx < stints.length - 1 ? `2px dashed ${info.color}` : "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "0 6px",
                overflow: "hidden",
                position: "relative",
                transition: "width 0.3s ease",
              }}
              title={`Stint ${idx + 1}: ${stint.compound} (L${stint.startLap} - L${stint.endLap}, ${stint.length} laps)`}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span
                  className="font-mono"
                  style={{
                    fontSize: compact ? "0.65rem" : "0.72rem",
                    fontWeight: 800,
                    color: info.color,
                  }}
                >
                  {info.letter}
                </span>
                {!compact && widthPct > 15 && (
                  <span
                    style={{
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      color: "var(--text-secondary)",
                      textTransform: "uppercase",
                    }}
                  >
                    {stint.compound} ({stint.length}L)
                  </span>
                )}
              </div>

              {/* Pit Marker on boundary */}
              {idx < stints.length - 1 && (
                <div
                  className="font-mono"
                  style={{
                    position: "absolute",
                    right: "2px",
                    bottom: "2px",
                    fontSize: "0.58rem",
                    fontWeight: 800,
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--text-primary)",
                    padding: "0 3px",
                    borderRadius: "2px",
                  }}
                >
                  PIT
                </div>
              )}
            </div>
          );
        })}

        {/* Current Race Position Marker */}
        {currentLap > 0 && currentLap <= maxLaps && (
          <div
            style={{
              position: "absolute",
              left: `${(currentLap / maxLaps) * 100}%`,
              top: 0,
              bottom: 0,
              width: "2px",
              background: "var(--f1-red)",
              boxShadow: "0 0 4px rgba(225, 6, 0, 0.4)",
              zIndex: 5,
            }}
            title={`Current Lap: ${currentLap}`}
          />
        )}
      </div>
    </div>
  );
}
