"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { ConfidenceIndicator } from "@/components/ui/ConfidenceIndicator";
import { formatScore, formatPosition } from "@/lib/formatters";

export function RecommendationCard() {
  const { lapData } = useRaceStore();
  const rec = lapData?.recommendation;

  if (!rec) {
    return (
      <div className="sis-card" style={{ border: "1px dashed var(--border)", textAlign: "center", padding: "28px 16px" }}>
        <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
          Advance to Lap 1 to generate live AI strategy recommendations.
        </div>
      </div>
    );
  }

  const isPit = rec.action.toUpperCase().includes("PIT");
  const actionText = isPit
    ? `PIT NOW → ${rec.target_compound || "HARD"}`
    : "STAY OUT";

  return (
    <div
      className="sis-card fade-in"
      style={{
        border: `1.5px solid ${isPit ? "var(--f1-red)" : "var(--border-strong)"}`,
        boxShadow: isPit ? "0 2px 8px rgba(225, 6, 0, 0.08)" : "var(--shadow-xs)",
        position: "relative",
        padding: "16px 18px",
      }}
    >
      {/* Top Bar with Changed Badge */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <span
          style={{
            fontSize: "0.72rem",
            fontWeight: 800,
            letterSpacing: "0.08em",
            color: "var(--text-muted)",
            textTransform: "uppercase",
          }}
        >
          AI Tactical Directive
        </span>

        {rec.changed_from_last_lap && (
          <span
            className="font-mono"
            style={{
              fontSize: "0.68rem",
              fontWeight: 800,
              background: "var(--color-warning-bg)",
              color: "var(--color-warning)",
              border: "1px solid var(--color-warning-border)",
              borderRadius: "var(--radius-pill)",
              padding: "1px 6px",
            }}
          >
            ↑ CALL CHANGED
          </span>
        )}
      </div>

      {/* Large Call-To-Action */}
      <div
        style={{
          background: isPit ? "var(--f1-red-subtle)" : "var(--surface-secondary)",
          padding: "14px 16px",
          borderRadius: "var(--radius-md)",
          border: `1px solid ${isPit ? "var(--f1-red-border)" : "var(--border)"}`,
          textAlign: "center",
          marginBottom: "12px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginBottom: "4px" }}>
          {rec.target_compound && isPit && (
            <CompoundBadge compound={rec.target_compound} size="md" />
          )}
          <span
            className="font-mono"
            style={{
              fontSize: "1.35rem",
              fontWeight: 900,
              letterSpacing: "0.02em",
              color: isPit ? "var(--f1-red)" : "var(--text-primary)",
            }}
          >
            {actionText}
          </span>
        </div>

        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "2px" }}>
          Target Finish: <strong style={{ color: "var(--text-primary)" }}>{formatPosition(rec.expected_position)}</strong>
        </div>

        {rec.planned_strategy_name && (
          <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginTop: "6px", borderTop: "1px dashed var(--border)", paddingTop: "6px" }}>
            Strategy: <strong style={{ color: "var(--text-primary)" }}>{rec.planned_compounds?.join(" ➔ ") || rec.planned_strategy_name}</strong>
            {rec.target_pit_lap ? ` (Pit Lap ${rec.target_pit_lap} for ${rec.target_compound || "HARD"})` : ""}
          </div>
        )}
      </div>

      <ConfidenceIndicator confidence={rec.confidence} size="sm" />

      {/* Score and Delta vs Runner Up */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
          fontSize: "0.75rem",
          marginTop: "12px",
        }}
      >
        <div
          style={{
            background: "var(--surface-secondary)",
            padding: "8px 10px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
            Utility Score
          </div>
          <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "1px" }}>
            {formatScore(rec.score)}
          </div>
        </div>

        <div
          style={{
            background: "var(--surface-secondary)",
            padding: "8px 10px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>
            Score Delta
          </div>
          <div
            className="font-mono"
            style={{
              fontSize: "0.95rem",
              fontWeight: 800,
              color: rec.score_delta >= 0 ? "var(--color-success)" : "var(--color-danger)",
              marginTop: "1px",
            }}
          >
            +{rec.score_delta.toFixed(3)} vs P2
          </div>
        </div>
      </div>
    </div>
  );
}
