"use client";

import React from "react";
import type { StrategyItem } from "@/types/race";
import { useRaceStore } from "@/store/raceStore";
import { useRaceSocket } from "@/hooks/useRaceSocket";
import { RiskIndicator } from "@/components/ui/RiskIndicator";
import { formatPosition, formatPct, formatScore } from "@/lib/formatters";

interface StrategyComparisonTableProps {
  strategies: StrategyItem[];
  recommendedId?: string;
}

export function StrategyComparisonTable({
  strategies,
  recommendedId,
}: StrategyComparisonTableProps) {
  const { adoptedStrategyId, isCalculating } = useRaceStore();
  const { armOverride } = useRaceSocket();

  if (!strategies || strategies.length === 0) {
    return (
      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
        No strategies available for comparison.
      </div>
    );
  }

  // Take top 3 strategies for side-by-side comparison
  const displayStrats = strategies.slice(0, 3);

  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.8rem",
          textAlign: "left",
        }}
      >
        <thead>
          <tr style={{ borderBottom: "1.5px solid var(--border)" }}>
            <th
              style={{
                padding: "8px 12px",
                fontSize: "0.7rem",
                fontWeight: 700,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              Metric / Strategy
            </th>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              const isAdopted = s.is_user_adopted || s.id === adoptedStrategyId;
              return (
                <th
                  key={s.id}
                  style={{
                    padding: "8px 14px",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                    borderRight: isRec ? "1px solid var(--f1-red-border)" : "none",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span
                      style={{
                        fontWeight: 800,
                        color: isRec ? "var(--f1-red)" : "var(--text-primary)",
                      }}
                    >
                      {s.name}
                    </span>
                    {isAdopted ? (
                      <span
                        className="font-mono"
                        style={{
                          fontSize: "0.62rem",
                          fontWeight: 800,
                          background: "var(--color-purple)",
                          color: "#fff",
                          padding: "1px 6px",
                          borderRadius: "var(--radius-pill)",
                        }}
                      >
                        ✓ ADOPTED
                      </span>
                    ) : isRec ? (
                      <span
                        className="font-mono"
                        style={{
                          fontSize: "0.62rem",
                          fontWeight: 800,
                          background: "var(--f1-red)",
                          color: "#fff",
                          padding: "1px 5px",
                          borderRadius: "var(--radius-pill)",
                        }}
                      >
                        RECOMMENDED
                      </span>
                    ) : null}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "2px" }}>
                    {s.tactical_intent.toUpperCase()} · Score {formatScore(s.score)}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {/* Expected Finish Position */}
          <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text-secondary)" }}>
              Expected Position
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              return (
                <td
                  key={s.id}
                  className="font-mono"
                  style={{
                    padding: "10px 14px",
                    fontWeight: 800,
                    fontSize: "0.95rem",
                    color: isRec ? "var(--f1-red)" : "var(--text-primary)",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  {formatPosition(s.expected_position)}
                </td>
              );
            })}
          </tr>

          {/* Win Probability */}
          <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text-secondary)" }}>
              Win Probability P(Win)
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              return (
                <td
                  key={s.id}
                  className="font-mono"
                  style={{
                    padding: "10px 14px",
                    fontWeight: 700,
                    color: s.p_win > 0.2 ? "var(--color-danger)" : "var(--text-primary)",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  {formatPct(s.p_win)}
                </td>
              );
            })}
          </tr>

          {/* Podium Probability */}
          <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text-secondary)" }}>
              Podium Probability P(Podium)
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              return (
                <td
                  key={s.id}
                  className="font-mono"
                  style={{
                    padding: "10px 14px",
                    fontWeight: 700,
                    color: s.p_podium > 0.5 ? "var(--color-warning)" : "var(--text-primary)",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  {formatPct(s.p_podium)}
                </td>
              );
            })}
          </tr>

          {/* Tyre Risk */}
          <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text-secondary)" }}>
              Tyre Cliff Exposure
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              const rLevel = s.tyre_risk > 0.65 ? "HIGH" : s.tyre_risk > 0.35 ? "MEDIUM" : "LOW";
              return (
                <td
                  key={s.id}
                  style={{
                    padding: "10px 14px",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  <RiskIndicator level={rLevel} score={s.tyre_risk} size="sm" />
                </td>
              );
            })}
          </tr>

          {/* Safety Car Delta Advantage */}
          <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text-secondary)" }}>
              SC Advantage
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              return (
                <td
                  key={s.id}
                  className="font-mono"
                  style={{
                    padding: "10px 14px",
                    fontWeight: 700,
                    color: s.sc_position_advantage > 0 ? "var(--color-success)" : "var(--text-secondary)",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  {s.sc_position_advantage > 0 ? `+${s.sc_position_advantage.toFixed(1)} pos` : `${s.sc_position_advantage.toFixed(1)} pos`}
                </td>
              );
            })}
          </tr>

          {/* Strategic Adoption Row */}
          <tr style={{ borderTop: "1.5px solid var(--border)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 700, color: "var(--text-secondary)", fontSize: "0.75rem", textTransform: "uppercase" }}>
              Strategic Directive
            </td>
            {displayStrats.map((s, idx) => {
              const isRec = s.id === recommendedId || idx === 0;
              const isAdopted = s.is_user_adopted || s.id === adoptedStrategyId;

              return (
                <td
                  key={s.id}
                  style={{
                    padding: "10px 14px",
                    background: isRec ? "var(--f1-red-subtle)" : "transparent",
                    borderLeft: "1px solid var(--border-subtle)",
                  }}
                >
                  {isAdopted ? (
                    <button
                      type="button"
                      onClick={() => armOverride("clear")}
                      disabled={isCalculating}
                      style={{
                        padding: "6px 12px",
                        fontSize: "0.72rem",
                        fontWeight: 700,
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--color-purple-border)",
                        background: "var(--surface)",
                        color: "var(--color-purple)",
                        cursor: isCalculating ? "not-allowed" : "pointer",
                        width: "100%",
                      }}
                    >
                      ✕ Revert to AI
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => armOverride("adopt_strategy", undefined, s.id)}
                      disabled={isCalculating}
                      style={{
                        padding: "6px 12px",
                        fontSize: "0.72rem",
                        fontWeight: 700,
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--border-strong)",
                        background: isRec ? "var(--surface-secondary)" : "var(--surface)",
                        color: "var(--text-primary)",
                        cursor: isCalculating ? "not-allowed" : "pointer",
                        width: "100%",
                      }}
                    >
                      {isRec ? "✓ Active AI Plan" : "🎯 Adopt Plan"}
                    </button>
                  )}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
