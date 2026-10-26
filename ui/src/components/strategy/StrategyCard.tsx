"use client";

import React from "react";
import type { StrategyItem } from "@/types/race";
import { useRaceStore } from "@/store/raceStore";
import { useRaceSocket } from "@/hooks/useRaceSocket";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { RiskIndicator } from "@/components/ui/RiskIndicator";
import { formatScore, formatPosition, formatPct } from "@/lib/formatters";

interface StrategyCardProps {
  strategy: StrategyItem;
  isTopRanked?: boolean;
}

export function StrategyCard({ strategy, isTopRanked = false }: StrategyCardProps) {
  const { adoptedStrategyId, isCalculating } = useRaceStore();
  const { armOverride } = useRaceSocket();

  const isAdopted = strategy.is_user_adopted || strategy.id === adoptedStrategyId;

  const intentColors: Record<string, { color: string; bg: string; border: string }> = {
    undercut: { color: "var(--color-info)", bg: "var(--color-info-bg)", border: "var(--color-info-border)" },
    overcut: { color: "var(--color-warning)", bg: "var(--color-warning-bg)", border: "var(--color-warning-border)" },
    cover: { color: "var(--color-purple)", bg: "var(--color-purple-bg)", border: "var(--color-purple-border)" },
    nominal: { color: "var(--text-secondary)", bg: "var(--surface-secondary)", border: "var(--border)" },
  };

  const intentStyle = intentColors[strategy.tactical_intent.toLowerCase()] || intentColors.nominal;

  const cardBorder = isAdopted
    ? "1.5px solid var(--color-purple)"
    : isTopRanked
    ? "1.5px solid var(--f1-red)"
    : "1.5px solid var(--border)";

  const cardShadow = isAdopted
    ? "0 2px 10px rgba(168, 85, 247, 0.12)"
    : isTopRanked
    ? "0 2px 8px rgba(225, 6, 0, 0.08)"
    : "var(--shadow-xs)";

  return (
    <div
      className="sis-card fade-in"
      style={{
        background: "var(--surface)",
        border: cardBorder,
        boxShadow: cardShadow,
        padding: "16px 18px",
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        transition: "all 0.15s ease",
      }}
    >
      {/* Top Header Row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            className="font-mono"
            style={{
              width: "22px",
              height: "22px",
              borderRadius: "var(--radius-xs)",
              background: isTopRanked ? "var(--f1-red)" : "var(--surface-secondary)",
              color: isTopRanked ? "#FFFFFF" : "var(--text-secondary)",
              border: `1px solid ${isTopRanked ? "var(--f1-red)" : "var(--border)"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "0.75rem",
              fontWeight: 800,
            }}
          >
            #{strategy.rank}
          </span>
          <span style={{ fontSize: "0.92rem", fontWeight: 700, color: "var(--text-primary)" }}>
            {strategy.name}
          </span>
          {isAdopted && (
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
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            className="font-mono"
            style={{
              fontSize: "0.68rem",
              fontWeight: 700,
              textTransform: "uppercase",
              padding: "2px 6px",
              borderRadius: "var(--radius-pill)",
              background: intentStyle.bg,
              color: intentStyle.color,
              border: `1px solid ${intentStyle.border}`,
            }}
          >
            {strategy.tactical_intent}
          </span>

          <span
            className="font-mono"
            style={{
              fontSize: "0.95rem",
              fontWeight: 800,
              color: isTopRanked ? "var(--f1-red)" : "var(--text-primary)",
            }}
          >
            Score: {formatScore(strategy.score)}
          </span>
        </div>
      </div>

      {/* Stint Sequence Bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          background: "var(--surface-secondary)",
          padding: "6px 10px",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
          Stints:
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {strategy.compounds.map((cmp, i) => (
            <React.Fragment key={i}>
              <CompoundBadge compound={cmp} size="sm" showLabel={false} />
              {i < strategy.compounds.length - 1 && (
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>→</span>
              )}
            </React.Fragment>
          ))}
        </div>
        {strategy.pit_laps.length > 0 && (
          <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginLeft: "auto", fontWeight: 600 }}>
            Pit L{strategy.pit_laps.join(", L")}
          </span>
        )}
      </div>

      {/* Key Metric Stats Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "8px",
          textAlign: "center",
          fontSize: "0.72rem",
          background: "var(--surface-secondary)",
          padding: "8px 10px",
          borderRadius: "var(--radius-sm)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div>
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>Exp Finish</div>
          <div className="font-mono" style={{ fontWeight: 800, color: "var(--text-primary)", marginTop: "2px", fontSize: "0.85rem" }}>
            {formatPosition(strategy.expected_position)}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>P(Win)</div>
          <div className="font-mono" style={{ fontWeight: 800, color: "var(--color-danger)", marginTop: "2px", fontSize: "0.85rem" }}>
            {formatPct(strategy.p_win)}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>P(Podium)</div>
          <div className="font-mono" style={{ fontWeight: 800, color: "var(--color-warning)", marginTop: "2px", fontSize: "0.85rem" }}>
            {formatPct(strategy.p_podium)}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>Tyre Risk</div>
          <div className="font-mono" style={{ fontWeight: 800, color: strategy.tyre_risk > 0.5 ? "var(--color-danger)" : "var(--color-success)", marginTop: "2px", fontSize: "0.85rem" }}>
            {formatPct(strategy.tyre_risk)}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--text-muted)", textTransform: "uppercase", fontSize: "0.66rem", fontWeight: 700 }}>SC Delta</div>
          <div className="font-mono" style={{ fontWeight: 800, color: strategy.sc_position_advantage > 0 ? "var(--color-success)" : "var(--text-muted)", marginTop: "2px", fontSize: "0.85rem" }}>
            {strategy.sc_position_advantage > 0 ? `+${strategy.sc_position_advantage.toFixed(1)}` : strategy.sc_position_advantage.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Action / Adoption Bar */}
      <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", paddingTop: "2px" }}>
        {isAdopted ? (
          <button
            type="button"
            onClick={() => armOverride("clear")}
            disabled={isCalculating}
            style={{
              padding: "4px 12px",
              fontSize: "0.72rem",
              fontWeight: 700,
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--color-purple-border)",
              background: "var(--surface-secondary)",
              color: "var(--color-purple)",
              cursor: isCalculating ? "not-allowed" : "pointer",
            }}
          >
            ✕ Revert to AI Recommendation
          </button>
        ) : (
          <button
            type="button"
            onClick={() => armOverride("adopt_strategy", undefined, strategy.id)}
            disabled={isCalculating}
            style={{
              padding: "4px 12px",
              fontSize: "0.72rem",
              fontWeight: 700,
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-strong)",
              background: isTopRanked ? "var(--surface-secondary)" : "var(--surface)",
              color: "var(--text-primary)",
              cursor: isCalculating ? "not-allowed" : "pointer",
              transition: "all 0.15s ease",
            }}
          >
            {isTopRanked ? "✓ Active Baseline Plan" : `🎯 Adopt Plan #${strategy.rank}`}
          </button>
        )}
      </div>
    </div>
  );
}
