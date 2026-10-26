"use client";

import React, { useState } from "react";
import { useRaceStore } from "@/store/raceStore";
import { useRaceSocket } from "@/hooks/useRaceSocket";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { ConfidenceIndicator } from "@/components/ui/ConfidenceIndicator";
import { RiskIndicator } from "@/components/ui/RiskIndicator";
import { formatScore, formatPosition } from "@/lib/formatters";

export function DecisionCenterpiece() {
  const {
    lapData,
    currentLap,
    totalLaps,
    config,
    lastEvaluationTimeMs,
    activeOverride,
    adoptedStrategyId,
    isCalculating,
  } = useRaceStore();
  const { armOverride } = useRaceSocket();

  const [overrideAction, setOverrideAction] = useState<string>("pit_next_lap");
  const [overrideCompound, setOverrideCompound] = useState<string>("MEDIUM");
  const [overrideArming, setOverrideArming] = useState<boolean>(false);
  const rec = lapData?.recommendation;

  if (!rec) {
    return (
      <div
        className="sis-card"
        style={{
          border: "1px dashed var(--border)",
          padding: "36px 24px",
          textAlign: "center",
          background: "var(--surface)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.08em" }}>
          Decision Engine Standby
        </div>
        <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "6px" }}>
          Awaiting Initial Race Observation
        </div>
        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          Advance to Lap 1 to begin Monte Carlo strategy evaluation and tactical directives.
        </p>
      </div>
    );
  }

  const isPit = rec.action.toUpperCase().includes("PIT");
  const actionText = isPit ? "PIT NOW" : "STAY OUT";
  const ego = lapData?.participants ? lapData.participants[config.driver] : null;
  const currentPos = ego?.position ?? rec.expected_position;
  const gain = currentPos - rec.expected_position;
  const gainFormatted = gain > 0.05 ? `+${gain.toFixed(1)} pos` : gain < -0.05 ? `${gain.toFixed(1)} pos` : "Track Pos";

  // Derive semantic risk level from top strategy tyre and traffic risk
  const topStrat = lapData?.strategies?.[0];
  const tyreRisk = topStrat?.tyre_risk ?? 0.3;
  const riskLevel = tyreRisk > 0.65 ? "HIGH" : tyreRisk > 0.35 ? "MEDIUM" : "LOW";

  // Active planned strategy derivation
  const plannedCompounds = rec.planned_compounds || topStrat?.compounds || [ego?.compound || "MEDIUM", rec.target_compound || "HARD"];
  const plannedPitLaps = rec.planned_pit_laps || topStrat?.pit_laps || (rec.target_pit_lap ? [rec.target_pit_lap] : []);
  const targetPitLap = rec.target_pit_lap ?? (plannedPitLaps.find((l) => l >= currentLap) ?? plannedPitLaps[0] ?? null);
  const targetCompound = rec.target_compound || (plannedCompounds.length > 1 ? plannedCompounds[plannedCompounds.length - 1] : "HARD");
  const stratName = rec.planned_strategy_name || topStrat?.name || (plannedPitLaps.length > 1 ? "2-Stop Strategy" : "1-Stop Strategy");
  const currentCompound = ego?.compound || plannedCompounds[0] || "MEDIUM";

  const compoundFlowText = plannedCompounds.join(" ➔ ");
  let strategyNarrative = "";
  let statusChip = "PLANNED";
  let statusChipColor = "var(--color-info)";

  if (targetPitLap) {
    if (currentLap < targetPitLap) {
      const lapsRemaining = targetPitLap - currentLap;
      strategyNarrative = `Using a ${compoundFlowText} strategy with pitting on Lap ${targetPitLap} for ${targetCompound} (${lapsRemaining} lap${lapsRemaining === 1 ? "" : "s"} until pit window).`;
      statusChip = `PIT LAP ${targetPitLap}`;
      statusChipColor = lapsRemaining <= 3 ? "var(--color-warning)" : "var(--color-info)";
    } else if (currentLap === targetPitLap || isPit) {
      strategyNarrative = `Executing pit stop on Lap ${currentLap}: Switching from ${currentCompound} to ${targetCompound} compound as planned.`;
      statusChip = "BOX NOW";
      statusChipColor = "var(--f1-red)";
    } else {
      strategyNarrative = `Final Stint Active: Running ${currentCompound} compound to the chequered flag (no further stops scheduled).`;
      statusChip = "FINAL STINT";
      statusChipColor = "var(--color-success)";
    }
  } else {
    strategyNarrative = `Running continuous stint on ${currentCompound} compound to race finish.`;
    statusChip = "NO STOPS";
    statusChipColor = "var(--color-success)";
  }

  // Synthesize impact factor weights based on model outputs
  const reasons = rec.explanation || [];
  const undercutProb = lapData?.event_probabilities?.undercut_success ?? (isPit ? 0.75 : 0.30);
  const undercutPct = Math.round(undercutProb * 100);
  const scProbH3 = lapData?.sc_probabilities?.p_sc_h3 ?? 0.05;
  const scExposurePct = Math.min(100, Math.round(scProbH3 * 100 * 2.5));

  const factors = [
    {
      factor: "Tyre Degradation",
      impact: tyreRisk > 0.5 ? "HIGH IMPACT" : "MODERATE IMPACT",
      pct: Math.min(95, Math.round(tyreRisk * 100)),
      color: tyreRisk > 0.5 ? "var(--color-danger)" : "var(--color-warning)",
    },
    {
      factor: "Undercut Opportunity",
      impact: undercutPct > 50 ? "HIGH IMPACT" : "LOW IMPACT",
      pct: undercutPct,
      color: undercutPct > 50 ? "var(--color-info)" : "var(--text-muted)",
    },
    {
      factor: "Pit-Exit Traffic Clearance",
      impact: (topStrat?.traffic_risk ?? 0.2) < 0.35 ? "CLEAR WINDOW" : "TRAFFIC RISK",
      pct: Math.round((1 - (topStrat?.traffic_risk ?? 0.2)) * 100),
      color: (topStrat?.traffic_risk ?? 0.2) < 0.35 ? "var(--color-success)" : "var(--color-warning)",
    },
    {
      factor: "Safety Car Exposure",
      impact: scProbH3 > 0.10 ? "ELEVATED" : "NOMINAL",
      pct: scExposurePct,
      color: scProbH3 > 0.10 ? "var(--color-warning)" : "var(--text-muted)",
    },
  ];

  return (
    <div
      className="sis-card fade-in"
      style={{
        border: `1.5px solid ${isPit ? "var(--f1-red)" : "var(--border-strong)"}`,
        boxShadow: isPit ? "0 2px 10px rgba(225, 6, 0, 0.08)" : "var(--shadow-sm)",
        background: "var(--surface)",
        position: "relative",
        padding: "20px 24px",
      }}
    >
      {/* Header Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              fontSize: "0.72rem",
              fontWeight: 800,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
            }}
          >
            DECISION ENGINE · LAP {currentLap}
          </span>
          {(rec.evaluation_time_ms ?? lastEvaluationTimeMs) && (
            <span
              className="font-mono"
              style={{
                fontSize: "0.68rem",
                fontWeight: 700,
                color: "var(--color-info)",
                background: "var(--surface-secondary)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-pill)",
                padding: "1px 6px",
              }}
              title={`Total Strategy Pipeline Latency: ${rec.evaluation_time_ms ?? lastEvaluationTimeMs}ms`}
            >
              ⚡ {(rec.evaluation_time_ms ?? lastEvaluationTimeMs)! < 1000
                ? `${Math.round((rec.evaluation_time_ms ?? lastEvaluationTimeMs)!)}ms`
                : `${(((rec.evaluation_time_ms ?? lastEvaluationTimeMs)!) / 1000).toFixed(2)}s`}
            </span>
          )}
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
                padding: "2px 8px",
              }}
            >
              CALL CHANGED
            </span>
          )}
        </div>

        <RiskIndicator level={riskLevel} size="sm" label="RISK" />
      </div>

      {/* Armed Override Banner */}
      {activeOverride && (
        <div
          className="fade-in"
          style={{
            background: "color-mix(in srgb, var(--f1-red) 12%, var(--surface))",
            border: "1.5px solid var(--f1-red)",
            borderRadius: "var(--radius-md)",
            padding: "12px 16px",
            marginBottom: "16px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
            boxShadow: "0 0 12px rgba(225, 6, 0, 0.15)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "1.3rem" }}>⚡</span>
            <div>
              <div
                style={{
                  fontSize: "0.7rem",
                  fontWeight: 900,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "var(--f1-red)",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span>USER TACTICAL OVERRIDE ARMED</span>
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: "var(--f1-red)",
                    display: "inline-block",
                  }}
                />
              </div>
              <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "2px" }}>
                {activeOverride.action === "stay_out"
                  ? `Forcing STAY OUT on Lap ${activeOverride.target_lap ?? currentLap + 1} (extending current stint)`
                  : `Forcing PIT STOP on Lap ${activeOverride.target_lap ?? currentLap + 1} for ${activeOverride.compound || "NEW"} tyre`}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={async () => {
              setOverrideArming(true);
              try {
                await armOverride("clear");
              } finally {
                setOverrideArming(false);
              }
            }}
            disabled={isCalculating || overrideArming}
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              color: "var(--text-primary)",
              padding: "6px 14px",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.75rem",
              fontWeight: 700,
              cursor: isCalculating ? "not-allowed" : "pointer",
              transition: "all 0.15s ease",
            }}
          >
            ✕ Cancel Override (Revert to AI)
          </button>
        </div>
      )}

      {/* Adopted Strategy Indicator */}
      {adoptedStrategyId && (
        <div
          className="fade-in"
          style={{
            background: "color-mix(in srgb, var(--color-purple) 12%, var(--surface))",
            border: "1px solid var(--color-purple-border)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 14px",
            marginBottom: "14px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "0.85rem" }}>🧭</span>
            <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--color-purple)" }}>
              User Adopted Plan Active: {stratName}
            </span>
          </div>
          <button
            type="button"
            onClick={async () => {
              setOverrideArming(true);
              try {
                await armOverride("clear");
              } finally {
                setOverrideArming(false);
              }
            }}
            disabled={isCalculating || overrideArming}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "0.72rem",
              cursor: "pointer",
              textDecoration: "underline",
              fontWeight: 600,
            }}
          >
            Reset to AI Recommended
          </button>
        </div>
      )}

      {/* Recommended Action Core */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "16px",
          padding: "16px 20px",
          background: isPit ? "var(--f1-red-subtle)" : "var(--surface-secondary)",
          borderRadius: "var(--radius-md)",
          border: `1px solid ${isPit ? "var(--f1-red-border)" : "var(--border)"}`,
          marginBottom: "16px",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.7rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: isPit ? "var(--f1-red)" : "var(--text-secondary)",
              marginBottom: "2px",
            }}
          >
            RECOMMENDED ACTION
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span
              className="font-mono"
              style={{
                fontSize: "1.85rem",
                fontWeight: 900,
                letterSpacing: "-0.01em",
                color: isPit ? "var(--f1-red)" : "var(--text-primary)",
              }}
            >
              {actionText}
            </span>
            {isPit && rec.target_compound && (
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ fontSize: "1.1rem", color: "var(--text-muted)" }}>→</span>
                <CompoundBadge compound={rec.target_compound} size="lg" showLabel />
              </div>
            )}
          </div>
        </div>

        {/* Quantified Metrics Box */}
        <div style={{ display: "flex", gap: "20px" }}>
          <div>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)" }}>
              Expected Position
            </div>
            <div className="font-mono" style={{ fontSize: "1.25rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "1px" }}>
              P{rec.expected_position.toFixed(1)} <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>± 0.6</span>
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)" }}>
              Projected Delta
            </div>
            <div
              className="font-mono"
              style={{
                fontSize: "1.25rem",
                fontWeight: 800,
                color: gain >= 0 ? "var(--color-success)" : "var(--color-danger)",
                marginTop: "1px",
              }}
            >
              {gainFormatted}
            </div>
          </div>
        </div>
      </div>

      {/* Active Strategy Directive & Gameplan Banner */}
      <div
        style={{
          background: "var(--surface-secondary)",
          border: "1px solid var(--border-strong)",
          borderRadius: "var(--radius-md)",
          padding: "14px 18px",
          marginBottom: "16px",
          position: "relative",
          boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "0.85rem" }}>🗺</span>
            <span
              style={{
                fontSize: "0.7rem",
                fontWeight: 800,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--text-secondary)",
              }}
            >
              ACTIVE STRATEGY DIRECTIVE
            </span>
            <span
              style={{
                fontSize: "0.68rem",
                color: "var(--text-muted)",
                fontWeight: 600,
              }}
            >
              · {stratName}
            </span>
          </div>

          <span
            className="font-mono"
            style={{
              fontSize: "0.68rem",
              fontWeight: 800,
              padding: "2px 8px",
              borderRadius: "var(--radius-pill)",
              background: `color-mix(in srgb, ${statusChipColor} 15%, transparent)`,
              color: statusChipColor,
              border: `1px solid color-mix(in srgb, ${statusChipColor} 35%, transparent)`,
            }}
          >
            {statusChip}
          </span>
        </div>

        {/* Compound Stint Progression Chain */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
            marginBottom: "10px",
            padding: "8px 12px",
            background: "var(--surface)",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          {plannedCompounds.map((comp, idx) => {
            const pitLap = plannedPitLaps[idx];
            const isCurrentStint =
              (idx === 0 && (!plannedPitLaps[0] || currentLap <= plannedPitLaps[0])) ||
              (idx > 0 && currentLap > (plannedPitLaps[idx - 1] ?? 0) && (!pitLap || currentLap <= pitLap));

            return (
              <React.Fragment key={idx}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    padding: "3px 8px",
                    borderRadius: "var(--radius-sm)",
                    background: isCurrentStint ? "var(--surface-secondary)" : "transparent",
                    border: isCurrentStint ? "1px solid var(--border-strong)" : "1px solid transparent",
                  }}
                >
                  <CompoundBadge compound={comp} size="sm" showLabel />
                  {isCurrentStint && (
                    <span
                      style={{
                        fontSize: "0.65rem",
                        fontWeight: 800,
                        color: "var(--color-success)",
                        textTransform: "uppercase",
                        letterSpacing: "0.04em",
                      }}
                    >
                      (ACTIVE)
                    </span>
                  )}
                </div>

                {idx < plannedCompounds.length - 1 && (
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>➔</span>
                    <span
                      className="font-mono"
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 700,
                        color: "var(--text-secondary)",
                        background: "var(--surface-secondary)",
                        padding: "2px 6px",
                        borderRadius: "4px",
                        border: "1px dashed var(--border)",
                      }}
                    >
                      🏁 Pit Lap {plannedPitLaps[idx] ?? targetPitLap ?? "?"}
                    </span>
                    <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>➔</span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Plain-English Strategy Narrative */}
        <div
          style={{
            fontSize: "0.82rem",
            color: "var(--text-primary)",
            lineHeight: 1.45,
            borderLeft: `3px solid ${statusChipColor}`,
            paddingLeft: "10px",
            fontWeight: 500,
          }}
        >
          {strategyNarrative}
        </div>
      </div>

      {/* Confidence and Score Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "14px",
          marginBottom: "16px",
          paddingBottom: "16px",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <ConfidenceIndicator confidence={rec.confidence} />

        <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.06em" }}>
              Score Advantage vs P2
            </span>
            <span className="font-mono" style={{ fontSize: "0.82rem", fontWeight: 800, color: "var(--color-success)" }}>
              +{rec.score_delta >= 0 ? rec.score_delta.toFixed(3) : "0.000"}
            </span>
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
            Utility Score: <strong className="font-mono" style={{ color: "var(--text-primary)" }}>{formatScore(rec.score)}</strong> (vs runner-up {rec.runner_up_name || "Alt"})
          </div>
        </div>
      </div>

      {/* Why This Decision? Explanatory Factors */}
      <div>
        <div
          style={{
            fontSize: "0.72rem",
            fontWeight: 800,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--text-muted)",
            marginBottom: "8px",
          }}
        >
          WHY THIS DECISION? (PRIMARY INFLUENCING FACTORS)
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {factors.map((f, idx) => (
            <div key={idx} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ width: "170px", fontSize: "0.78rem", fontWeight: 500, color: "var(--text-primary)" }}>
                {f.factor}
              </span>
              <div
                style={{
                  flex: 1,
                  height: "5px",
                  background: "var(--surface-secondary)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-pill)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${f.pct}%`,
                    height: "100%",
                    background: f.color,
                    borderRadius: "var(--radius-pill)",
                  }}
                />
              </div>
              <span
                className="font-mono"
                style={{
                  width: "120px",
                  textAlign: "right",
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  color: f.color,
                }}
              >
                {f.impact}
              </span>
            </div>
          ))}
        </div>

        {/* Natural Language Bullet Reasons */}
        {reasons.length > 0 && (
          <div
            style={{
              marginTop: "12px",
              padding: "10px 14px",
              background: "var(--surface-secondary)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "4px" }}>
              {reasons.map((r, i) => (
                <li key={i} style={{ fontSize: "0.78rem", color: "var(--text-secondary)", display: "flex", alignItems: "baseline", gap: "6px" }}>
                  <span style={{ color: "var(--f1-red)", fontSize: "0.85rem" }}>•</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Pit Wall Tactical Override Bar */}
      <div
        style={{
          marginTop: "20px",
          padding: "14px 18px",
          background: "var(--surface-secondary)",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border-strong)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "1rem" }}>🎮</span>
            <span
              style={{
                fontSize: "0.72rem",
                fontWeight: 800,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "var(--text-secondary)",
              }}
            >
              Pit Wall Tactical Directive (Manual Override)
            </span>
          </div>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
            Arm pit call or stint extension for next lap execution
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: overrideAction === "stay_out" ? "1.4fr 1fr" : "1.2fr 1fr 1fr",
            gap: "10px",
            alignItems: "end",
          }}
        >
          <div>
            <label
              style={{
                fontSize: "0.68rem",
                fontWeight: 700,
                textTransform: "uppercase",
                color: "var(--text-muted)",
                display: "block",
                marginBottom: "4px",
              }}
            >
              Action Directive
            </label>
            <select
              value={overrideAction}
              onChange={(e) => setOverrideAction(e.target.value)}
              disabled={isCalculating || overrideArming}
              style={{
                width: "100%",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                padding: "8px 12px",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.82rem",
                fontWeight: 600,
                outline: "none",
              }}
            >
              <option value="pit_next_lap">Pit Next Lap (Lap {currentLap + 1})</option>
              <option value="pit_this_lap">Box Immediately (Lap {currentLap})</option>
              <option value="stay_out">Stay Out (Extend Current Stint)</option>
            </select>
          </div>

          {overrideAction !== "stay_out" && (
            <div>
              <label
                style={{
                  fontSize: "0.68rem",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Fitted Compound
              </label>
              <select
                value={overrideCompound}
                onChange={(e) => setOverrideCompound(e.target.value)}
                disabled={isCalculating || overrideArming}
                style={{
                  width: "100%",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                  padding: "8px 12px",
                  borderRadius: "var(--radius-sm)",
                  fontSize: "0.82rem",
                  fontWeight: 600,
                  outline: "none",
                }}
              >
                <option value="HARD">HARD (C2 / White)</option>
                <option value="MEDIUM">MEDIUM (C3 / Yellow)</option>
                <option value="SOFT">SOFT (C4 / Red)</option>
              </select>
            </div>
          )}

          <div>
            <button
              type="button"
              onClick={async () => {
                setOverrideArming(true);
                try {
                  await armOverride(
                    overrideAction,
                    overrideAction !== "stay_out" ? overrideCompound : undefined
                  );
                } finally {
                  setOverrideArming(false);
                }
              }}
              disabled={isCalculating || overrideArming}
              className="btn-primary"
              style={{
                width: "100%",
                height: "36px",
                fontSize: "0.8rem",
                fontWeight: 800,
                letterSpacing: "0.04em",
                background: "var(--f1-red)",
              }}
            >
              {overrideArming ? "ARMING..." : "⚡ ARM OVERRIDE"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
