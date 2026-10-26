"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { RiskIndicator } from "@/components/ui/RiskIndicator";
import { ProbabilityBar } from "@/components/ui/ProbabilityBar";
import { formatGap, formatInterval } from "@/lib/formatters";

interface RivalData {
  driver: string;
  team: string;
  position: number;
  compound: string;
  tyre_life: number;
  gap_to_leader: number | null;
  interval: number | null;
  relationship: "ahead" | "behind";
  likelyAction: string;
  pitProb: number;
  threatLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  likelyStrategy: string;
  implication: string;
}

export function OpponentModelCard() {
  const { lapData, config } = useRaceStore();
  const participants = lapData?.participants ? Object.values(lapData.participants) : [];
  const ego = lapData?.participants ? lapData.participants[config.driver] : null;

  if (!lapData || !ego) {
    return (
      <div className="sis-card" style={{ padding: "24px", textAlign: "center" }}>
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          Awaiting race telemetry to model opponent pitstop probabilities.
        </div>
      </div>
    );
  }

  // Sort participants by position
  const sorted = [...participants].sort((a, b) => a.position - b.position);
  const egoPos = ego.position;

  // Identify rival ahead (egoPos - 1) and rival behind (egoPos + 1)
  const rivalAhead = sorted.find((p) => p.position === egoPos - 1);
  const rivalBehind = sorted.find((p) => p.position === egoPos + 1);

  const rivals: RivalData[] = [];

  if (rivalAhead) {
    const age = rivalAhead.tyre_life;
    const modelProb = typeof rivalAhead.pit_probability === "number" ? rivalAhead.pit_probability : (age > 20 ? 0.65 : 0.25);
    const threat: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" = rivalAhead.threat_level || ((ego.interval_ahead || 5.0) < 1.8 ? "HIGH" : "MEDIUM");
    const likelyAct = rivalAhead.likely_action || (modelProb > 0.55 ? "Pit Window Open" : "Extend Current Stint");
    const undercutLikelihood = Math.round((lapData?.event_probabilities?.undercut_success ?? 0.5) * 100);

    rivals.push({
      driver: rivalAhead.driver,
      team: rivalAhead.team,
      position: rivalAhead.position,
      compound: rivalAhead.compound,
      tyre_life: age,
      gap_to_leader: rivalAhead.gap_to_leader,
      interval: ego.interval_ahead,
      relationship: "ahead",
      likelyAction: likelyAct,
      pitProb: modelProb,
      threatLevel: threat,
      likelyStrategy: modelProb > 0.5 ? "Vulnerable to Undercut" : "Maintaining Pace",
      implication: modelProb > 0.4
        ? `Model predicts ${undercutLikelihood}% undercut success against ${rivalAhead.driver}`
        : `Track gap stable. No immediate undercut vulnerability.`,
    });
  }

  if (rivalBehind) {
    const gap = rivalBehind.interval_ahead || 2.4;
    const age = rivalBehind.tyre_life;
    const modelProb = typeof rivalBehind.pit_probability === "number" ? rivalBehind.pit_probability : (age > 18 ? 0.55 : 0.20);
    const threat: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" = rivalBehind.threat_level || (gap < 2.0 ? "HIGH" : gap < 3.5 ? "MEDIUM" : "LOW");
    const likelyAct = rivalBehind.likely_action || (modelProb > 0.5 ? "Pit Window Open" : "Stay Out (Pace Buffer)");

    rivals.push({
      driver: rivalBehind.driver,
      team: rivalBehind.team,
      position: rivalBehind.position,
      compound: rivalBehind.compound,
      tyre_life: age,
      gap_to_leader: rivalBehind.gap_to_leader,
      interval: gap,
      relationship: "behind",
      likelyAction: likelyAct,
      pitProb: modelProb,
      threatLevel: threat,
      likelyStrategy: threat === "HIGH" || threat === "CRITICAL" ? "Active Undercut Threat" : "Covering Delta",
      implication:
        threat === "HIGH" || threat === "CRITICAL"
          ? `${rivalBehind.driver} can undercut if pitting 1 lap earlier. Monitor pit window.`
          : `Delta cushion is +${gap.toFixed(2)}s. Pit window protected.`,
    });
  }

  if (rivals.length === 0) {
    return (
      <div className="sis-card" style={{ padding: "20px", textAlign: "center" }}>
        <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
          No immediate adjacent rivals identified for opponent modeling.
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {rivals.map((r) => (
        <div
          key={r.driver}
          className="sis-card fade-in"
          style={{
            padding: "16px 18px",
            borderLeft: `4px solid ${
              r.threatLevel === "HIGH"
                ? "var(--color-danger)"
                : r.threatLevel === "MEDIUM"
                ? "var(--color-warning)"
                : "var(--color-info)"
            }`,
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span
                className="font-mono"
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 900,
                  color: "var(--text-primary)",
                }}
              >
                P{r.position} · {r.driver}
              </span>
              <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
                {r.team}
              </span>
              <CompoundBadge compound={r.compound} size="sm" showLabel />
              <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                {Math.round(r.tyre_life)}L old
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="font-mono" style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)" }}>
                {r.relationship === "ahead" ? `-${formatInterval(r.interval)} ahead` : `+${formatInterval(r.interval)} behind`}
              </span>
              <RiskIndicator level={r.threatLevel} label="THREAT" size="sm" />
            </div>
          </div>

          {/* Opponent Model Predictions */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1.2fr 1fr",
              gap: "14px",
              background: "var(--surface-secondary)",
              padding: "12px 14px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
              marginBottom: "10px",
            }}
          >
            <div>
              <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "4px" }}>
                Predicted Action ({r.likelyAction})
              </div>
              <ProbabilityBar
                label="Pit Within 3 Laps"
                value={r.pitProb}
                color={r.pitProb > 0.6 ? "var(--color-danger)" : "var(--color-info)"}
                height={6}
              />
            </div>

            <div>
              <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", marginBottom: "2px" }}>
                Likely Tactical Strategy
              </div>
              <div className="font-mono" style={{ fontSize: "0.85rem", fontWeight: 800, color: "var(--text-primary)" }}>
                {r.likelyStrategy}
              </div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
                Stint extension likelihood: {Math.round((1 - r.pitProb) * 100)}%
              </div>
            </div>
          </div>

          {/* Strategic Implication for Ego Driver */}
          <div
            style={{
              fontSize: "0.78rem",
              color: "var(--text-primary)",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <span
              style={{
                fontSize: "0.68rem",
                fontWeight: 800,
                textTransform: "uppercase",
                padding: "2px 6px",
                background: "var(--surface-hover)",
                borderRadius: "var(--radius-xs)",
                color: "var(--text-secondary)",
              }}
            >
              IMPLICATION
            </span>
            <span>{r.implication}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
