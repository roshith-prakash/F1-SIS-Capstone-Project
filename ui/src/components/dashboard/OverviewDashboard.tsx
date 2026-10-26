"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { Metric, MetricGrid } from "@/components/ui/Metric";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { DecisionCenterpiece } from "@/components/strategy/DecisionCenterpiece";
import { StrategyTimeline } from "@/components/strategy/StrategyTimeline";
import { OpponentModelCard } from "@/components/strategy/OpponentModelCard";
import { RiskIndicator } from "@/components/ui/RiskIndicator";
import { formatGap, formatInterval, formatLapTime } from "@/lib/formatters";

export function OverviewDashboard() {
  const { lapData, currentLap, totalLaps, config } = useRaceStore();
  const ego = lapData?.participants ? lapData.participants[config.driver] : null;
  const strategies = lapData?.strategies || [];
  const topStrat = strategies[0];

  const position = ego?.position ?? 1;
  const gap = ego?.gap_to_leader;
  const compound = ego?.compound || "MEDIUM";
  const tyreLife = ego?.tyre_life || 1;
  const lastLap = ego?.last_lap_time;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* 1. KEY RACE METRICS (LEVEL 1 CRITICAL) */}
      <MetricGrid columns={4}>
        <Metric
          label="Track Position"
          value={`P${position}`}
          highlight={true}
          size="xl"
          subValue={ego?.team || "Focus Car"}
        />

        <Metric
          label="Gap to Leader"
          value={formatGap(gap)}
          size="xl"
          subValue={ego?.interval_ahead ? `Interval ahead: ${formatInterval(ego.interval_ahead)}` : "Leading current pack"}
        />

        <Metric
          label="Current Tyre Compound"
          value={
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <CompoundBadge compound={compound} size="lg" />
              <span>{compound}</span>
            </div>
          }
          size="lg"
          subValue={`${Math.round(tyreLife)} Laps on current set`}
        />

        <Metric
          label="Race Progression"
          value={`Lap ${currentLap}`}
          size="xl"
          subValue={`of ${totalLaps || "--"} total laps (${totalLaps > 0 ? Math.round((currentLap / totalLaps) * 100) : 0}%)`}
        />
      </MetricGrid>

      {/* 2. MAIN DECISION & SITUATION TWO-COLUMN DECK */}
      <div style={{ display: "grid", gridTemplateColumns: "1.25fr 1fr", gap: "20px" }}>
        {/* Left: The Visual Centerpiece - Decision Engine */}
        <DecisionCenterpiece />

        {/* Right: Race Situation & Real-Time Operational Risk */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Situation Snapshot Card */}
          <div className="sis-card">
            <div className="sis-card-header">
              <div className="sis-card-title">
                <span>📋</span>
                <span>Tactical Situation Snapshot</span>
              </div>
              <StatusBadge type="live" size="sm" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <div style={{ background: "var(--surface-secondary)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                  Last Lap Time
                </div>
                <div className="font-mono" style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
                  {formatLapTime(lastLap)}
                </div>
              </div>

              <div style={{ background: "var(--surface-secondary)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                  Stint Status
                </div>
                <div className="font-mono" style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
                  Stint {(ego?.pit_count || 0) + 1}
                </div>
              </div>
            </div>

            {/* Operational Risk Matrix */}
            <div style={{ marginTop: "14px", borderTop: "1px solid var(--border-subtle)", paddingTop: "12px" }}>
              <div style={{ fontSize: "0.68rem", fontWeight: 800, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "8px", letterSpacing: "0.06em" }}>
                Operational Risk Summary
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>Tyre Degradation Cliff</span>
                  <RiskIndicator
                    level={(topStrat?.tyre_risk ?? 0.3) > 0.65 ? "HIGH" : (topStrat?.tyre_risk ?? 0.3) > 0.35 ? "MEDIUM" : "LOW"}
                    size="sm"
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>Pit Window Undercut Threat</span>
                  <RiskIndicator
                    level={position > 1 && (ego?.interval_ahead || 5.0) < 2.0 ? "HIGH" : "LOW"}
                    size="sm"
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>Pit-Exit Traffic Congestion</span>
                  <RiskIndicator
                    level={(topStrat?.traffic_risk ?? 0.2) > 0.5 ? "HIGH" : "LOW"}
                    size="sm"
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>Safety Car Disruption Exposure</span>
                  <RiskIndicator
                    level={(lapData?.sc_probabilities?.p_sc_h5 ?? 0.08) > 0.15 ? "MEDIUM" : "LOW"}
                    size="sm"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Strategy Projection Mini */}
          {topStrat && (
            <div className="sis-card">
              <div className="sis-card-header" style={{ marginBottom: "10px" }}>
                <div className="sis-card-title">
                  <span>🗺</span>
                  <span>Optimal Plan Projection ({topStrat.name})</span>
                </div>
                <StatusBadge type="predicted" size="sm" />
              </div>
              <StrategyTimeline strategy={topStrat} totalLaps={totalLaps} currentLap={currentLap} compact={true} />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "8px" }}>
                <span>Target Finish: <strong style={{ color: "var(--text-primary)" }}>P{topStrat.expected_position.toFixed(1)}</strong></span>
                <span>P(Podium): <strong style={{ color: "var(--color-warning)" }}>{Math.round(topStrat.p_podium * 100)}%</strong></span>
                <span>Robustness: <strong style={{ color: "var(--color-success)" }}>{Math.round(topStrat.robustness * 100)}%</strong></span>
              </div>

              {/* Natural language planned strategy directive */}
              <div
                style={{
                  fontSize: "0.76rem",
                  color: "var(--text-secondary)",
                  background: "var(--surface-secondary)",
                  padding: "6px 10px",
                  borderRadius: "var(--radius-sm)",
                  marginTop: "8px",
                  borderLeft: "2px solid var(--f1-red)",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <span>🏎</span>
                <span>
                  {lapData?.recommendation?.target_pit_lap && lapData.recommendation.target_pit_lap > currentLap
                    ? `Using a ${(lapData.recommendation.planned_compounds || topStrat.compounds || []).join(" ➔ ")} strategy with pitting on Lap ${lapData.recommendation.target_pit_lap} for ${lapData.recommendation.target_compound || "HARD"}.`
                    : lapData?.recommendation?.target_pit_lap === currentLap
                    ? `Pitting on Lap ${currentLap} for ${lapData.recommendation.target_compound || "HARD"} compound.`
                    : `Active Stint: ${(lapData?.recommendation?.planned_compounds || topStrat.compounds || []).join(" ➔ ")} to the flag.`}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. OPPONENT THREAT MODELING DECK */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)" }}>
            Immediate Opponent Threats & Predicted Pit Windows
          </div>
          <StatusBadge type="predicted" size="sm" />
        </div>
        <OpponentModelCard />
      </div>
    </div>
  );
}
