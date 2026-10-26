"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { SCRiskGauge } from "./SCRiskGauge";
import { EventProbabilityPanel } from "./EventProbabilityPanel";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { RISK_PROFILES } from "@/lib/constants";

export function ModelInsightsPanel() {
  const { config, lapData } = useRaceStore();
  const profileInfo = RISK_PROFILES.find((p) => p.id === config.profile) || RISK_PROFILES[0];
  const topStrat = lapData?.strategies?.[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* 1. Header & Calibration Status */}
      <div className="sis-card" style={{ padding: "18px 20px" }}>
        <div className="sis-card-header" style={{ marginBottom: "8px" }}>
          <div className="sis-card-title">
            <span>🔬</span>
            <span>Predictive Intelligence & Calibration Engine</span>
          </div>
          <StatusBadge type="predicted" size="sm" />
        </div>
        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.45 }}>
          Real-time diagnostic metrics, model objective weights, Bayesian safety car hazard rates, and situational event probabilities.
        </p>

        {/* Diagnostic Metadata Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "12px",
            marginTop: "14px",
            padding: "12px",
            background: "var(--surface-secondary)",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Tactical Profile
            </div>
            <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
              {profileInfo.title}
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Rollout Depth
            </div>
            <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
              {config.rollouts} runs / strat
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Lookahead Horizon
            </div>
            <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--color-info)", marginTop: "2px" }}>
              {config.horizon_laps} Laps
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
              Variance Robustness
            </div>
            <div className="font-mono" style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--color-success)", marginTop: "2px" }}>
              {topStrat ? `${Math.round(topStrat.robustness * 100)}%` : "Calibrating"}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Objective Function Weights */}
      <div className="sis-card" style={{ padding: "18px 20px" }}>
        <div className="sis-card-header" style={{ marginBottom: "12px" }}>
          <div className="sis-card-title">
            <span>⚖</span>
            <span>Decision Engine Feature Importance & Objective Weights ({profileInfo.title})</span>
          </div>
          <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
            PROFILE WEIGHTS
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
          <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", fontWeight: 700, marginBottom: "4px" }}>
              <span style={{ color: "var(--text-secondary)" }}>P(Win) Weight</span>
              <span className="font-mono" style={{ color: "var(--color-danger)" }}>{Math.round(profileInfo.weights.win * 100)}%</span>
            </div>
            <div style={{ height: "4px", background: "var(--border)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
              <div style={{ width: `${profileInfo.weights.win * 100}%`, height: "100%", background: "var(--f1-red)" }} />
            </div>
          </div>

          <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", fontWeight: 700, marginBottom: "4px" }}>
              <span style={{ color: "var(--text-secondary)" }}>P(Podium) Weight</span>
              <span className="font-mono" style={{ color: "var(--color-warning)" }}>{Math.round(profileInfo.weights.podium * 100)}%</span>
            </div>
            <div style={{ height: "4px", background: "var(--border)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
              <div style={{ width: `${profileInfo.weights.podium * 100}%`, height: "100%", background: "var(--color-warning)" }} />
            </div>
          </div>

          <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", fontWeight: 700, marginBottom: "4px" }}>
              <span style={{ color: "var(--text-secondary)" }}>Tyre Cliff Preservation</span>
              <span className="font-mono" style={{ color: "var(--color-success)" }}>{Math.round(profileInfo.weights.tyre * 100)}%</span>
            </div>
            <div style={{ height: "4px", background: "var(--border)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
              <div style={{ width: `${profileInfo.weights.tyre * 100}%`, height: "100%", background: "var(--color-success)" }} />
            </div>
          </div>

          <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", fontWeight: 700, marginBottom: "4px" }}>
              <span style={{ color: "var(--text-secondary)" }}>Robustness / Invariance</span>
              <span className="font-mono" style={{ color: "var(--color-info)" }}>{Math.round(profileInfo.weights.robustness * 100)}%</span>
            </div>
            <div style={{ height: "4px", background: "var(--border)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
              <div style={{ width: `${profileInfo.weights.robustness * 100}%`, height: "100%", background: "var(--color-info)" }} />
            </div>
          </div>
        </div>
      </div>

      {/* 3. Safety Car Hazard Model & Situational Probabilities */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        <SCRiskGauge />
        <EventProbabilityPanel />
      </div>
    </div>
  );
}
