"use client";

import React from "react";
import { RISK_PROFILES } from "@/lib/constants";

interface RiskProfileSelectorProps {
  selectedProfile: string;
  onSelectProfile: (profile: string) => void;
}

export function RiskProfileSelector({
  selectedProfile,
  onSelectProfile,
}: RiskProfileSelectorProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <label
        style={{
          fontSize: "0.74rem",
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        Strategy Utility & Risk Profile
      </label>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "10px",
        }}
      >
        {RISK_PROFILES.map((p) => {
          const isSelected = p.id === selectedProfile;
          return (
            <div
              key={p.id}
              onClick={() => onSelectProfile(p.id)}
              style={{
                background: isSelected ? "var(--surface)" : "var(--surface-secondary)",
                border: `1.5px solid ${isSelected ? p.color : "var(--border)"}`,
                borderRadius: "var(--radius-md)",
                padding: "12px 14px",
                cursor: "pointer",
                transition: "all 0.15s ease",
                boxShadow: isSelected ? "var(--shadow-sm)" : "none",
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 800, fontSize: "0.9rem", color: "var(--text-primary)" }}>
                  {p.title}
                </span>
                <span
                  className="font-mono"
                  style={{
                    fontSize: "0.65rem",
                    fontWeight: 800,
                    letterSpacing: "0.05em",
                    padding: "2px 6px",
                    borderRadius: "var(--radius-pill)",
                    background: p.bg,
                    color: p.color,
                    border: `1px solid ${p.border}`,
                  }}
                >
                  {p.badge}
                </span>
              </div>

              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                {p.description}
              </p>

              {/* Weight breakdown mini bar */}
              <div style={{ marginTop: "auto", paddingTop: "6px" }}>
                <div style={{ display: "flex", height: "4px", borderRadius: "2px", overflow: "hidden", gap: "2px" }}>
                  <div style={{ width: `${p.weights.win * 100}%`, background: "var(--f1-red)" }} title={`Win: ${p.weights.win}`} />
                  <div style={{ width: `${p.weights.podium * 100}%`, background: "var(--color-warning)" }} title={`Podium: ${p.weights.podium}`} />
                  <div style={{ width: `${p.weights.tyre * 100}%`, background: "var(--color-success)" }} title={`Tyre: ${p.weights.tyre}`} />
                  <div style={{ width: `${p.weights.robustness * 100}%`, background: "var(--color-info)" }} title={`Robustness: ${p.weights.robustness}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
