"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function SCRiskGauge() {
  const { lapData } = useRaceStore();
  const sc = lapData?.sc_probabilities;

  const horizons = [
    { label: "1 Lap", sc: sc?.p_sc_h1 ?? 0.02, vsc: sc?.p_vsc_h1 ?? 0.01 },
    { label: "3 Laps", sc: sc?.p_sc_h3 ?? 0.05, vsc: sc?.p_vsc_h3 ?? 0.03 },
    { label: "5 Laps", sc: sc?.p_sc_h5 ?? 0.08, vsc: sc?.p_vsc_h5 ?? 0.05 },
    { label: "10 Laps", sc: sc?.p_sc_h10 ?? 0.15, vsc: sc?.p_vsc_h10 ?? 0.09 },
  ];

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "10px" }}>
        <div className="sis-card-title">
          <span>⚠</span>
          <span>Safety Car Disruption Risk Model</span>
        </div>
        <StatusBadge type="predicted" size="sm" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", textAlign: "center" }}>
        {horizons.map((h, idx) => {
          const scPct = Math.round(h.sc * 100);
          let scColor = "var(--color-success)";
          if (scPct > 20) scColor = "var(--color-danger)";
          else if (scPct > 10) scColor = "var(--color-warning)";

          return (
            <div
              key={idx}
              style={{
                background: "var(--surface-secondary)",
                padding: "8px 6px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                {h.label}
              </div>
              <div
                className="font-mono"
                style={{ fontSize: "0.98rem", fontWeight: 800, color: scColor, marginTop: "2px" }}
              >
                {scPct}%
              </div>
              <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                VSC {Math.round(h.vsc * 100)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
