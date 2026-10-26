"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function PositionDistributionChart() {
  const { lapData } = useRaceStore();
  const strategies = lapData?.strategies || [];

  if (strategies.length === 0) return null;

  return (
    <div className="sis-card" style={{ marginTop: "16px", padding: "18px 20px" }}>
      <div className="sis-card-header" style={{ marginBottom: "12px" }}>
        <div className="sis-card-title">
          <span>📊</span>
          <span>Monte Carlo Finishing Position Density</span>
        </div>
        <StatusBadge type="simulated" size="sm" />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {strategies.slice(0, 4).map((strat, idx) => {
          const pWin = Math.round(strat.p_win * 100);
          const pPodium = Math.max(0, Math.round(strat.p_podium * 100) - pWin);
          const pTop5 = Math.max(0, Math.round(strat.p_top5 * 100) - pWin - pPodium);
          const pPoints = Math.max(0, Math.round(strat.p_points * 100) - pWin - pPodium - pTop5);
          const pOther = Math.max(0, 100 - pWin - pPodium - pTop5 - pPoints);

          return (
            <div key={strat.id} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem" }}>
                <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>
                  #{idx + 1} · {strat.name}
                </span>
                <span className="font-mono" style={{ color: "var(--text-secondary)", fontWeight: 600 }}>
                  Exp P{strat.expected_position.toFixed(1)}
                </span>
              </div>

              {/* Stacked Outcome Distribution Bar */}
              <div
                style={{
                  height: "14px",
                  background: "var(--surface-secondary)",
                  borderRadius: "var(--radius-xs)",
                  border: "1px solid var(--border-subtle)",
                  overflow: "hidden",
                  display: "flex",
                }}
              >
                {pWin > 0 && (
                  <div
                    style={{ width: `${pWin}%`, background: "var(--f1-red)" }}
                    title={`P1 Win: ${pWin}%`}
                  />
                )}
                {pPodium > 0 && (
                  <div
                    style={{ width: `${pPodium}%`, background: "var(--color-warning)" }}
                    title={`P2-3 Podium: ${pPodium}%`}
                  />
                )}
                {pTop5 > 0 && (
                  <div
                    style={{ width: `${pTop5}%`, background: "var(--color-info)" }}
                    title={`P4-5 Top 5: ${pTop5}%`}
                  />
                )}
                {pPoints > 0 && (
                  <div
                    style={{ width: `${pPoints}%`, background: "var(--color-success)" }}
                    title={`P6-10 Points: ${pPoints}%`}
                  />
                )}
                {pOther > 0 && (
                  <div
                    style={{ width: `${pOther}%`, background: "var(--border-strong)" }}
                    title={`P11+ Outside Points: ${pOther}%`}
                  />
                )}
              </div>
            </div>
          );
        })}

        {/* Legend */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "16px",
            fontSize: "0.72rem",
            color: "var(--text-secondary)",
            marginTop: "6px",
            paddingTop: "8px",
            borderTop: "1px solid var(--border-subtle)",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", background: "var(--f1-red)", borderRadius: "2px" }} />
            <span>P1 Win</span>
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", background: "var(--color-warning)", borderRadius: "2px" }} />
            <span>P2–3 Podium</span>
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", background: "var(--color-info)", borderRadius: "2px" }} />
            <span>P4–5 Top 5</span>
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", background: "var(--color-success)", borderRadius: "2px" }} />
            <span>P6–10 Points</span>
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", background: "var(--border-strong)", borderRadius: "2px" }} />
            <span>P11+</span>
          </span>
        </div>
      </div>
    </div>
  );
}
