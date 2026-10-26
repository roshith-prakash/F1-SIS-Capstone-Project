"use client";

import React, { useState } from "react";
import { useRaceStore } from "@/store/raceStore";
import { useRaceSocket } from "@/hooks/useRaceSocket";
import { simulateSandboxScenario } from "@/lib/api";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { formatPosition, formatScore } from "@/lib/formatters";

export function StrategySandbox() {
  const { lapData, sandboxLoading, sandboxResult, setSandboxLoading, setSandboxResult, isCalculating } = useRaceStore();
  const { armOverride } = useRaceSocket();
  const [action, setAction] = useState<string>("pit_next_lap");
  const [compound, setCompound] = useState<string>("HARD");
  const [stayOutLaps, setStayOutLaps] = useState<number>(3);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [armSuccess, setArmSuccess] = useState<string | null>(null);

  const handleSimulate = async () => {
    if (!lapData) return;
    setSandboxLoading(true);
    setErrorMsg(null);
    try {
      const res = await simulateSandboxScenario(action, compound, stayOutLaps);
      setSandboxResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to run sandbox simulation");
    } finally {
      setSandboxLoading(false);
    }
  };

  const isPitAction = action === "pit_this_lap" || action === "pit_next_lap";

  return (
    <div className="sis-card" style={{ marginTop: "16px", padding: "18px 20px" }}>
      <div className="sis-card-header">
        <div className="sis-card-title">
          <span>📐</span>
          <span>Strategy Sandbox · What-If Hypothetical Scenario Tester</span>
        </div>
        <StatusBadge type="simulated" size="sm" />
      </div>

      <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "14px" }}>
        Simulate an alternative strategic call and evaluate its Monte Carlo outcome against the baseline AI directive.
      </div>

      {/* Control Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: isPitAction ? "1fr 140px 180px" : "1fr 180px",
          gap: "12px",
          alignItems: "end",
          background: "var(--surface-secondary)",
          padding: "14px 16px",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border)",
        }}
      >
        {/* Action Select */}
        <div>
          <label style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "4px", fontWeight: 700 }}>
            Hypothetical Tactical Move
          </label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value)}
            disabled={sandboxLoading || !lapData}
            style={{
              width: "100%",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
              padding: "8px 12px",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.85rem",
              fontWeight: 600,
              outline: "none",
            }}
          >
            <option value="pit_next_lap">Pit Next Lap</option>
            <option value="pit_this_lap">Pit This Lap (Immediate Box)</option>
            <option value="stay_out">Stay Out (Extend Current Stint)</option>
          </select>
        </div>

        {/* Compound Select (if pitting) */}
        {isPitAction && (
          <div>
            <label style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "4px", fontWeight: 700 }}>
              Fitted Compound
            </label>
            <select
              value={compound}
              onChange={(e) => setCompound(e.target.value)}
              disabled={sandboxLoading || !lapData}
              style={{
                width: "100%",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                padding: "8px 12px",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.85rem",
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

        {/* Action Button */}
        <div>
          <button
            type="button"
            onClick={handleSimulate}
            disabled={sandboxLoading || !lapData || isCalculating}
            className="btn-primary"
            style={{ width: "100%", height: "36px", fontSize: "0.82rem" }}
          >
            {sandboxLoading ? "Simulating..." : "▶ RUN SCENARIO"}
          </button>
        </div>
      </div>

      {errorMsg && (
        <div style={{ color: "var(--color-danger)", fontSize: "0.78rem", marginTop: "8px" }}>
          {errorMsg}
        </div>
      )}

      {/* Result Display */}
      {sandboxResult && (
        <div
          className="fade-in"
          style={{
            marginTop: "16px",
            background: "var(--surface-secondary)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)" }}>
              Scenario Analysis vs AI Directive
            </span>
            <span
              className="font-mono"
              style={{
                fontSize: "0.75rem",
                fontWeight: 800,
                color: sandboxResult.vs_ai_delta > 0 ? "var(--color-success)" : "var(--color-warning)",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                padding: "3px 8px",
                borderRadius: "var(--radius-pill)",
              }}
            >
              {sandboxResult.verdict}
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "12px",
            }}
          >
            {/* User Strategy Outcome */}
            <div
              style={{
                background: "var(--surface)",
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--color-info)",
              }}
            >
              <div style={{ fontSize: "0.7rem", color: "var(--color-info)", fontWeight: 700, textTransform: "uppercase" }}>
                Your Hypothetical
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "10px", marginTop: "4px" }}>
                <span className="font-mono" style={{ fontSize: "1.25rem", fontWeight: 900, color: "var(--text-primary)" }}>
                  {formatPosition(sandboxResult.expected_position)}
                </span>
                <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                  Score: {formatScore(sandboxResult.score)}
                </span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                P(Win) {Math.round(sandboxResult.p_win * 100)}% · P(Podium) {Math.round(sandboxResult.p_podium * 100)}%
              </div>
            </div>

            {/* AI Recommendation Outcome */}
            <div
              style={{
                background: "var(--surface)",
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--f1-red)",
              }}
            >
              <div style={{ fontSize: "0.7rem", color: "var(--f1-red)", fontWeight: 700, textTransform: "uppercase" }}>
                AI Recommendation
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "10px", marginTop: "4px" }}>
                <span className="font-mono" style={{ fontSize: "1.25rem", fontWeight: 900, color: "var(--text-primary)" }}>
                  {formatPosition(sandboxResult.vs_ai_position)}
                </span>
                <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                  Score: {formatScore(sandboxResult.vs_ai_score)}
                </span>
              </div>
              <div
                className="font-mono"
                style={{
                  fontSize: "0.75rem",
                  color: sandboxResult.vs_ai_delta <= 0 ? "var(--color-success)" : "var(--color-danger)",
                  marginTop: "4px",
                  fontWeight: 700,
                }}
              >
                AI Advantage: {sandboxResult.vs_ai_delta <= 0 ? `+${Math.abs(sandboxResult.vs_ai_delta).toFixed(3)}` : `-${sandboxResult.vs_ai_delta.toFixed(3)}`}
              </div>
            </div>
          </div>

          {/* Apply as Live Tactical Directive */}
          <div style={{ marginTop: "14px", display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "12px" }}>
            {armSuccess && (
              <span style={{ fontSize: "0.78rem", color: "var(--color-success)", fontWeight: 700 }}>
                ✓ {armSuccess}
              </span>
            )}
            <button
              type="button"
              onClick={async () => {
                try {
                  await armOverride(action, isPitAction ? compound : undefined);
                  setArmSuccess(
                    action === "stay_out"
                      ? "Stay Out armed for next lap!"
                      : `Pit call armed for ${compound} compound!`
                  );
                  setTimeout(() => setArmSuccess(null), 4000);
                } catch (err: any) {
                  setErrorMsg(err.message || "Failed to arm override");
                }
              }}
              disabled={isCalculating}
              className="btn-primary"
              style={{
                padding: "8px 16px",
                fontSize: "0.78rem",
                fontWeight: 800,
                background: "var(--f1-red)",
                letterSpacing: "0.04em",
              }}
            >
              ⚡ APPLY AS LIVE TACTICAL DIRECTIVE
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
