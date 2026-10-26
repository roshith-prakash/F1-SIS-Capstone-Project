"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRaceStore } from "@/store/raceStore";
import { LapScrubber } from "@/components/replay/LapScrubber";
import { StrategicDivergenceLog } from "@/components/replay/StrategicDivergenceLog";
import { CompoundBadge } from "@/components/ui/CompoundBadge";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { formatScore, formatLapTime } from "@/lib/formatters";

export function ReplayPage() {
  const { lapHistory, config, currentLap } = useRaceStore();
  const [selectedLap, setSelectedLap] = useState<number>(currentLap || 1);

  // Find the selected lap in history
  const activeRecord = lapHistory.find((h) => h.lap === selectedLap) || lapHistory[lapHistory.length - 1];

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-app)", padding: "24px 32px" }}>
      {/* Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <Link
            href="/race"
            className="btn-secondary"
            style={{
              textDecoration: "none",
              fontSize: "0.82rem",
              padding: "7px 12px",
            }}
          >
            <span>←</span>
            <span>Return to Live Pit Wall</span>
          </Link>

          <div>
            <h1 style={{ fontSize: "1.35rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
              Post-Race Telemetry Replay & Strategic Divergence
            </h1>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "1px" }}>
              {config.raceName || "Grand Prix"} · Driver: <strong style={{ color: "var(--text-primary)" }}>{config.driver}</strong> · Tactical Profile:{" "}
              <strong style={{ textTransform: "capitalize", color: "var(--color-info)" }}>{config.profile}</strong>
            </p>
          </div>
        </div>

        <StatusBadge type="historical" size="md" />
      </div>

      {/* Scrubber */}
      <LapScrubber selectedLap={selectedLap} onSelectLap={setSelectedLap} />

      {/* Replay Details Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "20px" }}>
        {/* Left: Lap Snapshot */}
        <div className="sis-card" style={{ padding: "18px 20px" }}>
          <div className="sis-card-header" style={{ marginBottom: "12px" }}>
            <div className="sis-card-title">
              <span>⏱</span>
              <span>Snapshot at Lap {selectedLap}</span>
            </div>
            <span className="font-mono" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              TELEMETRY RECORD
            </span>
          </div>

          {activeRecord ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <div
                style={{
                  background: activeRecord.recommendation_action.toUpperCase().includes("PIT")
                    ? "var(--f1-red-subtle)"
                    : "var(--surface-secondary)",
                  padding: "16px",
                  borderRadius: "var(--radius-md)",
                  border: `1px solid ${
                    activeRecord.recommendation_action.toUpperCase().includes("PIT")
                      ? "var(--f1-red-border)"
                      : "var(--border)"
                  }`,
                }}
              >
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>
                  AI Directive at this Lap
                </div>
                <div
                  className="font-mono"
                  style={{
                    fontSize: "1.35rem",
                    fontWeight: 900,
                    color: activeRecord.recommendation_action.toUpperCase().includes("PIT")
                      ? "var(--f1-red)"
                      : "var(--text-primary)",
                    marginTop: "3px",
                  }}
                >
                  {activeRecord.recommendation_action}
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  Utility Score: <strong className="font-mono" style={{ color: "var(--text-primary)" }}>{formatScore(activeRecord.recommendation_score)}</strong>
                </div>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "10px",
                }}
              >
                <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Actual Position</div>
                  <div className="font-mono" style={{ fontSize: "1.3rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
                    P{activeRecord.actual_position}
                  </div>
                </div>

                <div style={{ background: "var(--surface-secondary)", padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Lap Pace</div>
                  <div className="font-mono" style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", marginTop: "2px" }}>
                    {formatLapTime(activeRecord.actual_lap_time)}
                  </div>
                </div>
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  background: "var(--surface-secondary)",
                  padding: "12px",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <CompoundBadge compound={activeRecord.actual_compound} size="md" showLabel />
                <span className="font-mono" style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                  {Math.round(activeRecord.actual_tyre_life)} Laps on tyre
                </span>
              </div>
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", padding: "20px 0", textAlign: "center", fontSize: "0.82rem" }}>
              No telemetry data available for this lap.
            </div>
          )}
        </div>

        {/* Right: Divergence Log */}
        <StrategicDivergenceLog />
      </div>
    </div>
  );
}

export default ReplayPage;
