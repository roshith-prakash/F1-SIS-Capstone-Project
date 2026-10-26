"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";

interface LapScrubberProps {
  selectedLap: number;
  onSelectLap: (lap: number) => void;
}

export function LapScrubber({ selectedLap, onSelectLap }: LapScrubberProps) {
  const { lapHistory, totalLaps } = useRaceStore();
  const maxLap = lapHistory.length > 0 ? Math.max(...lapHistory.map((h) => h.lap)) : 1;

  return (
    <div className="sis-card" style={{ marginBottom: "16px", padding: "16px 20px" }}>
      <div className="sis-card-header" style={{ marginBottom: "8px" }}>
        <div className="sis-card-title">
          <span>⏪</span>
          <span>Race Lap Scrubber & Timeline Navigation</span>
        </div>
        <div className="font-mono" style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--f1-red)" }}>
          Lap {selectedLap} / {totalLaps || maxLap}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <input
          type="range"
          min={1}
          max={maxLap}
          value={selectedLap}
          onChange={(e) => onSelectLap(parseInt(e.target.value, 10))}
          style={{ flex: 1, accentColor: "var(--f1-red)", cursor: "pointer", height: "6px" }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "6px" }}>
        <span>Lap 1 (Race Start)</span>
        <span>Seek to any lap to inspect telemetry snapshot and AI decision state</span>
        <span>Lap {maxLap} (Latest Evaluated)</span>
      </div>
    </div>
  );
}
