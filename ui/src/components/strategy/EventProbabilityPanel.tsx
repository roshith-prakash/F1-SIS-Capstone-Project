"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { ProbabilityBar } from "@/components/ui/ProbabilityBar";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function EventProbabilityPanel() {
  const { lapData } = useRaceStore();
  const ep = lapData?.event_probabilities;

  const items = [
    { label: "Undercut Success Likelihood", val: ep?.undercut_success ?? 0.55, color: "var(--color-info)" },
    { label: "Overtake Probability This Lap", val: ep?.overtake_this_lap ?? 0.20, color: "var(--color-success)" },
    { label: "Precipitation Forecast (Next 5 Laps)", val: ep?.rain_5l ?? 0.0, color: "#0284C7" },
  ];

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "12px" }}>
        <div className="sis-card-title">
          <span>📊</span>
          <span>Situational Probabilities</span>
        </div>
        <StatusBadge type="predicted" size="sm" />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {items.map((it, idx) => (
          <ProbabilityBar
            key={idx}
            label={it.label}
            value={it.val}
            color={it.color}
            height={5}
          />
        ))}
      </div>
    </div>
  );
}
