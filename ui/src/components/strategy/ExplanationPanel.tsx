"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";

export function ExplanationPanel() {
  const { lapData } = useRaceStore();
  const rec = lapData?.recommendation;
  const reasons = rec?.explanation || [];

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "16px" }}>
      <div className="sis-card-header" style={{ marginBottom: "10px" }}>
        <div className="sis-card-title">
          <span>💡</span>
          <span>Tactical Rationale & Factor Influence</span>
        </div>
      </div>

      {reasons.length === 0 ? (
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", padding: "8px 0" }}>
          No tactical notes generated yet.
        </div>
      ) : (
        <ul
          style={{
            listStyle: "none",
            display: "flex",
            flexDirection: "column",
            gap: "6px",
          }}
        >
          {reasons.map((reason, idx) => (
            <li
              key={idx}
              style={{
                fontSize: "0.8rem",
                lineHeight: 1.4,
                color: "var(--text-primary)",
                display: "flex",
                alignItems: "flex-start",
                gap: "8px",
              }}
            >
              <span style={{ color: "var(--f1-red)", marginTop: "2px", fontWeight: 700 }}>▸</span>
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
