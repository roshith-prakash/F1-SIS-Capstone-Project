"use client";

import React, { useState } from "react";
import { useRaceStore } from "@/store/raceStore";

export function CounterfactualCard() {
  const { lapData } = useRaceStore();
  const rec = lapData?.recommendation;
  const [isOpen, setIsOpen] = useState(true);

  if (!rec || !rec.counterfactual_strategy || rec.counterfactual_strategy === "N/A") {
    return null;
  }

  return (
    <div className="sis-card" style={{ marginTop: "12px", padding: "14px 16px" }}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <div className="sis-card-title" style={{ marginBottom: 0 }}>
          <span>⚖</span>
          <span>Why not {rec.counterfactual_strategy}?</span>
        </div>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          {isOpen ? "▲" : "▼"}
        </span>
      </div>

      {isOpen && (
        <div style={{ marginTop: "10px", borderTop: "1px solid var(--border-subtle)", paddingTop: "8px" }}>
          <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: "6px", textTransform: "uppercase", fontWeight: 700 }}>
            Rejection analysis vs Runner-Up ({rec.runner_up_name || "Alternative"}):
          </div>
          <ul
            style={{
              listStyle: "none",
              display: "flex",
              flexDirection: "column",
              gap: "5px",
            }}
          >
            {rec.counterfactual_reasons.map((reason, idx) => (
              <li
                key={idx}
                style={{
                  fontSize: "0.78rem",
                  color: "var(--text-secondary)",
                  lineHeight: 1.35,
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "6px",
                }}
              >
                <span style={{ color: "var(--color-danger)", fontWeight: 700 }}>✕</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
