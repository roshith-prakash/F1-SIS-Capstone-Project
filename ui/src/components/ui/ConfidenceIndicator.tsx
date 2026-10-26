"use client";

import React from "react";

interface ConfidenceIndicatorProps {
  confidence: number; // 0.0 to 1.0 or 0 to 100
  showBar?: boolean;
  size?: "sm" | "md";
}

export function ConfidenceIndicator({
  confidence,
  showBar = true,
  size = "md",
}: ConfidenceIndicatorProps) {
  const norm = confidence > 1 ? confidence : confidence * 100;
  const rounded = Math.round(norm);

  let levelText = "HIGH";
  let color = "var(--color-success)";
  let bg = "var(--color-success-bg)";
  let border = "var(--color-success-border)";

  if (rounded < 60) {
    levelText = "LOW";
    color = "var(--color-danger)";
    bg = "var(--color-danger-bg)";
    border = "var(--color-danger-border)";
  } else if (rounded < 78) {
    levelText = "MODERATE";
    color = "var(--color-warning)";
    bg = "var(--color-warning-bg)";
    border = "var(--color-warning-border)";
  }

  const isSmall = size === "sm";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px" }}>
        <span
          style={{
            fontSize: isSmall ? "0.68rem" : "0.72rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
          }}
        >
          Model Confidence
        </span>
        <span
          className="font-mono"
          style={{
            fontSize: isSmall ? "0.75rem" : "0.82rem",
            fontWeight: 800,
            color: color,
          }}
        >
          {rounded}% <span style={{ fontSize: "0.68rem", fontWeight: 700 }}>({levelText})</span>
        </span>
      </div>

      {showBar && (
        <div
          style={{
            height: "4px",
            background: "var(--border)",
            borderRadius: "var(--radius-pill)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${Math.min(100, Math.max(0, rounded))}%`,
              height: "100%",
              background: color,
              borderRadius: "var(--radius-pill)",
              transition: "width 0.3s ease",
            }}
          />
        </div>
      )}
    </div>
  );
}
