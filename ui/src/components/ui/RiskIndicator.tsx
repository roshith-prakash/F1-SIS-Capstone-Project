"use client";

import React from "react";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

interface RiskIndicatorProps {
  level: RiskLevel | string;
  score?: number;
  label?: string;
  size?: "sm" | "md";
}

export function RiskIndicator({
  level,
  score,
  label,
  size = "md",
}: RiskIndicatorProps) {
  const normLevel = (level || "LOW").toUpperCase() as RiskLevel;

  let color = "var(--color-success)";
  let bg = "var(--color-success-bg)";
  let border = "var(--color-success-border)";
  let symbol = "✓";

  switch (normLevel) {
    case "LOW":
      color = "var(--color-success)";
      bg = "var(--color-success-bg)";
      border = "var(--color-success-border)";
      symbol = "✓";
      break;
    case "MEDIUM":
      color = "var(--color-warning)";
      bg = "var(--color-warning-bg)";
      border = "var(--color-warning-border)";
      symbol = "▲";
      break;
    case "HIGH":
      color = "var(--color-danger)";
      bg = "var(--color-danger-bg)";
      border = "var(--color-danger-border)";
      symbol = "⚠";
      break;
    case "CRITICAL":
      color = "#991B1B";
      bg = "#FEE2E2";
      border = "#F87171";
      symbol = "!";
      break;
  }

  const isSmall = size === "sm";

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "8px",
      }}
    >
      {label && (
        <span
          style={{
            fontSize: isSmall ? "0.75rem" : "0.8rem",
            color: "var(--text-secondary)",
            fontWeight: 500,
          }}
        >
          {label}
        </span>
      )}

      <span
        className="font-mono"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          padding: isSmall ? "1px 6px" : "2px 8px",
          borderRadius: "var(--radius-pill)",
          background: bg,
          color: color,
          border: `1px solid ${border}`,
          fontSize: isSmall ? "0.68rem" : "0.72rem",
          fontWeight: 700,
          letterSpacing: "0.04em",
          lineHeight: 1.2,
        }}
      >
        <span>{symbol}</span>
        <span>{normLevel}</span>
        {score !== undefined && (
          <span style={{ opacity: 0.8, marginLeft: "2px" }}>
            ({score.toFixed(2)})
          </span>
        )}
      </span>
    </div>
  );
}
