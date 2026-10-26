"use client";

import React from "react";

interface ProbabilityBarProps {
  label: string;
  value: number; // 0.0 to 1.0 or 0 to 100
  secondaryLabel?: string;
  color?: string;
  height?: number;
  showPercent?: boolean;
}

export function ProbabilityBar({
  label,
  value,
  secondaryLabel,
  color = "var(--color-info)",
  height = 5,
  showPercent = true,
}: ProbabilityBarProps) {
  const norm = value > 1 ? value : value * 100;
  const pct = Math.min(100, Math.max(0, Math.round(norm)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem" }}>
        <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>{label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {secondaryLabel && (
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{secondaryLabel}</span>
          )}
          {showPercent && (
            <span className="font-mono" style={{ color: "var(--text-primary)", fontWeight: 700 }}>
              {pct}%
            </span>
          )}
        </div>
      </div>

      <div
        style={{
          height: `${height}px`,
          background: "var(--border-subtle)",
          borderRadius: "var(--radius-pill)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: "var(--radius-pill)",
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}
