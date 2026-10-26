"use client";

import React from "react";

export type DataOriginType = "live" | "predicted" | "simulated" | "historical";

interface StatusBadgeProps {
  type: DataOriginType | "track_green" | "track_sc" | "track_vsc" | "track_red";
  label?: string;
  size?: "sm" | "md";
}

export function StatusBadge({ type, label, size = "md" }: StatusBadgeProps) {
  let text = label;
  let symbol = "●";
  let color = "var(--color-info)";
  let bg = "var(--color-info-bg)";
  let border = "var(--color-info-border)";

  switch (type) {
    case "live":
      text = text || "LIVE";
      symbol = "●";
      color = "var(--color-success)";
      bg = "var(--color-success-bg)";
      border = "var(--color-success-border)";
      break;
    case "predicted":
      text = text || "PREDICTED";
      symbol = "◉";
      color = "var(--color-info)";
      bg = "var(--color-info-bg)";
      border = "var(--color-info-border)";
      break;
    case "simulated":
      text = text || "SIMULATED";
      symbol = "◇";
      color = "var(--color-purple)";
      bg = "var(--color-purple-bg)";
      border = "var(--color-purple-border)";
      break;
    case "historical":
      text = text || "HISTORICAL";
      symbol = "○";
      color = "var(--text-secondary)";
      bg = "var(--surface-secondary)";
      border = "var(--border)";
      break;
    case "track_green":
      text = text || "GREEN TRACK";
      symbol = "●";
      color = "var(--color-success)";
      bg = "var(--color-success-bg)";
      border = "var(--color-success-border)";
      break;
    case "track_sc":
      text = text || "SAFETY CAR";
      symbol = "⚠";
      color = "var(--color-warning)";
      bg = "var(--color-warning-bg)";
      border = "var(--color-warning-border)";
      break;
    case "track_vsc":
      text = text || "VIRTUAL SC";
      symbol = "⚠";
      color = "#B45309";
      bg = "#FEF3C7";
      border = "#FCD34D";
      break;
    case "track_red":
      text = text || "RED FLAG";
      symbol = "■";
      color = "var(--color-danger)";
      bg = "var(--color-danger-bg)";
      border = "var(--color-danger-border)";
      break;
  }

  const isSmall = size === "sm";

  return (
    <span
      className="font-mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: isSmall ? "4px" : "6px",
        padding: isSmall ? "1px 6px" : "2px 8px",
        borderRadius: "var(--radius-pill)",
        background: bg,
        color: color,
        border: `1px solid ${border}`,
        fontSize: isSmall ? "0.68rem" : "0.72rem",
        fontWeight: 700,
        letterSpacing: "0.04em",
        whiteSpace: "nowrap",
        userSelect: "none",
      }}
    >
      <span style={{ fontSize: isSmall ? "0.65rem" : "0.75rem", lineHeight: 1 }}>{symbol}</span>
      <span>{text}</span>
    </span>
  );
}
