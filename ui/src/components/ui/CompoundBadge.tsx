"use client";

import React from "react";
import { TYRE_COMPOUNDS } from "@/lib/constants";

interface CompoundBadgeProps {
  compound: string;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export function CompoundBadge({ compound, size = "md", showLabel = false }: CompoundBadgeProps) {
  const norm = (compound || "").toUpperCase();
  const info = TYRE_COMPOUNDS[norm] || TYRE_COMPOUNDS.UNKNOWN;

  const sizeStyles = {
    sm: { width: "18px", height: "18px", fontSize: "10px", borderWidth: "1.5px" },
    md: { width: "22px", height: "22px", fontSize: "11px", borderWidth: "1.5px" },
    lg: { width: "28px", height: "28px", fontSize: "13px", borderWidth: "2px" },
  };

  const currentSize = sizeStyles[size] || sizeStyles.md;

  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
      <div
        className="font-mono"
        style={{
          width: currentSize.width,
          height: currentSize.height,
          fontSize: currentSize.fontSize,
          border: `${currentSize.borderWidth} solid ${info.color}`,
          color: info.color,
          backgroundColor: info.bg,
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: 800,
          flexShrink: 0,
        }}
        title={`Compound: ${info.label}`}
      >
        {info.letter}
      </div>
      {showLabel && (
        <span style={{ fontSize: "0.82rem", color: "var(--text-primary)", fontWeight: 600 }}>
          {info.label}
        </span>
      )}
    </div>
  );
}
