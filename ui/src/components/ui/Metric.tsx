"use client";

import React from "react";

interface MetricProps {
  label: string;
  value: React.ReactNode;
  subValue?: React.ReactNode;
  delta?: {
    value: string | number;
    positive?: boolean;
    neutral?: boolean;
  };
  highlight?: boolean;
  size?: "md" | "lg" | "xl";
}

export function Metric({
  label,
  value,
  subValue,
  delta,
  highlight = false,
  size = "md",
}: MetricProps) {
  const valueSizes = {
    md: "1.25rem",
    lg: "1.65rem",
    xl: "2.1rem",
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "3px",
      }}
    >
      <div
        style={{
          fontSize: "0.68rem",
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        {label}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: "8px",
        }}
      >
        <div
          className="font-mono"
          style={{
            fontSize: valueSizes[size],
            fontWeight: 800,
            lineHeight: 1.15,
            color: highlight ? "var(--f1-red)" : "var(--text-primary)",
            letterSpacing: "-0.02em",
          }}
        >
          {value}
        </div>

        {delta && (
          <span
            className="font-mono"
            style={{
              fontSize: "0.75rem",
              fontWeight: 700,
              color: delta.neutral
                ? "var(--text-muted)"
                : delta.positive
                ? "var(--color-success)"
                : "var(--color-danger)",
            }}
          >
            {delta.value}
          </span>
        )}
      </div>

      {subValue && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "var(--text-secondary)",
            marginTop: "1px",
          }}
        >
          {subValue}
        </div>
      )}
    </div>
  );
}

interface MetricGridProps {
  children: React.ReactNode;
  columns?: number;
}

export function MetricGrid({ children, columns = 4 }: MetricGridProps) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: "16px",
        padding: "14px 16px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-xs)",
      }}
    >
      {children}
    </div>
  );
}
