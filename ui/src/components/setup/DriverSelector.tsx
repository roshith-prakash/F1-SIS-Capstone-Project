"use client";

import React from "react";

interface DriverSelectorProps {
  drivers: string[];
  selectedDriver: string;
  onSelectDriver: (driver: string) => void;
}

export function DriverSelector({
  drivers,
  selectedDriver,
  onSelectDriver,
}: DriverSelectorProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <label
          style={{
            fontSize: "0.74rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
          }}
        >
          Select Focus Driver (Ego)
        </label>
        <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
          Telemetry focus locked for session
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "8px",
          background: "var(--surface-secondary)",
          padding: "12px",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border)",
          maxHeight: "130px",
          overflowY: "auto",
        }}
      >
        {drivers.map((drv) => {
          const isSelected = drv === selectedDriver;
          return (
            <button
              key={drv}
              type="button"
              onClick={() => onSelectDriver(drv)}
              className="font-mono"
              style={{
                background: isSelected ? "var(--f1-red)" : "var(--surface)",
                color: isSelected ? "#FFFFFF" : "var(--text-primary)",
                border: `1px solid ${isSelected ? "var(--f1-red)" : "var(--border)"}`,
                borderRadius: "var(--radius-sm)",
                padding: "6px 12px",
                fontSize: "0.85rem",
                fontWeight: 700,
                cursor: "pointer",
                transition: "all 0.15s ease",
                boxShadow: isSelected ? "var(--shadow-sm)" : "var(--shadow-xs)",
              }}
            >
              {drv}
            </button>
          );
        })}
      </div>
    </div>
  );
}
