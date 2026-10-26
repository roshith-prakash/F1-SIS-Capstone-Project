"use client";

import React from "react";
import type { RaceAvailableItem } from "@/types/race";

interface RaceSelectorProps {
  races: RaceAvailableItem[];
  selectedRace: string;
  onSelectRace: (file: string) => void;
  isLoading?: boolean;
}

export function RaceSelector({
  races,
  selectedRace,
  onSelectRace,
  isLoading = false,
}: RaceSelectorProps) {
  // Group races by year
  const years = Array.from(new Set(races.map((r) => r.year))).sort((a, b) => b - a);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <label
        style={{
          fontSize: "0.74rem",
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        Select Grand Prix Session Telemetry
      </label>

      <div style={{ position: "relative" }}>
        <select
          value={selectedRace}
          onChange={(e) => onSelectRace(e.target.value)}
          disabled={isLoading || races.length === 0}
          style={{
            width: "100%",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            color: "var(--text-primary)",
            padding: "10px 14px",
            fontSize: "0.92rem",
            fontWeight: 600,
            outline: "none",
            cursor: "pointer",
            appearance: "none",
            WebkitAppearance: "none",
            boxShadow: "var(--shadow-xs)",
          }}
        >
          {isLoading && <option>Loading available race sessions...</option>}
          {!isLoading && races.length === 0 && <option>No races found</option>}

          {years.map((year) => (
            <optgroup key={year} label={`${year} Season`}>
              {races
                .filter((r) => r.year === year)
                .map((race) => (
                  <option key={race.file} value={race.file}>
                    {race.name}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        <div
          style={{
            position: "absolute",
            right: "14px",
            top: "50%",
            transform: "translateY(-50%)",
            pointerEvents: "none",
            color: "var(--text-muted)",
            fontSize: "0.75rem",
          }}
        >
          ▼
        </div>
      </div>
    </div>
  );
}
