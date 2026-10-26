"use client";

import React from "react";
import { useRaceStore } from "@/store/raceStore";
import { StrategyCard } from "./StrategyCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { EmptyState } from "@/components/ui/EmptyState";

export function StrategyCardList() {
  const { lapData } = useRaceStore();
  const strategies = lapData?.strategies || [];

  return (
    <div className="sis-card" style={{ padding: "18px 20px" }}>
      <div className="sis-card-header">
        <div className="sis-card-title">
          <span>⚡</span>
          <span>Ranked Race Strategy Candidates</span>
        </div>
        <StatusBadge type="simulated" size="sm" />
      </div>

      {strategies.length === 0 ? (
        <EmptyState
          title="No Strategies Evaluated"
          description="Monte Carlo simulations will evaluate forward stints and compound lifecycles once telemetry is active."
          compact={true}
        />
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {strategies.slice(0, 5).map((strat, index) => (
            <StrategyCard
              key={strat.id || index}
              strategy={strat}
              isTopRanked={index === 0}
            />
          ))}
        </div>
      )}
    </div>
  );
}
