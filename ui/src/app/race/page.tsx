"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useRaceStore } from "@/store/raceStore";
import { useRaceSocket } from "@/hooks/useRaceSocket";
import { AppShell } from "@/components/layout/AppShell";
import { NavSection } from "@/components/layout/Sidebar";
import { OverviewDashboard } from "@/components/dashboard/OverviewDashboard";
import { RaceTower } from "@/components/dashboard/RaceTower";
import { ConditionsPanel } from "@/components/dashboard/ConditionsPanel";
import { EgoTelemetryCard } from "@/components/dashboard/EgoTelemetryCard";
import { StrategyCardList } from "@/components/strategy/StrategyCardList";
import { StrategyTimeline } from "@/components/strategy/StrategyTimeline";
import { StrategyComparisonTable } from "@/components/strategy/StrategyComparisonTable";
import { DecisionCenterpiece } from "@/components/strategy/DecisionCenterpiece";
import { ExplanationPanel } from "@/components/strategy/ExplanationPanel";
import { CounterfactualCard } from "@/components/strategy/CounterfactualCard";
import { OpponentModelCard } from "@/components/strategy/OpponentModelCard";
import { PositionDistributionChart } from "@/components/charts/PositionDistributionChart";
import { StrategySandbox } from "@/components/strategy/StrategySandbox";
import { LapTimeHistory } from "@/components/charts/LapTimeHistory";
import { ModelInsightsPanel } from "@/components/strategy/ModelInsightsPanel";
import { StatusBadge } from "@/components/ui/StatusBadge";

export default function LiveRaceDashboard() {
  const router = useRouter();
  const { isConfigured, currentLap, totalLaps, lapData } = useRaceStore();
  const { advance } = useRaceSocket();

  // If user accesses directly without configuring, redirect to setup
  useEffect(() => {
    if (!isConfigured && !lapData) {
      router.push("/");
    }
  }, [isConfigured, lapData, router]);

  // Initial trigger for Lap 1 if not yet started
  useEffect(() => {
    if (isConfigured && currentLap === 0 && !lapData) {
      advance();
    }
  }, [isConfigured, currentLap, lapData, advance]);

  const topStrat = lapData?.strategies?.[0];

  return (
    <AppShell defaultSection="overview">
      {(activeSection: NavSection) => {
        switch (activeSection) {
          case "overview":
            return <OverviewDashboard />;

          case "liverace":
            return (
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "20px" }}>
                <RaceTower />
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  <EgoTelemetryCard />
                  <ConditionsPanel />
                  <LapTimeHistory />
                </div>
              </div>
            );

          case "strategy":
            return (
              <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                {topStrat && (
                  <div className="sis-card" style={{ padding: "18px 20px" }}>
                    <div className="sis-card-header">
                      <div className="sis-card-title">
                        <span>🗺</span>
                        <span>Baseline Strategy Stint Progression ({topStrat.name})</span>
                      </div>
                      <StatusBadge type="predicted" size="sm" />
                    </div>
                    <StrategyTimeline strategy={topStrat} totalLaps={totalLaps} currentLap={currentLap} />
                  </div>
                )}

                <div className="sis-card" style={{ padding: "18px 20px" }}>
                  <div className="sis-card-header">
                    <div className="sis-card-title">
                      <span>⚖</span>
                      <span>Multi-Strategy Side-by-Side Comparison</span>
                    </div>
                    <StatusBadge type="simulated" size="sm" />
                  </div>
                  <StrategyComparisonTable
                    strategies={lapData?.strategies || []}
                    recommendedId={topStrat?.id}
                  />
                </div>

                <StrategyCardList />
              </div>
            );

          case "decision":
            return (
              <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: "20px" }}>
                <DecisionCenterpiece />
                <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                  <ExplanationPanel />
                  <CounterfactualCard />
                </div>
              </div>
            );

          case "opponents":
            return (
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div className="sis-card" style={{ padding: "18px 20px" }}>
                  <div className="sis-card-header" style={{ marginBottom: "6px" }}>
                    <div className="sis-card-title">
                      <span>🎯</span>
                      <span>Opponent Tactical Pit Window & Threat Modeling</span>
                    </div>
                    <StatusBadge type="predicted" size="sm" />
                  </div>
                  <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                    Predictive analysis of rival pit window entry, undercut likelihood, and track position threats.
                  </p>
                </div>
                <OpponentModelCard />
              </div>
            );

          case "simulation":
            return (
              <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                <PositionDistributionChart />
                <StrategySandbox />
              </div>
            );

          case "analytics":
            return (
              <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
                  <EgoTelemetryCard />
                  <ConditionsPanel />
                </div>
                <LapTimeHistory />
              </div>
            );

          case "modelinsights":
            return <ModelInsightsPanel />;

          default:
            return <OverviewDashboard />;
        }
      }}
    </AppShell>
  );
}
