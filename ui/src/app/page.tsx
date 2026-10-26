"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchAvailableRaces, setupRaceSession } from "@/lib/api";
import { useRaceStore } from "@/store/raceStore";
import { RaceSelector } from "@/components/setup/RaceSelector";
import { DriverSelector } from "@/components/setup/DriverSelector";
import { RiskProfileSelector } from "@/components/setup/RiskProfileSelector";
import { SimulationConfig } from "@/components/setup/SimulationConfig";
import { StatusBadge } from "@/components/ui/StatusBadge";
import type { RaceAvailableItem } from "@/types/race";

export default function SetupPage() {
  const router = useRouter();
  const { setConfig, setIsConfigured, setSessionInfo, resetSession } = useRaceStore();

  const [races, setRaces] = useState<RaceAvailableItem[]>([]);
  const [loadingRaces, setLoadingRaces] = useState<boolean>(true);
  const [selectedRaceFile, setSelectedRaceFile] = useState<string>("");
  const [availableDrivers, setAvailableDrivers] = useState<string[]>([]);
  const [selectedDriver, setSelectedDriver] = useState<string>("VER");
  const [selectedProfile, setSelectedProfile] = useState<string>("balanced");
  const [rollouts, setRollouts] = useState<number>(20);
  const [horizonLaps, setHorizonLaps] = useState<number>(8);
  const [isLaunching, setIsLaunching] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    resetSession();
    fetchAvailableRaces()
      .then((data) => {
        setRaces(data);
        if (data.length > 0) {
          // Default to Monza / Italian GP or first race
          const defaultRace =
            data.find((r) => r.name.toLowerCase().includes("italian") && r.year === 2023) ||
            data.find((r) => r.year === 2023) ||
            data[0];

          setSelectedRaceFile(defaultRace.file);
          setAvailableDrivers(defaultRace.drivers);
          if (defaultRace.drivers.includes("VER")) {
            setSelectedDriver("VER");
          } else if (defaultRace.drivers.length > 0) {
            setSelectedDriver(defaultRace.drivers[0]);
          }
        }
      })
      .catch(() => {
        setErrorMsg("Failed to connect to Python backend. Ensure the API server is running on port 8000.");
      })
      .finally(() => {
        setLoadingRaces(false);
      });
  }, [resetSession]);

  const handleSelectRace = (file: string) => {
    setSelectedRaceFile(file);
    const found = races.find((r) => r.file === file);
    if (found) {
      setAvailableDrivers(found.drivers);
      if (!found.drivers.includes(selectedDriver) && found.drivers.length > 0) {
        setSelectedDriver(found.drivers[0]);
      }
    }
  };

  const handleLaunch = async () => {
    if (!selectedRaceFile) return;
    setIsLaunching(true);
    setErrorMsg(null);

    const raceItem = races.find((r) => r.file === selectedRaceFile);

    try {
      const resp = await setupRaceSession({
        race_file: selectedRaceFile,
        driver: selectedDriver,
        profile: selectedProfile,
        rollouts,
        horizon_laps: horizonLaps,
      });

      setConfig({
        race_file: selectedRaceFile,
        driver: selectedDriver,
        profile: selectedProfile,
        rollouts,
        horizon_laps: horizonLaps,
        raceName: resp.race_name || raceItem?.name || "Grand Prix",
      });

      setSessionInfo(1, resp.total_laps, resp.driver);
      setIsConfigured(true);
      router.push("/race");
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to initialize race session");
      setIsLaunching(false);
    }
  };

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "var(--bg-app)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "40px 20px",
      }}
    >
      <div
        className="sis-card fade-in"
        style={{
          width: "100%",
          maxWidth: "840px",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-xl)",
          padding: "32px 36px",
          boxShadow: "var(--shadow-md)",
        }}
      >
        {/* Title Bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div
              style={{
                background: "var(--text-primary)",
                color: "#FFFFFF",
                fontWeight: 900,
                fontSize: "0.95rem",
                padding: "6px 12px",
                borderRadius: "var(--radius-sm)",
                letterSpacing: "0.08em",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <span>F1-SIS</span>
              <span
                style={{
                  width: "6px",
                  height: "6px",
                  borderRadius: "50%",
                  background: "var(--f1-red)",
                }}
              />
            </div>
            <div>
              <h1 style={{ fontSize: "1.35rem", fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
                Pit Wall Strategy Intelligence System
              </h1>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "1px" }}>
                Formula 1 Monte Carlo race decision-support and telemetry analytics platform
              </p>
            </div>
          </div>

          <StatusBadge type="historical" label="SETUP PHASE" size="sm" />
        </div>

        {errorMsg && (
          <div
            style={{
              background: "var(--color-danger-bg)",
              border: "1px solid var(--color-danger-border)",
              borderRadius: "var(--radius-md)",
              padding: "10px 14px",
              color: "var(--color-danger)",
              fontSize: "0.82rem",
              marginBottom: "20px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <span>⚠</span>
            <span>{errorMsg}</span>
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Race Selection */}
          <RaceSelector
            races={races}
            selectedRace={selectedRaceFile}
            onSelectRace={handleSelectRace}
            isLoading={loadingRaces}
          />

          {/* Ego Driver Selection */}
          <DriverSelector
            drivers={availableDrivers}
            selectedDriver={selectedDriver}
            onSelectDriver={setSelectedDriver}
          />

          {/* Risk Profile Selection */}
          <RiskProfileSelector
            selectedProfile={selectedProfile}
            onSelectProfile={setSelectedProfile}
          />

          {/* Simulation Config */}
          <SimulationConfig
            rollouts={rollouts}
            horizonLaps={horizonLaps}
            onChangeRollouts={setRollouts}
            onChangeHorizon={setHorizonLaps}
          />

          {/* Launch Button */}
          <button
            type="button"
            onClick={handleLaunch}
            disabled={isLaunching || !selectedRaceFile}
            className="btn-primary"
            style={{
              width: "100%",
              padding: "14px",
              fontSize: "0.95rem",
              fontWeight: 800,
              letterSpacing: "0.06em",
              borderRadius: "var(--radius-md)",
              marginTop: "4px",
            }}
          >
            {isLaunching ? (
              <>
                <span style={{ animation: "spin 1s linear infinite" }}>⚙</span>
                <span>INITIALISING PIT WALL TELEMETRY...</span>
              </>
            ) : (
              <>
                <span>▶</span>
                <span>LAUNCH STRATEGY COMMAND DECK</span>
              </>
            )}
          </button>
        </div>
      </div>
    </main>
  );
}
