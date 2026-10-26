import { create } from "zustand";
import type {
  LapCompletePayload,
  SandboxResult,
  HistoryLap,
  SetupRequest,
} from "@/types/race";

interface RaceConfig extends SetupRequest {
  raceName: string;
}

interface RaceStoreState {
  // Setup & Configuration
  config: RaceConfig;
  isConfigured: boolean;

  // Real-time Simulation Status
  isCalculating: boolean;
  calculatingLap: number | null;
  calculationStartTime: number | null;
  lastEvaluationTimeMs: number | null;
  progressMessage: string;
  progressPct: number;
  isRaceComplete: boolean;
  socketStatus: "disconnected" | "connecting" | "connected" | "error";
  error: string | null;

  // Live telemetry & strategy data
  currentLap: number;
  totalLaps: number;
  lapData: LapCompletePayload | null;
  lastAction: string | null;

  // History & Replay
  lapHistory: HistoryLap[];

  // Strategy Sandbox
  sandboxLoading: boolean;
  sandboxResult: SandboxResult | null;

  // User Strategic & Tactical Overrides
  activeOverride: { action: string; compound: string; target_lap: number } | null;
  adoptedStrategyId: string | null;

  // Actions
  setConfig: (config: Partial<RaceConfig>) => void;
  setSessionInfo: (currentLap: number, totalLaps: number, driver?: string) => void;
  setIsConfigured: (val: boolean) => void;
  startLapCalculation: (lap: number, total: number) => void;
  updateProgress: (message: string, pct: number) => void;
  applyLapComplete: (payload: LapCompletePayload) => void;
  setRaceComplete: (totalLaps?: number) => void;
  setSandboxLoading: (loading: boolean) => void;
  setSandboxResult: (res: SandboxResult | null) => void;
  setActiveOverride: (override: { action: string; compound: string; target_lap: number } | null) => void;
  setAdoptedStrategyId: (id: string | null) => void;
  setSocketStatus: (status: "disconnected" | "connecting" | "connected" | "error") => void;
  setError: (err: string | null) => void;
  resetSession: () => void;
}

const initialConfig: RaceConfig = {
  race_file: "",
  driver: "VER",
  profile: "balanced",
  rollouts: 20,
  horizon_laps: 8,
  raceName: "",
};

export const useRaceStore = create<RaceStoreState>((set) => ({
  config: initialConfig,
  isConfigured: false,

  isCalculating: false,
  calculatingLap: null,
  calculationStartTime: null,
  lastEvaluationTimeMs: null,
  progressMessage: "",
  progressPct: 0,
  isRaceComplete: false,
  socketStatus: "disconnected",
  error: null,

  currentLap: 0,
  totalLaps: 0,
  lapData: null,
  lastAction: null,

  lapHistory: [],

  sandboxLoading: false,
  sandboxResult: null,

  activeOverride: null,
  adoptedStrategyId: null,

  setConfig: (cfg) =>
    set((state) => ({ config: { ...state.config, ...cfg } })),

  setSessionInfo: (lap, total, driver) =>
    set((state) => ({
      currentLap: lap,
      totalLaps: total,
      config: driver ? { ...state.config, driver } : state.config,
    })),

  setIsConfigured: (val) => set({ isConfigured: val }),

  startLapCalculation: (lap, total) =>
    set({
      isCalculating: true,
      calculatingLap: lap,
      calculationStartTime: Date.now(),
      progressMessage: `Initializing simulation for Lap ${lap}...`,
      progressPct: 5,
      currentLap: lap,
      totalLaps: total,
    }),

  updateProgress: (message, pct) =>
    set({
      progressMessage: message,
      progressPct: pct,
    }),

  applyLapComplete: (payload) =>
    set((state) => {
      const clientElapsedMs = state.calculationStartTime ? Math.round(Date.now() - state.calculationStartTime) : null;
      const evalMs = payload.evaluation_time_ms ?? clientElapsedMs;

      const ego = payload.participants[state.config.driver] || null;
      const historyItem: HistoryLap = {
        lap: payload.lap,
        recommendation_action: payload.recommendation.action,
        recommendation_score: payload.recommendation.score,
        actual_compound: ego ? ego.compound : "MEDIUM",
        actual_tyre_life: ego ? ego.tyre_life : 1,
        actual_position: ego ? ego.position : 10,
        actual_lap_time: ego ? ego.last_lap_time : null,
        full_data: payload,
      };

      const existingIdx = state.lapHistory.findIndex((h) => h.lap === payload.lap);
      const nextHistory = [...state.lapHistory];
      if (existingIdx >= 0) {
        nextHistory[existingIdx] = historyItem;
      } else {
        nextHistory.push(historyItem);
      }

      return {
        isCalculating: false,
        calculatingLap: null,
        calculationStartTime: null,
        lastEvaluationTimeMs: evalMs,
        progressMessage: "",
        progressPct: 100,
        currentLap: payload.lap,
        totalLaps: payload.total_laps,
        lapData: payload,
        lastAction: payload.recommendation.action,
        isRaceComplete: payload.race_complete,
        activeOverride: payload.active_override ?? null,
        adoptedStrategyId: payload.adopted_strategy_id ?? state.adoptedStrategyId,
        lapHistory: nextHistory,
        error: null,
      };
    }),

  setRaceComplete: (totalLaps) =>
    set((state) => ({
      isRaceComplete: true,
      isCalculating: false,
      totalLaps: totalLaps ?? state.totalLaps,
    })),

  setSandboxLoading: (loading) => set({ sandboxLoading: loading }),

  setSandboxResult: (res) => set({ sandboxResult: res }),

  setActiveOverride: (override) => set({ activeOverride: override }),

  setAdoptedStrategyId: (id) => set({ adoptedStrategyId: id }),

  setSocketStatus: (status) => set({ socketStatus: status }),

  setError: (err) => set({ error: err, isCalculating: false }),

  resetSession: () =>
    set({
      config: initialConfig,
      isConfigured: false,
      isCalculating: false,
      calculatingLap: null,
      calculationStartTime: null,
      lastEvaluationTimeMs: null,
      progressMessage: "",
      progressPct: 0,
      isRaceComplete: false,
      currentLap: 0,
      totalLaps: 0,
      lapData: null,
      lastAction: null,
      lapHistory: [],
      sandboxLoading: false,
      sandboxResult: null,
      activeOverride: null,
      adoptedStrategyId: null,
      error: null,
    }),
}));
