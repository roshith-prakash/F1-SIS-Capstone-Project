import { API_BASE_URL } from "./constants";
import type { RaceAvailableItem, SetupRequest, SetupResponse, SandboxResult, HistoryLap } from "@/types/race";

export async function fetchAvailableRaces(): Promise<RaceAvailableItem[]> {
  const res = await fetch(`${API_BASE_URL}/api/race/available`);
  if (!res.ok) {
    throw new Error(`Failed to fetch races: ${res.statusText}`);
  }
  return res.json();
}

export async function setupRaceSession(req: SetupRequest): Promise<SetupResponse> {
  const res = await fetch(`${API_BASE_URL}/api/race/setup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Setup failed");
  }
  return res.json();
}

export async function advanceRaceLap(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/race/advance`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to advance lap");
  }
  return res.json();
}

export async function fetchRaceHistory(): Promise<HistoryLap[]> {
  const res = await fetch(`${API_BASE_URL}/api/race/history`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history: ${res.statusText}`);
  }
  return res.json();
}

export async function simulateSandboxScenario(
  action: string,
  compound: string,
  stay_out_laps: number = 3
): Promise<SandboxResult> {
  const res = await fetch(`${API_BASE_URL}/api/sandbox/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, compound, stay_out_laps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Sandbox simulation failed");
  }
  return res.json();
}

export async function setRaceOverride(
  action: string,
  compound?: string,
  strategy_id?: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/race/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, compound, strategy_id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to set override");
  }
  return res.json();
}

export async function resetRaceSession(): Promise<void> {
  await fetch(`${API_BASE_URL}/api/race/reset`, { method: "POST" });
}
