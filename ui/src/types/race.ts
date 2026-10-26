export interface ParticipantPayload {
  driver: string;
  team: string;
  position: number;
  compound: string;
  tyre_life: number;
  pit_count: number;
  last_lap_time: number | null;
  gap_to_leader: number | null;
  interval_ahead: number | null;
  is_pit_in_lap: boolean;
  is_pit_out_lap: boolean;
  pit_probability?: number | null;
  threat_level?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | null;
  likely_action?: string | null;
}

export interface ConditionsPayload {
  has_safety_car: boolean;
  has_vsc: boolean;
  has_red_flag: boolean;
  track_temp: number | null;
  air_temp: number | null;
  rainfall: boolean;
}

export interface StrategyItem {
  id: string;
  name: string;
  score: number;
  rank: number;
  tactical_intent: string;
  expected_position: number;
  p_win: number;
  p_podium: number;
  p_top5: number;
  p_points: number;
  tyre_risk: number;
  traffic_risk: number;
  robustness: number;
  position_distribution: Record<string, number>;
  expected_position_under_sc: number;
  expected_position_green: number;
  sc_position_advantage: number;
  compounds: string[];
  pit_laps: number[];
  is_user_adopted?: boolean;
}

export interface RecommendationPayload {
  action: string;
  target_compound: string | null;
  expected_position: number;
  confidence: number;
  score: number;
  runner_up_name: string;
  runner_up_score: number;
  score_delta: number;
  changed_from_last_lap: boolean;
  explanation: string[];
  counterfactual_strategy: string;
  counterfactual_reasons: string[];
  target_pit_lap?: number | null;
  planned_strategy_name?: string | null;
  planned_compounds?: string[] | null;
  planned_pit_laps?: number[] | null;
  evaluation_time_ms?: number | null;
  user_adopted_strategy_id?: string | null;
  active_override?: { action: string; compound: string; target_lap: number } | null;
}

export interface SCProbabilities {
  p_sc_h1: number;
  p_sc_h3: number;
  p_sc_h5: number;
  p_sc_h10: number;
  p_vsc_h1: number;
  p_vsc_h3: number;
  p_vsc_h5: number;
  p_vsc_h10: number;
}

export interface EventProbabilities {
  rain_5l: number;
  undercut_success: number;
  overtake_this_lap: number;
}

export interface LapCompletePayload {
  lap: number;
  total_laps: number;
  race_complete: boolean;
  conditions: ConditionsPayload;
  participants: Record<string, ParticipantPayload>;
  recommendation: RecommendationPayload;
  strategies: StrategyItem[];
  sc_probabilities: SCProbabilities;
  event_probabilities: EventProbabilities;
  evaluation_time_ms?: number | null;
  active_override?: { action: string; compound: string; target_lap: number } | null;
  adopted_strategy_id?: string | null;
}

export interface OverrideRequest {
  action: "pit_next_lap" | "pit_this_lap" | "stay_out" | "adopt_strategy" | "clear";
  compound?: string;
  strategy_id?: string;
}

export interface SandboxResult {
  action: string;
  compound: string;
  expected_position: number;
  p_win: number;
  p_podium: number;
  score: number;
  vs_ai_score: number;
  vs_ai_delta: number;
  vs_ai_position: number;
  verdict: string;
}

export interface HistoryLap {
  lap: number;
  recommendation_action: string;
  recommendation_score: number;
  actual_compound: string;
  actual_tyre_life: number;
  actual_position: number;
  actual_lap_time: number | null;
  full_data?: LapCompletePayload;
}

export interface RaceAvailableItem {
  year: number;
  name: string;
  file: string;
  drivers: string[];
}

export interface SetupRequest {
  race_file: string;
  driver: string;
  profile: string;
  rollouts: number;
  horizon_laps: number;
}

export interface SetupResponse {
  ok: boolean;
  driver: string;
  total_laps: number;
  race_name: string;
}
