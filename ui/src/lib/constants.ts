export const TYRE_COMPOUNDS: Record<
  string,
  { label: string; color: string; bg: string; border: string; letter: string }
> = {
  SOFT: { label: "Soft", color: "#DC2626", bg: "#FEF2F2", border: "#F87171", letter: "S" },
  MEDIUM: { label: "Medium", color: "#B45309", bg: "#FFFBEB", border: "#FBBF24", letter: "M" },
  HARD: { label: "Hard", color: "#374151", bg: "#F3F4F6", border: "#9CA3AF", letter: "H" },
  INTERMEDIATE: { label: "Intermediate", color: "#15803D", bg: "#F0FDF4", border: "#4ADE80", letter: "I" },
  INTER: { label: "Intermediate", color: "#15803D", bg: "#F0FDF4", border: "#4ADE80", letter: "I" },
  WET: { label: "Wet", color: "#0284C7", bg: "#F0F9FF", border: "#38BDF8", letter: "W" },
  UNKNOWN: { label: "Unknown", color: "#6B7280", bg: "#F3F4F6", border: "#D1D5DB", letter: "?" },
};

export const TEAM_COLORS: Record<string, string> = {
  "Red Bull Racing": "#1E3D7C",
  "Ferrari": "#E10600",
  "Mercedes": "#00A19B",
  "McLaren": "#FF8000",
  "Aston Martin": "#229971",
  "Alpine": "#0093CC",
  "Williams": "#005AFF",
  "RB": "#1E3D7C",
  "AlphaTauri": "#4E7C95",
  "Haas": "#6C757D",
  "Haas F1 Team": "#6C757D",
  "Kick Sauber": "#22C55E",
  "Alfa Romeo": "#981E32",
  "Unknown": "#9CA3AF",
};

export const RISK_PROFILES = [
  {
    id: "balanced",
    title: "Balanced",
    badge: "STANDARD",
    color: "#2563EB",
    bg: "#EFF6FF",
    border: "#BFDBFE",
    description: "Evenly balanced trade-off between track position, tyre preservation, and safety car readiness.",
    weights: { win: 0.25, podium: 0.25, tyre: 0.25, robustness: 0.25 },
  },
  {
    id: "aggressive",
    title: "Aggressive",
    badge: "ATTACK",
    color: "#DC2626",
    bg: "#FEF2F2",
    border: "#FECACA",
    description: "Maximises win and podium probabilities. Accepts higher tyre degradation and traffic exposure.",
    weights: { win: 0.45, podium: 0.35, tyre: 0.10, robustness: 0.10 },
  },
  {
    id: "conservative",
    title: "Conservative",
    badge: "DEFEND",
    color: "#16A34A",
    bg: "#F0FDF4",
    border: "#BBF7D0",
    description: "Minimises tyre cliff risks and unexpected variance. Prioritises steady, guaranteed points finishes.",
    weights: { win: 0.10, podium: 0.20, tyre: 0.40, robustness: 0.30 },
  },
  {
    id: "opportunistic",
    title: "Opportunistic",
    badge: "CHAOS",
    color: "#D97706",
    bg: "#FFFBEB",
    border: "#FDE68A",
    description: "Positions the car to exploit Safety Cars, VSC periods, and weather volatility with strategic flexibility.",
    weights: { win: 0.30, podium: 0.30, tyre: 0.15, robustness: 0.25 },
  },
  {
    id: "defensive",
    title: "Defensive",
    badge: "COVER",
    color: "#7C3AED",
    bg: "#FAF5FF",
    border: "#E9D5FF",
    description: "Covers off direct rival undercuts and prioritises track position retention over long-run tyre offset.",
    weights: { win: 0.20, podium: 0.30, tyre: 0.20, robustness: 0.30 },
  },
];

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/race";
