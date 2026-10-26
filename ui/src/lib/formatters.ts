export function formatLapTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || isNaN(seconds) || seconds <= 0) {
    return "--:--.---";
  }
  const mins = Math.floor(seconds / 60);
  const remainingSecs = seconds % 60;
  const secsFormatted = remainingSecs.toFixed(3).padStart(6, "0");
  if (mins > 0) {
    return `${mins}:${secsFormatted}`;
  }
  return secsFormatted;
}

export function formatGap(gap: number | null | undefined): string {
  if (gap === null || gap === undefined || isNaN(gap)) {
    return "--";
  }
  if (gap === 0) {
    return "LEADER";
  }
  return `+${gap.toFixed(3)}s`;
}

export function formatInterval(interval: number | null | undefined): string {
  if (interval === null || interval === undefined || isNaN(interval)) {
    return "--";
  }
  if (interval === 0) {
    return "--";
  }
  return `+${interval.toFixed(2)}s`;
}

export function formatPct(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) {
    return "0%";
  }
  return `${Math.round(val * 100)}%`;
}

export function formatScore(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) {
    return "0.000";
  }
  return val.toFixed(3);
}

export function formatPosition(pos: number | null | undefined): string {
  if (pos === null || pos === undefined || isNaN(pos)) {
    return "P--";
  }
  return `P${pos < 10 && Number.isInteger(pos) ? "" : ""}${pos.toFixed(Number.isInteger(pos) ? 0 : 1)}`;
}
