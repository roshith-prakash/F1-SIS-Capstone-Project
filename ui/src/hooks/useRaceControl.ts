"use client";

import { useEffect } from "react";
import { useRaceSocket } from "./useRaceSocket";
import { useRaceStore } from "@/store/raceStore";

export function useRaceControl() {
  const { advance, isConnected } = useRaceSocket();
  const { isCalculating, isRaceComplete, currentLap, totalLaps } = useRaceStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input or select
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (e.code === "Space" || e.code === "ArrowRight") {
        e.preventDefault();
        if (!isCalculating && !isRaceComplete) {
          advance();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [advance, isCalculating, isRaceComplete]);

  return {
    advance,
    isCalculating,
    isRaceComplete,
    currentLap,
    totalLaps,
    isConnected,
  };
}
