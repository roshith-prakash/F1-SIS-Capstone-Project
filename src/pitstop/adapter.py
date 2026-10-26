"""
src/pitstop/adapter.py
======================
Production PitstopAdapter interface class.

Serves data-driven pit stop duration estimates (mean and standard deviation)
and stochastic samples for any circuit, team, and caution state during race simulation.
Trained on 2022-2025 FastF1 ground-effect era data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "models" / "Pitstop Model" / "pitstop_params.json"

CIRCUIT_ALIASES: dict[str, str] = {
    "silverstone": "Great_Britain",
    "great britain": "Great_Britain",
    "great_britain": "Great_Britain",
    "british": "Great_Britain",
    "british grand prix": "Great_Britain",
    "monza": "Italy",
    "italian": "Italy",
    "italian grand prix": "Italy",
    "italy": "Italy",
    "imola": "Emilia_Romagna",
    "emilia romagna": "Emilia_Romagna",
    "emilia_romagna": "Emilia_Romagna",
    "emilia romagna grand prix": "Emilia_Romagna",
    "yas island": "Abu_Dhabi",
    "abu dhabi": "Abu_Dhabi",
    "abu_dhabi": "Abu_Dhabi",
    "abu dhabi grand prix": "Abu_Dhabi",
    "red bull ring": "Austria",
    "spielberg": "Austria",
    "austria": "Austria",
    "austrian": "Austria",
    "austrian grand prix": "Austria",
    "albert park": "Australia",
    "melbourne": "Australia",
    "australia": "Australia",
    "australian": "Australia",
    "australian grand prix": "Australia",
    "baku": "Azerbaijan",
    "azerbaijan": "Azerbaijan",
    "azerbaijan grand prix": "Azerbaijan",
    "sakhir": "Bahrain",
    "bahrain": "Bahrain",
    "bahrain grand prix": "Bahrain",
    "spa": "Belgium",
    "spa-francorchamps": "Belgium",
    "belgium": "Belgium",
    "belgian": "Belgium",
    "belgian grand prix": "Belgium",
    "interlagos": "Brazil",
    "sao paulo": "Brazil",
    "são paulo": "Brazil",
    "so paulo": "Brazil",
    "brazil": "Brazil",
    "sao paulo grand prix": "Brazil",
    "são paulo grand prix": "Brazil",
    "so paulo grand prix": "Brazil",
    "montreal": "Canada",
    "montréal": "Canada",
    "montral": "Canada",
    "circuit gilles villeneuve": "Canada",
    "canada": "Canada",
    "canadian": "Canada",
    "canadian grand prix": "Canada",
    "shanghai": "China",
    "china": "China",
    "chinese": "China",
    "chinese grand prix": "China",
    "hungaroring": "Hungary",
    "budapest": "Hungary",
    "hungary": "Hungary",
    "hungarian": "Hungary",
    "hungarian grand prix": "Hungary",
    "suzuka": "Japan",
    "japan": "Japan",
    "japanese": "Japan",
    "japanese grand prix": "Japan",
    "las vegas": "Las_Vegas",
    "las_vegas": "Las_Vegas",
    "las vegas grand prix": "Las_Vegas",
    "mexico city": "Mexico",
    "mexico": "Mexico",
    "mexican": "Mexico",
    "mexico city grand prix": "Mexico",
    "miami": "Miami",
    "miami gardens": "Miami",
    "miami grand prix": "Miami",
    "monaco": "Monaco",
    "monte carlo": "Monaco",
    "monaco grand prix": "Monaco",
    "zandvoort": "Netherlands",
    "netherlands": "Netherlands",
    "dutch": "Netherlands",
    "dutch grand prix": "Netherlands",
    "losail": "Qatar",
    "lusail": "Qatar",
    "qatar": "Qatar",
    "qatar grand prix": "Qatar",
    "jeddah": "Saudi_Arabia",
    "saudi arabia": "Saudi_Arabia",
    "saudi_arabia": "Saudi_Arabia",
    "saudi arabian": "Saudi_Arabia",
    "saudi arabian grand prix": "Saudi_Arabia",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "singapore grand prix": "Singapore",
    "barcelona": "Spain",
    "catalunya": "Spain",
    "circuit de barcelona-catalunya": "Spain",
    "spain": "Spain",
    "spanish": "Spain",
    "spanish grand prix": "Spain",
    "austin": "United_States",
    "cota": "United_States",
    "circuit of the americas": "United_States",
    "united states": "United_States",
    "united_states": "United_States",
    "united states grand prix": "United_States",
    "le castellet": "France",
    "paul ricard": "France",
    "france": "France",
    "french": "France",
    "french grand prix": "France",
}

TEAM_MAPPINGS: dict[str, str] = {
    "red bull racing": "Red Bull Racing",
    "red bull": "Red Bull Racing",
    "red_bull_racing": "Red Bull Racing",
    "red_bull": "Red Bull Racing",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "scuderia_ferrari": "Ferrari",
    "mercedes": "Mercedes",
    "mercedes amg": "Mercedes",
    "mercedes_amg": "Mercedes",
    "mclaren": "McLaren",
    "aston martin": "Aston Martin",
    "aston_martin": "Aston Martin",
    "alpine": "Alpine",
    "alphatauri": "Racing Bulls",
    "rb": "Racing Bulls",
    "racing bulls": "Racing Bulls",
    "racing_bulls": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "williams": "Williams",
    "williams racing": "Williams",
    "williams_racing": "Williams",
    "alfa romeo": "Sauber",
    "alfa_romeo": "Sauber",
    "kick sauber": "Sauber",
    "kick_sauber": "Sauber",
    "sauber": "Sauber",
    "haas": "Haas F1 Team",
    "haas f1 team": "Haas F1 Team",
    "haas_f1_team": "Haas F1 Team",
}


def resolve_circuit(raw: str | None) -> str:
    """Normalize circuit / venue string to canonical circuit key."""
    if not raw:
        return "Unknown"
    norm = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    for k, v in CIRCUIT_ALIASES.items():
        if k in norm:
            return v
    return raw.strip().replace(" ", "_")


def resolve_team(raw: str | None) -> str:
    """Normalize team name string to canonical team key."""
    if not raw:
        return "Unknown"
    norm = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    for k, v in TEAM_MAPPINGS.items():
        if k == norm or k in norm:
            return v
    return str(raw).strip()


class PitstopAdapter:
    """
    Production adapter providing data-driven pit stop duration estimates.
    Integrates directly into RaceScenarioSimulator and StrategyEngine.
    """

    def __init__(
        self,
        params_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.params_path = Path(params_path) if params_path else DEFAULT_PARAMS_PATH
        if params is not None:
            self.params = params
        else:
            self.params = self._load_params(self.params_path)

        self.global_mean: float = float(self.params.get("global_mean", 24.3))
        self.global_std: float = float(self.params.get("global_std", 4.5))
        self.circuit_params: dict[str, dict[str, float]] = self.params.get("circuit_params", {})
        self.team_offsets: dict[str, float] = self.params.get("team_offsets", {})
        self.sc_adjustment: float = float(self.params.get("sc_adjustment", -0.56))
        self.vsc_adjustment: float = float(self.params.get("vsc_adjustment", 0.88))
        self.min_clip: float = float(self.params.get("min_clip", 15.0))
        self.max_clip: float = float(self.params.get("max_clip", 55.0))

        # 2-Component Gaussian Mixture Model (GMM) parameters for clean vs botched pit stops
        self.gmm: dict[str, Any] = self.params.get("gmm", {})
        self.gmm_enabled: bool = bool(self.gmm.get("enabled", False))
        gmm_clean = self.gmm.get("clean", {})
        gmm_botched = self.gmm.get("botched", {})
        self.gmm_clean_weight: float = float(gmm_clean.get("weight", 0.7597))
        self.gmm_clean_mean: float = float(gmm_clean.get("mean_offset", -0.6521))
        self.gmm_clean_std: float = float(gmm_clean.get("std", 0.8096))
        self.gmm_botched_weight: float = float(gmm_botched.get("weight", 0.2403))
        self.gmm_botched_mean: float = float(gmm_botched.get("mean_offset", 2.0620))
        self.gmm_botched_std: float = float(gmm_botched.get("std", 6.0997))
        self.team_botched_probs: dict[str, float] = self.gmm.get("team_botched_probs", {})

        # Cache for stats queries
        self._stats_cache: dict[tuple[str, str, bool, bool, str], tuple[float, float]] = {}

    def _load_params(self, path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # Fallback parameters matching global ground-effect averages
        return {
            "version": "fallback",
            "global_mean": 24.3,
            "global_std": 4.5,
            "circuit_params": {},
            "team_offsets": {},
            "sc_adjustment": -0.5,
            "vsc_adjustment": 0.9,
            "min_clip": 15.0,
            "max_clip": 55.0,
            "gmm": {
                "enabled": False,
            },
        }

    def get_pit_stats(
        self,
        circuit: str | None,
        team: str | None,
        is_sc: bool = False,
        is_vsc: bool = False,
        mode: str = "clean",
    ) -> tuple[float, float]:
        """
        Returns (mean_seconds, std_seconds) for a pit stop.

        Parameters
        ----------
        circuit : str | None
            Circuit name, location, or Grand Prix title.
        team : str | None
            Team or constructor name.
        is_sc : bool
            True if pit stop occurs under full Safety Car.
        is_vsc : bool
            True if pit stop occurs under Virtual Safety Car.
        mode : str
            "clean" (default modal execution), "expected" (unconditional mean),
            "botched" (delayed execution), or "base" (circuit + team baseline).

        Returns
        -------
        tuple[float, float]
            (mean_duration, std_deviation) in seconds.
        """
        canon_circ = resolve_circuit(circuit)
        canon_team = resolve_team(team)
        cache_key = (canon_circ, canon_team, bool(is_sc), bool(is_vsc), str(mode))

        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        c_info = self.circuit_params.get(canon_circ)
        if c_info is not None:
            mean = float(c_info.get("mean", self.global_mean))
            std = float(c_info.get("std", self.global_std))
        else:
            mean = self.global_mean
            std = self.global_std

        team_offset = float(self.team_offsets.get(canon_team, 0.0))
        mean += team_offset

        if is_sc:
            mean += self.sc_adjustment
        elif is_vsc:
            mean += self.vsc_adjustment

        if self.gmm_enabled:
            if mode == "clean":
                mean += self.gmm_clean_mean
                std = self.gmm_clean_std
            elif mode == "botched":
                mean += self.gmm_botched_mean
                std = self.gmm_botched_std
            elif mode == "expected":
                mean += (self.gmm_clean_weight * self.gmm_clean_mean + self.gmm_botched_weight * self.gmm_botched_mean)
            elif mode == "base":
                pass

        res = (float(np.clip(mean, self.min_clip, self.max_clip)), max(0.5, std))
        self._stats_cache[cache_key] = res
        return res

    def predict_mean_duration(
        self,
        circuit: str | None,
        team: str | None,
        is_sc: bool = False,
        is_vsc: bool = False,
        mode: str = "clean",
    ) -> float:
        """Return expected deterministic pit duration (seconds)."""
        mean, _ = self.get_pit_stats(circuit, team, is_sc=is_sc, is_vsc=is_vsc, mode=mode)
        return mean

    def sample_pit_duration(
        self,
        circuit: str | None,
        team: str | None,
        is_sc: bool = False,
        is_vsc: bool = False,
        rng: Any = None,
    ) -> float:
        """
        Sample a single pit stop duration using the 2-component GMM (clean vs botched stop).

        Parameters
        ----------
        circuit : str | None
            Circuit name or Grand Prix title.
        team : str | None
            Team name.
        is_sc : bool
            Full Safety Car flag.
        is_vsc : bool
            Virtual Safety Car flag.
        rng : Any
            Random number generator (numpy RandomState, Generator, or random.Random).

        Returns
        -------
        float
            Simulated pit stop duration in seconds, clipped to [min_clip, max_clip].
        """
        canon_circ = resolve_circuit(circuit)
        canon_team = resolve_team(team)

        base_mean, base_std = self.get_pit_stats(circuit, team, is_sc=is_sc, is_vsc=is_vsc, mode="base")

        if self.gmm_enabled:
            p_botch = float(self.team_botched_probs.get(canon_team, self.gmm_botched_weight))

            # Draw uniform random number to choose clean vs botched component
            if rng is None:
                u = float(np.random.uniform(0.0, 1.0))
            elif hasattr(rng, "uniform"):
                u = float(rng.uniform(0.0, 1.0))
            elif hasattr(rng, "random"):
                u = float(rng.random())
            else:
                u = float(np.random.uniform(0.0, 1.0))

            if u < p_botch:
                comp_offset = self.gmm_botched_mean
                comp_std = self.gmm_botched_std
            else:
                comp_offset = self.gmm_clean_mean
                comp_std = self.gmm_clean_std

            if rng is None:
                noise = float(np.random.normal(comp_offset, comp_std))
            elif hasattr(rng, "normal"):
                noise = float(rng.normal(comp_offset, comp_std))
            elif hasattr(rng, "gauss"):
                noise = float(rng.gauss(comp_offset, comp_std))
            else:
                noise = float(np.random.normal(comp_offset, comp_std))

            sampled = base_mean + noise
            return float(np.clip(sampled, self.min_clip, self.max_clip))

        # Fallback to single Gaussian
        if rng is None:
            noise = float(np.random.normal(0.0, base_std))
        elif hasattr(rng, "normal"):
            noise = float(rng.normal(0.0, base_std))
        elif hasattr(rng, "gauss"):
            noise = float(rng.gauss(0.0, base_std))
        else:
            noise = float(np.random.normal(0.0, base_std))

        sampled = base_mean + noise
        return float(np.clip(sampled, self.min_clip, self.max_clip))
