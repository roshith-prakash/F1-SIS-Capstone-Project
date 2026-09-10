from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from race_state.models import ParticipantState, RaceState, normalize_driver
except ImportError:
    from src.race_state.models import ParticipantState, RaceState, normalize_driver


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Lap Time Estimation" / "laptime_model.pkl"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "Lap Time Estimation" / "laptime_metadata.pkl"

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
    "brazil": "Brazil",
    "sao paulo grand prix": "Brazil",
    "montreal": "Canada",
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
    "rodriguez": "Mexico",
    "mexico city": "Mexico",
    "mexico": "Mexico",
    "mexico city grand prix": "Mexico",
    "miami": "Miami",
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
    "saudi arabian grand prix": "Saudi_Arabia",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "singapore grand prix": "Singapore",
    "catalunya": "Spain",
    "barcelona": "Spain",
    "spain": "Spain",
    "spanish": "Spain",
    "spanish grand prix": "Spain",
    "cota": "United_States",
    "austin": "United_States",
    "united states": "United_States",
    "united_states": "United_States",
    "united states grand prix": "United_States",
    "usa": "United_States",
}

TEAM_ALIASES: dict[str, str] = {
    "red bull": "Red Bull Racing",
    "red bull racing": "Red Bull Racing",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "mclaren": "McLaren",
    "mclaren f1 team": "McLaren",
    "mercedes": "Mercedes",
    "mercedes amg": "Mercedes",
    "mercedes-amg": "Mercedes",
    "aston martin": "Aston Martin",
    "aston martin f1 team": "Aston Martin",
    "alpine": "Alpine",
    "alpine f1 team": "Alpine",
    "williams": "Williams",
    "williams racing": "Williams",
    "haas": "Haas F1 Team",
    "haas f1 team": "Haas F1 Team",
    "alfa romeo": "Alfa Romeo",
    "alfa romeo racing": "Alfa Romeo",
    "sauber": "Kick Sauber",
    "kick sauber": "Kick Sauber",
    "alphatauri": "AlphaTauri",
    "scuderia alphatauri": "AlphaTauri",
    "rb": "RB",
    "racing bulls": "Racing Bulls",
}


class LapTimeAdapter:
    """
    Adapter bridging ``RaceState`` runtime snapshots to the dry-lap relative pace
    prediction model trained in ``Lap_Time_Prediction.ipynb``.
    """

    def __init__(
        self,
        model: Any | None = None,
        meta: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
        meta_path: str | Path | None = None,
    ) -> None:
        if model is not None and meta is not None:
            self.model = model
            self.meta = meta
        else:
            mod_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            m_path = Path(meta_path) if meta_path else DEFAULT_META_PATH
            if not mod_path.exists():
                # Fallback to local relative path
                mod_path = Path("models") / "Lap Time Estimation" / "laptime_model.pkl"
            if not m_path.exists():
                m_path = Path("models") / "Lap Time Estimation" / "laptime_metadata.pkl"

            if not mod_path.exists():
                raise FileNotFoundError(f"Lap time model not found at {mod_path}")
            if not m_path.exists():
                raise FileNotFoundError(f"Lap time metadata not found at {m_path}")
            self.model = joblib.load(mod_path)
            self.meta = joblib.load(m_path)

        self.circuits_set = set(self.meta["circuits"])
        self.teams_set = set(self.meta["teams"])
        self.dry_compounds = set(self.meta.get("dry_compounds", ["SOFT", "MEDIUM", "HARD"]))
        self.feature_cols = list(self.meta["feature_cols"])
        self.auto_calibrate: bool = True
        self.calibrated_bases: dict[str, float] = {}
        self.implied_history: dict[str, list[float]] = {}

    def resolve_circuit(self, state: RaceState) -> str:
        """Resolve the canonical circuit name from RaceState attributes."""
        candidates = [
            getattr(state, "location", None),
            getattr(state, "country", None),
            getattr(state, "grand_prix", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            cleaned = str(candidate).strip()
            if cleaned in self.circuits_set:
                return cleaned
            low = cleaned.lower()
            if low in CIRCUIT_ALIASES:
                target = CIRCUIT_ALIASES[low]
                if target in self.circuits_set:
                    return target
            clean_under = cleaned.replace(" ", "_")
            if clean_under in self.circuits_set:
                return clean_under
            for circ in self.circuits_set:
                if circ.lower() in low or low in circ.lower():
                    return circ

        # Fallback: first circuit in metadata
        return self.meta["circuits"][0]

    def resolve_team(self, team_name: str | None) -> str:
        """Resolve the canonical constructor team name."""
        if not team_name:
            return "Red Bull Racing"
        cleaned = str(team_name).strip()
        if cleaned in self.teams_set:
            return cleaned
        low = cleaned.lower()
        if low in TEAM_ALIASES:
            target = TEAM_ALIASES[low]
            if target in self.teams_set:
                return target
        for t in self.teams_set:
            if t.lower() in low or low in t.lower():
                return t
        return "Red Bull Racing"

    def calibrate_from_state(self, state: RaceState) -> float | None:
        """
        Calibrate the track dry baseline dynamically from clean green-flag laps
        in the current RaceState snapshot without requiring practice/qualifying data.
        Returns the calibrated base in seconds, or None if not enough samples yet.
        """
        cond = state.current_conditions
        circuit = self.resolve_circuit(state)
        # Do not calibrate during full-course caution periods
        if cond.has_safety_car or cond.has_vsc or cond.has_red_flag:
            return self.calibrated_bases.get(circuit)

        static_base = float(self.meta["circuit_dry_bases"].get(circuit, 85.0))
        history = self.implied_history.setdefault(circuit, [])

        for driver_code, participant in state.participants.items():
            if not participant.is_active or participant.position is None or participant.position > 10:
                continue
            lt = participant.last_lap_time_seconds
            if lt is None or participant.is_pit_in_lap or participant.is_pit_out_lap:
                continue
            # Validate lap time is a realistic racing lap (within 15% of expected baseline)
            if not (static_base * 0.85 <= lt <= static_base * 1.25):
                continue

            feats = self.build_features(state, participant, use_calibrated=False)
            if feats is None:
                continue
            df_row = pd.DataFrame([feats])[self.feature_cols]
            delta = float(self.model.predict(df_row)[0])
            implied_base = lt - delta
            history.append(implied_base)

        if len(history) >= 5:
            # 15th percentile of implied base across clean top-half laps locks onto true dry base
            est_base = float(np.percentile(history, 15))
            self.calibrated_bases[circuit] = est_base
            return est_base

        return self.calibrated_bases.get(circuit)

    def build_features(
        self, state: RaceState, participant: ParticipantState, use_calibrated: bool = True
    ) -> dict[str, Any] | None:
        """
        Build the 15-feature dictionary for a specific participant on the current lap.
        Returns None if compound is not a dry slick (SOFT, MEDIUM, HARD).
        """
        compound = str(participant.compound or "MEDIUM").strip().upper()
        if compound not in self.dry_compounds:
            return None

        circuit = self.resolve_circuit(state)
        team = self.resolve_team(participant.team)

        lap_number = state.current_lap or 1
        total_laps = state.total_laps_expected or self.meta["race_laps"].get(circuit, 57)
        if total_laps is None or total_laps <= 0:
            total_laps = 57

        tl = participant.tyre_life
        tyre_life = float(tl) if tl is not None and not (isinstance(tl, float) and math.isnan(tl)) else 1.0

        pos = participant.position
        position = int(pos) if pos is not None and not (isinstance(pos, float) and math.isnan(pos)) else 10

        st = participant.stint
        stint_number = int(st) if st is not None and not (isinstance(st, float) and math.isnan(st)) else 1

        is_fresh_tyre = int(participant.fresh_tyre if participant.fresh_tyre is not None else (tyre_life <= 1))

        if use_calibrated and circuit in self.calibrated_bases:
            event_dry_base = self.calibrated_bases[circuit]
        else:
            event_dry_base = float(self.meta["circuit_dry_bases"].get(circuit, 85.0))

        progress_ratio = min(1.0, max(0.0, lap_number / total_laps))
        fuel_load = 110.0 * (1.0 - progress_ratio)
        race_progress_pct = progress_ratio * 100.0
        fuel_tyre_interaction = (fuel_load / 110.0) * tyre_life
        tyre_age_squared = tyre_life ** 2

        try:
            track_encoded = int(self.meta["le_track"].transform([circuit])[0])
        except (ValueError, KeyError):
            track_encoded = 0

        try:
            team_encoded = int(self.meta["le_team"].transform([team])[0])
        except (ValueError, KeyError):
            team_encoded = 0

        return {
            "Track_encoded": track_encoded,
            "Team_encoded": team_encoded,
            "LapNumber": lap_number,
            "TyreLife": tyre_life,
            "tyre_age_squared": tyre_age_squared,
            "Compound_HARD": int(compound == "HARD"),
            "Compound_MEDIUM": int(compound == "MEDIUM"),
            "Compound_SOFT": int(compound == "SOFT"),
            "fuel_load": fuel_load,
            "race_progress_pct": race_progress_pct,
            "stint_number": stint_number,
            "is_fresh_tyre": is_fresh_tyre,
            "Position": position,
            "fuel_tyre_interaction": fuel_tyre_interaction,
            "Event_Dry_Base": event_dry_base,
        }

    def predict_lap_time(self, state: RaceState, driver: str) -> float | None:
        """
        Predict expected dry lap time (seconds) for a driver given the current RaceState snapshot.
        Returns None if driver is inactive, unclassified, or running wet/intermediate tyres.
        """
        if self.auto_calibrate:
            self.calibrate_from_state(state)

        driver_code = normalize_driver(driver)
        if driver_code is None:
            return None

        participant = state.participants.get(driver_code)
        if participant is None:
            return None

        features = self.build_features(state, participant)
        if features is None:
            return None

        df_row = pd.DataFrame([features])[self.feature_cols]
        event_dry_base = features["Event_Dry_Base"]
        predicted_delta = float(self.model.predict(df_row)[0])
        return float(event_dry_base + predicted_delta)

    def predict_all(self, state: RaceState) -> dict[str, float]:
        """
        Predict expected dry lap times for all active participants in the current race state.
        Batches predictions for maximum vectorization and performance.
        Returns {driver_code: predicted_lap_time_seconds}.
        """
        if self.auto_calibrate:
            self.calibrate_from_state(state)

        rows: list[dict[str, Any]] = []
        driver_codes: list[str] = []
        bases: list[float] = []

        for driver_code, participant in state.participants.items():
            if not participant.is_active:
                continue
            features = self.build_features(state, participant)
            if features is not None:
                rows.append(features)
                driver_codes.append(driver_code)
                bases.append(features["Event_Dry_Base"])

        if not rows:
            return {}

        df_batch = pd.DataFrame(rows)[self.feature_cols]
        deltas = self.model.predict(df_batch)

        predictions: dict[str, float] = {}
        for driver_code, base, delta in zip(driver_codes, bases, deltas):
            predictions[driver_code] = float(base + delta)

        return predictions
