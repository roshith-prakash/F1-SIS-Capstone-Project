"""
src/overtake/features.py
========================
Feature definitions, circuit metadata, and feature engineering for the Overtake Model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Overtake Model"

# Feature set definitions for ablation study and production
FEATURES_MODEL_A: list[str] = [
    "gap_seconds",
    "gap_lt_1s",
    "gap_lt_0_5s",
    "is_sc",
    "is_vsc",
    "is_sc_restart",
]

FEATURES_MODEL_B: list[str] = FEATURES_MODEL_A + [
    "pace_delta",
    "rolling_3lap_pace_delta",
    "tyre_age_delta",
    "compound_delta",
    "is_fresh_tyre_behind",
]

FEATURES_MODEL_C: list[str] = FEATURES_MODEL_B + [
    "circuit_overtake_difficulty",
    "circuit_drs_zones",
    "pit_straight_length_m",
    "predicted_lap_time_behind",
    "predicted_lap_time_ahead",
    "predicted_pace_delta",
    "predicted_deg_behind",
    "predicted_deg_ahead",
    "predicted_deg_delta",
]

COMPOUND_ORDINAL: dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5,
}

# Static circuit properties (DRS zones & pit straight length in meters)
CIRCUIT_PROPERTIES: dict[str, dict[str, Any]] = {
    "Abu_Dhabi": {"drs_zones": 2, "pit_straight_length_m": 1200.0},
    "Australia": {"drs_zones": 4, "pit_straight_length_m": 800.0},
    "Austria": {"drs_zones": 3, "pit_straight_length_m": 800.0},
    "Azerbaijan": {"drs_zones": 2, "pit_straight_length_m": 2200.0},
    "Bahrain": {"drs_zones": 3, "pit_straight_length_m": 1090.0},
    "Belgium": {"drs_zones": 2, "pit_straight_length_m": 750.0},
    "Brazil": {"drs_zones": 2, "pit_straight_length_m": 650.0},
    "Canada": {"drs_zones": 2, "pit_straight_length_m": 1064.0},
    "China": {"drs_zones": 2, "pit_straight_length_m": 1170.0},
    "Emilia_Romagna": {"drs_zones": 1, "pit_straight_length_m": 550.0},
    "France": {"drs_zones": 2, "pit_straight_length_m": 900.0},
    "Great_Britain": {"drs_zones": 2, "pit_straight_length_m": 770.0},
    "Hungary": {"drs_zones": 2, "pit_straight_length_m": 790.0},
    "Italy": {"drs_zones": 2, "pit_straight_length_m": 1120.0},
    "Japan": {"drs_zones": 1, "pit_straight_length_m": 900.0},
    "Las_Vegas": {"drs_zones": 2, "pit_straight_length_m": 1900.0},
    "Mexico": {"drs_zones": 3, "pit_straight_length_m": 1200.0},
    "Miami": {"drs_zones": 3, "pit_straight_length_m": 1280.0},
    "Monaco": {"drs_zones": 1, "pit_straight_length_m": 510.0},
    "Netherlands": {"drs_zones": 2, "pit_straight_length_m": 680.0},
    "Qatar": {"drs_zones": 1, "pit_straight_length_m": 1068.0},
    "Saudi_Arabia": {"drs_zones": 3, "pit_straight_length_m": 1000.0},
    "Singapore": {"drs_zones": 3, "pit_straight_length_m": 500.0},
    "Spain": {"drs_zones": 2, "pit_straight_length_m": 1047.0},
    "United_States": {"drs_zones": 2, "pit_straight_length_m": 1000.0},
}

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
    "autodromo hermanos rodriguez": "Mexico",
    "mexico": "Mexico",
    "mexico city": "Mexico",
    "mexican": "Mexico",
    "mexican grand prix": "Mexico",
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
    "jeddah cornice": "Saudi_Arabia",
    "jeddah corniche": "Saudi_Arabia",
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
    "austin": "United_States",
    "cota": "United_States",
    "circuit of the americas": "United_States",
    "united states": "United_States",
    "united_states": "United_States",
    "united states grand prix": "United_States",
    "usa": "United_States",
    "paul ricard": "France",
    "france": "France",
    "french": "France",
    "french grand prix": "France",
}

DEFAULT_CIRCUIT_DIFFICULTY: float = 0.85  # Default ~15% baseline overtake success rate


def resolve_circuit_name(circuit: str | None) -> str:
    """Normalize input circuit name or location to canonical circuit key."""
    if not circuit:
        return "Bahrain"
    norm = str(circuit).strip().lower().replace("-", " ").replace("_", " ")
    if norm in CIRCUIT_ALIASES:
        return CIRCUIT_ALIASES[norm]
    # Check underscore variant
    norm_under = norm.replace(" ", "_")
    if norm_under in CIRCUIT_PROPERTIES:
        return norm_under
    # Substring search
    for alias, canon in CIRCUIT_ALIASES.items():
        if alias in norm or norm in alias:
            return canon
    return "Bahrain"


def load_circuit_overtake_difficulty(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, float]:
    """
    Load pre-computed circuit overtake difficulty scores from disk.
    If the file does not exist, return default dictionary.
    """
    path = Path(model_dir) / "circuit_overtake_difficulty.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Baseline defaults
    return {c: DEFAULT_CIRCUIT_DIFFICULTY for c in CIRCUIT_PROPERTIES}


def compute_overtake_features(
    gap_seconds: float,
    pace_delta: float,
    tyre_age_delta: float,
    compound_behind: str,
    compound_ahead: str,
    is_fresh_tyre_behind: bool | int,
    circuit: str,
    is_sc_restart: bool | int = False,
    is_sc: bool | int = False,
    is_vsc: bool | int = False,
    rolling_3lap_pace_delta: float | None = None,
    predicted_lap_time_behind: float | None = None,
    predicted_lap_time_ahead: float | None = None,
    predicted_deg_behind: float | None = None,
    predicted_deg_ahead: float | None = None,
    circuit_difficulty_map: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Construct the full feature dictionary (Model C) from live or replay battle state.
    """
    canon_circuit = resolve_circuit_name(circuit)
    c_props = CIRCUIT_PROPERTIES.get(canon_circuit, {"drs_zones": 2, "pit_straight_length_m": 900.0})
    
    diff_map = circuit_difficulty_map or {}
    difficulty = float(diff_map.get(canon_circuit, DEFAULT_CIRCUIT_DIFFICULTY))

    comp_b_ord = COMPOUND_ORDINAL.get(str(compound_behind).strip().upper(), 2)
    comp_a_ord = COMPOUND_ORDINAL.get(str(compound_ahead).strip().upper(), 2)
    compound_delta = float(comp_a_ord - comp_b_ord)

    r3_delta = pace_delta if rolling_3lap_pace_delta is None else rolling_3lap_pace_delta
    
    pred_lt_b = 85.0 if predicted_lap_time_behind is None else float(predicted_lap_time_behind)
    pred_lt_a = (pred_lt_b + pace_delta) if predicted_lap_time_ahead is None else float(predicted_lap_time_ahead)
    pred_pace_delta = float(pred_lt_a - pred_lt_b)

    pred_deg_b = 0.05 if predicted_deg_behind is None else float(predicted_deg_behind)
    pred_deg_a = 0.05 if predicted_deg_ahead is None else float(predicted_deg_ahead)
    pred_deg_delta = float(pred_deg_a - pred_deg_b)

    return {
        "gap_seconds": float(gap_seconds),
        "gap_lt_1s": 1.0 if gap_seconds < 1.0 else 0.0,
        "gap_lt_0_5s": 1.0 if gap_seconds < 0.5 else 0.0,
        "is_sc": 1.0 if is_sc else 0.0,
        "is_vsc": 1.0 if is_vsc else 0.0,
        "is_sc_restart": 1.0 if is_sc_restart else 0.0,
        "pace_delta": float(pace_delta),
        "rolling_3lap_pace_delta": float(r3_delta),
        "tyre_age_delta": float(tyre_age_delta),
        "compound_delta": compound_delta,
        "is_fresh_tyre_behind": 1.0 if is_fresh_tyre_behind else 0.0,
        "circuit_overtake_difficulty": difficulty,
        "circuit_drs_zones": float(c_props["drs_zones"]),
        "pit_straight_length_m": float(c_props["pit_straight_length_m"]),
        "predicted_lap_time_behind": pred_lt_b,
        "predicted_lap_time_ahead": pred_lt_a,
        "predicted_pace_delta": pred_pace_delta,
        "predicted_deg_behind": pred_deg_b,
        "predicted_deg_ahead": pred_deg_a,
        "predicted_deg_delta": pred_deg_delta,
    }
