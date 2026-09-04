import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

RACE_LAPS: dict[str, int] = {
    "Australia":    58,
    "Italy":        53,
    "Hungary":      70,
    "Saudi_Arabia": 50,
}

ALLOWED_RACES   = list(RACE_LAPS.keys())
ALLOWED_DRIVERS = ["ALB", "LEC", "NOR", "RUS", "VER"]
COMPOUNDS       = ["HARD", "MEDIUM", "SOFT"]

PITSTOP_TIME: dict[str, float] = {  # pit loss time (seconds) per circuit
    "Australia":    22.0,
    "Italy":        21.0,
    "Hungary":      22.0,
    "Saudi_Arabia": 19.0,
}
DEG_THRESHOLD:    float = 95.0   # % degradation that triggers a pit stop
TOP_N_STRATEGIES: int   = 3      # number of top strategies printed in summary

# Paths to model artefacts — place laptime_model.pkl and laptime_metadata.pkl
# in the model subfolder.
MODEL_DIR          = Path(__file__).parent
MODEL_SUBDIR       = MODEL_DIR / "models" / "Lap Time Estimation"
if not MODEL_SUBDIR.exists():
    MODEL_SUBDIR   = MODEL_DIR / "models"

LAPTIME_MODEL_PATH = MODEL_SUBDIR / "laptime_model.pkl"
LAPTIME_META_PATH  = MODEL_SUBDIR / "laptime_metadata.pkl"

# Optional tyre degradation model (built by teammate; empirical wear model used if absent)
TYREDEG_MODEL_PATH = MODEL_SUBDIR / "tyre_deg_model.pkl"
TYREDEG_META_PATH  = MODEL_SUBDIR / "tyre_deg_metadata.pkl"


# ---------------------------------------------------------------------------
# StrategySimulator
# ---------------------------------------------------------------------------

class StrategySimulator:
    """
    Simulates F1 race strategies for one driver at one track.

    Parameters
    ----------
    driver : str
        Three-letter driver code, e.g. "VER".
    race : str
        Race name, one of: Australia, Italy, Hungary, Saudi_Arabia.
    year : int
        Season year used as a model feature (e.g. 2023, 2024, 2025).
    """

    def __init__(self, driver: str, race: str, year: int) -> None:
        self._validate_inputs(driver, race, year)

        self.driver = driver
        self.race   = race
        self.year   = year

        # Populated after run_race()
        self._top_strategies: list[dict]    = []
        self.best_strategy_laps: pd.DataFrame = pd.DataFrame()

        self._load_models()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, driver: str, race: str, year: int) -> None:
        if driver not in ALLOWED_DRIVERS:
            raise ValueError(
                f"Driver '{driver}' not recognised. Allowed: {ALLOWED_DRIVERS}"
            )
        if race not in ALLOWED_RACES:
            raise ValueError(
                f"Race '{race}' not recognised. Allowed: {ALLOWED_RACES}"
            )
        if not isinstance(year, int) or year < 2022:
            raise ValueError("Year must be an integer >= 2022.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load the core lap time prediction model and metadata."""
        print("Loading lap time model...")
        self.lt_model = joblib.load(LAPTIME_MODEL_PATH)
        self.lt_meta  = joblib.load(LAPTIME_META_PATH)

        # Cache feature column order from metadata
        self.lt_feature_cols = self.lt_meta["feature_cols"]

        # Pre-encode driver and track — constant for the whole simulation run
        self._driver_enc    = int(self.lt_meta["le_driver"].transform([self.driver])[0])
        self._lt_track_enc  = int(self.lt_meta["le_track"].transform([self.race])[0])

        # Optional tyre degradation model (built by teammate; empirical wear model used if absent)
        self.td_model = None
        self.td_meta = None
        if TYREDEG_MODEL_PATH.exists() and TYREDEG_META_PATH.exists():
            try:
                self.td_model = joblib.load(TYREDEG_MODEL_PATH)
                self.td_meta  = joblib.load(TYREDEG_META_PATH)
                self.td_feature_cols = self.td_meta["feature_cols"]
                self._td_track_enc  = int(self.td_meta["le_track"].transform([self.race])[0])
            except Exception:
                self.td_model = None
                self.td_meta = None

        print("Lap time prediction model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    def generate_strategies(self) -> list[list[str]]:
        """
        Return all valid 2-stint and 3-stint compound sequences.

        Rules
        -----
        - At least 2 different compounds used across the race.
        - Maximum 3 stints (≤ 2 pit stops).
        - No two consecutive identical compounds.
        """
        strategies: list[list[str]] = []

        # 2-stint (1 pit stop) — all ordered pairs of distinct compounds
        for a, b in itertools.permutations(COMPOUNDS, 2):
            strategies.append([a, b])

        # 3-stint (2 pit stops) — repeated compounds allowed (e.g. SOFT->HARD->SOFT)
        # Identical compounds can also be used back-to-back as long as >=2 types used in total
        for combo in itertools.product(COMPOUNDS, repeat=3):
            if len(set(combo)) < 2:   # must use ≥ 2 different types overall
                continue
            strategies.append(list(combo))

        return strategies

    # ------------------------------------------------------------------
    # Per-lap ML predictions
    # ------------------------------------------------------------------

    def _predict_laptime(
        self,
        lap_number: int,
        tyre_life: int,
        compound: str,
    ) -> float:
        """Return predicted lap time (seconds) for a single lap."""
        row = {
            "Driver_encoded":  self._driver_enc,
            "Track_encoded":   self._lt_track_enc,
            "LapNumber":       lap_number,
            "TyreLife":        tyre_life,
            "Year":            self.year,
            "Compound_HARD":   int(compound == "HARD"),
            "Compound_MEDIUM": int(compound == "MEDIUM"),
            "Compound_SOFT":   int(compound == "SOFT"),
        }
        
        X = pd.DataFrame([row])[self.lt_feature_cols]
        return float(self.lt_model.predict(X)[0])

    def _predict_tyre_deg(
        self,
        lap_number: int,
        tyre_life: int,
        initial_life: int,
        compound: str,
    ) -> float:
        """Return tyre degradation (%) for the current lap."""
        if self.td_model is not None and self.td_meta is not None:
            comp_enc    = int(self.td_meta["le_comp"].transform([compound])[0])
            stint_usage = tyre_life - initial_life

            row = {
                "TyreLife":     tyre_life,
                "Initial_Life": initial_life,
                "Stint_Usage":  stint_usage,
                "LapNumber":    lap_number,
                "Compound_Enc": comp_enc,
                "Track_Enc":    self._td_track_enc,
                "Year":         self.year,
            }
            X = pd.DataFrame([row])[self.td_feature_cols]
            return float(self.td_model.predict(X)[0])
        else:
            # Physics-based empirical degradation based on tyre compound life limits
            # Reference maximum stint life (laps): Soft ~22, Medium ~32, Hard ~45
            compound_max_life = {"SOFT": 22.0, "MEDIUM": 32.0, "HARD": 45.0}
            max_life = compound_max_life.get(compound.upper(), 30.0)
            return min(100.0, ((tyre_life - initial_life) / max_life) * 100.0)

    # ------------------------------------------------------------------
    # Single strategy simulation
    # ------------------------------------------------------------------

    def simulate_strategy(
        self,
        strategy: list[str],
        record_laps: bool = False,
    ) -> dict | None:
        """
        Simulate one full race distance with the given compound sequence.

        Parameters
        ----------
        strategy : list[str]
            Ordered list of compounds, e.g. ["SOFT", "HARD"].
        record_laps : bool
            When True, collect a per-lap data record and include it in the
            returned dict as 'laps_df'. Used for the best-strategy table.

        Returns
        -------
        dict
            Keys: strategy, total_time, n_pitstops, [laps_df if record_laps]
        None
            Returned when compounds run out before the race ends (invalid).
        """
        total_laps = RACE_LAPS[self.race]
        total_time = 0.0
        n_pitstops = 0

        lap_number   = 1
        tyre_life    = 1   # 1-indexed: fresh tyres start at life = 1
        initial_life = 0   # tyre_life value at the start of the current stint
        stint_index  = 0   # which entry in the strategy list we are running
        stint_number = 1   # 1-based stint counter (increments on each pit stop)

        current_compound = strategy[stint_index]

        lap_records: list[dict] = []   # only populated when record_laps=True

        while lap_number <= total_laps:
            # -- 1. Predict lap time -----------------------------------
            lap_time    = self._predict_laptime(lap_number, tyre_life, current_compound)
            total_time += lap_time

            # -- 2. Predict tyre degradation ---------------------------
            deg = self._predict_tyre_deg(
                lap_number, tyre_life, initial_life, current_compound
            )

            # -- 3. Record this lap ------------------------------------
            if record_laps:
                lap_records.append(
                    {
                        "Lap":      lap_number,
                        "LapTime":  round(lap_time, 3),
                        "Compound": current_compound,
                        "TyreDeg":  round(deg, 2),
                        "Stint":    stint_number,
                    }
                )

            # -- 4. Pit stop check -------------------------------------
            # Pit only when: degradation threshold is crossed, laps remain,
            # AND there is a next compound to switch to.
            # If already on the last compound and deg crosses the threshold,
            # push on to the flag — do NOT discard the strategy.
            on_last_compound = (stint_index + 1 >= len(strategy))

            if deg >= DEG_THRESHOLD and lap_number < total_laps and not on_last_compound:
                stint_index      += 1
                stint_number     += 1
                current_compound  = strategy[stint_index]
                tyre_life         = 0   # will become 1 after increment below
                initial_life      = 0
                total_time       += PITSTOP_TIME[self.race]
                n_pitstops       += 1

            lap_number += 1
            tyre_life  += 1

        result: dict = {
            "strategy":   strategy,
            "total_time": total_time,
            "n_pitstops": n_pitstops,
        }

        if record_laps:
            result["laps_df"] = pd.DataFrame(lap_records)

        return result

    # ------------------------------------------------------------------
    # Race optimisation
    # ------------------------------------------------------------------

    def run_race(self) -> None:
        """
        Evaluate all valid strategies for the selected race.

        After this call:
          - self._top_strategies    : ranked list of top-N strategy summary dicts
          - self.best_strategy_laps : DataFrame with per-lap detail of the winner
        """
        strategies  = self.generate_strategies()
        all_results: list[dict] = []

        print(
            f"Simulating {len(strategies)} strategies for "
            f"{self.race} ({self.year}) - driver: {self.driver}..."
        )

        for strategy in strategies:
            outcome = self.simulate_strategy(strategy, record_laps=False)
            if outcome is not None:
                all_results.append(outcome)

        if not all_results:
            print("No valid strategies found for this race.")
            return

        # Sort ascending by total race time
        all_results.sort(key=lambda r: r["total_time"])
        self._top_strategies = all_results[:TOP_N_STRATEGIES]

        # Re-simulate the top 3 winners with full lap-by-lap recording
        for res in self._top_strategies:
            detailed = self.simulate_strategy(res["strategy"], record_laps=True)
            res["laps_df"] = detailed["laps_df"]

        # Keep a convenient reference to the absolute best
        self.best_strategy_laps = self._top_strategies[0]["laps_df"]

        print(f"Done - {len(all_results)} valid strategies evaluated.\n")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_results(self) -> None:
        """Print the strategy leaderboard and the best-strategy lap table."""
        if not self._top_strategies:
            print("No results yet. Call run_race() first.")
            return

        best = self._top_strategies[0]

        # -- Strategy leaderboard -------------------------------------
        print("=" * 62)
        print("         F1 RACE STRATEGY OPTIMISATION RESULTS")
        print("=" * 62)
        print(
            f"  Driver : {self.driver}    "
            f"Race : {self.race}    "
            f"Year : {self.year}"
        )
        print("=" * 62)

        for rank, result in enumerate(self._top_strategies, start=1):
            label        = " [BEST]" if rank == 1 else f"  #{rank} "
            compound_str = " -> ".join(result["strategy"])
            total_time   = result["total_time"]
            mins, secs   = divmod(total_time, 60)
            print(
                f"  {label}  {compound_str:<26}"
                f"  {int(mins)}m {secs:05.2f}s  ({total_time:.2f} s)"
                f"  |  Pit Stops: {result['n_pitstops']}"
            )

        # -- Per-lap detail for the best strategy ---------------------
        print(f"\n{'-' * 62}")
        print(
            f"  Best Strategy Lap Table : "
            f"{' -> '.join(best['strategy'])}"
            f"  ({best['n_pitstops']} pit stop(s))"
        )
        print(f"{'-' * 62}")

        # Format numeric columns for display
        df_display = self.best_strategy_laps.copy()
        df_display["LapTime"] = df_display["LapTime"].map(lambda t: f"{t:.3f}s")
        df_display["TyreDeg"] = df_display["TyreDeg"].map(lambda d: f"{d:.2f}%")
        print(df_display.to_string(index=False))
        print(f"{'=' * 62}\n")

    def get_actual_strategy(self) -> dict | None:
        """
        Fetch the actual pit strategy and total time for the driver from local datasets.
        
        Returns
        -------
        dict or None
            Keys: strategy (list), total_time (float), total_time_str (str).
            None if data file or driver not found.
        """
        # Check fastf1 laps: data_fastf1_v1/laps/{year}/{GP_Name}_Grand_Prix.csv
        race_file_names = {
            "Australia": "Australian_Grand_Prix.csv",
            "Italy": "Italian_Grand_Prix.csv",
            "Hungary": "Hungarian_Grand_Prix.csv",
            "Saudi_Arabia": "Saudi_Arabian_Grand_Prix.csv",
        }
        gp_filename = race_file_names.get(self.race, f"{self.race}_Grand_Prix.csv")
        data_file = MODEL_DIR / "data_fastf1_v1" / "laps" / str(self.year) / gp_filename
        
        if not data_file.exists():
            data_file = MODEL_DIR / "datasets" / self.race / f"{self.year}_{self.race}_Laps.csv"
        
        if not data_file.exists():
            print(f"Historical data file not found for {self.race} {self.year}")
            return None
            
        try:
            df = pd.read_csv(data_file)
            
            # Filter for specific driver
            driver_laps = df[df["Driver"] == self.driver].sort_values("LapNumber").copy()
            
            # 0. Filter out wet/intermediate compounds as requested
            driver_laps = driver_laps[driver_laps["Compound"].isin(COMPOUNDS)].copy()
            
            if driver_laps.empty:
                print(f"No dry compound data found for driver {self.driver} in {data_file}")
                return None

            # 1. Extract Strategy Sequence
            # We look at the first lap of each stint to identify the compound sequence
            strategy_laps = driver_laps.groupby("Stint")["Compound"].first().tolist()
            
            # 2. Calculate Total Race Time
            driver_laps["LapTime_dt"] = pd.to_timedelta(driver_laps["LapTime"], errors="coerce")
            total_time_td = driver_laps["LapTime_dt"].sum()
            total_seconds = total_time_td.total_seconds()
            
            if total_seconds == 0:
                return None
                
            mins, secs = divmod(total_seconds, 60)
            
            return {
                "strategy": " -> ".join(strategy_laps),
                "total_time": total_seconds,
                "total_time_str": f"{int(mins)}m {secs:05.2f}s",
                "n_pitstops": len(strategy_laps) - 1
            }
            
        except Exception as e:
            print(f"Error reading historical data: {e}")
            return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- CONFIGURE YOUR RUN HERE ------------------------------------------
    DRIVER = "VER"
    RACE   = "Italy"          # Australia | Italy | Hungary | Saudi_Arabia
    YEAR   = 2024
    # ---------------------------------------------------------------------

    simulator = StrategySimulator(driver=DRIVER, race=RACE, year=YEAR)
    simulator.run_race()
    simulator.get_results()

    # Test actual strategy fetch
    actual = simulator.get_actual_strategy()
    if actual:
        print("\n" + "=" * 62)
        print("         ACTUAL HISTORICAL RACE RESULTS")
        print("=" * 62)
        print(f"  Strategy      : {actual['strategy']}")
        print(f"  Total Time    : {actual['total_time_str']}")
        print(f"  Pit Stops     : {actual['n_pitstops']}")
        print("=" * 62)

        # -- Comparison -------------------------------------------------------
        if simulator._top_strategies:
            predicted_seconds = simulator._top_strategies[0]["total_time"]
            actual_seconds    = actual["total_time"]
            diff_seconds      = abs(predicted_seconds - actual_seconds)
            error_pct         = (diff_seconds / actual_seconds) * 100

            print(f"\n  [COMPARISON]")
            print(f"  Predicted Total : {predicted_seconds:.2f} s")
            print(f"  Actual Total    : {actual_seconds:.2f} s")
            print(f"  Difference      : {diff_seconds:.2f} s")
            print(f"  Error %         : {error_pct:.4f} %")
            print("=" * 62 + "\n")
