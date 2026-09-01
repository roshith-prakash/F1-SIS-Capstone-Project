"""
Clean integration module for F1 Lap Time Predictor.
Wraps the trained ensemble model into a simple prediction API.
"""

import os
import json
import logging
import warnings
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LapTimePredictor:
    """
    Clean wrapper for the F1 Lap Time Prediction ensemble model.
    """
    
    COMPOUND_ENCODING = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    
    def __init__(self, model_dir: str = 'models/ensemble_model'):
        """
        Initialize the predictor by loading the model and pipeline.
        
        Args:
            model_dir (str): Directory containing the saved model files.
        """
        self.model_dir = model_dir
        self.is_loaded = False
        self.ensemble = None
        self.feature_engineer = None
        self.config = {}
        
        self._load_models()
        
    def _load_models(self):
        """Loads models and feature pipeline with fallback."""
        try:
            # Lazy import of required custom classes
            try:
                from model_training import StackedEnsemble
                from feature_engineering import FeatureEngineer
            except ImportError as e:
                logger.error(f"Failed to import required modules: {e}")
                warnings.warn(f"Failed to import training modules. Predictor will not work. Error: {e}")
                return
            
            # Check for config
            config_path = os.path.join(self.model_dir, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file not found at {config_path}")
                
            # Load Feature Engineer
            pipeline_path = os.path.join(self.model_dir, 'feature_pipeline.pkl')
            if os.path.exists(pipeline_path):
                self.feature_engineer = joblib.load(pipeline_path)
            else:
                logger.error(f"Feature pipeline not found at {pipeline_path}")
                return
                
            # Load Ensemble Model
            self.ensemble = StackedEnsemble()
            self.ensemble.load(model_dir=self.model_dir)
                
            self.is_loaded = True
            logger.info("Successfully loaded LapTimePredictor.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            warnings.warn(f"Error loading models: {e}")
            self.is_loaded = False

    def predict(
        self,
        circuit: str,
        compound: str,
        tyre_life: int,
        lap_number: int,
        total_laps: int,
        year: int = 2025,
        stint_number: int = 1,
        is_fresh_tyre: bool = True,
        air_temp: float = 25.0,
        track_temp: float = 35.0,
        **kwargs
    ) -> float:
        """Predict a single lap time."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Cannot make predictions.")
            
        compound_upper = compound.upper()
        if compound_upper not in self.COMPOUND_ENCODING:
            raise ValueError(f"Invalid compound '{compound}'. Must be one of {list(self.COMPOUND_ENCODING.keys())}")
            
        fuel_load = max(0, total_laps - lap_number)
        race_progress_pct = (lap_number / total_laps) * 100 if total_laps > 0 else 0
        compound_encoded = self.COMPOUND_ENCODING[compound_upper]
        
        # Column names MUST match what FeatureEngineer.transform() expects
        feature_dict = {
            'Circuit': circuit,
            'Compound': compound_upper,
            'TyreLife': tyre_life,
            'LapNumber': lap_number,
            'Year': year,
            'stint_number': stint_number,
            'is_fresh_tyre': int(is_fresh_tyre),
            'fuel_load': fuel_load,
            'race_progress_pct': race_progress_pct,
            'compound_encoded': compound_encoded,
        }
        
        # Add any kwargs
        feature_dict.update(kwargs)
        
        df = pd.DataFrame([feature_dict])
        
        # Transform features using the fitted FeatureEngineer
        X = self.feature_engineer.transform(df)
            
        # Predict using the ensemble
        prediction = self.ensemble.predict(X)
            
        # Ensure returning a float
        if isinstance(prediction, (np.ndarray, list, pd.Series)):
            return float(prediction[0])
        return float(prediction)

    def predict_stint(
        self,
        circuit: str,
        compound: str,
        start_lap: int,
        end_lap: int,
        total_laps: int,
        year: int = 2025,
        stint_number: int = 1,
        **kwargs
    ) -> list[float]:
        """Predict all lap times for a given stint."""
        lap_times = []
        tyre_life = 1
        
        for lap in range(start_lap, end_lap + 1):
            is_fresh = (tyre_life == 1)
            
            lap_time = self.predict(
                circuit=circuit,
                compound=compound,
                tyre_life=tyre_life,
                lap_number=lap,
                total_laps=total_laps,
                year=year,
                stint_number=stint_number,
                is_fresh_tyre=is_fresh,
                **kwargs
            )
            lap_times.append(lap_time)
            tyre_life += 1
            
        return lap_times

    def predict_race(
        self,
        circuit: str,
        strategy: List[Tuple[str, int]],
        total_laps: int,
        year: int = 2025,
        pit_stop_time: float = 22.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict full race given a strategy."""
        all_lap_times = []
        strategy_used = []
        pit_stops = len(strategy) - 1
        
        current_lap = 1
        
        for stint_idx, (compound, laps_in_stint) in enumerate(strategy):
            stint_number = stint_idx + 1
            end_lap = current_lap + laps_in_stint - 1
            
            # Cap at total laps
            if end_lap > total_laps:
                end_lap = total_laps
                
            if current_lap > total_laps:
                break
                
            strategy_used.append(compound)
            
            stint_times = self.predict_stint(
                circuit=circuit,
                compound=compound,
                start_lap=current_lap,
                end_lap=end_lap,
                total_laps=total_laps,
                year=year,
                stint_number=stint_number,
                **kwargs
            )
            
            # Add pit stop time to the first lap of subsequent stints
            if stint_idx > 0 and stint_times:
                stint_times[0] += pit_stop_time
                
            all_lap_times.extend(stint_times)
            current_lap = end_lap + 1
            
        total_time = sum(all_lap_times)
        
        return {
            'total_time': float(total_time),
            'lap_times': all_lap_times,
            'strategy_used': strategy_used,
            'pit_stops': pit_stops
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return self.config
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance from the model."""
        if not self.is_loaded:
            return {}
            
        # Try shap first if ensemble supports it
        if hasattr(self.ensemble, 'get_shap_importance'):
            try:
                return self.ensemble.get_shap_importance()
            except Exception:
                pass
                
        # Fallback to built-in
        if hasattr(self.ensemble, 'get_feature_importance'):
            try:
                return self.ensemble.get_feature_importance()
            except Exception:
                pass
                
        logger.warning("Feature importance not supported by the underlying model structure.")
        return {}


def create_predictor_for_simulation(model_dir: str = 'models/ensemble_model') -> Optional[LapTimePredictor]:
    """
    Factory function for the simulation engine.
    
    Args:
        model_dir: Directory where models are stored.
        
    Returns:
        LapTimePredictor instance or None if failed.
    """
    predictor = LapTimePredictor(model_dir=model_dir)
    if predictor.is_loaded:
        return predictor
    
    logger.warning("Failed to create predictor for simulation. Missing files?")
    return None


if __name__ == "__main__":
    # Quick Demo
    print("--- F1 Lap Time Predictor Demo ---")
    
    # 1. Load Predictor
    model_directory = "models/ensemble_model" # Update as needed if models exist elsewhere
    predictor = create_predictor_for_simulation(model_dir=model_directory)
    
    if predictor is None:
        print(f"Predictor could not be loaded from '{model_directory}'.")
        print("Please ensure the models are trained and saved in this directory.")
    else:
        # 2. Predict a single lap
        try:
            lap_time = predictor.predict(
                circuit='Monza', 
                compound='SOFT', 
                tyre_life=5, 
                lap_number=10, 
                total_laps=53,
                air_temp=26.0,
                track_temp=38.0
            )
            print(f"Single Lap Prediction (Monza, SOFT, Lap 10, Tyre 5): {lap_time:.3f} s")
        except Exception as e:
            print(f"Single lap prediction failed: {e}")
            
        # 3. Predict a stint
        try:
            stint_times = predictor.predict_stint(
                circuit='Monza', 
                compound='MEDIUM', 
                start_lap=20, 
                end_lap=25, 
                total_laps=53
            )
            print(f"Stint Predictions (Laps 20-25): {[round(t, 3) for t in stint_times]}")
        except Exception as e:
            print(f"Stint prediction failed: {e}")
            
        # 4. Show model info
        try:
            info = predictor.get_model_info()
            print(f"Model Info: {info}")
        except Exception as e:
            print(f"Get model info failed: {e}")
