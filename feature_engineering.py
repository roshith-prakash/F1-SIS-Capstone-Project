import os
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    TargetEncoder = None

class FeatureEngineer:
    """
    Feature engineering pipeline for F1 lap time prediction.
    """
    def __init__(self, include_telemetry: bool = False):
        """
        Initialize the FeatureEngineer.
        
        Args:
            include_telemetry (bool): Whether to include telemetry features (SpeedI1, SpeedI2, etc.).
                                      Defaults to False as telemetry data is measured during the lap and can leak.
        """
        self.include_telemetry = include_telemetry
        self.target_encoder = None
        self.feature_names = []
        
        # Fallback for manual target encoding if sklearn.preprocessing.TargetEncoder is unavailable
        self.circuit_means = {}
        self.global_mean = 0.0
        
        self.telemetry_medians = {}

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Fits encoders and returns the feature matrix X, target y, and feature names.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            tuple: (X, y, feature_names)
        """
        df_proc = df.copy()
        y = df_proc['LapTime'].values
        
        # 0. Handle NaN in core numeric features BEFORE derived features
        numeric_impute_cols = ['TyreLife', 'stint_number', 'fuel_load', 'race_progress_pct', 
                               'compound_encoded', 'is_fresh_tyre', 'LapNumber', 'Year']
        self.impute_medians = {}
        for col in numeric_impute_cols:
            if col in df_proc.columns and df_proc[col].isna().any():
                median_val = df_proc[col].median()
                self.impute_medians[col] = median_val
                df_proc[col] = df_proc[col].fillna(median_val)
        
        # 1. Circuit features (Target Encoding)
        if TargetEncoder is not None:
            self.target_encoder = TargetEncoder(smooth='auto')
            circuit_enc = self.target_encoder.fit_transform(df_proc[['Circuit']], y)
            df_proc['circuit_target_enc'] = circuit_enc
        else:
            # Manual K-fold target encoding fallback
            self.global_mean = np.mean(y)
            df_proc['circuit_target_enc'] = np.nan
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(df_proc):
                X_tr, y_tr = df_proc.iloc[train_idx], y[train_idx]
                X_va = df_proc.iloc[val_idx]
                
                means = pd.Series(y_tr).groupby(X_tr['Circuit'].values).mean()
                df_proc.loc[val_idx, 'circuit_target_enc'] = X_va['Circuit'].map(means)
                
            df_proc['circuit_target_enc'].fillna(self.global_mean, inplace=True)
            self.circuit_means = pd.Series(y).groupby(df_proc['Circuit']).mean().to_dict()

        # 2. Tyre features
        df_proc['tyre_life_sq'] = df_proc['TyreLife'] ** 2
        df_proc['tyre_life_x_compound'] = df_proc['TyreLife'] * df_proc['compound_encoded']
        
        # 3. Fuel features
        df_proc['fuel_load_sq'] = df_proc['fuel_load'] ** 2

        # Base features list
        self.feature_names = [
            'circuit_target_enc',
            'TyreLife', 'tyre_life_sq', 'compound_encoded', 'stint_number', 'is_fresh_tyre', 'tyre_life_x_compound',
            'fuel_load', 'fuel_load_sq',
            'LapNumber', 'race_progress_pct', 'Year'
        ]
        
        # 5. Speed features
        if self.telemetry_cols_exist(df_proc) and self.include_telemetry:
            telemetry_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
            for col in telemetry_cols:
                if col in df_proc.columns:
                    median_val = df_proc[col].median()
                    self.telemetry_medians[col] = median_val
                    df_proc[col] = df_proc[col].fillna(median_val)
                    self.feature_names.append(col)
        
        # Final NaN safety check — fill any remaining NaN with 0
        X = df_proc[self.feature_names].values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data using fitted encoders.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            np.ndarray: Feature matrix.
        """
        df_proc = df.copy()
        
        # 0. Impute NaN in numeric features using saved medians
        if hasattr(self, 'impute_medians'):
            for col, median_val in self.impute_medians.items():
                if col in df_proc.columns and df_proc[col].isna().any():
                    df_proc[col] = df_proc[col].fillna(median_val)
        
        # 1. Circuit features (Target Encoding)
        if self.target_encoder is not None:
            circuit_enc = self.target_encoder.transform(df_proc[['Circuit']])
            df_proc['circuit_target_enc'] = circuit_enc
        else:
            df_proc['circuit_target_enc'] = df_proc['Circuit'].map(self.circuit_means).fillna(self.global_mean)
            
        # 2. Tyre features
        df_proc['tyre_life_sq'] = df_proc['TyreLife'] ** 2
        df_proc['tyre_life_x_compound'] = df_proc['TyreLife'] * df_proc.get('compound_encoded', 0)
        
        # 3. Fuel features
        df_proc['fuel_load_sq'] = df_proc['fuel_load'] ** 2
        
        # 5. Speed features
        if self.include_telemetry:
            for col, median_val in self.telemetry_medians.items():
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].fillna(median_val)
                else:
                    df_proc[col] = median_val

        # Ensure all columns exist
        for col in self.feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        # Final NaN safety check
        X = df_proc[self.feature_names].values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        return X

    def save(self, path: str):
        """Saves fitted state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Loads fitted state."""
        return joblib.load(path)

    def get_feature_names(self) -> list[str]:
        """Returns ordered list of feature names."""
        return self.feature_names
        
    def telemetry_cols_exist(self, df: pd.DataFrame) -> bool:
        cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        return any(c in df.columns for c in cols)

def prepare_train_test_split(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, test_season: int = 2025, val_fraction: float = 0.2):
    """
    Splits data into train, validation, and test sets.
    Test set is entirely from the `test_season`.
    The remainder is split into train and validation using GroupKFold.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        df (pd.DataFrame): Original dataframe for season and circuit info.
        test_season (int): Year to hold out as test set.
        val_fraction (float): Not strictly used if GroupKFold with 5 splits sets a 0.2 fraction, but kept for signature.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, group_kfold_splits)
    """
    is_test = df['Year'] == test_season
    
    X_test, y_test = X[is_test], y[is_test]
    
    X_rem = X[~is_test]
    y_rem = y[~is_test]
    df_rem = df[~is_test].reset_index(drop=True)
    
    # Group by (Year, Circuit)
    groups = df_rem['Year'].astype(str) + "_" + df_rem['Circuit'].astype(str)
    
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_rem, y_rem, groups))
    
    # Take the first split for train/val
    train_idx, val_idx = splits[0]
    
    X_train, y_train = X_rem[train_idx], y_rem[train_idx]
    X_val, y_val = X_rem[val_idx], y_rem[val_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, splits

if __name__ == "__main__":
    dataset_path = "datasets/f1_consolidated_data.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please generate it first.")
    else:
        print("Loading consolidated data...")
        df = pd.read_csv(dataset_path)
        
        fe = FeatureEngineer(include_telemetry=False)
        X, y, feature_names = fe.fit_transform(df)
        
        X_train, X_val, X_test, y_train, y_val, y_test, splits = prepare_train_test_split(X, y, df, test_season=2025)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature names ({len(feature_names)}): {feature_names}")
        print(f"Train size: {X_train.shape[0]}")
        print(f"Val size:   {X_val.shape[0]}")
        print(f"Test size:  {X_test.shape[0]}")
        
        model_dir = "models/ensemble_model"
        os.makedirs(model_dir, exist_ok=True)
        pipeline_path = os.path.join(model_dir, "feature_pipeline.pkl")
        fe.save(pipeline_path)
        print(f"Feature engineering pipeline saved to {pipeline_path}")
