"""
Model training script for F1 Lap Time Prediction.
Trains a Stacked Ensemble model (LightGBM + CatBoost + Ridge Regression).
"""

import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is not installed. Please install it using 'pip install lightgbm'.")

try:
    import catboost as cb
except ImportError:
    raise ImportError("CatBoost is not installed. Please install it using 'pip install catboost'.")

from feature_engineering import FeatureEngineer, prepare_train_test_split


class StackedEnsemble:
    """Stacked ensemble model for F1 lap time prediction."""
    
    def __init__(
        self,
        lgbm_params: Optional[Dict[str, Any]] = None,
        catboost_params: Optional[Dict[str, Any]] = None,
        ridge_alpha: float = 1.0,
        meta_alpha: float = 1.0,
        n_folds: int = 5
    ):
        """
        Initialize the stacked ensemble model.
        
        Args:
            lgbm_params: Hyperparameters for LightGBM
            catboost_params: Hyperparameters for CatBoost
            ridge_alpha: Regularization strength for base Ridge model
            meta_alpha: Regularization strength for meta-learner Ridge model
            n_folds: Number of folds for cross-validation
        """
        self.lgbm_params = lgbm_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 1500,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        
        self.catboost_params = catboost_params or {
            'iterations': 1500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'random_seed': 42,
            'verbose': 0,
        }
        
        self.ridge_alpha = ridge_alpha
        self.meta_alpha = meta_alpha
        self.n_folds = n_folds
        
        # Models
        self.lgbm_model = None
        self.catboost_model = None
        self.ridge_model = None
        self.ridge_scaler = None
        self.meta_learner = None
        
        self.feature_names_ = None
        self.metadata_ = {}
        
    def fit(self, X_train, y_train, groups=None, X_val=None, y_val=None, feature_names=None):
        """
        Train the stacked ensemble model.
        
        Args:
            X_train: Training features (numpy array or DataFrame)
            y_train: Training target (numpy array or Series)
            groups: Groups for GroupKFold (e.g., Year_Circuit combination)
            X_val: Validation features (for final models early stopping)
            y_val: Validation target
            feature_names: List of feature names (required if X_train is numpy array)
        """
        print("Starting training of Stacked Ensemble...")
        if hasattr(X_train, 'columns'):
            self.feature_names_ = X_train.columns.tolist()
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        X_train_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        
        # OOF predictions matrix
        oof_preds = np.zeros((len(X_train), 3))
        
        gkf = GroupKFold(n_splits=self.n_folds)
        
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold.")
            
        print(f"Generating out-of-fold predictions using {self.n_folds}-fold GroupKFold...")
        
        # Step 1: Generate out-of-fold predictions
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_np, y_train_np, groups=groups)):
            print(f"--- Fold {fold + 1}/{self.n_folds} ---")
            
            X_tr, y_tr = X_train_np[train_idx], y_train_np[train_idx]
            X_v, y_v = X_train_np[val_idx], y_train_np[val_idx]
            
            # LightGBM
            lgbm = lgb.LGBMRegressor(**self.lgbm_params)
            lgbm.fit(
                X_tr, y_tr,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
            oof_preds[val_idx, 0] = lgbm.predict(X_v)
            
            # CatBoost
            cb_model = cb.CatBoostRegressor(**self.catboost_params)
            cb_model.fit(
                X_tr, y_tr,
                eval_set=(X_v, y_v),
                early_stopping_rounds=100,
                verbose=False
            )
            oof_preds[val_idx, 1] = cb_model.predict(X_v)
            
            # Ridge
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_v_scaled = scaler.transform(X_v)
            
            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx, 2] = ridge.predict(X_v_scaled)
            
            # Fold evaluation
            fold_rmse = np.sqrt(mean_squared_error(y_v, oof_preds[val_idx].mean(axis=1)))
            print(f"Fold {fold + 1} Simple Average RMSE: {fold_rmse:.4f}")
            
        # Step 2: Train Meta-Learner (Ridge)
        print("Training Meta-Learner (Ridge) on OOF predictions...")
        self.meta_learner = Ridge(alpha=self.meta_alpha)
        self.meta_learner.fit(oof_preds, y_train_np)
        
        meta_rmse = np.sqrt(mean_squared_error(y_train_np, self.meta_learner.predict(oof_preds)))
        print(f"Meta-Learner OOF RMSE: {meta_rmse:.4f}")
        
        # Step 3: Retrain final Level-1 models on ALL training data
        print("Retraining Level-1 models on ALL training data...")
        
        eval_set_lgb = None
        callbacks_lgb = None
        eval_set_cb = None
        early_stopping_rounds_cb = None
        
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else np.array(X_val)
            y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
            eval_set_lgb = [(X_val_np, y_val_np)]
            callbacks_lgb = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
            eval_set_cb = (X_val_np, y_val_np)
            early_stopping_rounds_cb = 100
            
        self.lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.fit(
            X_train_np, y_train_np,
            eval_set=eval_set_lgb,
            callbacks=callbacks_lgb
        )
        
        self.catboost_model = cb.CatBoostRegressor(**self.catboost_params)
        self.catboost_model.fit(
            X_train_np, y_train_np,
            eval_set=eval_set_cb,
            early_stopping_rounds=early_stopping_rounds_cb,
            verbose=False
        )
        
        self.ridge_scaler = StandardScaler()
        X_train_scaled = self.ridge_scaler.fit_transform(X_train_np)
        self.ridge_model = Ridge(alpha=self.ridge_alpha)
        self.ridge_model.fit(X_train_scaled, y_train_np)
        
        print("Training completed successfully.")
        
        self.metadata_ = {
            'date': datetime.datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': len(self.feature_names_),
            'meta_oof_rmse': float(meta_rmse)
        }
        
    def predict(self, X) -> np.ndarray:
        """
        Predict lap times using the stacked ensemble.
        
        Args:
            X: Features (numpy array or DataFrame)
            
        Returns:
            Numpy array of predictions
        """
        if self.meta_learner is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        if hasattr(X, 'columns'):
            X_np = X[self.feature_names_].values
        else:
            X_np = np.array(X)
            
        preds = np.zeros((len(X_np), 3))
        
        # Level 1 predictions
        # Handle both LGBMRegressor (after training) and Booster (after loading)
        if isinstance(self.lgbm_model, lgb.LGBMRegressor):
            preds[:, 0] = self.lgbm_model.predict(X_np)
        else:
            preds[:, 0] = self.lgbm_model.predict(X_np)
        preds[:, 1] = self.catboost_model.predict(X_np)
        
        X_scaled = self.ridge_scaler.transform(X_np)
        preds[:, 2] = self.ridge_model.predict(X_scaled)
        
        # Level 2 predictions
        final_preds = self.meta_learner.predict(preds)
        
        return final_preds
        
    def evaluate(self, X, y, dataset_name: str = 'Test') -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features (numpy array or DataFrame)
            y: True targets (numpy array or Series)
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary of metrics
        """
        preds = self.predict(X)
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        rmse = np.sqrt(mean_squared_error(y_np, preds))
        mae = mean_absolute_error(y_np, preds)
        r2 = r2_score(y_np, preds)
        
        # MAPE logic
        # avoid division by zero or negative lap times theoretically
        valid_idx = y_np > 0
        mape = np.mean(np.abs((y_np[valid_idx] - preds[valid_idx]) / y_np[valid_idx])) * 100
        
        print(f"--- Evaluation on {dataset_name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        self.metadata_[f'{dataset_name.lower()}_metrics'] = metrics
        return metrics
        
    def save(self, model_dir: str = 'models/ensemble_model'):
        """
        Save all model artifacts.
        
        Args:
            model_dir: Directory to save the models in
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Level 1 models
        self.lgbm_model.booster_.save_model(os.path.join(model_dir, 'lgbm_model.txt'))
        self.catboost_model.save_model(os.path.join(model_dir, 'catboost_model.cbm'))
        joblib.dump(self.ridge_model, os.path.join(model_dir, 'ridge_model.pkl'))
        joblib.dump(self.ridge_scaler, os.path.join(model_dir, 'ridge_scaler.pkl'))
        
        # Save Level 2 model
        joblib.dump(self.meta_learner, os.path.join(model_dir, 'meta_learner.pkl'))
        
        # Save configuration and metadata
        config = {
            'feature_names': self.feature_names_,
            'hyperparameters': {
                'lgbm_params': self.lgbm_params,
                'catboost_params': self.catboost_params,
                'ridge_alpha': self.ridge_alpha,
                'meta_alpha': self.meta_alpha,
                'n_folds': self.n_folds
            },
            'metadata': self.metadata_
        }
        
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Model saved successfully to {model_dir}")
        
    def load(self, model_dir: str = 'models/ensemble_model'):
        """
        Load all model artifacts.
        
        Args:
            model_dir: Directory containing saved models
        """
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
            
        self.feature_names_ = config['feature_names']
        self.lgbm_params = config['hyperparameters']['lgbm_params']
        self.catboost_params = config['hyperparameters']['catboost_params']
        self.ridge_alpha = config['hyperparameters']['ridge_alpha']
        self.meta_alpha = config['hyperparameters']['meta_alpha']
        self.n_folds = config['hyperparameters']['n_folds']
        self.metadata_ = config['metadata']
        
        # Load Level 1 models
        self.lgbm_model = lgb.Booster(model_file=os.path.join(model_dir, 'lgbm_model.txt'))
        self.catboost_model = cb.CatBoostRegressor().load_model(os.path.join(model_dir, 'catboost_model.cbm'))
        self.ridge_model = joblib.load(os.path.join(model_dir, 'ridge_model.pkl'))
        self.ridge_scaler = joblib.load(os.path.join(model_dir, 'ridge_scaler.pkl'))
        
        # Load Level 2 model
        self.meta_learner = joblib.load(os.path.join(model_dir, 'meta_learner.pkl'))
        
        print(f"Model loaded successfully from {model_dir}")


def main():
    """Main function to train and evaluate the ensemble model."""
    print("Loading consolidated dataset...")
    data_path = 'datasets/f1_consolidated_data.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_collector_v2.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total records across seasons {sorted(df['Year'].unique())}.")
    
    # 1. Strictly separate 2025 as test set BEFORE feature engineering to prevent data leakage
    test_season = 2025
    is_test = df['Year'] == test_season
    df_train_val = df[~is_test].reset_index(drop=True)
    df_test = df[is_test].reset_index(drop=True)
    
    print(f"Held out {len(df_test)} records from season {test_season} as Test set.")
    print(f"Using {len(df_train_val)} records from seasons {sorted(df_train_val['Year'].unique())} for Training/Validation.")
    
    # 2. Fit feature engineering pipeline ONLY on training/validation data
    print("\nApplying feature engineering (fit_transform on Train/Val, transform on Test)...")
    engineer = FeatureEngineer()
    X_tv, y_tv, feature_names = engineer.fit_transform(df_train_val)
    X_test = engineer.transform(df_test)
    y_test = df_test['LapTime'].values
    
    # 3. Create GroupKFold splits on 2022-2024 train/val data grouped by (Year, Circuit)
    groups_tv = (df_train_val['Year'].astype(str) + '_' + df_train_val['Circuit'].astype(str)).values
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_tv, y_tv, groups_tv))
    
    train_idx, val_idx = splits[0]
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]
    groups_train = groups_tv[train_idx]
    
    print(f"Dataset split summary -> Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test (2025): {X_test.shape[0]}")
    
    # 4. Train stacked ensemble
    ensemble = StackedEnsemble()
    print("\nTraining the ensemble model...")
    ensemble.fit(
        X_train, y_train, 
        groups=groups_train, 
        X_val=X_val, y_val=y_val,
        feature_names=feature_names
    )
    
    # 5. Evaluate performance across Train, Validation, and Test sets
    print("\n" + "="*60)
    print("Evaluating models...")
    print("="*60)
    train_metrics = ensemble.evaluate(X_train, y_train, dataset_name='Train')
    val_metrics = ensemble.evaluate(X_val, y_val, dataset_name='Validation')
    test_metrics = ensemble.evaluate(X_test, y_test, dataset_name='Test (2025 Season)')
    
    # Overfitting check
    print("\n--- Overfitting Check ---")
    train_val_gap = train_metrics['r2'] - val_metrics['r2']
    val_test_gap = val_metrics['r2'] - test_metrics['r2']
    print(f"Train-Val R2 gap: {train_val_gap:.4f} {'[WARNING: OVERFITTING]' if train_val_gap > 0.02 else '[OK]'}")
    print(f"Val-Test R2 gap:  {val_test_gap:.4f} {'[WARNING: GENERALIZATION ISSUE]' if val_test_gap > 0.02 else '[OK]'}")
    
    # Compare with old model
    print("\n--- Comparison with Old XGBoost Model ---")
    old_metrics = {'rmse': 0.66, 'mae': 0.42, 'r2': 0.926}
    for metric in ['rmse', 'mae', 'r2']:
        old_val = old_metrics[metric]
        new_val = test_metrics[metric]
        if metric == 'r2':
            improvement = ((new_val - old_val) / old_val) * 100
            better = new_val > old_val
        else:
            improvement = ((old_val - new_val) / old_val) * 100
            better = new_val < old_val
        symbol = '[BETTER]' if better else '[WORSE]'
        print(f"{metric.upper():>5}: Old={old_val:.4f}  New={new_val:.4f}  {symbol} ({improvement:+.1f}%)")
    
    # Save everything
    print("\nSaving the ensemble model...")
    ensemble.save()
    
    # Save feature engineer pipeline
    model_dir = 'models/ensemble_model'
    engineer.save(os.path.join(model_dir, 'feature_pipeline.pkl'))
    print(f"Feature pipeline saved to {model_dir}/feature_pipeline.pkl")
    
    print("\n" + "="*60)
    print("Model Training and Evaluation Complete.")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}s (target: < 0.50s)")
    print(f"Test R2:   {test_metrics['r2']:.4f} (target: > 0.95)")
    print("="*60)


if __name__ == '__main__':
    main()
