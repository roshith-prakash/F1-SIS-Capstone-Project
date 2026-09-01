"""
Model Evaluation Module for F1 Lap Time Prediction.

This module provides comprehensive evaluation and diagnostics for the F1 lap time prediction model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error, median_absolute_error
from sklearn.model_selection import learning_curve, GroupKFold
import scipy.stats as stats
import shap

# Set plotting style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1c1c1c", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})


def compute_metrics(y_true, y_pred, dataset_name='Dataset') -> dict:
    """Compute and print comprehensive metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Max Error': max_err,
        'Median AE': med_ae
    }
    
    print(f"--- {dataset_name} Metrics ---")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("-" * 30)
    
    return metrics


def per_circuit_evaluation(y_true, y_pred, circuits, dataset_name='Test') -> pd.DataFrame:
    """Compute metrics per circuit."""
    results = []
    
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'circuit': circuits})
    
    for circuit in df['circuit'].unique():
        sub_df = df[df['circuit'] == circuit]
        rmse = np.sqrt(mean_squared_error(sub_df['y_true'], sub_df['y_pred']))
        mae = mean_absolute_error(sub_df['y_true'], sub_df['y_pred'])
        r2 = r2_score(sub_df['y_true'], sub_df['y_pred'])
        
        results.append({
            'Circuit': circuit,
            'Count': len(sub_df),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
    results_df = pd.DataFrame(results).sort_values('RMSE', ascending=True).reset_index(drop=True)
    
    print(f"\n--- Per-Circuit Evaluation ({dataset_name}) ---")
    print(results_df.to_string())
    
    return results_df


def check_overfitting(train_metrics: dict, val_metrics: dict, test_metrics: dict) -> None:
    """Compare train, val, and test metrics to diagnose overfitting/generalization."""
    print("\n--- Diagnostic Report ---")
    
    train_r2 = train_metrics['R2']
    val_r2 = val_metrics['R2']
    test_r2 = test_metrics['R2']
    
    train_rmse = train_metrics['RMSE']
    val_rmse = val_metrics['RMSE']
    test_rmse = test_metrics['RMSE']
    
    overfitting = False
    generalization = False
    
    if train_r2 - val_r2 > 0.02:
        print(f"WARNING: Overfitting detected! Train R2 ({train_r2:.4f}) is much higher than Val R2 ({val_r2:.4f}).")
        overfitting = True
        
    if val_r2 - test_r2 > 0.02:
        print(f"WARNING: Generalization issue! Val R2 ({val_r2:.4f}) is much higher than Test R2 ({test_r2:.4f}).")
        generalization = True
        
    if val_rmse - train_rmse > 0.05:
        print(f"WARNING: Noticeable RMSE gap. Train RMSE ({train_rmse:.4f}), Val RMSE ({val_rmse:.4f}).")
        
    if not overfitting and not generalization:
        print("PASS: No significant overfitting or generalization issues detected based on R2.")


def plot_residuals(y_true, y_pred, save_path=None) -> None:
    """Create a 2x2 diagnostic plot of residuals."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, color='#1f77b4')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Lap Time')
    axes[0, 0].set_ylabel('Predicted Lap Time')
    axes[0, 0].set_title('Predicted vs Actual')
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, color='#ff7f0e')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Lap Time')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    
    # 3. Residual Histogram
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#2ca02c')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_title('Residual Distribution')
    
    # 4. QQ Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_color('#9467bd')
    axes[1, 1].set_title('QQ Plot of Residuals')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residuals plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None) -> None:
    """Plot feature importance using SHAP or built-in model importance."""
    importances = None
    
    # Check if we can use SHAP directly
    # or fallback to model's feature_importances_
    try:
        # Check if stacked ensemble
        if hasattr(model, 'models') and 'lgb' in model.models:
            lgb_model = model.models['lgb']
            importances = lgb_model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        if importances is not None:
            # Simple feature importance bar plot
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='#1f77b4', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
            return
            
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")


def plot_learning_curves(model, X, y, groups=None, cv=5, save_path=None) -> None:
    """Plot learning curves for the model."""
    if groups is not None:
        cv = GroupKFold(n_splits=cv)
        
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, groups=groups, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))
    train_scores_std = np.sqrt(np.std(train_scores, axis=1))
    test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))
    test_scores_std = np.sqrt(np.std(test_scores, axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curves (RMSE)")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="#1f77b4")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#ff7f0e")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#1f77b4", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#ff7f0e", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def compare_models(old_metrics: dict, new_metrics: dict) -> None:
    """Print side-by-side comparison of old vs new model metrics."""
    print("\n--- Model Comparison ---")
    print(f"{'Metric':<12} | {'Old Model':<12} | {'New Model':<12} | {'Improvement':<12}")
    print("-" * 55)
    
    for metric in ['RMSE', 'MAE', 'R2']:
        if metric in old_metrics and metric in new_metrics:
            old_v = old_metrics[metric]
            new_v = new_metrics[metric]
            
            if metric in ['RMSE', 'MAE']:
                imp = ((old_v - new_v) / old_v) * 100
                imp_str = f"{imp:+.2f}%"
            else:
                imp = new_v - old_v
                imp_str = f"{imp:+.4f}"
                
            print(f"{metric:<12} | {old_v:<12.4f} | {new_v:<12.4f} | {imp_str:<12}")


def generate_evaluation_report(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                               feature_names, circuits_test, save_dir='models/ensemble_model/evaluation') -> dict:
    """Run all evaluations, generate plots, and return a summary."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, y_train_pred, "Train")
    val_metrics = compute_metrics(y_val, y_val_pred, "Validation")
    test_metrics = compute_metrics(y_test, y_test_pred, "Test")
    
    check_overfitting(train_metrics, val_metrics, test_metrics)
    
    print("\nEvaluating per circuit...")
    per_circuit_df = per_circuit_evaluation(y_test, y_test_pred, circuits_test)
    per_circuit_df.to_csv(os.path.join(save_dir, 'per_circuit_metrics.csv'), index=False)
    
    print("\nPlotting residuals...")
    plot_residuals(y_test, y_test_pred, save_path=os.path.join(save_dir, 'residuals.png'))
    
    print("Plotting feature importance...")
    plot_feature_importance(model, feature_names, save_path=os.path.join(save_dir, 'feature_importance.png'))
    
    report = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'per_circuit': per_circuit_df
    }
    
    # Compare with dummy old model metrics for example
    old_metrics = {'RMSE': 0.66, 'MAE': 0.42, 'R2': 0.926}
    compare_models(old_metrics, test_metrics)
    
    return report


if __name__ == '__main__':
    print("This module provides evaluation utilities for the F1 model.")
    print("Import it and use `generate_evaluation_report` for full diagnostics.")
    # Example logic to load and test can be placed here.
