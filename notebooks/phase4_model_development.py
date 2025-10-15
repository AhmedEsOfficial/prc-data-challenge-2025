#!/usr/bin/env python3
"""
Phase 4: Model Development - Aircraft Fuel Consumption Prediction
==================================================================

This script implements comprehensive model training, evaluation, and comparison
for predicting aircraft fuel consumption from trajectory features.

Models Implemented:
1. Baseline: Ridge Regression (linear model)
2. Baseline: Random Forest (tree ensemble)
3. Primary: LightGBM (gradient boosting)
4. Alternative: CatBoost (gradient boosting)

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime

# Model libraries
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Evaluation metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VIZ_DIR = BASE_DIR / "visualizations"

# Create directories if needed
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
#  SECTION 1: DATA LOADING
# ============================================================================

def load_data():
    """Load preprocessed training and validation datasets."""
    print("=" * 80)
    print("PHASE 4: MODEL DEVELOPMENT")
    print("=" * 80)
    print("\n[1/7] Loading preprocessed data...")

    start_time = time.time()

    # Load training data
    X_train = pd.read_parquet(CACHE_DIR / "X_train.parquet")
    y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
    train_ids = pd.read_parquet(CACHE_DIR / "train_ids.parquet")

    # Load validation data
    X_val = pd.read_parquet(CACHE_DIR / "X_val.parquet")
    y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']
    val_ids = pd.read_parquet(CACHE_DIR / "val_ids.parquet")

    load_time = time.time() - start_time

    print(f"✓ Data loaded in {load_time:.2f} seconds")
    print(f"\nDataset Summary:")
    print(f"  Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
    print(f"\nTarget Statistics:")
    print(f"  Train - Mean: {y_train.mean():.2f} kg, Std: {y_train.std():.2f} kg")
    print(f"  Val   - Mean: {y_val.mean():.2f} kg, Std: {y_val.std():.2f} kg")
    print(f"  Range: {y_train.min():.2f} - {y_train.max():.2f} kg")

    return X_train, y_train, X_val, y_val, train_ids, val_ids


# ============================================================================
#  SECTION 2: MODEL TRAINING FUNCTIONS
# ============================================================================

def train_ridge_baseline(X_train, y_train, X_val, y_val):
    """Train Ridge Regression baseline model."""
    print("\n" + "=" * 80)
    print("MODEL 1: Ridge Regression (Linear Baseline)")
    print("=" * 80)

    start_time = time.time()

    # Train with cross-validated alpha
    print("\nTraining Ridge Regression with alpha=1.0...")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    metrics = evaluate_model(y_train, y_train_pred, y_val, y_val_pred)

    train_time = time.time() - start_time
    metrics['train_time'] = train_time
    metrics['model_name'] = 'Ridge Regression'

    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print_metrics(metrics)

    return model, metrics, y_val_pred


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest baseline model."""
    print("\n" + "=" * 80)
    print("MODEL 2: Random Forest (Tree Ensemble Baseline)")
    print("=" * 80)

    start_time = time.time()

    print("\nTraining Random Forest...")
    print("  Parameters:")
    print("    - n_estimators: 100")
    print("    - max_depth: 15")
    print("    - min_samples_split: 20")
    print("    - n_jobs: -1 (all cores)")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    metrics = evaluate_model(y_train, y_train_pred, y_val, y_val_pred)

    train_time = time.time() - start_time
    metrics['train_time'] = train_time
    metrics['model_name'] = 'Random Forest'
    metrics['n_estimators'] = 100

    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print_metrics(metrics)

    return model, metrics, y_val_pred


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model (primary model)."""
    print("\n" + "=" * 80)
    print("MODEL 3: LightGBM (Primary Model)")
    print("=" * 80)

    start_time = time.time()

    print("\nPreparing LightGBM datasets...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Parameters optimized for laptop training
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }

    print("\nTraining LightGBM with parameters:")
    for key, value in params.items():
        if key not in ['verbose', 'n_jobs']:
            print(f"    - {key}: {value}")

    print("\nTraining progress:")
    print("-" * 80)

    # Train with early stopping
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50, verbose=True)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    print("-" * 80)

    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # Metrics
    metrics = evaluate_model(y_train, y_train_pred, y_val, y_val_pred)

    train_time = time.time() - start_time
    metrics['train_time'] = train_time
    metrics['model_name'] = 'LightGBM'
    metrics['best_iteration'] = model.best_iteration
    metrics['num_boost_round'] = 2000

    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Best iteration: {model.best_iteration}")
    print_metrics(metrics)

    return model, metrics, y_val_pred


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model (alternative model)."""
    print("\n" + "=" * 80)
    print("MODEL 4: CatBoost (Alternative Model)")
    print("=" * 80)

    start_time = time.time()

    print("\nTraining CatBoost...")
    print("  Parameters:")
    print("    - iterations: 2000")
    print("    - learning_rate: 0.05")
    print("    - depth: 6")
    print("    - l2_leaf_reg: 3.0")
    print("    - early_stopping_rounds: 50")

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=50,
        verbose=50,
        task_type='CPU',
        random_seed=RANDOM_STATE,
        thread_count=-1
    )

    print("\nTraining progress:")
    print("-" * 80)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=50
    )

    print("-" * 80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    metrics = evaluate_model(y_train, y_train_pred, y_val, y_val_pred)

    train_time = time.time() - start_time
    metrics['train_time'] = train_time
    metrics['model_name'] = 'CatBoost'
    metrics['best_iteration'] = model.get_best_iteration()

    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Best iteration: {model.get_best_iteration()}")
    print_metrics(metrics)

    return model, metrics, y_val_pred


# ============================================================================
#  SECTION 3: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(y_train, y_train_pred, y_val, y_val_pred):
    """Calculate comprehensive evaluation metrics."""

    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100

    # Validation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100

    # Overfitting indicator
    overfit_ratio = train_rmse / val_rmse

    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'overfit_ratio': overfit_ratio
    }

    return metrics


def print_metrics(metrics):
    """Pretty print evaluation metrics."""
    print("\nPerformance Metrics:")
    print("  " + "=" * 76)
    print(f"  {'Metric':<20} {'Train':>15} {'Validation':>15} {'Ratio':>15}")
    print("  " + "-" * 76)
    print(f"  {'RMSE (kg)':<20} {metrics['train_rmse']:>15.2f} {metrics['val_rmse']:>15.2f} {metrics['overfit_ratio']:>15.3f}")
    print(f"  {'MAE (kg)':<20} {metrics['train_mae']:>15.2f} {metrics['val_mae']:>15.2f} {metrics['train_mae']/metrics['val_mae']:>15.3f}")
    print(f"  {'R² Score':<20} {metrics['train_r2']:>15.4f} {metrics['val_r2']:>15.4f} {'':>15}")
    print(f"  {'MAPE (%)':<20} {metrics['train_mape']:>15.2f} {metrics['val_mape']:>15.2f} {'':>15}")
    print("  " + "=" * 76)

    # Interpretation
    if metrics['val_rmse'] < 400:
        print("  ✓ TARGET ACHIEVED: Validation RMSE < 400 kg")
    elif metrics['val_rmse'] < 500:
        print("  ⚠ CLOSE TO TARGET: Validation RMSE < 500 kg")
    else:
        print("  ✗ BELOW TARGET: Further optimization needed")

    if metrics['overfit_ratio'] > 1.1:
        print("  ⚠ WARNING: Significant overfitting detected (train/val ratio > 1.1)")
    elif metrics['overfit_ratio'] < 0.95:
        print("  ⚠ WARNING: Underfitting detected (train/val ratio < 0.95)")
    else:
        print("  ✓ Good generalization (train/val ratio healthy)")


# ============================================================================
#  SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_plots(models_dict, y_val):
    """Create comprehensive comparison visualizations."""
    print("\n[6/7] Creating visualizations...")

    # 1. Model comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(models_dict.keys())

    # RMSE comparison
    train_rmse = [models_dict[name]['metrics']['train_rmse'] for name in model_names]
    val_rmse = [models_dict[name]['metrics']['val_rmse'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x + width/2, val_rmse, width, label='Validation', alpha=0.8)
    axes[0, 0].set_xlabel('Model', fontsize=12)
    axes[0, 0].set_ylabel('RMSE (kg)', fontsize=12)
    axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=400, color='r', linestyle='--', label='Target (400 kg)', alpha=0.5)

    # MAE comparison
    train_mae = [models_dict[name]['metrics']['train_mae'] for name in model_names]
    val_mae = [models_dict[name]['metrics']['val_mae'] for name in model_names]

    axes[0, 1].bar(x - width/2, train_mae, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x + width/2, val_mae, width, label='Validation', alpha=0.8)
    axes[0, 1].set_xlabel('Model', fontsize=12)
    axes[0, 1].set_ylabel('MAE (kg)', fontsize=12)
    axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # R² comparison
    train_r2 = [models_dict[name]['metrics']['train_r2'] for name in model_names]
    val_r2 = [models_dict[name]['metrics']['val_r2'] for name in model_names]

    axes[1, 0].bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x + width/2, val_r2, width, label='Validation', alpha=0.8)
    axes[1, 0].set_xlabel('Model', fontsize=12)
    axes[1, 0].set_ylabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Training time comparison
    train_times = [models_dict[name]['metrics']['train_time'] for name in model_names]

    axes[1, 1].bar(model_names, train_times, alpha=0.8, color='coral')
    axes[1, 1].set_xlabel('Model', fontsize=12)
    axes[1, 1].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase4_model_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase4_model_comparison.png")
    plt.close()

    # 2. Predictions vs Actuals (for best model)
    best_model_name = min(models_dict.keys(),
                          key=lambda x: models_dict[x]['metrics']['val_rmse'])
    best_predictions = models_dict[best_model_name]['val_predictions']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    axes[0].scatter(y_val, best_predictions, alpha=0.3, s=10)
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Fuel (kg)', fontsize=12)
    axes[0].set_ylabel('Predicted Fuel (kg)', fontsize=12)
    axes[0].set_title(f'{best_model_name}: Predictions vs Actuals',
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residuals
    residuals = y_val - best_predictions
    axes[1].scatter(best_predictions, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Fuel (kg)', fontsize=12)
    axes[1].set_ylabel('Residuals (kg)', fontsize=12)
    axes[1].set_title(f'{best_model_name}: Residual Plot',
                     fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase4_best_model_predictions.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase4_best_model_predictions.png")
    plt.close()

    # 3. Residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Residual (kg)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{best_model_name}: Residual Distribution',
                     fontsize=14, fontweight='bold')
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].grid(alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f'{best_model_name}: Q-Q Plot (Normality Check)',
                     fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase4_residual_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase4_residual_analysis.png")
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""

    if model_name == 'LightGBM':
        importance = model.feature_importance(importance_type='gain')
    elif model_name == 'CatBoost':
        importance = model.feature_importances_
    elif model_name == 'Random Forest':
        importance = model.feature_importances_
    else:
        return  # Skip for linear models

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Plot top 20
    fig, ax = plt.subplots(figsize=(12, 10))

    top_20 = importance_df.head(20)
    ax.barh(range(len(top_20)), top_20['importance'], alpha=0.8)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'{model_name}: Top 20 Feature Importance',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    filename = f'phase4_{model_name.lower().replace(" ", "_")}_feature_importance.png'
    plt.savefig(VIZ_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()

    # Save importance dataframe
    csv_filename = f'phase4_{model_name.lower().replace(" ", "_")}_feature_importance.csv'
    importance_df.to_csv(REPORTS_DIR / csv_filename, index=False)
    print(f"  ✓ Saved: {csv_filename}")


# ============================================================================
#  SECTION 5: MODEL SAVING
# ============================================================================

def save_models(models_dict):
    """Save all trained models."""
    print("\n[7/7] Saving models and results...")

    for model_name, model_data in models_dict.items():
        # Save model
        model_filename = f'phase4_{model_name.lower().replace(" ", "_")}_model.pkl'
        with open(MODELS_DIR / model_filename, 'wb') as f:
            pickle.dump(model_data['model'], f)
        print(f"  ✓ Saved: {model_filename}")

        # Save metrics
        metrics_filename = f'phase4_{model_name.lower().replace(" ", "_")}_metrics.json'
        with open(REPORTS_DIR / metrics_filename, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in model_data['metrics'].items()
            }
            json.dump(metrics_serializable, f, indent=2)
        print(f"  ✓ Saved: {metrics_filename}")


def generate_summary_report(models_dict):
    """Generate comprehensive summary report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 4: MODEL DEVELOPMENT - SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall summary
    report_lines.append("=" * 80)
    report_lines.append("1. MODELS TRAINED")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Models: {len(models_dict)}")
    for name in models_dict.keys():
        report_lines.append(f"  - {name}")
    report_lines.append("")

    # Performance comparison
    report_lines.append("=" * 80)
    report_lines.append("2. PERFORMANCE COMPARISON")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'Model':<20} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<12} {'Train Time':<12}")
    report_lines.append("-" * 80)

    for name, data in models_dict.items():
        m = data['metrics']
        report_lines.append(
            f"{name:<20} {m['val_rmse']:<12.2f} {m['val_mae']:<12.2f} "
            f"{m['val_r2']:<12.4f} {m['train_time']:<12.2f}s"
        )
    report_lines.append("")

    # Best model
    best_model = min(models_dict.items(), key=lambda x: x[1]['metrics']['val_rmse'])
    report_lines.append("=" * 80)
    report_lines.append("3. BEST MODEL")
    report_lines.append("=" * 80)
    report_lines.append(f"Model: {best_model[0]}")
    report_lines.append(f"Validation RMSE: {best_model[1]['metrics']['val_rmse']:.2f} kg")
    report_lines.append(f"Validation MAE: {best_model[1]['metrics']['val_mae']:.2f} kg")
    report_lines.append(f"Validation R²: {best_model[1]['metrics']['val_r2']:.4f}")
    report_lines.append(f"Validation MAPE: {best_model[1]['metrics']['val_mape']:.2f}%")
    report_lines.append("")

    if best_model[1]['metrics']['val_rmse'] < 400:
        report_lines.append("✓ TARGET ACHIEVED: Validation RMSE < 400 kg")
    else:
        report_lines.append(f"⚠ Target not achieved. Gap: {best_model[1]['metrics']['val_rmse'] - 400:.2f} kg")
    report_lines.append("")

    # Detailed metrics for each model
    report_lines.append("=" * 80)
    report_lines.append("4. DETAILED METRICS")
    report_lines.append("=" * 80)

    for name, data in models_dict.items():
        report_lines.append("")
        report_lines.append(f"{name}")
        report_lines.append("-" * 40)
        m = data['metrics']
        report_lines.append(f"  Training RMSE:     {m['train_rmse']:.2f} kg")
        report_lines.append(f"  Validation RMSE:   {m['val_rmse']:.2f} kg")
        report_lines.append(f"  Training MAE:      {m['train_mae']:.2f} kg")
        report_lines.append(f"  Validation MAE:    {m['val_mae']:.2f} kg")
        report_lines.append(f"  Training R²:       {m['train_r2']:.4f}")
        report_lines.append(f"  Validation R²:     {m['val_r2']:.4f}")
        report_lines.append(f"  Training MAPE:     {m['train_mape']:.2f}%")
        report_lines.append(f"  Validation MAPE:   {m['val_mape']:.2f}%")
        report_lines.append(f"  Overfit Ratio:     {m['overfit_ratio']:.3f}")
        report_lines.append(f"  Training Time:     {m['train_time']:.2f}s")

        if 'best_iteration' in m:
            report_lines.append(f"  Best Iteration:    {m['best_iteration']}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("5. FILES GENERATED")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Models:")
    for name in models_dict.keys():
        filename = f"phase4_{name.lower().replace(' ', '_')}_model.pkl"
        report_lines.append(f"  - models/{filename}")
    report_lines.append("")
    report_lines.append("Metrics:")
    for name in models_dict.keys():
        filename = f"phase4_{name.lower().replace(' ', '_')}_metrics.json"
        report_lines.append(f"  - reports/{filename}")
    report_lines.append("")
    report_lines.append("Visualizations:")
    report_lines.append("  - visualizations/phase4_model_comparison.png")
    report_lines.append("  - visualizations/phase4_best_model_predictions.png")
    report_lines.append("  - visualizations/phase4_residual_analysis.png")
    report_lines.append("  - visualizations/phase4_*_feature_importance.png")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("PHASE 4 COMPLETE")
    report_lines.append("=" * 80)

    # Save report
    report_text = "\n".join(report_lines)
    with open(REPORTS_DIR / 'phase4_model_training_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✓ Summary report saved: reports/phase4_model_training_summary.txt")


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load data
    X_train, y_train, X_val, y_val, train_ids, val_ids = load_data()

    # Dictionary to store all models and results
    models_dict = {}

    # Train baseline models
    print("\n" + "=" * 80)
    print("[2/7] Training Baseline Models")
    print("=" * 80)

    ridge_model, ridge_metrics, ridge_pred = train_ridge_baseline(
        X_train, y_train, X_val, y_val
    )
    models_dict['Ridge'] = {
        'model': ridge_model,
        'metrics': ridge_metrics,
        'val_predictions': ridge_pred
    }

    rf_model, rf_metrics, rf_pred = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    models_dict['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'val_predictions': rf_pred
    }

    # Train primary model
    print("\n" + "=" * 80)
    print("[3/7] Training Primary Model (LightGBM)")
    print("=" * 80)

    lgb_model, lgb_metrics, lgb_pred = train_lightgbm(
        X_train, y_train, X_val, y_val
    )
    models_dict['LightGBM'] = {
        'model': lgb_model,
        'metrics': lgb_metrics,
        'val_predictions': lgb_pred
    }

    # Train alternative model
    print("\n" + "=" * 80)
    print("[4/7] Training Alternative Model (CatBoost)")
    print("=" * 80)

    cat_model, cat_metrics, cat_pred = train_catboost(
        X_train, y_train, X_val, y_val
    )
    models_dict['CatBoost'] = {
        'model': cat_model,
        'metrics': cat_metrics,
        'val_predictions': cat_pred
    }

    # Feature importance analysis
    print("\n" + "=" * 80)
    print("[5/7] Analyzing Feature Importance")
    print("=" * 80)

    feature_names = X_train.columns.tolist()

    plot_feature_importance(rf_model, feature_names, 'Random Forest')
    plot_feature_importance(lgb_model, feature_names, 'LightGBM')
    plot_feature_importance(cat_model, feature_names, 'CatBoost')

    # Create comparison visualizations
    create_comparison_plots(models_dict, y_val)

    # Save models
    save_models(models_dict)

    # Generate summary report
    generate_summary_report(models_dict)

    print("\n" + "=" * 80)
    print("✓ PHASE 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review phase4_model_training_summary.txt")
    print("  2. Examine visualizations in visualizations/")
    print("  3. Analyze feature importance reports")
    print("  4. Proceed to Phase 5: Optimization (if needed)")
    print("=" * 80)


if __name__ == "__main__":
    main()
