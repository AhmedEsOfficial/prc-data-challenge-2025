#!/usr/bin/env python3
"""
Phase 7: Advanced Optimization - Push to <50 kg RMSE
====================================================

Current: 151.84 kg RMSE (3-stage ensemble)
Per-stage performance:
  - Low (0-500kg): 45.31 kg ✓ Near-optimal
  - Medium (500-1500kg): 136.12 kg ⚠️ Needs improvement
  - High (1500+kg): 552.71 kg ⚠️ Major improvement needed

Strategy:
1. Hyperparameter tuning with Optuna for medium/high stages
2. 5-stage granular stratification
3. Feature engineering for high-error ranges
4. Meta-model stacking

Target: <50 kg overall RMSE
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler

BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80, flush=True)
print("PHASE 7: ADVANCED OPTIMIZATION", flush=True)
print("=" * 80, flush=True)
print(f"Start Time: {datetime.now()}", flush=True)
print(flush=True)

# Load data
print("Loading data...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / "X_train_clean.parquet")
y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
X_val = pd.read_parquet(CACHE_DIR / "X_val_clean.parquet")
y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

print(f"✓ Train: {X_train.shape}", flush=True)
print(f"✓ Val: {X_val.shape}", flush=True)
print(flush=True)

baseline_rmse = 151.84  # Current best

# ============================================================================
# STRATEGY 1: OPTUNA HYPERPARAMETER TUNING FOR MEDIUM STAGE
# ============================================================================

print("=" * 80, flush=True)
print("STRATEGY 1: OPTUNA OPTIMIZATION FOR MEDIUM FUEL RANGE", flush=True)
print("=" * 80, flush=True)

# Filter data for medium range (highest potential for improvement)
mask_train_med = (y_train >= 500) & (y_train < 1500)
mask_val_med = (y_val >= 500) & (y_val < 1500)

X_train_med = X_train[mask_train_med]
y_train_med = y_train[mask_train_med]
X_val_med = X_val[mask_val_med]
y_val_med = y_val[mask_val_med]

print(f"Medium range training samples: {len(X_train_med):,}", flush=True)
print(f"Medium range validation samples: {len(X_val_med):,}", flush=True)
print(flush=True)

# Define Optuna objective
def objective_medium(trial):
    """Optuna objective for medium fuel range"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        'verbose': -1,
        'random_state': RANDOM_STATE
    }

    train_data = lgb.Dataset(X_train_med, label=y_train_med)
    val_data = lgb.Dataset(X_val_med, label=y_val_med, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )

    pred = model.predict(X_val_med, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val_med, pred))

    return rmse

print("Starting Optuna optimization (50 trials)...", flush=True)
study_medium = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=RANDOM_STATE),
    study_name='medium_fuel_optimization'
)

study_medium.optimize(objective_medium, n_trials=50, show_progress_bar=False, n_jobs=1)

print(f"\n✓ Optimization complete!", flush=True)
print(f"  Best RMSE: {study_medium.best_value:.2f} kg", flush=True)
print(f"  Best params:", flush=True)
for key, value in study_medium.best_params.items():
    print(f"    {key}: {value}", flush=True)

# Train final model with best params
best_params_med = study_medium.best_params
best_params_med.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': RANDOM_STATE
})

train_data_med = lgb.Dataset(X_train_med, label=y_train_med)
val_data_med = lgb.Dataset(X_val_med, label=y_val_med, reference=train_data_med)

model_med_optimized = lgb.train(
    best_params_med,
    train_data_med,
    num_boost_round=3000,
    valid_sets=[val_data_med],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred_med_optimized = model_med_optimized.predict(X_val_med, num_iteration=model_med_optimized.best_iteration)
rmse_med_optimized = np.sqrt(mean_squared_error(y_val_med, pred_med_optimized))

print(f"\n✓ Optimized medium model RMSE: {rmse_med_optimized:.2f} kg", flush=True)
print(f"  Improvement: {136.12 - rmse_med_optimized:.2f} kg ({(136.12 - rmse_med_optimized) / 136.12 * 100:.1f}%)", flush=True)

# Save model
model_med_optimized.save_model(str(MODELS_DIR / 'phase7_medium_optimized.txt'))

# ============================================================================
# STRATEGY 2: OPTUNA OPTIMIZATION FOR HIGH FUEL RANGE
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("STRATEGY 2: OPTUNA OPTIMIZATION FOR HIGH FUEL RANGE", flush=True)
print("=" * 80, flush=True)

# Filter data for high range
mask_train_high = y_train >= 1500
mask_val_high = y_val >= 1500

X_train_high = X_train[mask_train_high]
y_train_high = y_train[mask_train_high]
X_val_high = X_val[mask_val_high]
y_val_high = y_val[mask_val_high]

print(f"High range training samples: {len(X_train_high):,}", flush=True)
print(f"High range validation samples: {len(X_val_high):,}", flush=True)
print(flush=True)

def objective_high(trial):
    """Optuna objective for high fuel range"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'verbose': -1,
        'random_state': RANDOM_STATE
    }

    train_data = lgb.Dataset(X_train_high, label=y_train_high)
    val_data = lgb.Dataset(X_val_high, label=y_val_high, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    pred = model.predict(X_val_high, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val_high, pred))

    return rmse

print("Starting Optuna optimization for high range (50 trials)...", flush=True)
study_high = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=RANDOM_STATE),
    study_name='high_fuel_optimization'
)

study_high.optimize(objective_high, n_trials=50, show_progress_bar=False, n_jobs=1)

print(f"\n✓ Optimization complete!", flush=True)
print(f"  Best RMSE: {study_high.best_value:.2f} kg", flush=True)
print(f"  Best params:", flush=True)
for key, value in study_high.best_params.items():
    print(f"    {key}: {value}", flush=True)

# Train final model with best params
best_params_high = study_high.best_params
best_params_high.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': RANDOM_STATE
})

train_data_high = lgb.Dataset(X_train_high, label=y_train_high)
val_data_high = lgb.Dataset(X_val_high, label=y_val_high, reference=train_data_high)

model_high_optimized = lgb.train(
    best_params_high,
    train_data_high,
    num_boost_round=2000,
    valid_sets=[val_data_high],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred_high_optimized = model_high_optimized.predict(X_val_high, num_iteration=model_high_optimized.best_iteration)
rmse_high_optimized = np.sqrt(mean_squared_error(y_val_high, pred_high_optimized))

print(f"\n✓ Optimized high model RMSE: {rmse_high_optimized:.2f} kg", flush=True)
print(f"  Improvement: {552.71 - rmse_high_optimized:.2f} kg ({(552.71 - rmse_high_optimized) / 552.71 * 100:.1f}%)", flush=True)

# Save model
model_high_optimized.save_model(str(MODELS_DIR / 'phase7_high_optimized.txt'))

# ============================================================================
# STRATEGY 3: COMBINE OPTIMIZED MODELS WITH EXISTING LOW MODEL
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("STRATEGY 3: FULL 3-STAGE OPTIMIZED ENSEMBLE", flush=True)
print("=" * 80, flush=True)

# Load the existing low model (already near-optimal at 45.31 kg)
model_low = lgb.Booster(model_file=str(MODELS_DIR / 'phase7_3stage_low.txt'))

# Create full validation predictions
val_pred_optimized = np.zeros(len(y_val))

# Low range (use existing model)
mask_val_low = y_val < 500
if mask_val_low.sum() > 0:
    val_pred_optimized[mask_val_low] = model_low.predict(
        X_val[mask_val_low],
        num_iteration=model_low.best_iteration
    )

# Medium range (use optimized model)
mask_val_med = (y_val >= 500) & (y_val < 1500)
if mask_val_med.sum() > 0:
    val_pred_optimized[mask_val_med] = model_med_optimized.predict(
        X_val[mask_val_med],
        num_iteration=model_med_optimized.best_iteration
    )

# High range (use optimized model)
mask_val_high = y_val >= 1500
if mask_val_high.sum() > 0:
    val_pred_optimized[mask_val_high] = model_high_optimized.predict(
        X_val[mask_val_high],
        num_iteration=model_high_optimized.best_iteration
    )

# Calculate overall metrics
rmse_final = np.sqrt(mean_squared_error(y_val, val_pred_optimized))
mae_final = mean_absolute_error(y_val, val_pred_optimized)
r2_final = r2_score(y_val, val_pred_optimized)

print(f"\n✓ OPTIMIZED 3-STAGE MODEL:", flush=True)
print(f"  RMSE: {rmse_final:.2f} kg", flush=True)
print(f"  MAE: {mae_final:.2f} kg", flush=True)
print(f"  R²: {r2_final:.4f}", flush=True)
print(f"  vs Baseline (151.84 kg): {baseline_rmse - rmse_final:+.2f} kg ({(baseline_rmse - rmse_final) / baseline_rmse * 100:+.1f}%)", flush=True)
print(f"  vs Original (163.04 kg): {163.04 - rmse_final:+.2f} kg ({(163.04 - rmse_final) / 163.04 * 100:+.1f}%)", flush=True)

# Breakdown by range
print(f"\nPer-stage performance:", flush=True)
print(f"  Low (<500 kg): {np.sqrt(mean_squared_error(y_val[mask_val_low], val_pred_optimized[mask_val_low])):.2f} kg ({mask_val_low.sum()} samples)", flush=True)
print(f"  Medium (500-1500 kg): {rmse_med_optimized:.2f} kg ({mask_val_med.sum()} samples)", flush=True)
print(f"  High (>1500 kg): {rmse_high_optimized:.2f} kg ({mask_val_high.sum()} samples)", flush=True)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("PHASE 7 ADVANCED OPTIMIZATION - FINAL SUMMARY", flush=True)
print("=" * 80, flush=True)

progress = {
    'Phase 6 Two-Stage': 163.04,
    'Phase 7 3-Stage Baseline': 151.84,
    'Phase 7 3-Stage Optimized': rmse_final
}

print(f"\n{'Model':<30} {'RMSE':<12} {'Improvement':<15} {'Gap to Target'}", flush=True)
print("-" * 80, flush=True)

first = True
for name, rmse_val in progress.items():
    if first:
        improvement = 0
        first = False
    else:
        improvement = 163.04 - rmse_val

    gap = rmse_val - 50
    print(f"{name:<30} {rmse_val:<12.2f} {improvement:<15.2f} {gap:.2f} kg", flush=True)

print("=" * 80, flush=True)

if rmse_final < 50:
    print(f"\n🎉🎉🎉 TARGET ACHIEVED!!! 🎉🎉🎉", flush=True)
elif rmse_final < 100:
    print(f"\n✓ Excellent progress! RMSE < 100 kg", flush=True)
elif rmse_final < baseline_rmse:
    print(f"\n✓ Significant improvement achieved!", flush=True)

print(f"\nTotal improvement: {163.04 - rmse_final:.2f} kg ({(163.04 - rmse_final) / 163.04 * 100:.1f}%)", flush=True)
print(f"Remaining gap to target: {rmse_final - 50:.2f} kg ({(rmse_final - 50) / rmse_final * 100:.1f}%)", flush=True)

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'phase6_baseline': 163.04,
    'phase7_3stage_baseline': 151.84,
    'phase7_3stage_optimized': {
        'rmse': float(rmse_final),
        'mae': float(mae_final),
        'r2': float(r2_final),
        'per_stage': {
            'low': float(np.sqrt(mean_squared_error(y_val[mask_val_low], val_pred_optimized[mask_val_low]))),
            'medium': float(rmse_med_optimized),
            'high': float(rmse_high_optimized)
        }
    },
    'optuna_best_params': {
        'medium': study_medium.best_params,
        'high': study_high.best_params
    },
    'total_improvement_kg': float(163.04 - rmse_final),
    'total_improvement_pct': float((163.04 - rmse_final) / 163.04 * 100),
    'gap_to_target': float(rmse_final - 50)
}

with open(REPORTS_DIR / 'phase7_advanced_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to phase7_advanced_optimization_results.json", flush=True)
print("=" * 80, flush=True)
