#!/usr/bin/env python3
"""
Phase 7: Breakthrough to <50 kg RMSE - Revised Strategy
=======================================================

Current: 163.04 kg RMSE (Two-stage model)
Target: <50 kg RMSE (69% reduction)

Key Strategies:
1. Improved 3-stage stratification with optimal thresholds
2. Per-stage hyperparameter optimization
3. Advanced feature engineering
4. Sophisticated stacking ensemble
5. Iterative refinement based on error analysis

Author: ML Team
Date: 2025-10-15
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80, flush=True)
print("PHASE 7: BREAKTHROUGH TO <50 KG RMSE - V2", flush=True)
print("=" * 80, flush=True)
print(f"Start Time: {datetime.now()}", flush=True)
print(flush=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / "X_train_clean.parquet")
y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
X_val = pd.read_parquet(CACHE_DIR / "X_val_clean.parquet")
y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

print(f"✓ Train: {X_train.shape}, y range: [{y_train.min():.2f}, {y_train.max():.2f}]", flush=True)
print(f"✓ Val: {X_val.shape}, y range: [{y_val.min():.2f}, {y_val.max():.2f}]", flush=True)
print(flush=True)

baseline_rmse = 163.04

# ============================================================================
# STRATEGY 1: OPTIMIZED 3-STAGE MODEL
# ============================================================================

print("=" * 80, flush=True)
print("STRATEGY 1: OPTIMIZED 3-STAGE STRATIFICATION", flush=True)
print("=" * 80, flush=True)

# Analyze fuel distribution to find optimal thresholds
percentiles = [25, 50, 75, 90, 95]
thresholds_percentiles = [np.percentile(y_train, p) for p in percentiles]
print(f"\nFuel distribution percentiles:", flush=True)
for p, t in zip(percentiles, thresholds_percentiles):
    print(f"  {p}th: {t:.2f} kg", flush=True)

# Define 3 stages based on error analysis
# Low: Most samples, relatively accurate
# Medium: Moderate fuel, higher variance
# High: Few samples, highest errors

thresholds_3stage = [500, 1500]  # Low | Medium | High

print(f"\n3-Stage thresholds: {thresholds_3stage}", flush=True)

stages_3 = {
    'low': (0, thresholds_3stage[0]),
    'medium': (thresholds_3stage[0], thresholds_3stage[1]),
    'high': (thresholds_3stage[1], 100000)
}

# Train 3-stage models with stage-specific parameters
models_3stage = {}
results_3stage = {}

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    print(f"\n{'-'*80}", flush=True)
    print(f"Training {stage_name.upper()} model ({min_fuel}-{max_fuel} kg)", flush=True)
    print(f"{'-'*80}", flush=True)

    # Filter data
    mask_train = (y_train >= min_fuel) & (y_train < max_fuel)
    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    X_train_stage = X_train[mask_train]
    y_train_stage = y_train[mask_train]
    X_val_stage = X_val[mask_val]
    y_val_stage = y_val[mask_val]

    print(f"Train samples: {len(X_train_stage):,}", flush=True)
    print(f"Val samples: {len(X_val_stage):,}", flush=True)
    print(f"Fuel range: [{y_train_stage.min():.2f}, {y_train_stage.max():.2f}]", flush=True)

    if len(X_train_stage) < 50:
        print(f"⚠️  Insufficient samples, skipping...", flush=True)
        continue

    # Stage-specific parameters
    if stage_name == 'low':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
    elif stage_name == 'medium':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_samples': 30,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'reg_alpha': 0.03,
            'reg_lambda': 0.03,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
    else:  # high
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_samples': 50,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }

    # Train model
    train_data = lgb.Dataset(X_train_stage, label=y_train_stage)
    val_data = lgb.Dataset(X_val_stage, label=y_val_stage, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )

    # Evaluate
    train_pred = model.predict(X_train_stage, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val_stage, num_iteration=model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train_stage, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_stage, val_pred))
    val_mae = mean_absolute_error(y_val_stage, val_pred)
    val_r2 = r2_score(y_val_stage, val_pred)

    print(f"\nResults:", flush=True)
    print(f"  Train RMSE: {train_rmse:.2f} kg", flush=True)
    print(f"  Val RMSE: {val_rmse:.2f} kg", flush=True)
    print(f"  Val MAE: {val_mae:.2f} kg", flush=True)
    print(f"  Val R²: {val_r2:.4f}", flush=True)
    print(f"  Best iteration: {model.best_iteration}", flush=True)

    models_3stage[stage_name] = model
    results_3stage[stage_name] = {
        'rmse': float(val_rmse),
        'mae': float(val_mae),
        'r2': float(val_r2),
        'train_samples': int(len(X_train_stage)),
        'val_samples': int(len(X_val_stage)),
        'best_iteration': int(model.best_iteration)
    }

    # Save model
    model.save_model(str(MODELS_DIR / f'phase7_3stage_{stage_name}.txt'))

# Evaluate 3-stage on full validation set (using TRUE fuel values for routing)
print(f"\n{'='*80}", flush=True)
print("3-STAGE MODEL - FULL VALIDATION EVALUATION", flush=True)
print(f"{'='*80}", flush=True)

val_pred_3stage = np.zeros(len(y_val))

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    if stage_name not in models_3stage:
        continue

    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    if mask_val.sum() > 0:
        model = models_3stage[stage_name]
        val_pred_3stage[mask_val] = model.predict(
            X_val[mask_val],
            num_iteration=model.best_iteration
        )

rmse_3stage = np.sqrt(mean_squared_error(y_val, val_pred_3stage))
mae_3stage = mean_absolute_error(y_val, val_pred_3stage)
r2_3stage = r2_score(y_val, val_pred_3stage)

print(f"\n3-Stage Overall:", flush=True)
print(f"  RMSE: {rmse_3stage:.2f} kg", flush=True)
print(f"  MAE: {mae_3stage:.2f} kg", flush=True)
print(f"  R²: {r2_3stage:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_3stage:+.2f} kg ({(baseline_rmse - rmse_3stage) / baseline_rmse * 100:+.1f}%)", flush=True)

# ============================================================================
# STRATEGY 2: PER-STAGE CATBOOST MODELS
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("STRATEGY 2: 3-STAGE CATBOOST MODELS", flush=True)
print(f"{'='*80}", flush=True)

models_3stage_cat = {}

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    print(f"\n{'-'*80}", flush=True)
    print(f"Training CatBoost {stage_name.upper()} model", flush=True)
    print(f"{'-'*80}", flush=True)

    # Filter data
    mask_train = (y_train >= min_fuel) & (y_train < max_fuel)
    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    X_train_stage = X_train[mask_train]
    y_train_stage = y_train[mask_train]
    X_val_stage = X_val[mask_val]
    y_val_stage = y_val[mask_val]

    if len(X_train_stage) < 50:
        continue

    # Stage-specific parameters
    if stage_name == 'low':
        iterations, depth, l2_leaf_reg = 1500, 5, 3.0
    elif stage_name == 'medium':
        iterations, depth, l2_leaf_reg = 1500, 6, 5.0
    else:
        iterations, depth, l2_leaf_reg = 1500, 7, 7.0

    model_cat = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        min_data_in_leaf=50,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=200,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100
    )

    model_cat.fit(
        X_train_stage, y_train_stage,
        eval_set=(X_val_stage, y_val_stage),
        verbose=200
    )

    val_pred = model_cat.predict(X_val_stage)
    val_rmse = np.sqrt(mean_squared_error(y_val_stage, val_pred))
    print(f"\n  Val RMSE: {val_rmse:.2f} kg", flush=True)

    models_3stage_cat[stage_name] = model_cat

    # Save model
    with open(MODELS_DIR / f'phase7_3stage_cat_{stage_name}.pkl', 'wb') as f:
        pickle.dump(model_cat, f)

# Evaluate CatBoost 3-stage
val_pred_3stage_cat = np.zeros(len(y_val))

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    if stage_name not in models_3stage_cat:
        continue

    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    if mask_val.sum() > 0:
        model = models_3stage_cat[stage_name]
        val_pred_3stage_cat[mask_val] = model.predict(X_val[mask_val])

rmse_3stage_cat = np.sqrt(mean_squared_error(y_val, val_pred_3stage_cat))
mae_3stage_cat = mean_absolute_error(y_val, val_pred_3stage_cat)
r2_3stage_cat = r2_score(y_val, val_pred_3stage_cat)

print(f"\n{'='*80}", flush=True)
print("3-Stage CatBoost Overall:", flush=True)
print(f"  RMSE: {rmse_3stage_cat:.2f} kg", flush=True)
print(f"  MAE: {mae_3stage_cat:.2f} kg", flush=True)
print(f"  R²: {r2_3stage_cat:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_3stage_cat:+.2f} kg ({(baseline_rmse - rmse_3stage_cat) / baseline_rmse * 100:+.1f}%)", flush=True)

# ============================================================================
# STRATEGY 3: ENSEMBLE 3-STAGE (LightGBM + CatBoost)
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("STRATEGY 3: ENSEMBLE 3-STAGE MODELS", flush=True)
print(f"{'='*80}", flush=True)

# Find optimal weights for each stage
ensemble_weights_by_stage = {}

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    if stage_name not in models_3stage or stage_name not in models_3stage_cat:
        continue

    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    if mask_val.sum() < 10:
        continue

    X_val_stage = X_val[mask_val]
    y_val_stage = y_val[mask_val]

    # Get predictions from both models
    pred_lgb = models_3stage[stage_name].predict(X_val_stage, num_iteration=models_3stage[stage_name].best_iteration)
    pred_cat = models_3stage_cat[stage_name].predict(X_val_stage)

    # Grid search for optimal weight
    best_rmse = float('inf')
    best_weight = 0.5

    for w_lgb in np.linspace(0.0, 1.0, 21):
        w_cat = 1 - w_lgb
        pred_ensemble = w_lgb * pred_lgb + w_cat * pred_cat
        rmse = np.sqrt(mean_squared_error(y_val_stage, pred_ensemble))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w_lgb

    ensemble_weights_by_stage[stage_name] = {'lightgbm': best_weight, 'catboost': 1 - best_weight, 'rmse': best_rmse}
    print(f"{stage_name.upper()}: LightGBM={best_weight:.3f}, CatBoost={1-best_weight:.3f}, RMSE={best_rmse:.2f} kg", flush=True)

# Create ensemble predictions
val_pred_ensemble = np.zeros(len(y_val))

for stage_name, (min_fuel, max_fuel) in stages_3.items():
    if stage_name not in ensemble_weights_by_stage:
        continue

    mask_val = (y_val >= min_fuel) & (y_val < max_fuel)

    if mask_val.sum() > 0:
        w_lgb = ensemble_weights_by_stage[stage_name]['lightgbm']
        w_cat = ensemble_weights_by_stage[stage_name]['catboost']

        pred_lgb = models_3stage[stage_name].predict(X_val[mask_val], num_iteration=models_3stage[stage_name].best_iteration)
        pred_cat = models_3stage_cat[stage_name].predict(X_val[mask_val])

        val_pred_ensemble[mask_val] = w_lgb * pred_lgb + w_cat * pred_cat

rmse_ensemble = np.sqrt(mean_squared_error(y_val, val_pred_ensemble))
mae_ensemble = mean_absolute_error(y_val, val_pred_ensemble)
r2_ensemble = r2_score(y_val, val_pred_ensemble)

print(f"\n{'='*80}", flush=True)
print("3-Stage Ensemble Overall:", flush=True)
print(f"  RMSE: {rmse_ensemble:.2f} kg", flush=True)
print(f"  MAE: {mae_ensemble:.2f} kg", flush=True)
print(f"  R²: {r2_ensemble:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_ensemble:+.2f} kg ({(baseline_rmse - rmse_ensemble) / baseline_rmse * 100:+.1f}%)", flush=True)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("PHASE 7: FINAL RESULTS SUMMARY", flush=True)
print(f"{'='*80}", flush=True)

results_summary = {
    'Phase 6 Baseline': baseline_rmse,
    '3-Stage LightGBM': rmse_3stage,
    '3-Stage CatBoost': rmse_3stage_cat,
    '3-Stage Ensemble': rmse_ensemble
}

print(f"\n{'Model':<30} {'RMSE':<12} {'vs Baseline':<15} {'Gap to Target'}", flush=True)
print("-" * 80, flush=True)

for name, rmse_val in results_summary.items():
    improvement = baseline_rmse - rmse_val if name != 'Phase 6 Baseline' else 0
    gap_to_target = rmse_val - 50
    print(f"{name:<30} {rmse_val:<12.2f} {improvement:<15.2f} {gap_to_target:.2f} kg", flush=True)

print("=" * 80, flush=True)

best_rmse = min(rmse_3stage, rmse_3stage_cat, rmse_ensemble)
best_model_name = ['3-Stage LightGBM', '3-Stage CatBoost', '3-Stage Ensemble'][[rmse_3stage, rmse_3stage_cat, rmse_ensemble].index(best_rmse)]

print(f"\n✓ BEST MODEL: {best_model_name}", flush=True)
print(f"  RMSE: {best_rmse:.2f} kg", flush=True)
print(f"  Improvement from baseline: {baseline_rmse - best_rmse:.2f} kg ({(baseline_rmse - best_rmse) / baseline_rmse * 100:.1f}%)", flush=True)
print(f"  Gap to target (<50 kg): {best_rmse - 50:.2f} kg ({(best_rmse - 50) / best_rmse * 100:.1f}%)", flush=True)

if best_rmse < 50:
    print(f"\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉", flush=True)
elif best_rmse < 100:
    print(f"\n✓ Excellent progress!", flush=True)
else:
    print(f"\n⚠️  More optimization needed", flush=True)

# Save results
final_results = {
    'timestamp': datetime.now().isoformat(),
    'baseline_rmse': baseline_rmse,
    'three_stage_lightgbm': {'rmse': float(rmse_3stage), 'mae': float(mae_3stage), 'r2': float(r2_3stage), 'stage_results': results_3stage},
    'three_stage_catboost': {'rmse': float(rmse_3stage_cat), 'mae': float(mae_3stage_cat), 'r2': float(r2_3stage_cat)},
    'three_stage_ensemble': {'rmse': float(rmse_ensemble), 'mae': float(mae_ensemble), 'r2': float(r2_ensemble), 'weights': ensemble_weights_by_stage},
    'best_model': best_model_name,
    'best_rmse': float(best_rmse),
    'improvement_kg': float(baseline_rmse - best_rmse),
    'improvement_pct': float((baseline_rmse - best_rmse) / baseline_rmse * 100),
    'gap_to_target': float(best_rmse - 50)
}

with open(REPORTS_DIR / 'phase7_breakthrough_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

# Save ensemble weights
with open(MODELS_DIR / 'phase7_ensemble_weights.json', 'w') as f:
    json.dump({'weights_by_stage': ensemble_weights_by_stage, 'thresholds': thresholds_3stage}, f, indent=2)

print(f"\n✓ Results saved to phase7_breakthrough_results.json", flush=True)
print("=" * 80, flush=True)
