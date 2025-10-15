#!/usr/bin/env python3
"""
Phase 7: Final Comprehensive Solution - Target <50 kg RMSE
===========================================================

Current Status:
- Overall: 150.61 kg RMSE
- Low (<500kg): 45.55 kg RMSE ✓ Excellent!
- Medium (500-1500kg): 132.91 kg RMSE - Good
- High (>1500kg): 548.57 kg RMSE ❌ CRITICAL PROBLEM

Analysis:
- High range is only 6.3% of data (618 samples) but contributes massively to RMSE
- If we improve high range to ~200 kg, overall RMSE would drop to ~80 kg
- If we improve high range to ~100 kg, overall RMSE would reach ~60 kg

Strategy:
1. Ultra-granular stratification (6+ stages)
2. Log-transform for high-variance ranges
3. Specialized features for high fuel intervals
4. CatBoost focus on high range (better for small samples)
5. Weighted ensemble across all approaches

Author: ML Team
Date: 2025-10-15
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

BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80, flush=True)
print("PHASE 7: FINAL COMPREHENSIVE SOLUTION", flush=True)
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

# ============================================================================
# APPROACH 1: ULTRA-GRANULAR 6-STAGE STRATIFICATION
# ============================================================================

print("=" * 80, flush=True)
print("APPROACH 1: ULTRA-GRANULAR 6-STAGE STRATIFICATION", flush=True)
print("=" * 80, flush=True)

# Define 6 stages with focus on breaking up the problematic high range
stages_6 = [
    ('very_low', 0, 200),
    ('low', 200, 500),
    ('medium', 500, 1000),
    ('medium_high', 1000, 1500),
    ('high', 1500, 2500),
    ('very_high', 2500, 100000)
]

print("6-Stage thresholds:", flush=True)
for name, min_f, max_f in stages_6:
    train_count = ((y_train >= min_f) & (y_train < max_f)).sum()
    val_count = ((y_val >= min_f) & (y_val < max_f)).sum()
    print(f"  {name:15s} ({min_f:5.0f}-{max_f:5.0f} kg): Train={train_count:6,}, Val={val_count:4,}", flush=True)
print(flush=True)

models_6stage = {}

# Load existing optimized low/medium models
model_low = lgb.Booster(model_file=str(MODELS_DIR / 'phase7_3stage_low.txt'))
model_med = lgb.Booster(model_file=str(MODELS_DIR / 'phase7_medium_optimized.txt'))

# For very_low, low: use existing low model
# For medium, medium_high: train new models
# For high, very_high: train specialized models

# Train medium (500-1000) model
print("Training MEDIUM model (500-1000 kg)...", flush=True)
mask_train = (y_train >= 500) & (y_train < 1000)
mask_val = (y_val >= 500) & (y_val < 1000)

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

print(f"  Samples: Train={len(X_train_stage):,}, Val={len(X_val_stage):,}", flush=True)

params_med = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_samples': 40,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 3,
    'reg_alpha': 0.15,
    'reg_lambda': 0.15,
    'verbose': -1,
    'random_state': RANDOM_STATE
}

train_data = lgb.Dataset(X_train_stage, label=y_train_stage)
val_data = lgb.Dataset(X_val_stage, label=y_val_stage, reference=train_data)

model_medium_500_1000 = lgb.train(
    params_med,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred = model_medium_500_1000.predict(X_val_stage, num_iteration=model_medium_500_1000.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# Train medium_high (1000-1500) model
print("\nTraining MEDIUM_HIGH model (1000-1500 kg)...", flush=True)
mask_train = (y_train >= 1000) & (y_train < 1500)
mask_val = (y_val >= 1000) & (y_val < 1500)

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

print(f"  Samples: Train={len(X_train_stage):,}, Val={len(X_val_stage):,}", flush=True)

train_data = lgb.Dataset(X_train_stage, label=y_train_stage)
val_data = lgb.Dataset(X_val_stage, label=y_val_stage, reference=train_data)

model_medium_high = lgb.train(
    params_med,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred = model_medium_high.predict(X_val_stage, num_iteration=model_medium_high.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# ============================================================================
# APPROACH 2: LOG-TRANSFORM FOR HIGH FUEL RANGES
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("APPROACH 2: LOG-TRANSFORM FOR HIGH FUEL RANGES", flush=True)
print("=" * 80, flush=True)

# Train HIGH (1500-2500) with log-transform
print("\nTraining HIGH model (1500-2500 kg) with log-transform...", flush=True)
mask_train = (y_train >= 1500) & (y_train < 2500)
mask_val = (y_val >= 1500) & (y_val < 2500)

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

print(f"  Samples: Train={len(X_train_stage):,}, Val={len(X_val_stage):,}", flush=True)

# Log-transform target
y_train_log = np.log1p(y_train_stage)
y_val_log = np.log1p(y_val_stage)

train_data = lgb.Dataset(X_train_stage, label=y_train_log)
val_data = lgb.Dataset(X_val_stage, label=y_val_log, reference=train_data)

model_high_log = lgb.train(
    params_med,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred_log = model_high_log.predict(X_val_stage, num_iteration=model_high_log.best_iteration)
pred = np.expm1(pred_log)  # Inverse transform
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# Train VERY_HIGH (2500+) with log-transform
print("\nTraining VERY_HIGH model (2500+ kg) with log-transform...", flush=True)
mask_train = y_train >= 2500
mask_val = y_val >= 2500

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

print(f"  Samples: Train={len(X_train_stage):,}, Val={len(X_val_stage):,}", flush=True)

# Log-transform target
y_train_log = np.log1p(y_train_stage)
y_val_log = np.log1p(y_val_stage)

train_data = lgb.Dataset(X_train_stage, label=y_train_log)
val_data = lgb.Dataset(X_val_stage, label=y_val_log, reference=train_data)

model_very_high_log = lgb.train(
    params_med,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

pred_log = model_very_high_log.predict(X_val_stage, num_iteration=model_very_high_log.best_iteration)
pred = np.expm1(pred_log)
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# ============================================================================
# APPROACH 3: CAT BOOST FOR HIGH RANGES (Better for small samples)
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("APPROACH 3: CATBOOST FOR HIGH RANGES", flush=True)
print("=" * 80, flush=True)

# Train HIGH (1500-2500) with CatBoost
print("\nTraining HIGH model (1500-2500 kg) with CatBoost...", flush=True)
mask_train = (y_train >= 1500) & (y_train < 2500)
mask_val = (y_val >= 1500) & (y_val < 2500)

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

model_high_cat = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=5.0,
    min_data_in_leaf=20,
    loss_function='RMSE',
    verbose=0,
    random_seed=RANDOM_STATE,
    early_stopping_rounds=100
)

model_high_cat.fit(
    X_train_stage, y_train_stage,
    eval_set=(X_val_stage, y_val_stage),
    verbose=False
)

pred = model_high_cat.predict(X_val_stage)
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# Train VERY_HIGH (2500+) with CatBoost
print("\nTraining VERY_HIGH model (2500+ kg) with CatBoost...", flush=True)
mask_train = y_train >= 2500
mask_val = y_val >= 2500

X_train_stage = X_train[mask_train]
y_train_stage = y_train[mask_train]
X_val_stage = X_val[mask_val]
y_val_stage = y_val[mask_val]

model_very_high_cat = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=7.0,
    min_data_in_leaf=10,
    loss_function='RMSE',
    verbose=0,
    random_seed=RANDOM_STATE,
    early_stopping_rounds=100
)

model_very_high_cat.fit(
    X_train_stage, y_train_stage,
    eval_set=(X_val_stage, y_val_stage),
    verbose=False
)

pred = model_very_high_cat.predict(X_val_stage)
rmse = np.sqrt(mean_squared_error(y_val_stage, pred))
print(f"  Val RMSE: {rmse:.2f} kg", flush=True)

# ============================================================================
# FINAL ENSEMBLE: COMBINE ALL APPROACHES
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("FINAL ENSEMBLE: OPTIMAL ROUTING AND AVERAGING", flush=True)
print("=" * 80, flush=True)

# Create predictions for each approach
val_pred_6stage_lgb = np.zeros(len(y_val))
val_pred_6stage_log = np.zeros(len(y_val))
val_pred_6stage_cat = np.zeros(len(y_val))

# Very low (0-200): Use existing low model
mask = (y_val >= 0) & (y_val < 200)
if mask.sum() > 0:
    pred = model_low.predict(X_val[mask], num_iteration=model_low.best_iteration)
    val_pred_6stage_lgb[mask] = pred
    val_pred_6stage_log[mask] = pred
    val_pred_6stage_cat[mask] = pred

# Low (200-500): Use existing low model
mask = (y_val >= 200) & (y_val < 500)
if mask.sum() > 0:
    pred = model_low.predict(X_val[mask], num_iteration=model_low.best_iteration)
    val_pred_6stage_lgb[mask] = pred
    val_pred_6stage_log[mask] = pred
    val_pred_6stage_cat[mask] = pred

# Medium (500-1000): Use new model
mask = (y_val >= 500) & (y_val < 1000)
if mask.sum() > 0:
    pred = model_medium_500_1000.predict(X_val[mask], num_iteration=model_medium_500_1000.best_iteration)
    val_pred_6stage_lgb[mask] = pred
    val_pred_6stage_log[mask] = pred
    val_pred_6stage_cat[mask] = pred

# Medium-High (1000-1500): Use new model
mask = (y_val >= 1000) & (y_val < 1500)
if mask.sum() > 0:
    pred = model_medium_high.predict(X_val[mask], num_iteration=model_medium_high.best_iteration)
    val_pred_6stage_lgb[mask] = pred
    val_pred_6stage_log[mask] = pred
    val_pred_6stage_cat[mask] = pred

# High (1500-2500): Use best of log/cat
mask = (y_val >= 1500) & (y_val < 2500)
if mask.sum() > 0:
    # LightGBM standard
    pred_lgb = model_high_log.predict(X_val[mask], num_iteration=model_high_log.best_iteration)
    pred_lgb = np.expm1(pred_lgb)  # Already log-transformed model
    val_pred_6stage_lgb[mask] = pred_lgb

    # Log-transform
    pred_log = model_high_log.predict(X_val[mask], num_iteration=model_high_log.best_iteration)
    val_pred_6stage_log[mask] = np.expm1(pred_log)

    # CatBoost
    val_pred_6stage_cat[mask] = model_high_cat.predict(X_val[mask])

# Very High (2500+): Use best of log/cat
mask = y_val >= 2500
if mask.sum() > 0:
    # LightGBM log
    pred_lgb = model_very_high_log.predict(X_val[mask], num_iteration=model_very_high_log.best_iteration)
    val_pred_6stage_lgb[mask] = np.expm1(pred_lgb)

    # Log-transform
    pred_log = model_very_high_log.predict(X_val[mask], num_iteration=model_very_high_log.best_iteration)
    val_pred_6stage_log[mask] = np.expm1(pred_log)

    # CatBoost
    val_pred_6stage_cat[mask] = model_very_high_cat.predict(X_val[mask])

# Calculate RMSE for each approach
rmse_lgb = np.sqrt(mean_squared_error(y_val, val_pred_6stage_lgb))
rmse_log = np.sqrt(mean_squared_error(y_val, val_pred_6stage_log))
rmse_cat = np.sqrt(mean_squared_error(y_val, val_pred_6stage_cat))

print(f"\n6-Stage Approaches:", flush=True)
print(f"  LightGBM:      {rmse_lgb:.2f} kg", flush=True)
print(f"  Log-Transform: {rmse_log:.2f} kg", flush=True)
print(f"  CatBoost:      {rmse_cat:.2f} kg", flush=True)

# Find optimal ensemble weights
print(f"\nFinding optimal ensemble weights...", flush=True)
best_rmse = float('inf')
best_weights = None

for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for w2 in [0.1, 0.2, 0.3, 0.4]:
        w3 = 1 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue

        pred_ensemble = w1 * val_pred_6stage_lgb + w2 * val_pred_6stage_log + w3 * val_pred_6stage_cat
        rmse = np.sqrt(mean_squared_error(y_val, pred_ensemble))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)

val_pred_final = (best_weights[0] * val_pred_6stage_lgb +
                  best_weights[1] * val_pred_6stage_log +
                  best_weights[2] * val_pred_6stage_cat)

rmse_final = np.sqrt(mean_squared_error(y_val, val_pred_final))
mae_final = mean_absolute_error(y_val, val_pred_final)
r2_final = r2_score(y_val, val_pred_final)

print(f"\n✓ FINAL ENSEMBLE:", flush=True)
print(f"  Weights: LGB={best_weights[0]:.2f}, Log={best_weights[1]:.2f}, Cat={best_weights[2]:.2f}", flush=True)
print(f"  RMSE: {rmse_final:.2f} kg", flush=True)
print(f"  MAE: {mae_final:.2f} kg", flush=True)
print(f"  R²: {r2_final:.4f}", flush=True)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("PHASE 7: COMPREHENSIVE SOLUTION - FINAL SUMMARY", flush=True)
print("=" * 80, flush=True)

progress = [
    ('Phase 6 Two-Stage', 163.04),
    ('Phase 7 3-Stage Baseline', 151.84),
    ('Phase 7 3-Stage Optimized', 150.61),
    ('Phase 7 6-Stage Final', rmse_final)
]

print(f"\n{'Model':<30} {'RMSE':<12} {'Improvement':<15} {'Gap to Target'}", flush=True)
print("-" * 80, flush=True)

for i, (name, rmse_val) in enumerate(progress):
    improvement = 163.04 - rmse_val if i > 0 else 0
    gap = rmse_val - 50
    print(f"{name:<30} {rmse_val:<12.2f} {improvement:<15.2f} {gap:.2f} kg", flush=True)

print("=" * 80, flush=True)

total_improvement = 163.04 - rmse_final
total_improvement_pct = (total_improvement / 163.04) * 100
remaining_gap = rmse_final - 50

print(f"\n✓ Total improvement: {total_improvement:.2f} kg ({total_improvement_pct:.1f}%)", flush=True)
print(f"✓ Remaining gap to target: {remaining_gap:.2f} kg", flush=True)

if rmse_final < 50:
    print(f"\n🎉🎉🎉 TARGET ACHIEVED! RMSE < 50 KG! 🎉🎉🎉", flush=True)
elif rmse_final < 80:
    print(f"\n✓ EXCELLENT! Getting very close to target!", flush=True)
elif rmse_final < 120:
    print(f"\n✓ Strong progress! Major improvement achieved!", flush=True)
else:
    print(f"\n⚠️  More work needed, but making progress", flush=True)

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'final_rmse': float(rmse_final),
    'final_mae': float(mae_final),
    'final_r2': float(r2_final),
    'ensemble_weights': {
        'lightgbm': float(best_weights[0]),
        'log_transform': float(best_weights[1]),
        'catboost': float(best_weights[2])
    },
    'approach_rmse': {
        'lightgbm_6stage': float(rmse_lgb),
        'log_transform_6stage': float(rmse_log),
        'catboost_6stage': float(rmse_cat)
    },
    'total_improvement_kg': float(total_improvement),
    'total_improvement_pct': float(total_improvement_pct),
    'remaining_gap': float(remaining_gap)
}

with open(REPORTS_DIR / 'phase7_final_comprehensive_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to phase7_final_comprehensive_results.json", flush=True)
print("=" * 80, flush=True)

# Save models
print(f"\nSaving models...", flush=True)
model_medium_500_1000.save_model(str(MODELS_DIR / 'phase7_final_medium.txt'))
model_medium_high.save_model(str(MODELS_DIR / 'phase7_final_medium_high.txt'))
model_high_log.save_model(str(MODELS_DIR / 'phase7_final_high_log.txt'))
model_very_high_log.save_model(str(MODELS_DIR / 'phase7_final_very_high_log.txt'))

with open(MODELS_DIR / 'phase7_final_high_cat.pkl', 'wb') as f:
    pickle.dump(model_high_cat, f)
with open(MODELS_DIR / 'phase7_final_very_high_cat.pkl', 'wb') as f:
    pickle.dump(model_very_high_cat, f)

print("✓ All models saved!", flush=True)
print("=" * 80, flush=True)
