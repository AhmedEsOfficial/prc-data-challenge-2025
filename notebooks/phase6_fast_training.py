#!/usr/bin/env python3
"""
Phase 6: Fast Model Training - Aircraft Fuel Consumption Prediction
===================================================================

Quick training of improved models with enhanced features.

Author: ML Team
Date: 2025-10-15
"""

import sys
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

# Model libraries
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths
BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="  * 80, flush=True)
print("PHASE 6: FAST MODEL TRAINING WITH ENHANCED FEATURES", flush=True)
print("=" * 80, flush=True)

# Load data
print("\n[1/4] Loading enhanced features...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / "X_train_enhanced.parquet")
y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
X_val = pd.read_parquet(CACHE_DIR / "X_val_enhanced.parquet")
y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

print(f"✓ Training: {X_train.shape}", flush=True)
print(f"✓ Validation: {X_val.shape}", flush=True)
print(f"✓ Features: {X_train.shape[1]}", flush=True)

baseline_rmse = 189.29

# Train LightGBM
print("\n" + "=" * 80, flush=True)
print("[2/4] Training LightGBM with Enhanced Features", flush=True)
print("=" * 80, flush=True)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.08,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'max_depth': 8,
    'verbose': -1,
    'random_state': RANDOM_STATE
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

print("Training...", flush=True)
start_time = time.time()

lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

train_time = time.time() - start_time

y_val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
lgb_mae = mean_absolute_error(y_val, y_val_pred_lgb)
lgb_r2 = r2_score(y_val, y_val_pred_lgb)

print(f"\n✓ LightGBM Results:", flush=True)
print(f"  Val RMSE:   {lgb_rmse:.2f} kg", flush=True)
print(f"  Val MAE:    {lgb_mae:.2f} kg", flush=True)
print(f"  Val R²:     {lgb_r2:.4f}", flush=True)
print(f"  Training time: {train_time:.2f}s", flush=True)
print(f"  Improvement: {((baseline_rmse - lgb_rmse) / baseline_rmse * 100):.1f}%", flush=True)

# Save LightGBM
with open(MODELS_DIR / 'phase6_fast_lightgbm.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
print("  ✓ Saved model", flush=True)

# Train CatBoost
print("\n" + "=" * 80, flush=True)
print("[3/4] Training CatBoost with Enhanced Features", flush=True)
print("=" * 80, flush=True)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.08,
    depth=6,
    l2_leaf_reg=5.0,
    min_data_in_leaf=50,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=50,
    random_seed=RANDOM_STATE,
    early_stopping_rounds=50
)

print("Training...", flush=True)
start_time = time.time()

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=50
)

train_time = time.time() - start_time

y_val_pred_cat = cat_model.predict(X_val)
cat_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_cat))
cat_mae = mean_absolute_error(y_val, y_val_pred_cat)
cat_r2 = r2_score(y_val, y_val_pred_cat)

print(f"\n✓ CatBoost Results:", flush=True)
print(f"  Val RMSE:   {cat_rmse:.2f} kg", flush=True)
print(f"  Val MAE:    {cat_mae:.2f} kg", flush=True)
print(f"  Val R²:     {cat_r2:.4f}", flush=True)
print(f"  Training time: {train_time:.2f}s", flush=True)
print(f"  Improvement: {((baseline_rmse - cat_rmse) / baseline_rmse * 100):.1f}%", flush=True)

# Save CatBoost
with open(MODELS_DIR / 'phase6_fast_catboost.pkl', 'wb') as f:
    pickle.dump(cat_model, f)
print("  ✓ Saved model", flush=True)

# Simple Ensemble
print("\n" + "=" * 80, flush=True)
print("[4/4] Creating Simple Ensemble", flush=True)
print("=" * 80, flush=True)

# Grid search for optimal weights
best_rmse = float('inf')
best_weights = None

for w1 in np.linspace(0.3, 0.7, 9):
    w2 = 1 - w1
    ensemble_pred = w1 * y_val_pred_lgb + w2 * y_val_pred_cat
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

    if rmse < best_rmse:
        best_rmse = rmse
        best_weights = (w1, w2)

ensemble_pred = best_weights[0] * y_val_pred_lgb + best_weights[1] * y_val_pred_cat
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
ensemble_r2 = r2_score(y_val, ensemble_pred)

print(f"Optimal weights: LightGBM={best_weights[0]:.3f}, CatBoost={best_weights[1]:.3f}", flush=True)
print(f"\n✓ Ensemble Results:", flush=True)
print(f"  Val RMSE:   {ensemble_rmse:.2f} kg", flush=True)
print(f"  Val MAE:    {ensemble_mae:.2f} kg", flush=True)
print(f"  Val R²:     {ensemble_r2:.4f}", flush=True)
print(f"  Improvement: {((baseline_rmse - ensemble_rmse) / baseline_rmse * 100):.1f}%", flush=True)

# Save ensemble weights
ensemble_info = {
    'weights': {'LightGBM': best_weights[0], 'CatBoost': best_weights[1]},
    'rmse': ensemble_rmse
}
with open(MODELS_DIR / 'phase6_fast_ensemble_weights.json', 'w') as f:
    json.dump(ensemble_info, f, indent=2)
print("  ✓ Saved ensemble weights", flush=True)

# Final Summary
print("\n" + "=" * 80, flush=True)
print("FINAL RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)
print(f"\n{'Model':<20} {'RMSE (kg)':<15} {'vs Baseline':<15} {'Improvement'}", flush=True)
print("-" * 80, flush=True)
print(f"{'Phase 5 Baseline':<20} {baseline_rmse:<15.2f} {'---':<15} {'---'}", flush=True)
print(f"{'LightGBM':<20} {lgb_rmse:<15.2f} {baseline_rmse - lgb_rmse:<15.2f} {((baseline_rmse - lgb_rmse) / baseline_rmse * 100):.1f}%", flush=True)
print(f"{'CatBoost':<20} {cat_rmse:<15.2f} {baseline_rmse - cat_rmse:<15.2f} {((baseline_rmse - cat_rmse) / baseline_rmse * 100):.1f}%", flush=True)
print(f"{'Ensemble':<20} {ensemble_rmse:<15.2f} {baseline_rmse - ensemble_rmse:<15.2f} {((baseline_rmse - ensemble_rmse) / baseline_rmse * 100):.1f}%", flush=True)
print("=" * 80, flush=True)

best_rmse = min(lgb_rmse, cat_rmse, ensemble_rmse)
best_model = ['LightGBM', 'CatBoost', 'Ensemble'][[lgb_rmse, cat_rmse, ensemble_rmse].index(best_rmse)]

print(f"\n✓ BEST MODEL: {best_model}", flush=True)
print(f"  Final RMSE: {best_rmse:.2f} kg", flush=True)
print(f"  Improvement from Phase 5: {((baseline_rmse - best_rmse) / baseline_rmse * 100):.1f}% ({baseline_rmse - best_rmse:.2f} kg)", flush=True)

if best_rmse < 50:
    print(f"\n🎉🎉🎉 TARGET ACHIEVED! RMSE < 50 kg 🎉🎉🎉", flush=True)
elif best_rmse < 100:
    print(f"\n✓ Excellent progress! RMSE < 100 kg", flush=True)
    print(f"  Gap to target (50 kg): {best_rmse - 50:.2f} kg", flush=True)
else:
    print(f"\n⚠️  Target not yet reached", flush=True)
    print(f"  Current: {best_rmse:.2f} kg", flush=True)
    print(f"  Target: 50 kg", flush=True)
    print(f"  Gap: {best_rmse - 50:.2f} kg", flush=True)

print("\n" + "=" * 80, flush=True)
print("✓ PHASE 6: FAST TRAINING COMPLETE", flush=True)
print("=" * 80, flush=True)

# Save final results
results = {
    'baseline_rmse': baseline_rmse,
    'lightgbm_rmse': float(lgb_rmse),
    'catboost_rmse': float(cat_rmse),
    'ensemble_rmse': float(ensemble_rmse),
    'best_model': best_model,
    'best_rmse': float(best_rmse),
    'improvement_pct': float((baseline_rmse - best_rmse) / baseline_rmse * 100),
    'timestamp': datetime.now().isoformat()
}

with open(REPORTS_DIR / 'phase6_fast_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to phase6_fast_results.json", flush=True)
