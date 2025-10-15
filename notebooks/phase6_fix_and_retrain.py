#!/usr/bin/env python3
"""Fix data leakage and retrain models"""

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

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42

print("=" * 80, flush=True)
print("PHASE 6: FIX DATA LEAKAGE AND RETRAIN", flush=True)
print("=" * 80, flush=True)

# Load data
print("\nLoading data...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / "X_train_enhanced.parquet")
y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
X_val = pd.read_parquet(CACHE_DIR / "X_val_enhanced.parquet")
y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

print(f"Original shape: {X_train.shape}", flush=True)

# Remove data leakage features
leakage_features = [
    'target_fuel_per_min',
    'target_fuel_per_km',
    'aircraft_fuel_mean',
    'aircraft_fuel_std',
    'aircraft_fuel_median',
    'aircraft_fuel_min',
    'aircraft_fuel_max',
    'aircraft_fuel_per_km_mean',
    'aircraft_fuel_per_km_std',
    'fuel_deviation_from_aircraft_avg'
]

# Remove features that exist
features_to_remove = [f for f in leakage_features if f in X_train.columns]
print(f"\nRemoving {len(features_to_remove)} leakage features:", flush=True)
for f in features_to_remove:
    print(f"  - {f}", flush=True)

X_train_clean = X_train.drop(columns=features_to_remove)
X_val_clean = X_val.drop(columns=features_to_remove)

print(f"\nCleaned shape: {X_train_clean.shape}", flush=True)

baseline_rmse = 189.29

# Train LightGBM
print("\n" + "=" * 80, flush=True)
print("Training LightGBM (No Leakage)", flush=True)
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

train_data = lgb.Dataset(X_train_clean, label=y_train)
val_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data)

start_time = time.time()
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)
train_time = time.time() - start_time

y_val_pred_lgb = lgb_model.predict(X_val_clean, num_iteration=lgb_model.best_iteration)
lgb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
lgb_mae = mean_absolute_error(y_val, y_val_pred_lgb)
lgb_r2 = r2_score(y_val, y_val_pred_lgb)

print(f"\n✓ LightGBM Results:", flush=True)
print(f"  Val RMSE:   {lgb_rmse:.2f} kg", flush=True)
print(f"  Val MAE:    {lgb_mae:.2f} kg", flush=True)
print(f"  Val R²:     {lgb_r2:.4f}", flush=True)
print(f"  Improvement vs baseline: {((baseline_rmse - lgb_rmse) / baseline_rmse * 100):.1f}%", flush=True)

with open(MODELS_DIR / 'phase6_clean_lightgbm.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

# Train CatBoost
print("\n" + "=" * 80, flush=True)
print("Training CatBoost (No Leakage)", flush=True)
print("=" * 80, flush=True)

cat_model = CatBoostRegressor(
    iterations=1500,
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

start_time = time.time()
cat_model.fit(
    X_train_clean, y_train,
    eval_set=(X_val_clean, y_val),
    verbose=50
)
train_time = time.time() - start_time

y_val_pred_cat = cat_model.predict(X_val_clean)
cat_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_cat))
cat_mae = mean_absolute_error(y_val, y_val_pred_cat)
cat_r2 = r2_score(y_val, y_val_pred_cat)

print(f"\n✓ CatBoost Results:", flush=True)
print(f"  Val RMSE:   {cat_rmse:.2f} kg", flush=True)
print(f"  Val MAE:    {cat_mae:.2f} kg", flush=True)
print(f"  Val R²:     {cat_r2:.4f}", flush=True)
print(f"  Improvement vs baseline: {((baseline_rmse - cat_rmse) / baseline_rmse * 100):.1f}%", flush=True)

with open(MODELS_DIR / 'phase6_clean_catboost.pkl', 'wb') as f:
    pickle.dump(cat_model, f)

# Ensemble
print("\n" + "=" * 80, flush=True)
print("Creating Ensemble", flush=True)
print("=" * 80, flush=True)

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
print(f"  Improvement vs baseline: {((baseline_rmse - ensemble_rmse) / baseline_rmse * 100):.1f}%", flush=True)

ensemble_info = {
    'weights': {'LightGBM': best_weights[0], 'CatBoost': best_weights[1]},
    'rmse': ensemble_rmse
}
with open(MODELS_DIR / 'phase6_clean_ensemble_weights.json', 'w') as f:
    json.dump(ensemble_info, f, indent=2)

# Save cleaned features
X_train_clean.to_parquet(CACHE_DIR / 'X_train_clean.parquet', index=False)
X_val_clean.to_parquet(CACHE_DIR / 'X_val_clean.parquet', index=False)
print(f"\n✓ Saved cleaned feature sets", flush=True)

# Final Summary
print("\n" + "=" * 80, flush=True)
print("FINAL RESULTS (NO DATA LEAKAGE)", flush=True)
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
print(f"  Improvement: {((baseline_rmse - best_rmse) / baseline_rmse * 100):.1f}% ({baseline_rmse - best_rmse:.2f} kg)", flush=True)

if best_rmse < 50:
    print(f"\n🎉 TARGET ACHIEVED! RMSE < 50 kg", flush=True)
elif best_rmse < 100:
    print(f"\n✓ Excellent! RMSE < 100 kg (target: 50 kg)", flush=True)
    print(f"  Gap: {best_rmse - 50:.2f} kg", flush=True)
elif best_rmse < baseline_rmse:
    print(f"\n✓ Improvement achieved!", flush=True)
    print(f"  Gap to target: {best_rmse - 50:.2f} kg", flush=True)
else:
    print(f"\n⚠️  No improvement from baseline", flush=True)

results = {
    'baseline_rmse': baseline_rmse,
    'lightgbm_rmse': float(lgb_rmse),
    'catboost_rmse': float(cat_rmse),
    'ensemble_rmse': float(ensemble_rmse),
    'best_model': best_model,
    'best_rmse': float(best_rmse),
    'improvement_pct': float((baseline_rmse - best_rmse) / baseline_rmse * 100),
    'features_used': int(X_train_clean.shape[1]),
    'leakage_features_removed': len(features_to_remove),
    'timestamp': datetime.now().isoformat()
}

with open(REPORTS_DIR / 'phase6_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to phase6_final_results.json", flush=True)
print("=" * 80, flush=True)
