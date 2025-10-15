#!/usr/bin/env python3
"""
Phase 6: Alternative Modeling Approaches
========================================

Explore advanced techniques to achieve <50 kg RMSE:
1. Log-transform target (handle skewness)
2. Two-stage modeling (separate models by fuel range)
3. Residual boosting (train on previous model's errors)
4. Quantile regression (handle heteroscedasticity)
5. Polynomial feature interactions
6. XGBoost with advanced parameters

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

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80, flush=True)
print("PHASE 6: ALTERNATIVE MODELING APPROACHES", flush=True)
print("=" * 80, flush=True)

# Load clean data
print("\n[1/7] Loading data...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / "X_train_clean.parquet")
y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
X_val = pd.read_parquet(CACHE_DIR / "X_val_clean.parquet")
y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

print(f"✓ Training: {X_train.shape}", flush=True)
print(f"✓ Validation: {X_val.shape}", flush=True)

baseline_rmse = 189.29
results = {}


# ============================================================================
#  APPROACH 1: LOG-TRANSFORMED TARGET
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[2/7] Approach 1: Log-Transformed Target", flush=True)
print("=" * 80, flush=True)
print("Rationale: Target is highly skewed (0.45 - 32,205 kg)", flush=True)

# Log transform
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

print("\nTraining LightGBM on log-transformed target...", flush=True)

params_log = {
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

train_data = lgb.Dataset(X_train, label=y_train_log)
val_data = lgb.Dataset(X_val, label=y_val_log, reference=train_data)

model_log = lgb.train(
    params_log,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

# Predict and inverse transform
y_val_pred_log = model_log.predict(X_val, num_iteration=model_log.best_iteration)
y_val_pred_log = np.expm1(y_val_pred_log)  # Inverse log transform

rmse_log = np.sqrt(mean_squared_error(y_val, y_val_pred_log))
mae_log = mean_absolute_error(y_val, y_val_pred_log)
r2_log = r2_score(y_val, y_val_pred_log)

print(f"\n✓ Log-Transform Results:", flush=True)
print(f"  RMSE: {rmse_log:.2f} kg", flush=True)
print(f"  MAE:  {mae_log:.2f} kg", flush=True)
print(f"  R²:   {r2_log:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_log:+.2f} kg ({((baseline_rmse - rmse_log) / baseline_rmse * 100):+.1f}%)", flush=True)

results['log_transform'] = {
    'rmse': float(rmse_log),
    'mae': float(mae_log),
    'r2': float(r2_log),
    'improvement': float(baseline_rmse - rmse_log)
}


# ============================================================================
#  APPROACH 2: TWO-STAGE MODELING (BY FUEL RANGE)
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[3/7] Approach 2: Two-Stage Modeling", flush=True)
print("=" * 80, flush=True)
print("Rationale: Errors highest in >2000 kg range", flush=True)

# Split data by fuel range
threshold = 1000  # kg
mask_train_low = y_train <= threshold
mask_train_high = y_train > threshold
mask_val_low = y_val <= threshold
mask_val_high = y_val > threshold

print(f"\nLow fuel (<={threshold} kg): Train={mask_train_low.sum()}, Val={mask_val_low.sum()}", flush=True)
print(f"High fuel (>{threshold} kg): Train={mask_train_high.sum()}, Val={mask_val_high.sum()}", flush=True)

# Train separate models
print("\nTraining model for LOW fuel range...", flush=True)
train_data_low = lgb.Dataset(X_train[mask_train_low], label=y_train[mask_train_low])
val_data_low = lgb.Dataset(X_val[mask_val_low], label=y_val[mask_val_low], reference=train_data_low)

model_low = lgb.train(
    params_log,
    train_data_low,
    num_boost_round=1500,
    valid_sets=[val_data_low],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

print("\nTraining model for HIGH fuel range...", flush=True)
train_data_high = lgb.Dataset(X_train[mask_train_high], label=y_train[mask_train_high])
val_data_high = lgb.Dataset(X_val[mask_val_high], label=y_val[mask_val_high], reference=train_data_high)

model_high = lgb.train(
    params_log,
    train_data_high,
    num_boost_round=1500,
    valid_sets=[val_data_high],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

# Predict with appropriate model
y_val_pred_twostage = np.zeros(len(y_val))
y_val_pred_twostage[mask_val_low] = model_low.predict(X_val[mask_val_low], num_iteration=model_low.best_iteration)
y_val_pred_twostage[mask_val_high] = model_high.predict(X_val[mask_val_high], num_iteration=model_high.best_iteration)

rmse_twostage = np.sqrt(mean_squared_error(y_val, y_val_pred_twostage))
mae_twostage = mean_absolute_error(y_val, y_val_pred_twostage)
r2_twostage = r2_score(y_val, y_val_pred_twostage)

print(f"\n✓ Two-Stage Results:", flush=True)
print(f"  RMSE: {rmse_twostage:.2f} kg", flush=True)
print(f"  MAE:  {mae_twostage:.2f} kg", flush=True)
print(f"  R²:   {r2_twostage:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_twostage:+.2f} kg ({((baseline_rmse - rmse_twostage) / baseline_rmse * 100):+.1f}%)", flush=True)

results['two_stage'] = {
    'rmse': float(rmse_twostage),
    'mae': float(mae_twostage),
    'r2': float(r2_twostage),
    'improvement': float(baseline_rmse - rmse_twostage)
}


# ============================================================================
#  APPROACH 3: RESIDUAL BOOSTING
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[3/7] Approach 3: Residual Boosting", flush=True)
print("=" * 80, flush=True)
print("Rationale: Train second model on first model's errors", flush=True)

# Use Phase 5 predictions as starting point
print("\nLoading Phase 5 best model...", flush=True)
try:
    with open(MODELS_DIR / 'phase5_catboost_optimized.pkl', 'rb') as f:
        base_model = pickle.load(f)

    # Get residuals
    y_train_pred_base = base_model.predict(X_train)
    y_val_pred_base = base_model.predict(X_val)

    residuals_train = y_train - y_train_pred_base
    residuals_val = y_val - y_val_pred_base

    print(f"Base model RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred_base)):.2f} kg", flush=True)
    print(f"Residual mean: {residuals_val.mean():.2f} kg", flush=True)
    print(f"Residual std: {residuals_val.std():.2f} kg", flush=True)

    # Train residual model
    print("\nTraining residual correction model...", flush=True)

    train_data_res = lgb.Dataset(X_train, label=residuals_train)
    val_data_res = lgb.Dataset(X_val, label=residuals_val, reference=train_data_res)

    model_residual = lgb.train(
        params_log,
        train_data_res,
        num_boost_round=500,
        valid_sets=[val_data_res],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    )

    # Combine predictions
    residual_correction = model_residual.predict(X_val, num_iteration=model_residual.best_iteration)
    y_val_pred_residual = y_val_pred_base + residual_correction

    rmse_residual = np.sqrt(mean_squared_error(y_val, y_val_pred_residual))
    mae_residual = mean_absolute_error(y_val, y_val_pred_residual)
    r2_residual = r2_score(y_val, y_val_pred_residual)

    print(f"\n✓ Residual Boosting Results:", flush=True)
    print(f"  RMSE: {rmse_residual:.2f} kg", flush=True)
    print(f"  MAE:  {mae_residual:.2f} kg", flush=True)
    print(f"  R²:   {r2_residual:.4f}", flush=True)
    print(f"  vs Baseline: {baseline_rmse - rmse_residual:+.2f} kg ({((baseline_rmse - rmse_residual) / baseline_rmse * 100):+.1f}%)", flush=True)

    results['residual_boosting'] = {
        'rmse': float(rmse_residual),
        'mae': float(mae_residual),
        'r2': float(r2_residual),
        'improvement': float(baseline_rmse - rmse_residual)
    }

except Exception as e:
    print(f"⚠️  Could not load Phase 5 model: {e}", flush=True)
    results['residual_boosting'] = {'error': str(e)}


# ============================================================================
#  APPROACH 4: XGBOOST WITH HUBER LOSS
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[5/7] Approach 4: XGBoost with Huber Loss", flush=True)
print("=" * 80, flush=True)
print("Rationale: Huber loss more robust to outliers", flush=True)

print("\nTraining XGBoost...", flush=True)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params_xgb = {
    'objective': 'reg:pseudohubererror',  # Huber loss (correct name)
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'min_child_weight': 50,
    'alpha': 0.1,
    'lambda': 0.1,
    'seed': RANDOM_STATE
}

evals = [(dval, 'validation')]
model_xgb = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=1500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

y_val_pred_xgb = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration + 1))

rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
r2_xgb = r2_score(y_val, y_val_pred_xgb)

print(f"\n✓ XGBoost (Huber) Results:", flush=True)
print(f"  RMSE: {rmse_xgb:.2f} kg", flush=True)
print(f"  MAE:  {mae_xgb:.2f} kg", flush=True)
print(f"  R²:   {r2_xgb:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_xgb:+.2f} kg ({((baseline_rmse - rmse_xgb) / baseline_rmse * 100):+.1f}%)", flush=True)

results['xgboost_huber'] = {
    'rmse': float(rmse_xgb),
    'mae': float(mae_xgb),
    'r2': float(r2_xgb),
    'improvement': float(baseline_rmse - rmse_xgb)
}


# ============================================================================
#  APPROACH 5: POLYNOMIAL FEATURES (Top Features Only)
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[6/7] Approach 5: Polynomial Features", flush=True)
print("=" * 80, flush=True)
print("Rationale: Capture higher-order interactions", flush=True)

# Select top features for polynomial expansion
top_features = [
    'interval_duration_min',
    'distance_covered_km',
    'aircraft_type_target_enc',
    'groundspeed_mean',
    'altitude_mean',
    'vertical_rate_mean'
]

top_features_present = [f for f in top_features if f in X_train.columns]
print(f"\nCreating degree-2 polynomials for {len(top_features_present)} top features...", flush=True)

X_train_top = X_train[top_features_present]
X_val_top = X_val[top_features_present]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_top)
X_val_poly = poly.transform(X_val_top)

# Combine with original features
X_train_combined = np.hstack([X_train.values, X_train_poly])
X_val_combined = np.hstack([X_val.values, X_val_poly])

print(f"Combined feature count: {X_train_combined.shape[1]}", flush=True)
print("\nTraining LightGBM with polynomial features...", flush=True)

train_data_poly = lgb.Dataset(X_train_combined, label=y_train)
val_data_poly = lgb.Dataset(X_val_combined, label=y_val, reference=train_data_poly)

model_poly = lgb.train(
    params_log,
    train_data_poly,
    num_boost_round=1000,
    valid_sets=[val_data_poly],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

y_val_pred_poly = model_poly.predict(X_val_combined, num_iteration=model_poly.best_iteration)

rmse_poly = np.sqrt(mean_squared_error(y_val, y_val_pred_poly))
mae_poly = mean_absolute_error(y_val, y_val_pred_poly)
r2_poly = r2_score(y_val, y_val_pred_poly)

print(f"\n✓ Polynomial Features Results:", flush=True)
print(f"  RMSE: {rmse_poly:.2f} kg", flush=True)
print(f"  MAE:  {mae_poly:.2f} kg", flush=True)
print(f"  R²:   {r2_poly:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_poly:+.2f} kg ({((baseline_rmse - rmse_poly) / baseline_rmse * 100):+.1f}%)", flush=True)

results['polynomial'] = {
    'rmse': float(rmse_poly),
    'mae': float(mae_poly),
    'r2': float(r2_poly),
    'improvement': float(baseline_rmse - rmse_poly)
}


# ============================================================================
#  APPROACH 6: SUPER ENSEMBLE (ALL APPROACHES)
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("[7/7] Approach 6: Super Ensemble", flush=True)
print("=" * 80, flush=True)
print("Combining all approaches...", flush=True)

# Collect all predictions
all_predictions = []
all_names = []

if 'log_transform' in results and results['log_transform'].get('rmse'):
    all_predictions.append(y_val_pred_log)
    all_names.append('LogTransform')

if 'two_stage' in results and results['two_stage'].get('rmse'):
    all_predictions.append(y_val_pred_twostage)
    all_names.append('TwoStage')

if 'residual_boosting' in results and results['residual_boosting'].get('rmse'):
    all_predictions.append(y_val_pred_residual)
    all_names.append('ResidualBoost')

if 'xgboost_huber' in results and results['xgboost_huber'].get('rmse'):
    all_predictions.append(y_val_pred_xgb)
    all_names.append('XGBoost')

if 'polynomial' in results and results['polynomial'].get('rmse'):
    all_predictions.append(y_val_pred_poly)
    all_names.append('Polynomial')

print(f"\nEnsembling {len(all_predictions)} approaches:", flush=True)
for name in all_names:
    print(f"  - {name}", flush=True)

# Simple average
super_ensemble_pred = np.mean(all_predictions, axis=0)

rmse_super = np.sqrt(mean_squared_error(y_val, super_ensemble_pred))
mae_super = mean_absolute_error(y_val, super_ensemble_pred)
r2_super = r2_score(y_val, super_ensemble_pred)

print(f"\n✓ Super Ensemble Results:", flush=True)
print(f"  RMSE: {rmse_super:.2f} kg", flush=True)
print(f"  MAE:  {mae_super:.2f} kg", flush=True)
print(f"  R²:   {r2_super:.4f}", flush=True)
print(f"  vs Baseline: {baseline_rmse - rmse_super:+.2f} kg ({((baseline_rmse - rmse_super) / baseline_rmse * 100):+.1f}%)", flush=True)

results['super_ensemble'] = {
    'rmse': float(rmse_super),
    'mae': float(mae_super),
    'r2': float(r2_super),
    'improvement': float(baseline_rmse - rmse_super),
    'components': all_names
}


# ============================================================================
#  FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("ALTERNATIVE APPROACHES SUMMARY", flush=True)
print("=" * 80, flush=True)

print(f"\n{'Approach':<25} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'vs Baseline'}", flush=True)
print("-" * 80, flush=True)
print(f"{'Phase 5 Baseline':<25} {baseline_rmse:<12.2f} {'---':<12} {'---':<10} {'---'}", flush=True)

for name, result in results.items():
    if 'rmse' in result:
        rmse_val = result['rmse']
        mae_val = result.get('mae', 0)
        r2_val = result.get('r2', 0)
        improvement = result['improvement']

        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {rmse_val:<12.2f} {mae_val:<12.2f} {r2_val:<10.4f} {improvement:+.2f} kg", flush=True)

print("=" * 80, flush=True)

# Find best approach
best_approach = min(
    [(name, r) for name, r in results.items() if 'rmse' in r],
    key=lambda x: x[1]['rmse']
)

print(f"\n✓ BEST APPROACH: {best_approach[0].replace('_', ' ').title()}", flush=True)
print(f"  RMSE: {best_approach[1]['rmse']:.2f} kg", flush=True)
print(f"  Improvement: {best_approach[1]['improvement']:+.2f} kg ({best_approach[1]['improvement'] / baseline_rmse * 100:+.1f}%)", flush=True)

if best_approach[1]['rmse'] < 50:
    print(f"\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉", flush=True)
elif best_approach[1]['rmse'] < 100:
    print(f"\n✓ Great progress! Under 100 kg", flush=True)
    print(f"  Gap to target: {best_approach[1]['rmse'] - 50:.2f} kg", flush=True)
elif best_approach[1]['rmse'] < baseline_rmse:
    print(f"\n✓ Improvement achieved!", flush=True)
    print(f"  Gap to target: {best_approach[1]['rmse'] - 50:.2f} kg", flush=True)
else:
    print(f"\n⚠️  No improvement from baseline", flush=True)

# Save results
results['baseline_rmse'] = baseline_rmse
results['best_approach'] = best_approach[0]
results['timestamp'] = datetime.now().isoformat()

with open(REPORTS_DIR / 'phase6_alternative_approaches_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ Results saved to phase6_alternative_approaches_results.json", flush=True)
print("=" * 80, flush=True)
