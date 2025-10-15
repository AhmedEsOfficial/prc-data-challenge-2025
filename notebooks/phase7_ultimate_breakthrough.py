"""
Phase 7: Ultimate Breakthrough - Push to < 50 kg RMSE

Current: 163.04 kg RMSE
Target: < 50 kg RMSE
Required: 69% reduction

Strategy:
1. Sequential/Temporal Features - Model interval dependencies within flights
2. Advanced Physics Features - Energy, momentum, aerodynamics
3. Multi-level Stratification - 4+ stage models by fuel, phase, aircraft
4. Advanced Neural Networks - Proper architecture with embeddings
5. Sophisticated Stacking - Ensemble with meta-learning
6. Aggressive Hyperparameter Tuning - Optuna with 500+ trials
"""

#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

print("="*80, flush=True)
print("PHASE 7: ULTIMATE BREAKTHROUGH TO < 50 KG RMSE", flush=True)
print("="*80, flush=True)
print(f"Start Time: {datetime.now()}", flush=True)
print(flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path('.')
CACHE_DIR = BASE_DIR / 'cache'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("Loading data...", flush=True)
X_train = pd.read_parquet(CACHE_DIR / 'X_train_clean.parquet')
X_val = pd.read_parquet(CACHE_DIR / 'X_val_clean.parquet')
y_train = pd.read_parquet(CACHE_DIR / 'y_train.parquet')['fuel_kg'].values
y_val = pd.read_parquet(CACHE_DIR / 'y_val.parquet')['fuel_kg'].values
train_ids = pd.read_parquet(CACHE_DIR / 'train_ids.parquet')
val_ids = pd.read_parquet(CACHE_DIR / 'val_ids.parquet')

print(f"Train shape: {X_train.shape}", flush=True)
print(f"Val shape: {X_val.shape}", flush=True)
print(f"Features: {X_train.shape[1]}", flush=True)
print(flush=True)

# ============================================================================
# STEP 2: ADVANCED SEQUENTIAL FEATURES
# ============================================================================

print("="*80)
print("CREATING ADVANCED SEQUENTIAL FEATURES")
print("="*80)

def add_sequential_features(X, ids_df, y=None, is_train=True):
    """
    Add features that capture temporal patterns within flights.
    Key insight: Fuel consumption in interval i depends on intervals i-1, i-2, etc.
    """
    X = X.copy()

    # Sort by flight_id and interval order
    if 'flight_id' in X.columns and 'start' in ids_df.columns:
        X['flight_id'] = ids_df['flight_id'].values
        X['start_time'] = ids_df['start'].values
        X = X.sort_values(['flight_id', 'start_time'])

        if y is not None:
            y_sorted = y[X.index]

        # === ROLLING FEATURES WITHIN FLIGHT ===

        # Previous interval duration
        X['prev_duration'] = X.groupby('flight_id')['interval_duration_min'].shift(1)
        X['prev_duration_delta'] = X['interval_duration_min'] - X['prev_duration']

        # Previous 3 intervals average duration
        X['prev3_duration_mean'] = X.groupby('flight_id')['interval_duration_min'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )

        # Cumulative distance in flight
        X['cumulative_distance'] = X.groupby('flight_id')['distance_covered_km'].cumsum()

        # Cumulative time in flight (approximation)
        X['cumulative_time'] = X.groupby('flight_id')['interval_duration_min'].cumsum()

        # Distance acceleration (rate of change of distance covered)
        X['distance_acceleration'] = X.groupby('flight_id')['distance_covered_km'].diff()

        # Previous altitude
        X['prev_altitude'] = X.groupby('flight_id')['altitude_mean'].shift(1)
        X['altitude_trend'] = X['altitude_mean'] - X['prev_altitude']

        # Previous speed
        X['prev_groundspeed'] = X.groupby('flight_id')['groundspeed_mean'].shift(1)
        X['speed_trend'] = X['groundspeed_mean'] - X['prev_groundspeed']

        # Interval number in flight (position)
        X['interval_number'] = X.groupby('flight_id').cumcount() + 1
        X['intervals_in_flight'] = X.groupby('flight_id')['flight_id'].transform('count')
        X['interval_position_ratio'] = X['interval_number'] / X['intervals_in_flight']

        # === PHASE DETECTION ===

        # Improved flight phase detection
        X['phase_climb'] = ((X['altitude_mean'] < 30000) & (X['vertical_rate_mean'] > 500)).astype(int)
        X['phase_cruise'] = ((X['altitude_mean'] >= 30000) & (np.abs(X['vertical_rate_mean']) < 500)).astype(int)
        X['phase_descent'] = ((X['altitude_mean'] < 30000) & (X['vertical_rate_mean'] < -500)).astype(int)
        X['phase_maneuvering'] = ((X['groundspeed_mean'] < 200) | (X['altitude_mean'] < 10000)).astype(int)

        # Phase transitions
        X['prev_phase_cruise'] = X.groupby('flight_id')['phase_cruise'].shift(1)
        X['phase_transition_to_cruise'] = ((X['phase_cruise'] == 1) & (X['prev_phase_cruise'] == 0)).astype(int)
        X['phase_transition_from_cruise'] = ((X['phase_cruise'] == 0) & (X['prev_phase_cruise'] == 1)).astype(int)

        # === FUEL-RELATED FEATURES (TRAIN ONLY) ===

        if is_train and y is not None:
            # Previous fuel consumption (LAG FEATURE)
            # CRITICAL: This captures fuel consumption patterns
            X['prev_fuel'] = pd.Series(y_sorted, index=X.index).groupby(X['flight_id']).shift(1)
            X['prev2_fuel'] = pd.Series(y_sorted, index=X.index).groupby(X['flight_id']).shift(2)
            X['prev3_fuel'] = pd.Series(y_sorted, index=X.index).groupby(X['flight_id']).shift(3)

            # Rolling average fuel (previous 3 intervals)
            X['prev3_fuel_mean'] = pd.Series(y_sorted, index=X.index).groupby(X['flight_id']).transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )

            # Fuel rate trend
            X['prev_fuel_rate'] = X['prev_fuel'] / X['prev_duration']

            # Cumulative fuel (helps predict remaining flight fuel needs)
            X['cumulative_fuel'] = pd.Series(y_sorted, index=X.index).groupby(X['flight_id']).cumsum()

        else:
            # For validation, we'll use a different strategy (predict iteratively)
            # For now, fill with NaN and handle later
            X['prev_fuel'] = np.nan
            X['prev2_fuel'] = np.nan
            X['prev3_fuel'] = np.nan
            X['prev3_fuel_mean'] = np.nan
            X['prev_fuel_rate'] = np.nan
            X['cumulative_fuel'] = np.nan

        # === ADVANCED PHYSICS FEATURES ===

        # Specific Air Range (SAR) - distance per fuel (proxy)
        # Using duration and distance as proxies
        X['sar_proxy'] = X['distance_covered_km'] / (X['interval_duration_min'] + 1)

        # Thrust-specific fuel consumption proxy
        # Higher at low altitudes and high speeds
        X['tsfc_proxy'] = X['groundspeed_mean'] / (X['altitude_mean'] + 1000)

        # Drag coefficient proxy (function of speed and altitude)
        # Drag = 0.5 * rho * v^2 * Cd * A
        # Rho decreases with altitude
        X['drag_proxy_v2'] = (X['groundspeed_mean'] ** 2) / (X['altitude_mean'] + 1000)

        # Energy state (kinetic + potential)
        # E = 0.5 * m * v^2 + m * g * h
        # Without mass, use normalized version
        KE_FACTOR = 0.5
        PE_FACTOR = 0.00981  # g = 9.81 m/s^2, convert to km

        X['kinetic_energy_proxy'] = KE_FACTOR * (X['groundspeed_mean'] ** 2)
        X['potential_energy_proxy'] = PE_FACTOR * X['altitude_mean']
        X['total_energy_proxy'] = X['kinetic_energy_proxy'] + X['potential_energy_proxy']

        # Energy change rate (power proxy)
        X['prev_total_energy'] = X.groupby('flight_id')['total_energy_proxy'].shift(1)
        X['energy_change_rate'] = (X['total_energy_proxy'] - X['prev_total_energy']) / (X['interval_duration_min'] + 0.01)

        # Work done (climbing)
        X['climb_work_rate'] = X['altitude_delta'] * X['vertical_rate_mean'] / (X['interval_duration_min'] + 0.01)

        # === ADVANCED INTERACTIONS ===

        # Distance-duration efficiency
        X['distance_per_duration'] = X['distance_covered_km'] / (X['interval_duration_min'] + 0.01)

        # Altitude-distance interaction
        X['altitude_distance_interaction'] = X['altitude_mean'] * X['distance_covered_km']

        # Speed-vertical rate interaction (indicates aggressive maneuvers)
        X['speed_vrate_interaction'] = X['groundspeed_mean'] * np.abs(X['vertical_rate_mean'])

        # Duration-altitude interaction
        X['duration_altitude_interaction'] = X['interval_duration_min'] * X['altitude_mean']

        # === STATISTICAL FEATURES ===

        # Coefficient of variation for speed
        X['speed_cv'] = X['groundspeed_std'] / (X['groundspeed_mean'] + 1)

        # Altitude variation intensity
        X['altitude_variation'] = X['altitude_std'] / (X['altitude_mean'] + 1)

        # Clean up
        X = X.drop(['flight_id', 'start_time'], axis=1, errors='ignore')

        if is_train and y is not None:
            # Return sorted y as well
            return X, y_sorted
        else:
            return X

    else:
        print("Warning: Cannot create sequential features without flight_id and start time")
        return X if not is_train else (X, y)

print("Adding sequential features to training set...")
X_train_seq, y_train_sorted = add_sequential_features(X_train, train_ids, y_train, is_train=True)

print("Adding sequential features to validation set...")
# For validation, we need a special approach for fuel lag features
X_val_seq = add_sequential_features(X_val, val_ids, y_val, is_train=False)

# Handle NaN values from sequential features
print("\nHandling missing values from sequential features...")
print(f"Train NaNs before: {X_train_seq.isna().sum().sum()}")
print(f"Val NaNs before: {X_val_seq.isna().sum().sum()}")

# Fill NaN for fuel lag features in validation (we'll predict iteratively later)
# For now, use median imputation
for col in X_train_seq.columns:
    if X_train_seq[col].isna().any():
        median_val = X_train_seq[col].median()
        X_train_seq[col].fillna(median_val, inplace=True)
        X_val_seq[col].fillna(median_val, inplace=True)

print(f"Train NaNs after: {X_train_seq.isna().sum().sum()}")
print(f"Val NaNs after: {X_val_seq.isna().sum().sum()}")

print(f"\nNew feature count: {X_train_seq.shape[1]} (was {X_train.shape[1]})")
print(f"Added {X_train_seq.shape[1] - X_train.shape[1]} sequential features")
print()

# Save feature names
feature_names = list(X_train_seq.columns)
with open(MODELS_DIR / 'phase7_feature_names.json', 'w') as f:
    json.dump({'features': feature_names, 'count': len(feature_names)}, f, indent=2)

# ============================================================================
# STEP 3: STRATIFIED MODELING - 4-STAGE APPROACH
# ============================================================================

print("="*80)
print("TRAINING 4-STAGE STRATIFIED MODELS")
print("="*80)

# Define fuel ranges for stratification
fuel_ranges = [
    ('very_low', 0, 200),
    ('low', 200, 600),
    ('medium', 600, 1500),
    ('high', 1500, 100000)
]

# LightGBM parameters (tuned for each stage)
base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

stage_params = {
    'very_low': {**base_params, 'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 6,
                 'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
                 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5},
    'low': {**base_params, 'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7,
            'min_child_samples': 30, 'reg_alpha': 0.05, 'reg_lambda': 0.05,
            'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5},
    'medium': {**base_params, 'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 8,
               'min_child_samples': 50, 'reg_alpha': 0.01, 'reg_lambda': 0.01,
               'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5},
    'high': {**base_params, 'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 8,
             'min_child_samples': 50, 'reg_alpha': 0.01, 'reg_lambda': 0.01,
             'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
}

# Train separate model for each fuel range
stage_models = {}
stage_results = {}

for stage_name, min_fuel, max_fuel in fuel_ranges:
    print(f"\n{'='*80}")
    print(f"Training {stage_name.upper()} model ({min_fuel}-{max_fuel} kg)")
    print(f"{'='*80}")

    # Filter data for this fuel range
    train_mask = (y_train_sorted >= min_fuel) & (y_train_sorted < max_fuel)
    val_mask = (y_val >= min_fuel) & (y_val < max_fuel)

    X_train_stage = X_train_seq[train_mask]
    y_train_stage = y_train_sorted[train_mask]
    X_val_stage = X_val_seq[val_mask]
    y_val_stage = y_val[val_mask]

    print(f"Train samples: {len(X_train_stage):,}")
    print(f"Val samples: {len(X_val_stage):,}")

    if len(X_train_stage) < 100:
        print(f"⚠️  Insufficient samples for {stage_name}, skipping...")
        continue

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_stage, label=y_train_stage)
    val_data = lgb.Dataset(X_val_stage, label=y_val_stage, reference=train_data)

    # Train model
    params = stage_params[stage_name]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # Predictions
    train_pred = model.predict(X_train_stage, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val_stage, num_iteration=model.best_iteration)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_stage, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_stage, val_pred))
    val_mae = mean_absolute_error(y_val_stage, val_pred)
    val_r2 = r2_score(y_val_stage, val_pred)

    print(f"\n{stage_name.upper()} Results:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE: {val_rmse:.2f} kg")
    print(f"  Val MAE: {val_mae:.2f} kg")
    print(f"  Val R²: {val_r2:.4f}")
    print(f"  Best iteration: {model.best_iteration}")

    # Save model and results
    stage_models[stage_name] = model
    stage_results[stage_name] = {
        'rmse': float(val_rmse),
        'mae': float(val_mae),
        'r2': float(val_r2),
        'train_samples': int(len(X_train_stage)),
        'val_samples': int(len(X_val_stage)),
        'best_iteration': int(model.best_iteration)
    }

    # Save model to disk
    model.save_model(str(MODELS_DIR / f'phase7_4stage_{stage_name}.txt'))

print("\n" + "="*80)
print("4-STAGE MODEL COMPLETE - EVALUATING ON FULL VALIDATION SET")
print("="*80)

# Predict on full validation set using appropriate model
val_predictions_4stage = np.zeros(len(y_val))

for stage_name, min_fuel, max_fuel in fuel_ranges:
    if stage_name not in stage_models:
        continue

    # For each sample, use model based on predicted fuel range
    # Strategy: Use predicted fuel from previous stage to decide which model to use
    # For first pass, use heuristics based on features

    # Simple heuristic: duration-based assignment
    # This is imperfect but necessary since we don't know y_val for routing
    duration_thresholds = {
        'very_low': (0, 8),      # < 8 min
        'low': (8, 15),          # 8-15 min
        'medium': (15, 25),      # 15-25 min
        'high': (25, 100)        # > 25 min
    }

    min_dur, max_dur = duration_thresholds[stage_name]

    # Use duration as proxy for fuel range
    if 'interval_duration_min' in X_val_seq.columns:
        stage_mask = (X_val_seq['interval_duration_min'] >= min_dur) & (X_val_seq['interval_duration_min'] < max_dur)
    else:
        # Fallback: split by position
        n_samples = len(X_val_seq)
        if stage_name == 'very_low':
            stage_mask = np.arange(n_samples) < n_samples * 0.4
        elif stage_name == 'low':
            stage_mask = (np.arange(n_samples) >= n_samples * 0.4) & (np.arange(n_samples) < n_samples * 0.7)
        elif stage_name == 'medium':
            stage_mask = (np.arange(n_samples) >= n_samples * 0.7) & (np.arange(n_samples) < n_samples * 0.9)
        else:
            stage_mask = np.arange(n_samples) >= n_samples * 0.9

    if stage_mask.sum() > 0:
        model = stage_models[stage_name]
        val_predictions_4stage[stage_mask] = model.predict(
            X_val_seq[stage_mask],
            num_iteration=model.best_iteration
        )

# Calculate overall metrics
rmse_4stage = np.sqrt(mean_squared_error(y_val, val_predictions_4stage))
mae_4stage = mean_absolute_error(y_val, val_predictions_4stage)
r2_4stage = r2_score(y_val, val_predictions_4stage)

print(f"\n4-STAGE MODEL - OVERALL VALIDATION METRICS:")
print(f"  RMSE: {rmse_4stage:.2f} kg")
print(f"  MAE: {mae_4stage:.2f} kg")
print(f"  R²: {r2_4stage:.4f}")
print()

# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================

results = {
    'timestamp': str(datetime.now()),
    'baseline_rmse': 163.04,
    'four_stage': {
        'overall_rmse': float(rmse_4stage),
        'overall_mae': float(mae_4stage),
        'overall_r2': float(r2_4stage),
        'improvement_kg': float(163.04 - rmse_4stage),
        'improvement_pct': float((163.04 - rmse_4stage) / 163.04 * 100),
        'stage_results': stage_results
    },
    'features_used': len(feature_names),
    'sequential_features_added': X_train_seq.shape[1] - X_train.shape[1]
}

# Save results
with open(REPORTS_DIR / 'phase7_initial_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("PHASE 7 - INITIAL RESULTS SAVED")
print("="*80)
print(f"4-Stage RMSE: {rmse_4stage:.2f} kg")
print(f"Baseline: 163.04 kg")
print(f"Improvement: {163.04 - rmse_4stage:.2f} kg ({(163.04 - rmse_4stage)/163.04*100:.1f}%)")
print(f"Target: 50 kg")
print(f"Remaining gap: {rmse_4stage - 50:.2f} kg")
print("="*80)
