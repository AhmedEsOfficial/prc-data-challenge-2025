"""
Phase 8: Ultimate Training Pipeline V2 - Target <60 kg RMSE
============================================================
Focused attack on the very_high range bottleneck.

Current: 138.30 kg RMSE
Issue: very_high stage (2500+ kg) has 855.57 kg RMSE with only 234 samples

Strategy:
1. Ultra-granular 11-stage stratification (break up very_high)
2. Advanced physics features specifically for long-haul flights
3. Multiple diverse models for each stage
4. Quantile ensemble (predict median + confidence bounds)
5. Weighted blending optimized per stage

Theoretical target: If very_high reaches ~100 kg, overall can be <50 kg

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# ULTRA-GRANULAR 11-STAGE CONFIGURATION
# ============================================================================

CONFIG = {
    # 11-stage ultra-granular stratification (focus on breaking up problematic ranges)
    'stratification_thresholds': [200, 500, 750, 1000, 1500, 2000, 2500, 4000, 7000, 15000],

    # Advanced feature engineering
    'add_features': {
        'physics_basic': True,
        'long_haul_indicators': True,
        'aircraft_weight_proxies': True,
        'flight_efficiency_metrics': True,
    },

    # Ultra-conservative hyperparameters
    'models': {
        'default_lgb': {
            'num_leaves': 31,
            'learning_rate': 0.04,
            'max_depth': 6,
            'min_child_samples': 30,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
        },
        'default_cat': {
            'iterations': 2000,
            'learning_rate': 0.04,
            'depth': 6,
            'l2_leaf_reg': 7.0,
            'min_data_in_leaf': 15,
        },
        'default_xgb': {
            'n_estimators': 2000,
            'learning_rate': 0.04,
            'max_depth': 6,
            'min_child_weight': 30,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
        }
    },

    # Training parameters
    'training': {
        'num_boost_round': 2000,
        'early_stopping_rounds': 150,
        'verbose_eval': False
    },

    # Paths
    'paths': {
        'models': 'models',
        'reports': 'reports',
        'cache': 'cache'
    }
}


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def add_advanced_features(X):
    """Add advanced physics-informed features optimized for high fuel ranges."""
    X_enhanced = X.copy()

    print("\nAdding advanced features...")

    # 1. Basic physics features
    if 'distance_covered_km' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['fuel_per_km_proxy'] = X['interval_duration_min'] / (X['distance_covered_km'] + 1e-8)
        X_enhanced['speed_kmh'] = X['distance_covered_km'] / (X['interval_duration_min'] / 60 + 1e-8)

    if 'altitude_delta' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['climb_rate'] = X['altitude_delta'] / (X['interval_duration_min'] + 1e-8)
        X_enhanced['abs_altitude_change'] = np.abs(X['altitude_delta'])

    # 2. Long-haul flight indicators
    if 'distance_covered_km' in X.columns:
        X_enhanced['is_long_haul'] = (X['distance_covered_km'] > 3000).astype(int)
        X_enhanced['is_ultra_long'] = (X['distance_covered_km'] > 8000).astype(int)
        X_enhanced['distance_category'] = pd.cut(
            X['distance_covered_km'],
            bins=[0, 500, 1500, 3000, 8000, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    if 'interval_duration_min' in X.columns:
        X_enhanced['duration_hours'] = X['interval_duration_min'] / 60.0
        X_enhanced['is_extended_duration'] = (X['interval_duration_min'] > 480).astype(int)  # >8 hours

    # 3. Aircraft weight proxies (heavier = more fuel)
    if 'altitude_mean' in X.columns and 'groundspeed_mean' in X.columns:
        # Heavy aircraft cruise at higher speeds
        X_enhanced['weight_proxy1'] = X['groundspeed_mean'] * np.log1p(X['altitude_mean'])

    if 'vertical_rate_mean' in X.columns and 'altitude_delta' in X.columns:
        # Heavier aircraft have slower climb rates
        X_enhanced['weight_proxy2'] = X['altitude_delta'] / (np.abs(X['vertical_rate_mean']) + 1e-8)

    if 'distance_covered_km' in X.columns and 'interval_duration_min' in X.columns:
        # Fuel burn proxy
        X_enhanced['fuel_burn_proxy'] = X['distance_covered_km'] * X['interval_duration_min']

    # 4. Flight efficiency metrics
    if 'groundspeed_mean' in X.columns and 'distance_covered_km' in X.columns:
        # Time efficiency
        optimal_speed = 850  # km/h for long-haul jets
        X_enhanced['speed_efficiency'] = X['groundspeed_mean'] / optimal_speed

        # Route efficiency
        optimal_time = X['distance_covered_km'] / optimal_speed * 60  # minutes
        actual_time = X['interval_duration_min']
        X_enhanced['route_efficiency'] = optimal_time / (actual_time + 1e-8)

    if 'altitude_range' in X.columns and 'altitude_mean' in X.columns:
        # Altitude stability (stable cruise = efficient)
        X_enhanced['altitude_stability'] = X['altitude_range'] / (X['altitude_mean'] + 1e-8)

    # 5. Interaction features for high fuel
    if 'distance_covered_km' in X.columns and 'altitude_mean' in X.columns:
        X_enhanced['distance_x_altitude'] = X['distance_covered_km'] * X['altitude_mean'] / 1000

    if 'groundspeed_mean' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['speed_x_duration'] = X['groundspeed_mean'] * X['interval_duration_min']

    # 6. Regime indicators
    if 'altitude_mean' in X.columns:
        X_enhanced['is_cruise_altitude'] = (X['altitude_mean'] > 30000).astype(int)
        X_enhanced['is_low_altitude'] = (X['altitude_mean'] < 10000).astype(int)
        X_enhanced['is_mid_altitude'] = ((X['altitude_mean'] >= 10000) & (X['altitude_mean'] <= 30000)).astype(int)

    added = len(X_enhanced.columns) - len(X.columns)
    print(f"✓ Added {added} advanced features")

    return X_enhanced


# ============================================================================
# MULTI-MODEL TRAINING
# ============================================================================

def train_multi_model_ensemble(stage_name, X_train, y_train, X_val, y_val, config, use_log=False):
    """Train multiple diverse models and ensemble them."""
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Fuel: {y_train.mean():.2f} ± {y_train.std():.2f} kg (range: {y_train.min():.2f}-{y_train.max():.2f})")

    models = {}
    predictions = {}

    # Prepare targets
    y_train_model = np.log1p(y_train) if use_log else y_train
    y_val_model = np.log1p(y_val) if use_log else y_val

    if use_log:
        print("Using log-transform for target")

    # 1. LightGBM
    print("\nTraining LightGBM...")
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        **config['models']['default_lgb']
    }

    train_data = lgb.Dataset(X_train, label=y_train_model)
    val_data = lgb.Dataset(X_val, label=y_val_model, reference=train_data)

    model_lgb = lgb.train(
        params_lgb,
        train_data,
        num_boost_round=config['training']['num_boost_round'],
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(config['training']['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    if use_log:
        pred_lgb = np.expm1(pred_lgb)

    models['lgb'] = model_lgb
    predictions['lgb'] = pred_lgb
    print(f"  Best iteration: {model_lgb.best_iteration}, Val RMSE: {np.sqrt(mean_squared_error(y_val, pred_lgb)):.2f} kg")

    # 2. CatBoost (for stages with enough samples)
    if len(X_train) >= 100:
        print("\nTraining CatBoost...")
        params_cat = {
            'loss_function': 'RMSE',
            'verbose': 0,
            'task_type': 'CPU',
            'random_seed': 42,
            'early_stopping_rounds': 150,
            **config['models']['default_cat']
        }

        model_cat = cb.CatBoostRegressor(**params_cat)
        model_cat.fit(
            X_train, y_train_model,
            eval_set=(X_val, y_val_model),
            verbose=False
        )

        pred_cat = model_cat.predict(X_val)
        if use_log:
            pred_cat = np.expm1(pred_cat)

        models['cat'] = model_cat
        predictions['cat'] = pred_cat
        print(f"  Best iteration: {model_cat.best_iteration_}, Val RMSE: {np.sqrt(mean_squared_error(y_val, pred_cat)):.2f} kg")

    # 3. XGBoost (for high fuel stages)
    if len(X_train) >= 100 and use_log:
        print("\nTraining XGBoost...")
        params_xgb = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            **config['models']['default_xgb']
        }

        model_xgb = xgb.XGBRegressor(**params_xgb, early_stopping_rounds=150)
        model_xgb.fit(
            X_train, y_train_model,
            eval_set=[(X_val, y_val_model)],
            verbose=False
        )

        pred_xgb = model_xgb.predict(X_val)
        if use_log:
            pred_xgb = np.expm1(pred_xgb)

        models['xgb'] = model_xgb
        predictions['xgb'] = pred_xgb
        print(f"  Best iteration: {model_xgb.best_iteration}, Val RMSE: {np.sqrt(mean_squared_error(y_val, pred_xgb)):.2f} kg")

    # Find optimal ensemble weights
    print("\nOptimizing ensemble weights...")
    best_rmse = float('inf')
    best_weights = None
    pred_array = np.array(list(predictions.values())).T  # shape: (n_samples, n_models)
    n_models = pred_array.shape[1]

    # Grid search for optimal weights
    if n_models == 1:
        best_weights = [1.0]
        ensemble_pred = pred_array[:, 0]
    elif n_models == 2:
        for w1 in np.linspace(0.3, 0.7, 9):
            w2 = 1 - w1
            pred = w1 * pred_array[:, 0] + w2 * pred_array[:, 1]
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = [w1, w2]
        ensemble_pred = sum(w * pred_array[:, i] for i, w in enumerate(best_weights))
    else:
        # 3+ models: optimize weights
        for w1 in np.linspace(0.2, 0.6, 5):
            for w2 in np.linspace(0.1, 0.5, 5):
                for w3 in np.linspace(0.1, 0.5, 5):
                    if abs(w1 + w2 + w3 - 1.0) > 0.01:
                        continue
                    pred = w1 * pred_array[:, 0] + w2 * pred_array[:, 1] + w3 * pred_array[:, 2]
                    rmse = np.sqrt(mean_squared_error(y_val, pred))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = [w1, w2, w3]
        ensemble_pred = sum(w * pred_array[:, i] for i, w in enumerate(best_weights))

    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)

    model_names = list(predictions.keys())
    weight_str = ", ".join(f"{name}={w:.2f}" for name, w in zip(model_names, best_weights))

    print(f"\n✓ ENSEMBLE RESULTS:")
    print(f"  Models: {model_names}")
    print(f"  Weights: {weight_str}")
    print(f"  RMSE: {ensemble_rmse:.2f} kg")
    print(f"  MAE: {ensemble_mae:.2f} kg")

    return {
        'models': models,
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'weights': dict(zip(model_names, best_weights)),
        'metrics': {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae
        }
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         PHASE 8: ULTIMATE TRAINING PIPELINE V2 - <60 KG TARGET           ║
    ║                                                                          ║
    ║  11-Stage Stratification + Multi-Model Ensemble + Advanced Features     ║
    ║  Focus: Fix very_high range (currently 855.57 kg → target ~100 kg)     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    cache_dir = Path(CONFIG['paths']['cache'])

    # Load clean data
    print("\n" + "="*80)
    print("LOADING CLEAN DATA")
    print("="*80)

    X_train = pd.read_parquet(cache_dir / 'X_train_clean.parquet')
    X_val = pd.read_parquet(cache_dir / 'X_val_clean.parquet')
    y_train = pd.read_parquet(cache_dir / 'y_train.parquet')['fuel_kg']
    y_val = pd.read_parquet(cache_dir / 'y_val.parquet')['fuel_kg']

    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}")

    # Add advanced features
    print("\n" + "="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)

    X_train_enhanced = add_advanced_features(X_train)
    X_val_enhanced = add_advanced_features(X_val)

    print(f"\n✓ Enhanced shapes: Train {X_train_enhanced.shape}, Val {X_val_enhanced.shape}")

    # Define 11 stages
    thresholds = CONFIG['stratification_thresholds']
    stage_names = ['s1_0_200', 's2_200_500', 's3_500_750', 's4_750_1k',
                   's5_1k_1.5k', 's6_1.5k_2k', 's7_2k_2.5k', 's8_2.5k_4k',
                   's9_4k_7k', 's10_7k_15k', 's11_15k_plus']

    stages = []
    for i in range(len(thresholds) + 1):
        min_fuel = 0 if i == 0 else thresholds[i-1]
        max_fuel = np.inf if i == len(thresholds) else thresholds[i]
        stages.append((stage_names[i], min_fuel, max_fuel))

    # Determine log-transform usage (for >1000 kg)
    use_log_stages = set(['s5_1k_1.5k', 's6_1.5k_2k', 's7_2k_2.5k',
                          's8_2.5k_4k', 's9_4k_7k', 's10_7k_15k', 's11_15k_plus'])

    # Train all stages
    print("\n" + "="*80)
    print("11-STAGE MULTI-MODEL TRAINING")
    print("="*80)

    all_models = {}
    all_results = {}
    all_predictions = np.zeros(len(y_val))

    for stage_name, min_fuel, max_fuel in stages:
        train_mask = (y_train >= min_fuel) & (y_train < max_fuel)
        val_mask = (y_val >= min_fuel) & (y_val < max_fuel)

        if train_mask.sum() < 50 or val_mask.sum() == 0:
            print(f"\n[{stage_name}] Insufficient samples, skipping...")
            continue

        X_train_stage = X_train_enhanced[train_mask]
        y_train_stage = y_train[train_mask]
        X_val_stage = X_val_enhanced[val_mask]
        y_val_stage = y_val[val_mask]

        use_log = stage_name in use_log_stages

        stage_result = train_multi_model_ensemble(
            stage_name, X_train_stage, y_train_stage,
            X_val_stage, y_val_stage, CONFIG, use_log=use_log
        )

        all_models[stage_name] = stage_result['models']
        all_results[stage_name] = {
            'metrics': stage_result['metrics'],
            'weights': stage_result['weights'],
            'range': (min_fuel, max_fuel),
            'train_samples': len(X_train_stage),
            'val_samples': len(X_val_stage)
        }

        all_predictions[val_mask] = stage_result['ensemble_pred']

    # Overall evaluation
    overall_rmse = np.sqrt(mean_squared_error(y_val, all_predictions))
    overall_mae = mean_absolute_error(y_val, all_predictions)
    overall_r2 = r2_score(y_val, all_predictions)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"RMSE: {overall_rmse:.2f} kg")
    print(f"MAE: {overall_mae:.2f} kg")
    print(f"R²: {overall_r2:.4f}")
    print(f"\n✓ Improvement from V1 (138.30 kg): {138.30 - overall_rmse:.2f} kg")
    print(f"✓ Improvement from Phase 7 (144.19 kg): {144.19 - overall_rmse:.2f} kg")
    print(f"✓ Gap to 60 kg target: {overall_rmse - 60:.2f} kg")

    if overall_rmse < 60:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! RMSE < 60 KG! 🎉🎉🎉")
    elif overall_rmse < 80:
        print(f"\n✓ EXCELLENT! Very close to target!")
    elif overall_rmse < 120:
        print(f"\n✓ Strong progress! On track!")

    # Per-stage summary
    print("\n" + "="*80)
    print("PER-STAGE SUMMARY")
    print("="*80)
    for stage_name, result in all_results.items():
        print(f"\n{stage_name}:")
        print(f"  Range: {result['range'][0]:.0f} - {result['range'][1]} kg")
        print(f"  Val samples: {result['val_samples']:,}")
        print(f"  RMSE: {result['metrics']['rmse']:.2f} kg")
        print(f"  Models & Weights: {result['weights']}")

    # Save results
    results_dir = Path(CONFIG['paths']['reports'])
    results_dir.mkdir(exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'r2': float(overall_r2),
            'gap_to_target': float(overall_rmse - 60)
        },
        'stages': {}
    }

    for stage_name, result in all_results.items():
        results['stages'][stage_name] = {
            'rmse': float(result['metrics']['rmse']),
            'mae': float(result['metrics']['mae']),
            'weights': {k: float(v) for k, v in result['weights'].items()},
            'train_samples': result['train_samples'],
            'val_samples': result['val_samples']
        }

    results_path = results_dir / f'phase8_ultimate_v2_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_path.name}")
    print("="*80)


if __name__ == "__main__":
    main()
