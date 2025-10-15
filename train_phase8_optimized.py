"""
Phase 8: Ultra-Optimized Training Pipeline - Target <60 kg RMSE
===============================================================
Based on Phase 7 learnings (144.19 kg) with targeted improvements.

Key Strategy:
1. Start from clean features (no over-preprocessing)
2. Add minimal physics-informed features
3. More granular 8-stage stratification
4. Log-transform for high fuel ranges (>1000 kg)
5. Conservative hyperparameters with strong regularization
6. Multi-model ensemble (LightGBM + CatBoost + Log-transform)
7. Optimal blending weights

Target: <60 kg RMSE (vs current 224.90 kg, Phase 7: 144.19 kg)

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ULTRA-OPTIMIZED
# ============================================================================

CONFIG = {
    # 8-stage ultra-granular stratification
    'stratification_thresholds': [200, 500, 750, 1000, 1500, 2000, 2500],

    # Physics-informed feature engineering (minimal, high-value only)
    'add_features': {
        'fuel_efficiency': True,  # fuel per km
        'climb_efficiency': True,  # fuel per altitude change
        'speed_time_ratio': True,  # actual time vs optimal
        'load_proxy': True,  # inferred from fuel and distance
    },

    # Model configurations per stage (conservative, strong regularization)
    'models': {
        'very_low': {  # 0-200 kg
            'lgb': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_samples': 50,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
            }
        },
        'low': {  # 200-500 kg
            'lgb': {
                'num_leaves': 40,
                'learning_rate': 0.05,
                'max_depth': 7,
                'min_child_samples': 40,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.15,
                'reg_lambda': 0.15,
            }
        },
        'medium_low': {  # 500-750 kg
            'lgb': {
                'num_leaves': 40,
                'learning_rate': 0.04,
                'max_depth': 7,
                'min_child_samples': 35,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.15,
                'reg_lambda': 0.15,
            }
        },
        'medium': {  # 750-1000 kg
            'lgb': {
                'num_leaves': 40,
                'learning_rate': 0.04,
                'max_depth': 7,
                'min_child_samples': 30,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
            },
            'use_log_transform': True  # Log for better variance handling
        },
        'medium_high': {  # 1000-1500 kg
            'lgb': {
                'num_leaves': 35,
                'learning_rate': 0.04,
                'max_depth': 7,
                'min_child_samples': 25,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.25,
                'reg_lambda': 0.25,
            },
            'use_log_transform': True
        },
        'high': {  # 1500-2000 kg
            'lgb': {
                'num_leaves': 35,
                'learning_rate': 0.03,
                'max_depth': 6,
                'min_child_samples': 20,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            },
            'cat': {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 7,
                'l2_leaf_reg': 5.0,
                'min_data_in_leaf': 20,
            },
            'use_log_transform': True
        },
        'high_plus': {  # 2000-2500 kg
            'lgb': {
                'num_leaves': 30,
                'learning_rate': 0.03,
                'max_depth': 6,
                'min_child_samples': 15,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.35,
                'reg_lambda': 0.35,
            },
            'cat': {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 8,
                'l2_leaf_reg': 7.0,
                'min_data_in_leaf': 15,
            },
            'use_log_transform': True
        },
        'very_high': {  # 2500+ kg
            'lgb': {
                'num_leaves': 30,
                'learning_rate': 0.03,
                'max_depth': 6,
                'min_child_samples': 10,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.4,
                'reg_lambda': 0.4,
            },
            'cat': {
                'iterations': 2000,
                'learning_rate': 0.04,
                'depth': 8,
                'l2_leaf_reg': 10.0,
                'min_data_in_leaf': 10,
            },
            'use_log_transform': True
        }
    },

    # Ensemble weights per stage
    'ensemble_weights': {
        'very_low': {'lgb': 1.0},
        'low': {'lgb': 1.0},
        'medium_low': {'lgb': 1.0},
        'medium': {'lgb': 1.0},
        'medium_high': {'lgb': 1.0},
        'high': {'lgb': 0.6, 'cat': 0.4},
        'high_plus': {'lgb': 0.5, 'cat': 0.5},
        'very_high': {'lgb': 0.4, 'cat': 0.6}
    },

    # Training parameters
    'training': {
        'num_boost_round': 2000,
        'early_stopping_rounds': 100,
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
# PHYSICS-INFORMED FEATURE ENGINEERING
# ============================================================================

def add_physics_features(X):
    """Add minimal, high-value physics-informed features."""
    X_enhanced = X.copy()

    print("\nAdding physics-informed features...")

    # 1. Fuel efficiency: fuel per km (if we have fuel predictions, skip for now)
    if 'distance_covered_km' in X.columns:
        # Inverse efficiency - higher means more fuel needed per km
        X_enhanced['fuel_per_km_proxy'] = X['interval_duration_min'] / (X['distance_covered_km'] + 1e-8)

    # 2. Climb efficiency: work done climbing
    if 'altitude_delta' in X.columns and 'interval_duration_min' in X.columns:
        # Climb rate
        X_enhanced['climb_rate'] = X['altitude_delta'] / (X['interval_duration_min'] + 1e-8)
        # Absolute climb work
        X_enhanced['abs_altitude_change'] = np.abs(X['altitude_delta'])

    # 3. Speed efficiency
    if 'groundspeed_mean' in X.columns and 'distance_covered_km' in X.columns:
        # Optimal time for distance
        optimal_time = X['distance_covered_km'] / (550 + 1e-8)  # Assume optimal ~550 km/h cruise
        actual_time = X['interval_duration_min'] / 60.0  # Convert to hours
        X_enhanced['time_efficiency'] = actual_time / (optimal_time + 1e-8)

    # 4. Load proxy (heavier aircraft = more fuel)
    if 'altitude_mean' in X.columns and 'groundspeed_mean' in X.columns:
        # Higher speed at lower altitude suggests heavier aircraft
        X_enhanced['load_proxy'] = X['groundspeed_mean'] * (10000 / (X['altitude_mean'] + 1e-8))

    # 5. Flight regime indicators
    if 'altitude_mean' in X.columns:
        X_enhanced['is_cruise_altitude'] = (X['altitude_mean'] > 30000).astype(int)
        X_enhanced['is_low_altitude'] = (X['altitude_mean'] < 10000).astype(int)

    added = len(X_enhanced.columns) - len(X.columns)
    print(f"✓ Added {added} physics-informed features")
    print(f"  New features: {list(X_enhanced.columns[len(X.columns):][:10])}")

    return X_enhanced


# ============================================================================
# STAGE HELPER FUNCTIONS
# ============================================================================

def get_stage_info(thresholds):
    """Get stage names and ranges."""
    stages = []
    stage_names = ['very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'high_plus', 'very_high']

    for i in range(len(thresholds) + 1):
        min_fuel = 0 if i == 0 else thresholds[i-1]
        max_fuel = np.inf if i == len(thresholds) else thresholds[i]
        stage_name = stage_names[i] if i < len(stage_names) else f'stage_{i}'
        stages.append((stage_name, min_fuel, max_fuel))

    return stages


def get_stage_mask(y, min_fuel, max_fuel):
    """Get boolean mask for samples in fuel range."""
    return (y >= min_fuel) & (y < max_fuel)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_stage_model(stage_name, X_train, y_train, X_val, y_val, config):
    """Train models for a specific stage."""
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
    print(f"Fuel range: {y_train.min():.2f} - {y_train.max():.2f} kg")
    print(f"Fuel mean: {y_train.mean():.2f} ± {y_train.std():.2f} kg")

    stage_config = config['models'].get(stage_name, config['models']['low'])
    use_log = stage_config.get('use_log_transform', False)

    stage_models = {}
    stage_predictions = {}

    # Prepare targets (log-transform if specified)
    y_train_model = np.log1p(y_train) if use_log else y_train
    y_val_model = np.log1p(y_val) if use_log else y_val

    if use_log:
        print("Using log-transform for target")

    # Train LightGBM
    if 'lgb' in stage_config:
        print(f"\nTraining LightGBM...")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            **stage_config['lgb']
        }

        train_data = lgb.Dataset(X_train, label=y_train_model)
        val_data = lgb.Dataset(X_val, label=y_val_model, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=config['training']['num_boost_round'],
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(config['training']['early_stopping_rounds'], verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        stage_models['lgb'] = model

        # Predict (inverse transform if needed)
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        if use_log:
            pred = np.expm1(pred)
        stage_predictions['lgb'] = pred

        train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        if use_log:
            train_pred = np.expm1(train_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, pred))
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Train RMSE: {train_rmse:.2f} kg, Val RMSE: {val_rmse:.2f} kg")

    # Train CatBoost (for high fuel stages)
    if 'cat' in stage_config:
        print(f"\nTraining CatBoost...")

        cat_params = {
            'loss_function': 'RMSE',
            'verbose': 0,
            'task_type': 'CPU',
            'random_seed': 42,
            'early_stopping_rounds': 100,
            **stage_config['cat']
        }

        model = cb.CatBoostRegressor(**cat_params)
        model.fit(
            X_train, y_train_model,
            eval_set=(X_val, y_val_model),
            verbose=False
        )

        stage_models['cat'] = model

        # Predict (inverse transform if needed)
        pred = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred)
        stage_predictions['cat'] = pred

        train_pred = model.predict(X_train)
        if use_log:
            train_pred = np.expm1(train_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, pred))
        print(f"  Best iteration: {model.best_iteration_}")
        print(f"  Train RMSE: {train_rmse:.2f} kg, Val RMSE: {val_rmse:.2f} kg")

    # Create ensemble prediction
    weights = config['ensemble_weights'].get(stage_name, {'lgb': 1.0})
    ensemble_pred = np.zeros(len(y_val))

    for model_name, weight in weights.items():
        if model_name in stage_predictions:
            ensemble_pred += weight * stage_predictions[model_name]

    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_r2 = r2_score(y_val, ensemble_pred)

    print(f"\n✓ ENSEMBLE RESULTS:")
    print(f"  Weights: {weights}")
    print(f"  RMSE: {ensemble_rmse:.2f} kg")
    print(f"  MAE: {ensemble_mae:.2f} kg")
    print(f"  R²: {ensemble_r2:.4f}")

    return {
        'models': stage_models,
        'predictions': stage_predictions,
        'ensemble_pred': ensemble_pred,
        'metrics': {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        }
    }


def main():
    """Main training pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              PHASE 8: ULTRA-OPTIMIZED TRAINING PIPELINE                  ║
    ║                                                                          ║
    ║  Based on Phase 7 learnings + Physics-informed features                 ║
    ║  Target: < 60 kg RMSE (from 224.90 kg current, 144.19 kg Phase 7)      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    cache_dir = Path(CONFIG['paths']['cache'])

    # Load CLEAN data (no over-preprocessing)
    print("\n" + "="*80)
    print("LOADING CLEAN DATA (Phase 7 Format)")
    print("="*80)

    X_train = pd.read_parquet(cache_dir / 'X_train_clean.parquet')
    X_val = pd.read_parquet(cache_dir / 'X_val_clean.parquet')
    y_train = pd.read_parquet(cache_dir / 'y_train.parquet')['fuel_kg']
    y_val = pd.read_parquet(cache_dir / 'y_val.parquet')['fuel_kg']

    print(f"✓ Train: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"✓ Val: X_val {X_val.shape}, y_val {y_val.shape}")

    # Add physics-informed features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    X_train_enhanced = add_physics_features(X_train)
    X_val_enhanced = add_physics_features(X_val)

    print(f"\n✓ Enhanced shapes: Train {X_train_enhanced.shape}, Val {X_val_enhanced.shape}")

    # Train all stages
    print("\n" + "="*80)
    print("8-STAGE STRATIFIED TRAINING")
    print("="*80)

    stages = get_stage_info(CONFIG['stratification_thresholds'])
    all_models = {}
    all_results = {}
    all_predictions = np.zeros(len(y_val))

    for stage_name, min_fuel, max_fuel in stages:
        # Get stage data
        train_mask = get_stage_mask(y_train, min_fuel, max_fuel)
        val_mask = get_stage_mask(y_val, min_fuel, max_fuel)

        if train_mask.sum() == 0:
            print(f"\n[{stage_name.upper()}] No training samples, skipping...")
            continue

        if val_mask.sum() == 0:
            print(f"\n[{stage_name.upper()}] No validation samples, skipping...")
            continue

        X_train_stage = X_train_enhanced[train_mask]
        y_train_stage = y_train[train_mask]
        X_val_stage = X_val_enhanced[val_mask]
        y_val_stage = y_val[val_mask]

        # Train stage models
        stage_result = train_stage_model(
            stage_name, X_train_stage, y_train_stage,
            X_val_stage, y_val_stage, CONFIG
        )

        # Store results
        all_models[stage_name] = stage_result['models']
        all_results[stage_name] = {
            'metrics': stage_result['metrics'],
            'range': (min_fuel, max_fuel),
            'train_samples': len(X_train_stage),
            'val_samples': len(X_val_stage)
        }

        # Add predictions to full array
        all_predictions[val_mask] = stage_result['ensemble_pred']

    # Overall evaluation
    overall_rmse = np.sqrt(mean_squared_error(y_val, all_predictions))
    overall_mae = mean_absolute_error(y_val, all_predictions)
    overall_r2 = r2_score(y_val, all_predictions)

    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"RMSE: {overall_rmse:.2f} kg")
    print(f"MAE: {overall_mae:.2f} kg")
    print(f"R²: {overall_r2:.4f}")
    print(f"\n✓ Improvement from Phase 8 baseline: {224.90 - overall_rmse:.2f} kg")
    print(f"✓ Improvement from Phase 7: {144.19 - overall_rmse:.2f} kg")
    print(f"✓ Gap to 60 kg target: {overall_rmse - 60:.2f} kg")

    if overall_rmse < 60:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! RMSE < 60 KG! 🎉🎉🎉")
    elif overall_rmse < 80:
        print(f"\n✓ EXCELLENT! Very close to target!")
    elif overall_rmse < 120:
        print(f"\n✓ Strong improvement! On track to target!")

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)

    models_dir = Path(CONFIG['paths']['models'])
    models_dir.mkdir(exist_ok=True)

    for stage_name, stage_models in all_models.items():
        for model_type, model in stage_models.items():
            if model_type == 'lgb':
                path = models_dir / f'phase8_opt_{stage_name}_{model_type}.txt'
                model.save_model(str(path))
            elif model_type == 'cat':
                path = models_dir / f'phase8_opt_{stage_name}_{model_type}.cbm'
                model.save_model(str(path))
            print(f"✓ Saved {stage_name} {model_type}: {path.name}")

    # Save results
    results_dir = Path(CONFIG['paths']['reports'])
    results_dir.mkdir(exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'r2': float(overall_r2),
            'improvement_from_phase8': float(224.90 - overall_rmse),
            'improvement_from_phase7': float(144.19 - overall_rmse),
            'gap_to_target': float(overall_rmse - 60)
        },
        'stages': {}
    }

    for stage_name, stage_result in all_results.items():
        results['stages'][stage_name] = {
            'rmse': float(stage_result['metrics']['rmse']),
            'mae': float(stage_result['metrics']['mae']),
            'r2': float(stage_result['metrics']['r2']),
            'train_samples': stage_result['train_samples'],
            'val_samples': stage_result['val_samples'],
            'range_min': float(stage_result['range'][0]),
            'range_max': float(stage_result['range'][1]) if stage_result['range'][1] != np.inf else 'inf'
        }

    results_path = results_dir / f'phase8_optimized_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_path.name}")
    print("="*80)


if __name__ == "__main__":
    main()
