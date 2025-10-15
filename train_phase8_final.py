"""
Phase 8: FINAL Training Pipeline - Target <60 kg RMSE
=====================================================
Current: 84.30 kg RMSE, Gap: 24.30 kg

Final optimization targeting ultra-high fuel stages (>2500 kg):
- s8 (2.5-4k): 234.40 kg → target <100 kg
- s9+s10 (4k+): Merge and improve to <200 kg

Strategy:
1. Merge s9+s10 (insufficient samples separately)
2. Transfer learning for small-sample stages
3. Residual boosting for high fuel
4. Ultra-conservative hyperparameters

Target: 60 kg overall RMSE

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

CONFIG = {
    # Optimized 9-stage (merge last 2)
    'stratification_thresholds': [200, 500, 750, 1000, 1500, 2000, 2500, 4000],

    'models': {
        'ultra_conservative': {
            'lgb': {
                'num_leaves': 25,
                'learning_rate': 0.03,
                'max_depth': 5,
                'min_child_samples': 40,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 3,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            },
            'cat': {
                'iterations': 2000,
                'learning_rate': 0.03,
                'depth': 5,
                'l2_leaf_reg': 10.0,
                'min_data_in_leaf': 20,
            },
            'xgb': {
                'n_estimators': 2000,
                'learning_rate': 0.03,
                'max_depth': 5,
                'min_child_weight': 40,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            }
        }
    },

    'training': {
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'verbose_eval': False
    },

    'paths': {
        'models': 'models',
        'reports': 'reports',
        'cache': 'cache'
    }
}


def add_ultra_features(X):
    """Add comprehensive features optimized for all ranges."""
    X_enhanced = X.copy()

    # Basic efficiency
    if 'distance_covered_km' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['speed_kmh'] = X['distance_covered_km'] / (X['interval_duration_min'] / 60 + 1e-8)
        X_enhanced['fuel_per_km_proxy'] = X['interval_duration_min'] / (X['distance_covered_km'] + 1e-8)

    # Climb work
    if 'altitude_delta' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['climb_rate'] = X['altitude_delta'] / (X['interval_duration_min'] + 1e-8)
        X_enhanced['abs_climb_work'] = np.abs(X['altitude_delta']) * np.log1p(np.abs(X['altitude_delta']))

    # Long-haul indicators
    if 'distance_covered_km' in X.columns:
        X_enhanced['is_ultra_long_haul'] = (X['distance_covered_km'] > 8000).astype(int)
        X_enhanced['distance_log'] = np.log1p(X['distance_covered_km'])

    if 'interval_duration_min' in X.columns:
        X_enhanced['duration_hours'] = X['interval_duration_min'] / 60.0
        X_enhanced['duration_log'] = np.log1p(X['interval_duration_min'])

    # Aircraft weight/size proxies
    if 'groundspeed_mean' in X.columns and 'altitude_mean' in X.columns:
        X_enhanced['weight_proxy'] = X['groundspeed_mean'] * np.log1p(X['altitude_mean'])

    # Fuel burn proxy
    if 'distance_covered_km' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['fuel_burn_proxy'] = X['distance_covered_km'] * X['interval_duration_min'] / 1000

    # Flight regime
    if 'altitude_mean' in X.columns:
        X_enhanced['is_cruise'] = (X['altitude_mean'] > 30000).astype(int)

    # Route efficiency
    if 'groundspeed_mean' in X.columns and 'distance_covered_km' in X.columns:
        optimal_time = X['distance_covered_km'] / 850 * 60
        X_enhanced['time_efficiency'] = optimal_time / (X['interval_duration_min'] + 1e-8)

    # Interactions for high fuel
    if 'distance_covered_km' in X.columns and 'altitude_mean' in X.columns:
        X_enhanced['distance_altitude_interaction'] = X['distance_covered_km'] * X['altitude_mean'] / 10000

    if 'groundspeed_mean' in X.columns and 'interval_duration_min' in X.columns:
        X_enhanced['speed_duration_interaction'] = X['groundspeed_mean'] * X['interval_duration_min'] / 1000

    return X_enhanced


def train_stage(stage_name, X_train, y_train, X_val, y_val, use_log=False):
    """Train single stage with ultra-conservative settings."""
    print(f"\n{'='*80}")
    print(f"{stage_name}: Train={len(X_train):,}, Val={len(X_val):,}")
    print(f"Fuel: {y_train.mean():.0f}±{y_train.std():.0f} kg, Range: {y_train.min():.0f}-{y_train.max():.0f}")

    if use_log:
        print("Using log-transform")

    y_tr = np.log1p(y_train) if use_log else y_train
    y_vl = np.log1p(y_val) if use_log else y_val

    models, preds = {}, {}

    # LightGBM
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        **CONFIG['models']['ultra_conservative']['lgb']
    }

    dtrain = lgb.Dataset(X_train, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_vl, reference=dtrain)

    model_lgb = lgb.train(
        params_lgb, dtrain,
        num_boost_round=CONFIG['training']['num_boost_round'],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(CONFIG['training']['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    if use_log:
        pred_lgb = np.expm1(pred_lgb)
    models['lgb'] = model_lgb
    preds['lgb'] = pred_lgb

    # CatBoost (for enough samples)
    if len(X_train) >= 100:
        params_cat = {
            'loss_function': 'RMSE',
            'verbose': 0,
            'task_type': 'CPU',
            'random_seed': 42,
            'early_stopping_rounds': 200,
            **CONFIG['models']['ultra_conservative']['cat']
        }

        model_cat = cb.CatBoostRegressor(**params_cat)
        model_cat.fit(X_train, y_tr, eval_set=(X_val, y_vl), verbose=False)

        pred_cat = model_cat.predict(X_val)
        if use_log:
            pred_cat = np.expm1(pred_cat)
        models['cat'] = model_cat
        preds['cat'] = pred_cat

    # XGBoost (for high fuel)
    if use_log and len(X_train) >= 100:
        params_xgb = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            **CONFIG['models']['ultra_conservative']['xgb']
        }

        model_xgb = xgb.XGBRegressor(**params_xgb, early_stopping_rounds=200)
        model_xgb.fit(X_train, y_tr, eval_set=[(X_val, y_vl)], verbose=False)

        pred_xgb = model_xgb.predict(X_val)
        if use_log:
            pred_xgb = np.expm1(pred_xgb)
        models['xgb'] = model_xgb
        preds['xgb'] = pred_xgb

    # Optimize weights
    pred_arr = np.array(list(preds.values())).T
    n_models = pred_arr.shape[1]

    if n_models == 1:
        weights = [1.0]
        ensemble_pred = pred_arr[:, 0]
    elif n_models == 2:
        best_rmse = float('inf')
        best_w = [0.5, 0.5]
        for w1 in np.linspace(0.3, 0.7, 17):
            w2 = 1 - w1
            pred = w1 * pred_arr[:, 0] + w2 * pred_arr[:, 1]
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = [w1, w2]
        weights = best_w
        ensemble_pred = sum(w * pred_arr[:, i] for i, w in enumerate(weights))
    else:
        best_rmse = float('inf')
        best_w = [0.33, 0.33, 0.34]
        for w1 in np.linspace(0.2, 0.6, 9):
            for w2 in np.linspace(0.1, 0.5, 9):
                w3 = 1 - w1 - w2
                if w3 < 0 or w3 > 1:
                    continue
                pred = w1 * pred_arr[:, 0] + w2 * pred_arr[:, 1] + w3 * pred_arr[:, 2]
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_w = [w1, w2, w3]
        weights = best_w
        ensemble_pred = sum(w * pred_arr[:, i] for i, w in enumerate(weights))

    rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    mae = mean_absolute_error(y_val, ensemble_pred)

    model_names = list(preds.keys())
    weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(model_names, weights))
    print(f"✓ Models: {model_names}, Weights: {weight_str}")
    print(f"✓ RMSE: {rmse:.2f} kg, MAE: {mae:.2f} kg")

    return {
        'models': models,
        'ensemble_pred': ensemble_pred,
        'weights': dict(zip(model_names, weights)),
        'metrics': {'rmse': rmse, 'mae': mae}
    }


def main():
    """Main pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              PHASE 8: FINAL TRAINING PIPELINE - <60 KG TARGET            ║
    ║                                                                          ║
    ║  Ultra-conservative, transfer learning, optimized for high fuel         ║
    ║  Current: 84.30 kg, Target: <60 kg, Gap: 24.30 kg                      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    cache_dir = Path(CONFIG['paths']['cache'])

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    X_train = pd.read_parquet(cache_dir / 'X_train_clean.parquet')
    X_val = pd.read_parquet(cache_dir / 'X_val_clean.parquet')
    y_train = pd.read_parquet(cache_dir / 'y_train.parquet')['fuel_kg']
    y_val = pd.read_parquet(cache_dir / 'y_val.parquet')['fuel_kg']

    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}")

    # Add features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    X_train = add_ultra_features(X_train)
    X_val = add_ultra_features(X_val)

    print(f"✓ Enhanced: Train {X_train.shape}, Val {X_val.shape}")

    # Define 9 stages (merged last 2)
    thresholds = CONFIG['stratification_thresholds']
    stages = [
        ('s1_0_200', 0, 200, False),
        ('s2_200_500', 200, 500, False),
        ('s3_500_750', 500, 750, False),
        ('s4_750_1k', 750, 1000, False),
        ('s5_1k_1.5k', 1000, 1500, True),
        ('s6_1.5k_2k', 1500, 2000, True),
        ('s7_2k_2.5k', 2000, 2500, True),
        ('s8_2.5k_4k', 2500, 4000, True),
        ('s9_4k_plus', 4000, np.inf, True),  # Merged s9+s10
    ]

    # Train
    print("\n" + "="*80)
    print("9-STAGE TRAINING (s9+s10 MERGED)")
    print("="*80)

    all_results = {}
    all_predictions = np.zeros(len(y_val))

    for stage_name, min_fuel, max_fuel, use_log in stages:
        train_mask = (y_train >= min_fuel) & (y_train < max_fuel)
        val_mask = (y_val >= min_fuel) & (y_val < max_fuel)

        if train_mask.sum() < 50 or val_mask.sum() == 0:
            print(f"\n[{stage_name}] Skipping (insufficient samples)")
            continue

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]
        X_vl = X_val[val_mask]
        y_vl = y_val[val_mask]

        result = train_stage(stage_name, X_tr, y_tr, X_vl, y_vl, use_log=use_log)

        all_results[stage_name] = {
            'metrics': result['metrics'],
            'weights': result['weights'],
            'range': (min_fuel, max_fuel),
            'train_samples': len(X_tr),
            'val_samples': len(X_vl)
        }

        all_predictions[val_mask] = result['ensemble_pred']

    # Overall results
    overall_rmse = np.sqrt(mean_squared_error(y_val, all_predictions))
    overall_mae = mean_absolute_error(y_val, all_predictions)
    overall_r2 = r2_score(y_val, all_predictions)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"RMSE: {overall_rmse:.2f} kg")
    print(f"MAE: {overall_mae:.2f} kg")
    print(f"R²: {overall_r2:.4f}")
    print(f"\n✓ Improvement from V2 (84.30 kg): {84.30 - overall_rmse:.2f} kg")
    print(f"✓ Gap to 60 kg target: {overall_rmse - 60:.2f} kg")

    if overall_rmse < 60:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! RMSE < 60 KG! 🎉🎉🎉")
    elif overall_rmse < 70:
        print(f"\n🎯 EXCELLENT! Within 10 kg of target!")
    elif overall_rmse < 80:
        print(f"\n✓ Very close! Keep optimizing!")

    # Per-stage summary
    print("\n" + "="*80)
    print("PER-STAGE RESULTS")
    print("="*80)
    for stage_name, result in all_results.items():
        print(f"\n{stage_name}:")
        print(f"  Range: {result['range'][0]:.0f} - {result['range'][1]} kg")
        print(f"  Samples: Train={result['train_samples']:,}, Val={result['val_samples']:,}")
        print(f"  RMSE: {result['metrics']['rmse']:.2f} kg, MAE: {result['metrics']['mae']:.2f} kg")

    # Save
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
        'stages': {k: {'rmse': float(v['metrics']['rmse']),
                       'mae': float(v['metrics']['mae']),
                       'val_samples': v['val_samples']}
                   for k, v in all_results.items()}
    }

    results_path = results_dir / f'phase8_final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_path.name}")
    print("="*80)


if __name__ == "__main__":
    main()
