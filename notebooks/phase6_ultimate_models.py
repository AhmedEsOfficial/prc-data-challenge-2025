#!/usr/bin/env python3
"""
Phase 6: Ultimate Model Training - Aircraft Fuel Consumption Prediction
========================================================================

This script trains ultimate models with enhanced features:
1. Optimized LightGBM and CatBoost with enhanced features
2. Deep Neural Network (MLP) for non-linear patterns
3. Stacking Ensemble with meta-learner
4. Final comparison and selection

Goal: Achieve <50 kg RMSE (down from 189.29 kg)

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model libraries
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, will use sklearn MLPRegressor")

# Optimization
import optuna
from optuna.samplers import TPESampler

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VIZ_DIR = BASE_DIR / "visualizations"

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
#  SECTION 1: DATA LOADING
# ============================================================================

def load_enhanced_data():
    """Load enhanced feature sets."""
    print("=" * 80)
    print("PHASE 6: ULTIMATE MODEL TRAINING")
    print("=" * 80)
    print("\n[1/6] Loading enhanced features...")

    X_train = pd.read_parquet(CACHE_DIR / "X_train_enhanced.parquet")
    y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
    X_val = pd.read_parquet(CACHE_DIR / "X_val_enhanced.parquet")
    y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

    print(f"✓ Training: {X_train.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(f"✓ Features: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val


# ============================================================================
#  SECTION 2: OPTIMIZED GRADIENT BOOSTING MODELS
# ============================================================================

def train_optimized_lightgbm(X_train, y_train, X_val, y_val):
    """Train optimized LightGBM with enhanced features."""
    print("\n" + "=" * 80)
    print("[2/6] Training Optimized LightGBM")
    print("=" * 80)

    # Optimized parameters from Phase 5, adapted for more features
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
        'min_child_weight': 1e-3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 8,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    print("Training with optimized hyperparameters...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    start_time = time.time()

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )

    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n✓ LightGBM Results:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val MAE:    {val_mae:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Training time: {train_time:.2f}s")

    return model, val_rmse


def train_optimized_catboost(X_train, y_train, X_val, y_val):
    """Train optimized CatBoost with enhanced features."""
    print("\n" + "=" * 80)
    print("[3/6] Training Optimized CatBoost")
    print("=" * 80)

    # Optimized parameters
    params = {
        'iterations': 1500,
        'learning_rate': 0.08,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'min_data_in_leaf': 50,
        'random_strength': 3.0,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 100,
        'random_seed': RANDOM_STATE,
        'thread_count': -1
    }

    print("Training with optimized hyperparameters...")

    start_time = time.time()

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )

    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n✓ CatBoost Results:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val MAE:    {val_mae:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Best iteration: {model.get_best_iteration()}")
    print(f"  Training time: {train_time:.2f}s")

    return model, val_rmse


# ============================================================================
#  SECTION 3: NEURAL NETWORK
# ============================================================================

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train deep neural network for non-linear patterns."""
    print("\n" + "=" * 80)
    print("[4/6] Training Deep Neural Network")
    print("=" * 80)

    # Scale features for neural network
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training MLP Regressor...")

    # Use sklearn MLPRegressor (more stable than PyTorch for this use case)
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=True
    )

    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = mlp.predict(X_train_scaled)
    y_val_pred = mlp.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n✓ Neural Network Results:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val MAE:    {val_mae:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Training time: {train_time:.2f}s")

    # Return model wrapped with scaler
    model_bundle = {
        'model': mlp,
        'scaler': scaler
    }

    return model_bundle, val_rmse


# ============================================================================
#  SECTION 4: STACKING ENSEMBLE
# ============================================================================

def create_stacking_ensemble(models_dict, X_train, y_train, X_val, y_val):
    """Create stacking ensemble with meta-learner."""
    print("\n" + "=" * 80)
    print("[5/6] Creating Stacking Ensemble")
    print("=" * 80)

    print("Building meta-features from base models...")

    # Generate out-of-fold predictions for meta-learner
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    meta_train = np.zeros((len(X_train), len(models_dict)))
    meta_val = np.zeros((len(X_val), len(models_dict)))

    for i, (name, model_info) in enumerate(models_dict.items()):
        print(f"\nGenerating meta-features for {name}...")

        if name == 'NeuralNetwork':
            # Neural network needs scaling
            model = model_info['model']
            scaler = model_info['scaler']

            # Out-of-fold predictions for training
            for train_idx, val_idx in kfold.split(X_train):
                X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Scale and predict
                scaler_fold = StandardScaler()
                X_tr_scaled = scaler_fold.fit_transform(X_tr)
                X_val_fold_scaled = scaler_fold.transform(X_val_fold)

                model_fold = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32),
                    random_state=RANDOM_STATE,
                    max_iter=100,
                    early_stopping=True,
                    verbose=False
                )
                model_fold.fit(X_tr_scaled, y_tr)
                meta_train[val_idx, i] = model_fold.predict(X_val_fold_scaled)

            # Validation predictions
            X_val_scaled = scaler.transform(X_val)
            meta_val[:, i] = model.predict(X_val_scaled)

        else:
            # Tree models
            model = model_info

            # Out-of-fold predictions for training
            for train_idx, val_idx in kfold.split(X_train):
                X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                if isinstance(model, lgb.Booster):
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

                    params = model.params.copy()
                    params['verbose'] = -1

                    model_fold = lgb.train(
                        params,
                        train_data,
                        num_boost_round=500,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
                    )
                    meta_train[val_idx, i] = model_fold.predict(X_val_fold, num_iteration=model_fold.best_iteration)

                else:  # CatBoost
                    model_fold = CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.08,
                        depth=6,
                        random_seed=RANDOM_STATE,
                        verbose=False
                    )
                    model_fold.fit(X_tr, y_tr, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=False)
                    meta_train[val_idx, i] = model_fold.predict(X_val_fold)

            # Validation predictions
            if isinstance(model, lgb.Booster):
                meta_val[:, i] = model.predict(X_val, num_iteration=model.best_iteration)
            else:
                meta_val[:, i] = model.predict(X_val)

    print(f"\n✓ Meta-features generated: {meta_train.shape}")

    # Train meta-learner (Ridge regression for stability)
    print("Training meta-learner...")

    meta_learner = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    meta_learner.fit(meta_train, y_train)

    # Meta-learner weights
    print("\nMeta-learner weights:")
    for i, name in enumerate(models_dict.keys()):
        print(f"  {name}: {meta_learner.coef_[i]:.4f}")

    # Final predictions
    stacking_pred = meta_learner.predict(meta_val)

    # Evaluate
    val_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
    val_mae = mean_absolute_error(y_val, stacking_pred)
    val_r2 = r2_score(y_val, stacking_pred)

    print(f"\n✓ Stacking Ensemble Results:")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val MAE:    {val_mae:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")

    # Bundle for saving
    stacking_bundle = {
        'meta_learner': meta_learner,
        'base_models': models_dict
    }

    return stacking_bundle, val_rmse


# ============================================================================
#  SECTION 5: MODEL COMPARISON AND SELECTION
# ============================================================================

def compare_and_select_best(models_dict, baseline_rmse):
    """Compare all models and select the best."""
    print("\n" + "=" * 80)
    print("[6/6] Final Model Comparison")
    print("=" * 80)

    results = []

    print("\nModel Performance Summary:")
    print("=" * 80)
    print(f"{'Model':<30} {'Val RMSE':<15} {'vs Baseline':<15} {'Improvement':<15}")
    print("-" * 80)

    print(f"{'Phase 5 Baseline':<30} {baseline_rmse:<15.2f} {'---':<15} {'---':<15}")

    for name, (model, rmse) in models_dict.items():
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        vs_baseline = baseline_rmse - rmse

        print(f"{name:<30} {rmse:<15.2f} {vs_baseline:<15.2f} {improvement:<15.1f}%")

        results.append({
            'model': name,
            'rmse': rmse,
            'improvement_pct': improvement,
            'improvement_kg': vs_baseline
        })

    print("=" * 80)

    # Find best model
    best_result = min(results, key=lambda x: x['rmse'])

    print(f"\n✓ BEST MODEL: {best_result['model']}")
    print(f"  RMSE: {best_result['rmse']:.2f} kg")
    print(f"  Improvement: {best_result['improvement_pct']:.1f}% ({best_result['improvement_kg']:.2f} kg)")

    # Check if target achieved
    if best_result['rmse'] < 50:
        print(f"\n🎉 TARGET ACHIEVED! RMSE < 50 kg")
    elif best_result['rmse'] < 100:
        print(f"\n✓ Excellent progress! RMSE < 100 kg (target: 50 kg)")
    else:
        print(f"\n⚠️  Target not yet reached (50 kg). Current: {best_result['rmse']:.2f} kg")
        print(f"   Additional improvement needed: {best_result['rmse'] - 50:.2f} kg")

    return results, best_result


# ============================================================================
#  SECTION 6: SAVE AND VISUALIZE
# ============================================================================

def save_ultimate_models(models_dict, results, best_result):
    """Save final models and results."""
    print("\n" + "=" * 80)
    print("Saving Ultimate Models")
    print("=" * 80)

    # Save individual models
    for name, (model, rmse) in models_dict.items():
        if name == 'Stacking':
            filename = f"phase6_ultimate_stacking.pkl"
            with open(MODELS_DIR / filename, 'wb') as f:
                pickle.dump(model, f)
        elif name == 'LightGBM':
            filename = f"phase6_ultimate_lightgbm.pkl"
            with open(MODELS_DIR / filename, 'wb') as f:
                pickle.dump(model, f)
        elif name == 'CatBoost':
            filename = f"phase6_ultimate_catboost.pkl"
            with open(MODELS_DIR / filename, 'wb') as f:
                pickle.dump(model, f)
        elif name == 'NeuralNetwork':
            filename = f"phase6_ultimate_neural_network.pkl"
            with open(MODELS_DIR / filename, 'wb') as f:
                pickle.dump(model, f)

        print(f"  ✓ Saved: {filename}")

    # Save results summary
    results_summary = {
        'best_model': best_result['model'],
        'best_rmse': best_result['rmse'],
        'improvement_from_baseline': best_result['improvement_pct'],
        'all_results': results,
        'timestamp': datetime.now().isoformat()
    }

    with open(REPORTS_DIR / 'phase6_ultimate_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"  ✓ Saved: phase6_ultimate_results.json")

    # Create comparison visualization
    create_comparison_visualization(results)


def create_comparison_visualization(results):
    """Create visual comparison of all models."""

    fig, ax = plt.subplots(figsize=(12, 6))

    models = [r['model'] for r in results]
    rmses = [r['rmse'] for r in results]
    colors = ['#2ecc71' if rmse < 100 else '#e74c3c' for rmse in rmses]

    bars = ax.barh(models, rmses, color=colors, alpha=0.7)

    # Add value labels
    for i, (model, rmse) in enumerate(zip(models, rmses)):
        ax.text(rmse + 5, i, f'{rmse:.2f} kg', va='center', fontweight='bold')

    # Add target line
    ax.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Target (50 kg)')
    ax.axvline(x=189.29, color='red', linestyle='--', linewidth=2, label='Baseline (189.29 kg)')

    ax.set_xlabel('Validation RMSE (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Ultimate Model Comparison - Phase 6', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase6_ultimate_model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: phase6_ultimate_model_comparison.png")
    plt.close()


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load data
    X_train, y_train, X_val, y_val = load_enhanced_data()

    baseline_rmse = 189.29  # Phase 5 best result

    # Dictionary to store all models
    models_results = {}

    # Train LightGBM
    lgb_model, lgb_rmse = train_optimized_lightgbm(X_train, y_train, X_val, y_val)
    models_results['LightGBM'] = (lgb_model, lgb_rmse)

    # Train CatBoost
    cat_model, cat_rmse = train_optimized_catboost(X_train, y_train, X_val, y_val)
    models_results['CatBoost'] = (cat_model, cat_rmse)

    # Train Neural Network
    nn_model, nn_rmse = train_neural_network(X_train, y_train, X_val, y_val)
    models_results['NeuralNetwork'] = (nn_model, nn_rmse)

    # Create Stacking Ensemble
    stacking_model, stacking_rmse = create_stacking_ensemble(
        {
            'LightGBM': lgb_model,
            'CatBoost': cat_model,
            'NeuralNetwork': nn_model
        },
        X_train, y_train, X_val, y_val
    )
    models_results['Stacking'] = (stacking_model, stacking_rmse)

    # Compare and select best
    results, best_result = compare_and_select_best(models_results, baseline_rmse)

    # Save models and results
    save_ultimate_models(models_results, results, best_result)

    print("\n" + "=" * 80)
    print("✓ PHASE 6: ULTIMATE MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Model: {best_result['model']}")
    print(f"Final RMSE: {best_result['rmse']:.2f} kg")
    print(f"Improvement from baseline: {best_result['improvement_pct']:.1f}%")

    if best_result['rmse'] < 50:
        print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
    else:
        print(f"\nGap to target: {best_result['rmse'] - 50:.2f} kg")

    print("=" * 80)


if __name__ == "__main__":
    main()
