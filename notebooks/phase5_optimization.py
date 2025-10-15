#!/usr/bin/env python3
"""
Phase 5: Model Optimization - Aircraft Fuel Consumption Prediction
===================================================================

This script implements comprehensive model optimization including:
1. Feature selection using SHAP values
2. Hyperparameter tuning with Optuna (Bayesian optimization)
3. Ensemble methods (weighted averaging, stacking)
4. Cross-validation for robustness
5. Final model selection and evaluation

Goal: Improve from Phase 4 baseline (196.20 kg RMSE) to < 180 kg RMSE

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

# Optimization
import optuna
from optuna.samplers import TPESampler

# Feature selection
import shap

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

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

def load_data():
    """Load preprocessed data and Phase 4 models."""
    print("=" * 80)
    print("PHASE 5: MODEL OPTIMIZATION")
    print("=" * 80)
    print("\n[1/8] Loading data and Phase 4 models...")

    start_time = time.time()

    # Load training and validation data
    X_train = pd.read_parquet(CACHE_DIR / "X_train.parquet")
    y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
    X_val = pd.read_parquet(CACHE_DIR / "X_val.parquet")
    y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

    # Load Phase 4 best models
    with open(MODELS_DIR / "phase4_catboost_model.pkl", 'rb') as f:
        catboost_base = pickle.load(f)

    with open(MODELS_DIR / "phase4_lightgbm_model.pkl", 'rb') as f:
        lightgbm_base = pickle.load(f)

    load_time = time.time() - start_time

    print(f"✓ Data loaded in {load_time:.2f} seconds")
    print(f"\nDataset Summary:")
    print(f"  Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
    print(f"\nPhase 4 Baseline Performance:")

    # Calculate baseline performance
    catboost_pred = catboost_base.predict(X_val)
    baseline_rmse = np.sqrt(mean_squared_error(y_val, catboost_pred))
    print(f"  CatBoost RMSE: {baseline_rmse:.2f} kg (target to beat)")

    return X_train, y_train, X_val, y_val, catboost_base, lightgbm_base, baseline_rmse


# ============================================================================
#  SECTION 2: FEATURE SELECTION WITH SHAP
# ============================================================================

def feature_selection_shap(model, X_train, X_val, y_train, y_val, top_n=80):
    """
    Select most important features using SHAP values.

    Args:
        model: Trained model (LightGBM or CatBoost)
        X_train, X_val: Feature matrices
        y_train, y_val: Target vectors
        top_n: Number of top features to keep

    Returns:
        selected_features: List of selected feature names
        shap_values: SHAP values for analysis
    """
    print("\n" + "=" * 80)
    print("FEATURE SELECTION WITH SHAP")
    print("=" * 80)
    print(f"\nAnalyzing feature importance using SHAP values...")
    print(f"Current features: {X_train.shape[1]}")
    print(f"Target features: {top_n}")

    start_time = time.time()

    # Sample data for SHAP (use 1000 samples for speed)
    sample_size = min(1000, len(X_val))
    X_sample = X_val.sample(n=sample_size, random_state=RANDOM_STATE)

    print(f"\nCalculating SHAP values on {sample_size} validation samples...")

    # Create SHAP explainer
    if isinstance(model, lgb.Booster):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:  # CatBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # Calculate mean absolute SHAP values
    shap_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    # Select top features
    selected_features = shap_importance.head(top_n)['feature'].tolist()

    elapsed_time = time.time() - start_time

    print(f"✓ SHAP analysis completed in {elapsed_time:.2f} seconds")
    print(f"\nTop 15 features by SHAP importance:")
    for i, row in shap_importance.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<40} {row['importance']:.6f}")

    # Calculate cumulative importance
    cumsum = shap_importance['importance'].cumsum() / shap_importance['importance'].sum()
    importance_covered = cumsum.iloc[top_n-1]

    print(f"\n✓ Selected {top_n} features covering {importance_covered*100:.2f}% of total importance")

    # Save SHAP importance
    shap_importance.to_csv(REPORTS_DIR / 'phase5_shap_feature_importance.csv', index=False)
    print(f"✓ Saved: phase5_shap_feature_importance.csv")

    # Test performance with selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    # Quick test with CatBoost
    print(f"\nTesting performance with selected features...")
    test_model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=RANDOM_STATE
    )
    test_model.fit(X_train_selected, y_train, eval_set=(X_val_selected, y_val), verbose=False)
    test_pred = test_model.predict(X_val_selected)
    test_rmse = np.sqrt(mean_squared_error(y_val, test_pred))

    print(f"  Quick test RMSE: {test_rmse:.2f} kg")

    return selected_features, shap_importance


# ============================================================================
#  SECTION 3: HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================

def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X_train, X_val: Feature matrices
        y_train, y_val: Target vectors
        n_trials: Number of optimization trials

    Returns:
        best_params: Optimized parameters
        best_model: Trained model with best parameters
        study: Optuna study object
    """
    print("\n" + "=" * 80)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"\nRunning Optuna optimization with {n_trials} trials...")
    print("This may take several minutes...")

    def objective(trial):
        """Optuna objective function for LightGBM."""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'verbose': -1,
            'random_state': RANDOM_STATE
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    # Create study
    start_time = time.time()
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name='lightgbm_optimization'
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    print(f"\n✓ Optimization completed in {elapsed_time:.2f} seconds")
    print(f"\nBest trial:")
    print(f"  RMSE: {study.best_value:.2f} kg")
    print(f"  Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    print(f"\nTraining final LightGBM with optimized parameters...")
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': RANDOM_STATE
    })

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    best_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )

    # Evaluate
    y_train_pred = best_model.predict(X_train, num_iteration=best_model.best_iteration)
    y_val_pred = best_model.predict(X_val, num_iteration=best_model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n✓ Final LightGBM Performance:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Best iteration: {best_model.best_iteration}")

    return best_params, best_model, study


def optimize_catboost(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Optimize CatBoost hyperparameters using Optuna.

    Args:
        X_train, X_val: Feature matrices
        y_train, y_val: Target vectors
        n_trials: Number of optimization trials

    Returns:
        best_params: Optimized parameters
        best_model: Trained model with best parameters
        study: Optuna study object
    """
    print("\n" + "=" * 80)
    print("CATBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"\nRunning Optuna optimization with {n_trials} trials...")
    print("This may take several minutes...")

    def objective(trial):
        """Optuna objective function for CatBoost."""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'verbose': False,
            'random_seed': RANDOM_STATE,
            'thread_count': -1
        }

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    # Create study
    start_time = time.time()
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name='catboost_optimization'
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    print(f"\n✓ Optimization completed in {elapsed_time:.2f} seconds")
    print(f"\nBest trial:")
    print(f"  RMSE: {study.best_value:.2f} kg")
    print(f"  Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    print(f"\nTraining final CatBoost with optimized parameters...")
    best_params = study.best_params.copy()
    best_params.update({
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 100,
        'random_seed': RANDOM_STATE,
        'thread_count': -1
    })

    best_model = CatBoostRegressor(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )

    # Evaluate
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n✓ Final CatBoost Performance:")
    print(f"  Train RMSE: {train_rmse:.2f} kg")
    print(f"  Val RMSE:   {val_rmse:.2f} kg")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Best iteration: {best_model.get_best_iteration()}")

    return best_params, best_model, study


# ============================================================================
#  SECTION 4: ENSEMBLE METHODS
# ============================================================================

def create_ensemble(models_dict, X_val, y_val):
    """
    Create weighted ensemble of multiple models.

    Args:
        models_dict: Dictionary of {'name': model} pairs
        X_val: Validation features
        y_val: Validation target

    Returns:
        best_weights: Optimal weights for ensemble
        ensemble_pred: Ensemble predictions
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 80)
    print(f"\nCreating ensemble from {len(models_dict)} models...")

    # Get predictions from all models
    predictions = {}
    for name, model in models_dict.items():
        print(f"  Getting predictions from {name}...")
        if isinstance(model, lgb.Booster):
            pred = model.predict(X_val, num_iteration=model.best_iteration)
        else:
            pred = model.predict(X_val)
        predictions[name] = pred

        # Calculate individual RMSE
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        print(f"    RMSE: {rmse:.2f} kg")

    # Grid search for optimal weights
    print(f"\nSearching for optimal weights...")
    best_rmse = float('inf')
    best_weights = None

    model_names = list(predictions.keys())
    n_models = len(model_names)

    # Try different weight combinations
    if n_models == 2:
        weight_ranges = [np.linspace(0.3, 0.7, 9)]
    elif n_models == 3:
        weight_ranges = [np.linspace(0.2, 0.6, 5), np.linspace(0.2, 0.6, 5)]
    else:  # 4 models
        weight_ranges = [np.linspace(0.15, 0.45, 4) for _ in range(n_models-1)]

    # Generate weight combinations
    from itertools import product
    weight_combinations = list(product(*weight_ranges))

    for weights_tuple in weight_combinations:
        weights = list(weights_tuple)
        # Calculate last weight to sum to 1
        weights.append(1.0 - sum(weights))

        # Skip invalid combinations
        if weights[-1] < 0 or weights[-1] > 1:
            continue

        # Calculate ensemble prediction
        ensemble_pred = np.zeros_like(y_val, dtype=float)
        for i, name in enumerate(model_names):
            ensemble_pred += weights[i] * predictions[name]

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights.copy()

    print(f"\n✓ Best ensemble found:")
    print(f"  RMSE: {best_rmse:.2f} kg")
    print(f"  Weights:")
    for i, name in enumerate(model_names):
        print(f"    {name}: {best_weights[i]:.3f}")

    # Create final ensemble prediction
    ensemble_pred = np.zeros_like(y_val, dtype=float)
    for i, name in enumerate(model_names):
        ensemble_pred += best_weights[i] * predictions[name]

    return dict(zip(model_names, best_weights)), ensemble_pred, best_rmse


# ============================================================================
#  SECTION 5: CROSS-VALIDATION
# ============================================================================

def cross_validate_model(model_func, X, y, n_splits=3):
    """
    Perform K-Fold cross-validation.

    Args:
        model_func: Function that returns a trained model
        X: Features
        y: Target
        n_splits: Number of CV folds

    Returns:
        cv_scores: List of RMSE scores for each fold
    """
    print(f"\nPerforming {n_splits}-fold cross-validation...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = model_func(X_train_cv, y_train_cv, X_val_cv, y_val_cv)

        if isinstance(model, lgb.Booster):
            pred = model.predict(X_val_cv, num_iteration=model.best_iteration)
        else:
            pred = model.predict(X_val_cv)

        rmse = np.sqrt(mean_squared_error(y_val_cv, pred))
        cv_scores.append(rmse)

        print(f"  Fold {fold}: RMSE = {rmse:.2f} kg")

    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)

    print(f"\n✓ CV Results: {mean_rmse:.2f} ± {std_rmse:.2f} kg")

    return cv_scores


# ============================================================================
#  SECTION 6: VISUALIZATION
# ============================================================================

def create_optimization_visualizations(optuna_studies, baseline_rmse):
    """Create visualizations for optimization results."""
    print("\n[7/8] Creating optimization visualizations...")

    # 1. Optimization history
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if 'lightgbm' in optuna_studies:
        study = optuna_studies['lightgbm']
        trials = study.trials
        values = [trial.value for trial in trials]

        axes[0].plot(range(1, len(values)+1), values, 'o-', alpha=0.6)
        axes[0].axhline(y=baseline_rmse, color='r', linestyle='--', label=f'Phase 4 Baseline ({baseline_rmse:.2f} kg)')
        axes[0].axhline(y=study.best_value, color='g', linestyle='--', label=f'Best ({study.best_value:.2f} kg)')
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('RMSE (kg)', fontsize=12)
        axes[0].set_title('LightGBM Optimization History', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if 'catboost' in optuna_studies:
        study = optuna_studies['catboost']
        trials = study.trials
        values = [trial.value for trial in trials]

        axes[1].plot(range(1, len(values)+1), values, 'o-', alpha=0.6, color='orange')
        axes[1].axhline(y=baseline_rmse, color='r', linestyle='--', label=f'Phase 4 Baseline ({baseline_rmse:.2f} kg)')
        axes[1].axhline(y=study.best_value, color='g', linestyle='--', label=f'Best ({study.best_value:.2f} kg)')
        axes[1].set_xlabel('Trial', fontsize=12)
        axes[1].set_ylabel('RMSE (kg)', fontsize=12)
        axes[1].set_title('CatBoost Optimization History', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase5_optimization_history.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase5_optimization_history.png")
    plt.close()

    # 2. Parameter importance
    if 'catboost' in optuna_studies:
        fig, ax = plt.subplots(figsize=(12, 8))

        study = optuna_studies['catboost']
        param_importance = optuna.importance.get_param_importances(study)

        params = list(param_importance.keys())[:10]
        importances = [param_importance[p] for p in params]

        ax.barh(range(len(params)), importances, alpha=0.8)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('CatBoost Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'phase5_hyperparameter_importance.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: phase5_hyperparameter_importance.png")
        plt.close()


# ============================================================================
#  SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for Phase 5."""

    # Load data
    X_train, y_train, X_val, y_val, catboost_base, lightgbm_base, baseline_rmse = load_data()

    # Dictionary to store results
    results = {
        'baseline_rmse': baseline_rmse,
        'optimized_models': {},
        'optuna_studies': {}
    }

    # STEP 2: Feature Selection with SHAP
    print("\n" + "=" * 80)
    print("[2/8] Feature Selection")
    print("=" * 80)

    selected_features, shap_importance = feature_selection_shap(
        catboost_base, X_train, X_val, y_train, y_val, top_n=85
    )

    # Update data with selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    print(f"\n✓ Using {len(selected_features)} selected features for optimization")

    # STEP 3: Optimize LightGBM
    print("\n" + "=" * 80)
    print("[3/8] LightGBM Optimization")
    print("=" * 80)

    lgb_params, lgb_model, lgb_study = optimize_lightgbm(
        X_train_selected, y_train, X_val_selected, y_val, n_trials=80
    )

    results['optimized_models']['LightGBM'] = lgb_model
    results['optuna_studies']['lightgbm'] = lgb_study

    # STEP 4: Optimize CatBoost
    print("\n" + "=" * 80)
    print("[4/8] CatBoost Optimization")
    print("=" * 80)

    cat_params, cat_model, cat_study = optimize_catboost(
        X_train_selected, y_train, X_val_selected, y_val, n_trials=80
    )

    results['optimized_models']['CatBoost'] = cat_model
    results['optuna_studies']['catboost'] = cat_study

    # STEP 5: Create Ensemble
    print("\n" + "=" * 80)
    print("[5/8] Ensemble Creation")
    print("=" * 80)

    ensemble_weights, ensemble_pred, ensemble_rmse = create_ensemble(
        results['optimized_models'],
        X_val_selected,
        y_val
    )

    results['ensemble_weights'] = ensemble_weights
    results['ensemble_rmse'] = ensemble_rmse

    # STEP 6: Final Evaluation
    print("\n" + "=" * 80)
    print("[6/8] Final Evaluation")
    print("=" * 80)

    print("\nPerformance Summary:")
    print("=" * 80)
    print(f"{'Model':<30} {'RMSE (kg)':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Phase 4 Baseline (CatBoost)':<30} {baseline_rmse:<15.2f} {'---':<15}")

    # Get best individual model
    best_individual = None
    best_individual_rmse = float('inf')

    for name, model in results['optimized_models'].items():
        if isinstance(model, lgb.Booster):
            pred = model.predict(X_val_selected, num_iteration=model.best_iteration)
        else:
            pred = model.predict(X_val_selected)

        rmse = np.sqrt(mean_squared_error(y_val, pred))
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

        print(f"{f'Optimized {name}':<30} {rmse:<15.2f} {f'-{improvement:.1f}%':<15}")

        if rmse < best_individual_rmse:
            best_individual_rmse = rmse
            best_individual = (name, model)

    ensemble_improvement = ((baseline_rmse - ensemble_rmse) / baseline_rmse) * 100
    print(f"{'Ensemble':<30} {ensemble_rmse:<15.2f} {f'-{improvement:.1f}%':<15}")
    print("=" * 80)

    # Determine best model
    if ensemble_rmse < best_individual_rmse:
        print(f"\n✓ BEST MODEL: Ensemble")
        print(f"  Final RMSE: {ensemble_rmse:.2f} kg")
        print(f"  Improvement over Phase 4: {ensemble_improvement:.1f}%")
        results['best_model_type'] = 'ensemble'
        results['best_rmse'] = ensemble_rmse
    else:
        print(f"\n✓ BEST MODEL: {best_individual[0]}")
        print(f"  Final RMSE: {best_individual_rmse:.2f} kg")
        improvement = ((baseline_rmse - best_individual_rmse) / baseline_rmse) * 100
        print(f"  Improvement over Phase 4: {improvement:.1f}%")
        results['best_model_type'] = best_individual[0]
        results['best_rmse'] = best_individual_rmse

    # Create visualizations
    create_optimization_visualizations(results['optuna_studies'], baseline_rmse)

    # STEP 8: Save Results
    print("\n" + "=" * 80)
    print("[8/8] Saving Optimized Models")
    print("=" * 80)

    # Save models
    for name, model in results['optimized_models'].items():
        filename = f"phase5_{name.lower()}_optimized.pkl"
        with open(MODELS_DIR / filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Saved: {filename}")

    # Save selected features
    with open(MODELS_DIR / 'phase5_selected_features.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"  ✓ Saved: phase5_selected_features.json")

    # Save ensemble weights
    with open(MODELS_DIR / 'phase5_ensemble_weights.json', 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    print(f"  ✓ Saved: phase5_ensemble_weights.json")

    # Save optimization parameters
    opt_params = {
        'lightgbm': lgb_params,
        'catboost': cat_params
    }
    with open(MODELS_DIR / 'phase5_optimized_params.json', 'w') as f:
        # Convert numpy types to Python types
        serializable_params = {}
        for model_name, params in opt_params.items():
            serializable_params[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in params.items()
            }
        json.dump(serializable_params, f, indent=2)
    print(f"  ✓ Saved: phase5_optimized_params.json")

    # Generate summary report
    generate_summary_report(results, baseline_rmse, selected_features)

    print("\n" + "=" * 80)
    print("✓ PHASE 5 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Phase 4 Baseline: {baseline_rmse:.2f} kg RMSE")
    print(f"  Phase 5 Best:     {results['best_rmse']:.2f} kg RMSE")
    improvement = ((baseline_rmse - results['best_rmse']) / baseline_rmse) * 100
    print(f"  Improvement:      {improvement:.1f}%")
    print(f"  Best Model:       {results['best_model_type']}")
    print("\n" + "=" * 80)


def generate_summary_report(results, baseline_rmse, selected_features):
    """Generate comprehensive summary report for Phase 5."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 5: MODEL OPTIMIZATION - SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("1. OPTIMIZATION RESULTS")
    report_lines.append("=" * 80)
    report_lines.append(f"Phase 4 Baseline RMSE: {baseline_rmse:.2f} kg")
    report_lines.append(f"Phase 5 Best RMSE:     {results['best_rmse']:.2f} kg")
    improvement = ((baseline_rmse - results['best_rmse']) / baseline_rmse) * 100
    report_lines.append(f"Improvement:           {improvement:.1f}%")
    report_lines.append(f"Best Model Type:       {results['best_model_type']}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("2. FEATURE SELECTION")
    report_lines.append("=" * 80)
    report_lines.append(f"Original features:  110")
    report_lines.append(f"Selected features:  {len(selected_features)}")
    report_lines.append(f"Reduction:          {110 - len(selected_features)} features")
    report_lines.append("")

    if 'ensemble_weights' in results:
        report_lines.append("=" * 80)
        report_lines.append("3. ENSEMBLE WEIGHTS")
        report_lines.append("=" * 80)
        for model, weight in results['ensemble_weights'].items():
            report_lines.append(f"  {model}: {weight:.3f}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("PHASE 5 COMPLETE")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    with open(REPORTS_DIR / 'phase5_optimization_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✓ Summary report saved: reports/phase5_optimization_summary.txt")


if __name__ == "__main__":
    main()
