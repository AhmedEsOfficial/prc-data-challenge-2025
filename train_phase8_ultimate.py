"""
Phase 8: Ultimate Training Pipeline
====================================
Integrates advanced preprocessing with Phase 7 best practices.

Target: < 130 kg RMSE (from 144.19 kg baseline)

Architecture:
- Advanced preprocessing (9 modules)
- 6-stage stratification (0-200, 200-500, 500-1000, 1000-1500, 1500-2500, 2500+)
- Per-stage data augmentation
- Per-stage hyperparameter optimization
- Multi-model ensemble (LightGBM + CatBoost + Log-transform)

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
import sys

sys.path.append('.')

from src.preprocessing_advanced import (
    AdvancedMissingValueHandler,
    OutlierHandler,
    PolynomialFeatureGenerator,
    RatioFeatureGenerator,
    TemporalFeatureEngineer,
    LeaveOneOutEncoder,
    AdvancedScaler,
    IntelligentFeatureSelector
)

from src.data_augmentation import (
    FuelRangeAugmenter,
    validate_augmentation
)

# Not needed - using pre-split Phase 7 data
# from src.preprocessing import create_temporal_split, validate_no_leakage

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE8_CONFIG = {
    # Stratification thresholds (6 stages)
    'stratification_thresholds': [200, 500, 1000, 1500, 2500],

    # Preprocessing configuration
    'preprocessing': {
        'missing_strategy': 'hybrid',
        'outlier_method': 'percentile',
        'outlier_percentile_range': (0.5, 99.5),
        'scaling_strategy': 'hybrid',
        'polynomial_degree': 2,
        'enable_polynomial': True,
        'enable_ratios': True,
        'enable_temporal': False,  # Set True if temporal features available
        'enable_feature_selection': True,
        'variance_threshold': 0.005,
        'correlation_threshold': 0.95,
        'mutual_info_percentile': 85,
        'verbose': True
    },

    # Augmentation per stage (min_fuel, max_fuel, multiplier, strategy)
    'augmentation': {
        'very_low': None,  # No augmentation needed
        'low': None,
        'medium': {'multiplier': 1.3, 'strategy': 'noise'},
        'medium_high': {'multiplier': 1.5, 'strategy': 'noise'},
        'high': {'multiplier': 2.0, 'strategy': 'interpolate'},
        'very_high': {'multiplier': 4.0, 'strategy': 'interpolate'}
    },

    # Model configurations per stage
    'models': {
        'very_low': {
            'lgb': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'min_child_samples': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        },
        'low': {
            'lgb': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'min_child_samples': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        },
        'medium': {
            'lgb': {
                'num_leaves': 50,
                'learning_rate': 0.03,
                'min_child_samples': 15,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2
            }
        },
        'medium_high': {
            'lgb': {
                'num_leaves': 63,
                'learning_rate': 0.03,
                'min_child_samples': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3
            }
        },
        'high': {
            'lgb': {
                'num_leaves': 50,
                'learning_rate': 0.02,
                'min_child_samples': 8,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5
            },
            'cat': {
                'iterations': 1500,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 5.0,
                'random_seed': 42
            }
        },
        'very_high': {
            'lgb': {
                'num_leaves': 40,
                'learning_rate': 0.02,
                'min_child_samples': 5,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0
            },
            'cat': {
                'iterations': 2000,
                'learning_rate': 0.02,
                'depth': 5,
                'l2_leaf_reg': 10.0,
                'random_seed': 42
            }
        }
    },

    # Ensemble weights per stage
    'ensemble_weights': {
        'very_low': {'lgb': 1.0},
        'low': {'lgb': 1.0},
        'medium': {'lgb': 1.0},
        'medium_high': {'lgb': 1.0},
        'high': {'lgb': 0.5, 'cat': 0.5},
        'very_high': {'lgb': 0.4, 'cat': 0.6}
    },

    # Training parameters
    'training': {
        'num_boost_round': 2000,
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    },

    # Paths
    'paths': {
        'models': 'models',
        'reports': 'reports',
        'cache': 'cache'
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_stage_name(fuel_value, thresholds):
    """Get stage name for a fuel value."""
    if fuel_value < thresholds[0]:
        return 'very_low'
    elif fuel_value < thresholds[1]:
        return 'low'
    elif fuel_value < thresholds[2]:
        return 'medium'
    elif fuel_value < thresholds[3]:
        return 'medium_high'
    elif fuel_value < thresholds[4]:
        return 'high'
    else:
        return 'very_high'


def get_stage_mask(y, min_fuel, max_fuel):
    """Get boolean mask for samples in fuel range."""
    if isinstance(y, pd.Series):
        return (y >= min_fuel) & (y < max_fuel)
    else:
        return (y >= min_fuel) & (y < max_fuel)


def print_stage_statistics(stage_name, X, y):
    """Print statistics for a stage."""
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Fuel range: {y.min():.2f} - {y.max():.2f} kg")
    print(f"Fuel mean: {y.mean():.2f} kg")
    print(f"Fuel std: {y.std():.2f} kg")
    print(f"Fuel median: {y.median():.2f} kg")


# ============================================================================
# ADVANCED PREPROCESSING PIPELINE
# ============================================================================

class AdvancedPreprocessingPipeline:
    """Complete preprocessing pipeline."""

    def __init__(self, config):
        self.config = config
        self.transformers = {}
        self.feature_names = None

    def fit(self, X_train, y_train):
        """Fit all preprocessing transformers."""
        print("\n" + "="*80)
        print("ADVANCED PREPROCESSING PIPELINE - FITTING")
        print("="*80)

        X_work = X_train.copy()

        # 1. Missing value handling
        print("\n[1/7] Missing Value Handling...")
        self.transformers['missing'] = AdvancedMissingValueHandler(
            strategy=self.config['missing_strategy'],
            create_indicators=True,
            indicator_threshold=0.10,
            group_by_aircraft='aircraft_type' in X_work.columns,
            verbose=self.config['verbose']
        )
        self.transformers['missing'].fit(X_work, y_train)
        X_work = self.transformers['missing'].transform(X_work)

        # 2. Outlier handling
        print("\n[2/7] Outlier Handling...")
        exclude_features = [col for col in X_work.columns
                          if '_is_missing' in col or col in ['flight_id', 'aircraft_type']]
        self.transformers['outliers'] = OutlierHandler(
            method=self.config['outlier_method'],
            percentile_range=self.config.get('outlier_percentile_range', (1, 99)),
            exclude_features=exclude_features,
            verbose=self.config['verbose']
        )
        self.transformers['outliers'].fit(X_work, y_train)
        X_work = self.transformers['outliers'].transform(X_work)

        # 3. Ratio features
        if self.config.get('enable_ratios', True):
            print("\n[3/7] Ratio Features...")
            ratio_pairs = [
                ('distance_covered_km', 'interval_duration_min'),
                ('altitude_delta', 'interval_duration_min'),
                ('climb_work_proxy', 'interval_duration_min'),
                ('altitude_mean', 'groundspeed_mean'),
                ('altitude_range', 'altitude_mean'),
            ]
            # Filter to existing features
            ratio_pairs = [(n, d) for n, d in ratio_pairs
                         if n in X_work.columns and d in X_work.columns]

            self.transformers['ratios'] = RatioFeatureGenerator(
                ratio_pairs=ratio_pairs,
                verbose=self.config['verbose']
            )
            self.transformers['ratios'].fit(X_work)
            X_work = self.transformers['ratios'].transform(X_work)

        # 4. Categorical encoding (LOO)
        print("\n[4/7] Advanced Categorical Encoding...")
        loo_cols = [col for col in ['aircraft_type', 'origin_icao', 'destination_icao']
                   if col in X_work.columns]

        if len(loo_cols) > 0:
            self.transformers['loo_encoder'] = LeaveOneOutEncoder(
                cols=loo_cols,
                smoothing=10.0,
                verbose=self.config['verbose']
            )
            self.transformers['loo_encoder'].fit(X_work, y_train)
            X_work = self.transformers['loo_encoder'].transform(X_work)

        # 5. Polynomial features
        if self.config.get('enable_polynomial', True):
            print("\n[5/7] Polynomial Features...")
            poly_features = [
                'altitude_mean', 'altitude_delta', 'groundspeed_mean',
                'vertical_rate_mean', 'interval_duration_min', 'distance_covered_km',
                'climb_work_proxy'
            ]
            poly_features = [f for f in poly_features if f in X_work.columns]

            if len(poly_features) > 0:
                self.transformers['polynomial'] = PolynomialFeatureGenerator(
                    degree=self.config['polynomial_degree'],
                    interaction_only=True,
                    include_features=poly_features,
                    max_features=12,
                    verbose=self.config['verbose']
                )
                self.transformers['polynomial'].fit(X_work)
                X_work = self.transformers['polynomial'].transform(X_work)

        # 6. Scaling
        print("\n[6/7] Advanced Scaling...")
        self.transformers['scaler'] = AdvancedScaler(
            strategy=self.config['scaling_strategy'],
            exclude_binary=True,
            verbose=self.config['verbose']
        )
        self.transformers['scaler'].fit(X_work, y_train)
        X_work = self.transformers['scaler'].transform(X_work)

        # 7. Feature selection
        if self.config.get('enable_feature_selection', True):
            print("\n[7/7] Feature Selection...")
            self.transformers['selector'] = IntelligentFeatureSelector(
                variance_threshold=self.config.get('variance_threshold', 0.005),
                correlation_threshold=self.config.get('correlation_threshold', 0.95),
                mutual_info_percentile=self.config.get('mutual_info_percentile', 85),
                verbose=self.config['verbose']
            )
            self.transformers['selector'].fit(X_work, y_train)
            X_work = self.transformers['selector'].transform(X_work)

        self.feature_names = X_work.columns.tolist()

        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE - Final shape: {X_work.shape}")
        print("="*80)

        return self

    def transform(self, X):
        """Transform data using fitted transformers."""
        X_work = X.copy()

        for name, transformer in self.transformers.items():
            if hasattr(transformer, 'transform'):
                X_work = transformer.transform(X_work)

        # Ensure same features as training
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(X_work.columns)
            if missing_cols:
                for col in missing_cols:
                    X_work[col] = 0
            X_work = X_work[self.feature_names]

        return X_work


# ============================================================================
# STRATIFIED TRAINING PIPELINE
# ============================================================================

class StratifiedTrainingPipeline:
    """6-stage stratified training with advanced preprocessing."""

    def __init__(self, config):
        self.config = config
        self.stages = {}
        self.preprocessing_pipeline = None
        self.models = {}
        self.results = {}

    def prepare_data(self, X_train=None, y_train=None, X_val=None, y_val=None):
        """Prepare data (uses already-split data from Phase 7 caching)."""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)

        print(f"\nData already split (Phase 7 format):")
        print(f"  Train: {len(X_train):,} samples, {X_train.shape[1]} features")
        print(f"  Val: {len(X_val):,} samples, {X_val.shape[1]} features")

        return X_train, X_val, y_train, y_val

    def fit_preprocessing(self, X_train, y_train):
        """Fit preprocessing pipeline."""
        print("\n" + "="*80)
        print("FITTING PREPROCESSING PIPELINE")
        print("="*80)

        self.preprocessing_pipeline = AdvancedPreprocessingPipeline(
            self.config['preprocessing']
        )
        self.preprocessing_pipeline.fit(X_train, y_train)

        return self

    def apply_preprocessing(self, X_train, X_val):
        """Apply preprocessing to train and validation data."""
        print("\n" + "="*80)
        print("APPLYING PREPROCESSING")
        print("="*80)

        X_train_proc = self.preprocessing_pipeline.transform(X_train)
        X_val_proc = self.preprocessing_pipeline.transform(X_val)

        print(f"\nPreprocessed shapes:")
        print(f"  Train: {X_train_proc.shape}")
        print(f"  Val: {X_val_proc.shape}")

        return X_train_proc, X_val_proc

    def augment_stage_data(self, X_train, y_train, stage_name, min_fuel, max_fuel):
        """Apply data augmentation for a specific stage."""
        aug_config = self.config['augmentation'].get(stage_name)

        if aug_config is None:
            print(f"\n[{stage_name.upper()}] No augmentation configured")
            return X_train, y_train

        print(f"\n[{stage_name.upper()}] Applying augmentation...")
        print(f"  Strategy: {aug_config['strategy']}")
        print(f"  Multiplier: {aug_config['multiplier']}x")

        augmenter = FuelRangeAugmenter(
            augment_ranges=[(min_fuel, max_fuel, aug_config['multiplier'])],
            strategy=aug_config['strategy'],
            random_state=42,
            verbose=False
        )

        X_aug, y_aug = augmenter.fit_resample(X_train, y_train)

        print(f"  Original samples: {len(X_train):,}")
        print(f"  Augmented samples: {len(X_aug):,}")
        print(f"  Added: {len(X_aug) - len(X_train):,}")

        return X_aug, y_aug

    def train_stage_models(self, stage_name, X_train, y_train, X_val, y_val):
        """Train models for a specific stage."""
        print(f"\n{'='*80}")
        print(f"TRAINING STAGE: {stage_name.upper()}")
        print(f"{'='*80}")

        stage_models = {}
        stage_predictions = {}

        model_configs = self.config['models'].get(stage_name, {})

        # Train LightGBM
        if 'lgb' in model_configs:
            print(f"\n[{stage_name.upper()}] Training LightGBM...")

            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                **model_configs['lgb']
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=self.config['training']['num_boost_round'],
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(self.config['training']['early_stopping_rounds']),
                    lgb.log_evaluation(self.config['training']['verbose_eval'])
                ]
            )

            stage_models['lgb'] = model
            stage_predictions['lgb'] = model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            val_rmse = np.sqrt(mean_squared_error(y_val, stage_predictions['lgb']))
            print(f"  LightGBM - Train RMSE: {train_rmse:.2f} kg, Val RMSE: {val_rmse:.2f} kg")

        # Train CatBoost (for high and very_high stages)
        if 'cat' in model_configs:
            print(f"\n[{stage_name.upper()}] Training CatBoost...")

            cat_params = {
                'loss_function': 'RMSE',
                'verbose': 100,
                'task_type': 'CPU',
                **model_configs['cat']
            }

            model = cb.CatBoostRegressor(**cat_params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=100
            )

            stage_models['cat'] = model
            stage_predictions['cat'] = model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            val_rmse = np.sqrt(mean_squared_error(y_val, stage_predictions['cat']))
            print(f"  CatBoost - Train RMSE: {train_rmse:.2f} kg, Val RMSE: {val_rmse:.2f} kg")

        # Create ensemble prediction
        weights = self.config['ensemble_weights'].get(stage_name, {'lgb': 1.0})
        ensemble_pred = np.zeros(len(y_val))

        for model_name, weight in weights.items():
            if model_name in stage_predictions:
                ensemble_pred += weight * stage_predictions[model_name]

        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_r2 = r2_score(y_val, ensemble_pred)

        print(f"\n[{stage_name.upper()}] ENSEMBLE RESULTS:")
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

    def train(self, X_train, y_train, X_val, y_val):
        """Train all stages."""
        print("\n" + "="*80)
        print("STRATIFIED TRAINING PIPELINE")
        print("="*80)

        thresholds = self.config['stratification_thresholds']
        stage_ranges = [
            ('very_low', 0, thresholds[0]),
            ('low', thresholds[0], thresholds[1]),
            ('medium', thresholds[1], thresholds[2]),
            ('medium_high', thresholds[2], thresholds[3]),
            ('high', thresholds[3], thresholds[4]),
            ('very_high', thresholds[4], np.inf)
        ]

        all_predictions = np.zeros(len(y_val))

        for stage_name, min_fuel, max_fuel in stage_ranges:
            # Get stage data
            train_mask = get_stage_mask(y_train, min_fuel, max_fuel)
            val_mask = get_stage_mask(y_val, min_fuel, max_fuel)

            if train_mask.sum() == 0:
                print(f"\n[{stage_name.upper()}] No training samples, skipping...")
                continue

            X_train_stage = X_train[train_mask]
            y_train_stage = y_train[train_mask] if isinstance(y_train, np.ndarray) else y_train[train_mask]
            X_val_stage = X_val[val_mask]
            y_val_stage = y_val[val_mask] if isinstance(y_val, np.ndarray) else y_val[val_mask]

            print_stage_statistics(stage_name, X_train_stage, y_train_stage)
            print(f"Val samples: {len(X_val_stage):,}")

            # Augment training data
            X_train_aug, y_train_aug = self.augment_stage_data(
                X_train_stage, y_train_stage, stage_name, min_fuel, max_fuel
            )

            # Train models
            stage_result = self.train_stage_models(
                stage_name, X_train_aug, y_train_aug, X_val_stage, y_val_stage
            )

            # Store results
            self.models[stage_name] = stage_result['models']
            self.results[stage_name] = {
                'metrics': stage_result['metrics'],
                'range': (min_fuel, max_fuel),
                'train_samples': len(X_train_stage),
                'train_samples_augmented': len(X_train_aug),
                'val_samples': len(X_val_stage)
            }

            # Add predictions to full array
            all_predictions[val_mask] = stage_result['ensemble_pred']

        # Overall evaluation
        overall_rmse = np.sqrt(mean_squared_error(y_val, all_predictions))
        overall_mae = mean_absolute_error(y_val, all_predictions)
        overall_r2 = r2_score(y_val, all_predictions)

        print("\n" + "="*80)
        print("OVERALL ENSEMBLE RESULTS")
        print("="*80)
        print(f"RMSE: {overall_rmse:.2f} kg")
        print(f"MAE: {overall_mae:.2f} kg")
        print(f"R²: {overall_r2:.4f}")

        self.results['overall'] = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'r2': overall_r2,
            'predictions': all_predictions
        }

        return self

    def save_models(self):
        """Save all trained models."""
        models_dir = Path(self.config['paths']['models'])
        models_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("SAVING MODELS")
        print("="*80)

        # Save preprocessing pipeline
        preprocessing_path = models_dir / 'phase8_preprocessing_pipeline.pkl'
        with open(preprocessing_path, 'wb') as f:
            pickle.dump(self.preprocessing_pipeline, f)
        print(f"✓ Saved preprocessing pipeline: {preprocessing_path}")

        # Save stage models
        for stage_name, stage_models in self.models.items():
            for model_type, model in stage_models.items():
                if model_type == 'lgb':
                    model_path = models_dir / f'phase8_{stage_name}_{model_type}.txt'
                    model.save_model(str(model_path))
                elif model_type == 'cat':
                    model_path = models_dir / f'phase8_{stage_name}_{model_type}.cbm'
                    model.save_model(str(model_path))

                print(f"✓ Saved {stage_name} {model_type}: {model_path}")

    def save_results(self):
        """Save training results."""
        reports_dir = Path(self.config['paths']['reports'])
        reports_dir.mkdir(exist_ok=True)

        # Save detailed results
        results_path = reports_dir / f'phase8_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        # Convert numpy types to native Python types
        results_serializable = {}
        for stage_name, stage_results in self.results.items():
            if stage_name == 'overall':
                results_serializable[stage_name] = {
                    'rmse': float(stage_results['rmse']),
                    'mae': float(stage_results['mae']),
                    'r2': float(stage_results['r2'])
                }
            else:
                results_serializable[stage_name] = {
                    'metrics': {
                        'rmse': float(stage_results['metrics']['rmse']),
                        'mae': float(stage_results['metrics']['mae']),
                        'r2': float(stage_results['metrics']['r2'])
                    },
                    'range': [float(stage_results['range'][0]),
                             float(stage_results['range'][1]) if stage_results['range'][1] != np.inf else 'inf'],
                    'train_samples': int(stage_results['train_samples']),
                    'train_samples_augmented': int(stage_results['train_samples_augmented']),
                    'val_samples': int(stage_results['val_samples'])
                }

        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n✓ Saved results: {results_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                     PHASE 8: ULTIMATE TRAINING PIPELINE                  ║
    ║                                                                          ║
    ║  Advanced Preprocessing + 6-Stage Stratification + Ensemble              ║
    ║  Target: < 130 kg RMSE (from 144.19 kg baseline)                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Load data (Phase 7 format - already split)
    print("\n" + "="*80)
    print("LOADING DATA (Phase 7 Format)")
    print("="*80)

    cache_dir = Path('cache')

    # Load enhanced features (or fall back to clean)
    if (cache_dir / 'X_train_enhanced.parquet').exists():
        print("Using enhanced features...")
        X_train = pd.read_parquet(cache_dir / 'X_train_enhanced.parquet')
        X_val = pd.read_parquet(cache_dir / 'X_val_enhanced.parquet')
    elif (cache_dir / 'X_train_clean.parquet').exists():
        print("Using clean features...")
        X_train = pd.read_parquet(cache_dir / 'X_train_clean.parquet')
        X_val = pd.read_parquet(cache_dir / 'X_val_clean.parquet')
    else:
        print(f"\n❌ ERROR: No cached features found in {cache_dir}")
        print("\nExpected files (one of):")
        print("  - X_train_enhanced.parquet / X_val_enhanced.parquet")
        print("  - X_train_clean.parquet / X_val_clean.parquet")
        print("\nPlease run Phase 7 feature extraction first.")
        return

    y_train = pd.read_parquet(cache_dir / 'y_train.parquet')['fuel_kg']
    y_val = pd.read_parquet(cache_dir / 'y_val.parquet')['fuel_kg']

    print(f"✓ Loaded training data: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"✓ Loaded validation data: X_val {X_val.shape}, y_val {y_val.shape}")

    # Initialize pipeline
    pipeline = StratifiedTrainingPipeline(PHASE8_CONFIG)

    # Prepare data (just validation, data already split)
    X_train, X_val, y_train, y_val = pipeline.prepare_data(X_train, y_train, X_val, y_val)

    # Fit and apply preprocessing
    pipeline.fit_preprocessing(X_train, y_train)
    X_train_proc, X_val_proc = pipeline.apply_preprocessing(X_train, X_val)

    # Train all stages
    pipeline.train(X_train_proc, y_train, X_val_proc, y_val)

    # Save models and results
    pipeline.save_models()
    pipeline.save_results()

    print("\n" + "="*80)
    print("PHASE 8 TRAINING COMPLETE!")
    print("="*80)
    print("\nFinal Results:")
    print(f"  RMSE: {pipeline.results['overall']['rmse']:.2f} kg")
    print(f"  MAE: {pipeline.results['overall']['mae']:.2f} kg")
    print(f"  R²: {pipeline.results['overall']['r2']:.4f}")
    print(f"\nImprovement from Phase 7: {144.19 - pipeline.results['overall']['rmse']:.2f} kg")

    # Print per-stage results
    print("\n" + "="*80)
    print("PER-STAGE RESULTS")
    print("="*80)

    for stage_name in ['very_low', 'low', 'medium', 'medium_high', 'high', 'very_high']:
        if stage_name in pipeline.results:
            result = pipeline.results[stage_name]
            print(f"\n{stage_name.upper()}:")
            print(f"  Range: {result['range'][0]:.0f} - {result['range'][1]} kg")
            print(f"  Val samples: {result['val_samples']:,}")
            print(f"  RMSE: {result['metrics']['rmse']:.2f} kg")
            print(f"  MAE: {result['metrics']['mae']:.2f} kg")


if __name__ == "__main__":
    main()
