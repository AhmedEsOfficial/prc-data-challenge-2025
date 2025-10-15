"""
Comprehensive Preprocessing Integration Guide
==============================================
Shows how to use all advanced preprocessing modules for optimal results.

This guide demonstrates the COMPLETE preprocessing workflow to push toward
theoretical practical limits (target: ~130-135 kg RMSE).

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
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
    IntelligentFeatureSelector,
    create_preprocessing_pipeline,
    apply_preprocessing_pipeline
)

from src.data_augmentation import (
    FuelRangeAugmenter,
    StratifiedAugmenter,
    validate_augmentation
)

from src.preprocessing import (
    create_temporal_split,
    validate_no_leakage
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    # Missing value handling
    'missing_strategy': 'hybrid',  # hybrid, knn, iterative, median

    # Outlier handling
    'outlier_method': 'percentile',  # percentile, iqr, zscore, none
    'outlier_percentile_range': (0.5, 99.5),  # Slightly more aggressive

    # Scaling
    'scaling_strategy': 'hybrid',  # hybrid, robust, quantile, power, standard

    # Feature engineering
    'polynomial_degree': 2,
    'enable_polynomial': True,
    'enable_ratios': True,
    'enable_temporal': True,

    # Feature selection
    'enable_feature_selection': True,
    'variance_threshold': 0.005,
    'correlation_threshold': 0.95,
    'mutual_info_percentile': 85,

    # Data augmentation
    'enable_augmentation': True,
    'augmentation_strategy': 'stratified',  # stratified, simple

    # Verbose output
    'verbose': True
}


# ============================================================================
# STEP 1: DEFINE KEY FEATURES FOR ADVANCED PROCESSING
# ============================================================================

def get_feature_groups(df):
    """
    Define feature groups for intelligent processing.

    Returns:
        Dictionary with feature categories
    """
    groups = {
        # Features for polynomial interactions (most important)
        'polynomial_features': [
            'altitude_mean',
            'altitude_delta',
            'groundspeed_mean',
            'vertical_rate_mean',
            'interval_duration_min',
            'distance_covered_km',
            'climb_work_proxy',
            'altitude_mean_normalized'
        ],

        # Ratio pairs (numerator, denominator)
        'ratio_pairs': [
            ('distance_covered_km', 'interval_duration_min'),  # Speed proxy
            ('altitude_delta', 'interval_duration_min'),  # Climb rate
            ('climb_work_proxy', 'interval_duration_min'),  # Power proxy
            ('altitude_mean', 'groundspeed_mean'),  # Flight regime
            ('altitude_range', 'altitude_mean'),  # Altitude variability
        ],

        # Datetime columns for temporal features
        'datetime_cols': [
            'start_time' if 'start_time' in df.columns else None
        ],

        # Features to exclude from outlier handling (binary/categorical)
        'outlier_exclude': [
            'flight_id',
            'aircraft_type_target_enc',
            'detected_flight_phase'
        ] + [col for col in df.columns if '_is_missing' in col or col.startswith('detected_flight_phase_')],

        # High-cardinality categorical for LOO encoding
        'loo_encode_features': [
            'aircraft_type',
            'origin_icao',
            'destination_icao'
        ]
    }

    # Filter out None values
    groups['datetime_cols'] = [col for col in groups['datetime_cols'] if col is not None]

    return groups


# ============================================================================
# STEP 2: COMPREHENSIVE PREPROCESSING FUNCTION
# ============================================================================

def preprocess_data_advanced(X_train, X_val, y_train, config=None):
    """
    Apply comprehensive advanced preprocessing.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        config: Configuration dictionary

    Returns:
        X_train_processed, X_val_processed, preprocessing_pipeline
    """
    config = config or PREPROCESSING_CONFIG

    print("\n" + "="*80)
    print("ADVANCED PREPROCESSING PIPELINE")
    print("="*80)
    print(f"\nInitial shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")

    # Get feature groups
    feature_groups = get_feature_groups(X_train)

    # ========================================================================
    # PHASE 1: MISSING VALUE HANDLING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: MISSING VALUE HANDLING")
    print("="*80)

    missing_handler = AdvancedMissingValueHandler(
        strategy=config['missing_strategy'],
        create_indicators=True,
        indicator_threshold=0.10,
        group_by_aircraft='aircraft_type' in X_train.columns,
        verbose=config['verbose']
    )

    missing_handler.fit(X_train, y_train)
    X_train = missing_handler.transform(X_train)
    X_val = missing_handler.transform(X_val)

    print(f"\nShapes after missing value handling:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")

    # ========================================================================
    # PHASE 2: OUTLIER HANDLING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: OUTLIER HANDLING")
    print("="*80)

    outlier_handler = OutlierHandler(
        method=config['outlier_method'],
        percentile_range=config.get('outlier_percentile_range', (1, 99)),
        exclude_features=feature_groups['outlier_exclude'],
        verbose=config['verbose']
    )

    outlier_handler.fit(X_train, y_train)
    X_train = outlier_handler.transform(X_train)
    X_val = outlier_handler.transform(X_val)

    # ========================================================================
    # PHASE 3: TEMPORAL FEATURE ENGINEERING
    # ========================================================================
    if config.get('enable_temporal', True) and len(feature_groups['datetime_cols']) > 0:
        print("\n" + "="*80)
        print("PHASE 3: TEMPORAL FEATURE ENGINEERING")
        print("="*80)

        temporal_engineer = TemporalFeatureEngineer(
            datetime_cols=feature_groups['datetime_cols'],
            cyclical_encoding=True,
            verbose=config['verbose']
        )

        temporal_engineer.fit(X_train)
        X_train = temporal_engineer.transform(X_train)
        X_val = temporal_engineer.transform(X_val)

        print(f"\nShapes after temporal features:")
        print(f"  X_train: {X_train.shape}")

    # ========================================================================
    # PHASE 4: RATIO FEATURES
    # ========================================================================
    if config.get('enable_ratios', True):
        print("\n" + "="*80)
        print("PHASE 4: RATIO FEATURES")
        print("="*80)

        ratio_generator = RatioFeatureGenerator(
            ratio_pairs=feature_groups['ratio_pairs'],
            verbose=config['verbose']
        )

        ratio_generator.fit(X_train)
        X_train = ratio_generator.transform(X_train)
        X_val = ratio_generator.transform(X_val)

        print(f"\nShapes after ratio features:")
        print(f"  X_train: {X_train.shape}")

    # ========================================================================
    # PHASE 5: CATEGORICAL ENCODING (LOO for high cardinality)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: ADVANCED CATEGORICAL ENCODING")
    print("="*80)

    loo_cols = [col for col in feature_groups['loo_encode_features'] if col in X_train.columns]

    if len(loo_cols) > 0:
        loo_encoder = LeaveOneOutEncoder(
            cols=loo_cols,
            smoothing=10.0,  # Strong smoothing to prevent overfitting
            verbose=config['verbose']
        )

        loo_encoder.fit(X_train, y_train)
        X_train = loo_encoder.transform(X_train)
        X_val = loo_encoder.transform(X_val)

        print(f"\nShapes after LOO encoding:")
        print(f"  X_train: {X_train.shape}")

    # ========================================================================
    # PHASE 6: POLYNOMIAL FEATURES (Carefully!)
    # ========================================================================
    if config.get('enable_polynomial', True):
        print("\n" + "="*80)
        print("PHASE 6: POLYNOMIAL FEATURES")
        print("="*80)

        # Filter polynomial features to those that exist
        poly_features = [col for col in feature_groups['polynomial_features']
                        if col in X_train.columns]

        if len(poly_features) > 0:
            poly_generator = PolynomialFeatureGenerator(
                degree=config['polynomial_degree'],
                interaction_only=True,  # Only interactions, not higher powers
                include_features=poly_features,
                max_features=15,  # Limit to prevent explosion
                verbose=config['verbose']
            )

            poly_generator.fit(X_train)
            X_train = poly_generator.transform(X_train)
            X_val = poly_generator.transform(X_val)

            print(f"\nShapes after polynomial features:")
            print(f"  X_train: {X_train.shape}")

    # ========================================================================
    # PHASE 7: ADVANCED SCALING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: ADVANCED SCALING")
    print("="*80)

    scaler = AdvancedScaler(
        strategy=config['scaling_strategy'],
        exclude_binary=True,
        verbose=config['verbose']
    )

    scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # ========================================================================
    # PHASE 8: FEATURE SELECTION
    # ========================================================================
    if config.get('enable_feature_selection', True):
        print("\n" + "="*80)
        print("PHASE 8: INTELLIGENT FEATURE SELECTION")
        print("="*80)

        feature_selector = IntelligentFeatureSelector(
            variance_threshold=config.get('variance_threshold', 0.005),
            correlation_threshold=config.get('correlation_threshold', 0.95),
            mutual_info_percentile=config.get('mutual_info_percentile', 85),
            verbose=config['verbose']
        )

        feature_selector.fit(X_train, y_train)
        X_train = feature_selector.transform(X_train)
        X_val = feature_selector.transform(X_val)

        print(f"\nShapes after feature selection:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"\nFeature expansion: {X_train.shape[1]} features (from initial)")

    # Store pipeline for reuse
    pipeline = {
        'missing': missing_handler,
        'outliers': outlier_handler,
        'temporal': temporal_engineer if config.get('enable_temporal') else None,
        'ratios': ratio_generator if config.get('enable_ratios') else None,
        'loo_encoder': loo_encoder if len(loo_cols) > 0 else None,
        'polynomial': poly_generator if config.get('enable_polynomial') else None,
        'scaler': scaler,
        'selector': feature_selector if config.get('enable_feature_selection') else None
    }

    # Remove None entries
    pipeline = {k: v for k, v in pipeline.items() if v is not None}

    return X_train, X_val, pipeline


# ============================================================================
# STEP 3: DATA AUGMENTATION FOR EXTREME RANGES
# ============================================================================

def augment_training_data(X_train, y_train, config=None):
    """
    Apply data augmentation to handle extreme fuel ranges.

    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary

    Returns:
        X_train_augmented, y_train_augmented
    """
    config = config or PREPROCESSING_CONFIG

    if not config.get('enable_augmentation', True):
        print("\n[Augmentation] Disabled by config")
        return X_train, y_train

    print("\n" + "="*80)
    print("DATA AUGMENTATION")
    print("="*80)

    if config['augmentation_strategy'] == 'stratified':
        # Stratified augmentation with different strategies per range
        augmentation_config = [
            {
                'range': (1000, 1500),
                'strategy': 'noise',
                'multiplier': 1.5
            },
            {
                'range': (1500, 2500),
                'strategy': 'noise',
                'multiplier': 2.0
            },
            {
                'range': (2500, np.inf),
                'strategy': 'interpolate',
                'multiplier': 4.0
            }
        ]

        augmenter = StratifiedAugmenter(
            augmentation_config=augmentation_config,
            verbose=config['verbose']
        )

        X_train_aug, y_train_aug = augmenter.fit_resample(X_train, y_train)

    else:
        # Simple augmentation
        augmenter = FuelRangeAugmenter(
            augment_ranges=[(2500, np.inf, 4.0)],
            strategy='interpolate',
            verbose=config['verbose']
        )

        X_train_aug, y_train_aug = augmenter.fit_resample(X_train, y_train)

    # Validate augmentation
    validate_augmentation(X_train, y_train, X_train_aug, y_train_aug)

    return X_train_aug, y_train_aug


# ============================================================================
# STEP 4: COMPLETE WORKFLOW EXAMPLE
# ============================================================================

def complete_preprocessing_workflow(df, split_month=8):
    """
    Complete end-to-end preprocessing workflow.

    Args:
        df: Full dataset with features and target
        split_month: Month to use for validation split

    Returns:
        X_train, X_val, y_train, y_val, pipeline
    """
    print("\n" + "="*80)
    print("COMPLETE PREPROCESSING WORKFLOW")
    print("="*80)

    # Step 1: Split data
    print("\nStep 1: Temporal split...")
    train_df, val_df = create_temporal_split(
        df,
        split_month=split_month,
        date_column='start_month',
        flight_id_col='flight_id'
    )

    # Validate no leakage
    validate_no_leakage(train_df, val_df, identifier_col='flight_id')

    # Separate features and target
    target_col = 'fuel_kg'

    # Remove non-feature columns
    exclude_cols = [target_col, 'flight_id', 'idx']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    print(f"\nInitial split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val: {len(X_val):,} samples")

    # Step 2: Advanced preprocessing
    print("\nStep 2: Advanced preprocessing...")
    X_train_proc, X_val_proc, pipeline = preprocess_data_advanced(
        X_train, X_val, y_train, config=PREPROCESSING_CONFIG
    )

    # Step 3: Data augmentation (optional but recommended)
    print("\nStep 3: Data augmentation...")
    X_train_aug, y_train_aug = augment_training_data(
        X_train_proc, y_train, config=PREPROCESSING_CONFIG
    )

    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print(f"\nFinal training set:")
    print(f"  Samples: {len(X_train_aug):,}")
    print(f"  Features: {X_train_aug.shape[1]}")
    print(f"\nFinal validation set:")
    print(f"  Samples: {len(X_val_proc):,}")
    print(f"  Features: {X_val_proc.shape[1]}")

    return X_train_aug, X_val_proc, y_train_aug, y_val, pipeline


# ============================================================================
# STEP 5: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the advanced preprocessing pipeline.
    """

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                   ADVANCED PREPROCESSING PIPELINE                         ║
    ║                 Pushing Toward Theoretical Practical Limits               ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    This script demonstrates how to use all advanced preprocessing modules:

    1. Advanced Missing Value Handling (KNN, Iterative, Hybrid)
    2. Outlier Detection and Handling
    3. Temporal Feature Engineering (Cyclical encoding)
    4. Ratio Features (Domain-informed ratios)
    5. Advanced Categorical Encoding (Leave-One-Out)
    6. Polynomial Features (Intelligent interactions)
    7. Advanced Scaling (Robust, Quantile, Power transforms)
    8. Intelligent Feature Selection
    9. Data Augmentation for Extreme Ranges

    Expected Improvements:
    - Better handling of missing data patterns
    - Reduced impact of outliers
    - More informative features
    - Better generalization to extreme cases
    - Target: ~130-135 kg RMSE (from current 144.19 kg)

    To use this pipeline with your data:

    ```python
    # Load your processed features
    df = pd.read_parquet('cache/processed_features.parquet')

    # Run complete workflow
    X_train, X_val, y_train, y_val, pipeline = complete_preprocessing_workflow(
        df, split_month=8
    )

    # Train your model
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Evaluate
    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"\\nValidation RMSE: {rmse:.2f} kg")
    ```

    For stratified modeling (recommended):
    - Apply this preprocessing to each fuel range separately
    - Use augmentation strategically for high-variance ranges
    - Combine predictions with ensemble approach

    """)
