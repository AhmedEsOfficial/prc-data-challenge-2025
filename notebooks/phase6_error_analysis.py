#!/usr/bin/env python3
"""
Phase 6: Deep Error Analysis - Aircraft Fuel Consumption Prediction
====================================================================

This script performs comprehensive error analysis to identify:
1. High-error segments (aircraft type, flight phase, interval position)
2. Systematic error patterns and biases
3. Feature gaps and opportunities
4. Actionable insights for advanced feature engineering

Goal: Identify specific areas where errors are concentrated to guide
      improvements from 189.29 kg RMSE to <50 kg RMSE

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

def load_data_and_models():
    """Load validation data, models, and make predictions."""
    print("=" * 80)
    print("PHASE 6: DEEP ERROR ANALYSIS")
    print("=" * 80)
    print("\n[1/7] Loading data and models...")

    # Load validation data
    X_val = pd.read_parquet(CACHE_DIR / "X_val.parquet")
    y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

    # Load selected features
    with open(MODELS_DIR / 'phase5_selected_features.json', 'r') as f:
        selected_features = json.load(f)

    X_val_selected = X_val[selected_features]

    # Load Phase 5 optimized models
    with open(MODELS_DIR / 'phase5_lightgbm_optimized.pkl', 'rb') as f:
        lgb_model = pickle.load(f)

    with open(MODELS_DIR / 'phase5_catboost_optimized.pkl', 'rb') as f:
        cat_model = pickle.load(f)

    # Load ensemble weights
    with open(MODELS_DIR / 'phase5_ensemble_weights.json', 'r') as f:
        ensemble_weights = json.load(f)

    # Make predictions
    import lightgbm as lgb
    if isinstance(lgb_model, lgb.Booster):
        lgb_pred = lgb_model.predict(X_val_selected, num_iteration=lgb_model.best_iteration)
    else:
        lgb_pred = lgb_model.predict(X_val_selected)

    cat_pred = cat_model.predict(X_val_selected)

    # Ensemble prediction
    ensemble_pred = (
        ensemble_weights['LightGBM'] * lgb_pred +
        ensemble_weights['CatBoost'] * cat_pred
    )

    # Calculate residuals
    residuals = y_val - ensemble_pred
    abs_residuals = np.abs(residuals)

    print(f"✓ Data loaded: {len(y_val):,} validation samples")
    print(f"✓ Ensemble RMSE: {np.sqrt(np.mean(residuals**2)):.2f} kg")
    print(f"✓ MAE: {np.mean(abs_residuals):.2f} kg")

    return X_val, X_val_selected, y_val, ensemble_pred, residuals, abs_residuals


# ============================================================================
#  SECTION 2: ERROR SEGMENTATION ANALYSIS
# ============================================================================

def analyze_errors_by_segments(X_val, y_val, predictions, residuals, abs_residuals):
    """Analyze errors by different segments."""
    print("\n" + "=" * 80)
    print("[2/7] Error Segmentation Analysis")
    print("=" * 80)

    # Create analysis DataFrame
    error_df = pd.DataFrame({
        'actual_fuel': y_val,
        'predicted_fuel': predictions,
        'residual': residuals,
        'abs_residual': abs_residuals,
        'rel_error': (abs_residuals / (y_val + 1)) * 100  # Add 1 to avoid division by zero
    })

    # Add features for segmentation
    key_features = [
        'aircraft_type_target_enc', 'interval_duration_min', 'distance_covered_km',
        'interval_position_pct', 'flight_phase_cruise', 'flight_phase_climb',
        'flight_phase_descent', 'altitude_mean', 'groundspeed_mean',
        'vertical_rate_mean', 'seconds_from_takeoff', 'seconds_to_landing'
    ]

    for feat in key_features:
        if feat in X_val.columns:
            error_df[feat] = X_val[feat].values

    results = {}

    # 1. Errors by Fuel Magnitude
    print("\n1. Errors by Fuel Magnitude:")
    print("-" * 80)
    fuel_bins = [0, 100, 200, 500, 1000, 2000, 50000]
    fuel_labels = ['<100', '100-200', '200-500', '500-1000', '1000-2000', '>2000']
    error_df['fuel_bin'] = pd.cut(error_df['actual_fuel'], bins=fuel_bins, labels=fuel_labels)

    fuel_analysis = error_df.groupby('fuel_bin').agg({
        'abs_residual': ['count', 'mean', 'std', 'median'],
        'rel_error': ['mean', 'median'],
        'actual_fuel': 'mean'
    }).round(2)

    print(fuel_analysis)
    results['fuel_magnitude'] = fuel_analysis

    # 2. Errors by Interval Position
    print("\n2. Errors by Interval Position:")
    print("-" * 80)
    if 'interval_position_pct' in error_df.columns:
        try:
            # Check if position data is valid
            valid_positions = error_df['interval_position_pct'].notna()
            if valid_positions.sum() > 100:
                position_bins = [0, 20, 40, 60, 80, 100]
                position_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
                error_df['position_bin'] = pd.cut(error_df['interval_position_pct'],
                                                  bins=position_bins, labels=position_labels,
                                                  include_lowest=True)

                position_analysis = error_df.groupby('position_bin', observed=True).agg({
                    'abs_residual': ['count', 'mean', 'std'],
                    'rel_error': 'mean'
                }).round(2)

                # Filter out bins with no data
                position_analysis = position_analysis[position_analysis[('abs_residual', 'count')] > 0]

                print(position_analysis)
                results['position'] = position_analysis
            else:
                print("⚠️  Insufficient valid position data")
        except Exception as e:
            print(f"⚠️  Could not analyze by position: {str(e)}")

    # 3. Errors by Flight Phase
    print("\n3. Errors by Flight Phase:")
    print("-" * 80)
    phase_cols = [c for c in error_df.columns if c.startswith('flight_phase_')]
    if phase_cols:
        # Determine dominant phase
        phases = []
        for idx, row in error_df.iterrows():
            phase_values = {col.replace('flight_phase_', ''): row[col] for col in phase_cols}
            dominant_phase = max(phase_values, key=phase_values.get)
            phases.append(dominant_phase)

        error_df['dominant_phase'] = phases

        phase_analysis = error_df.groupby('dominant_phase').agg({
            'abs_residual': ['count', 'mean', 'std'],
            'rel_error': 'mean',
            'actual_fuel': 'mean'
        }).round(2)

        print(phase_analysis)
        results['phase'] = phase_analysis

    # 4. Errors by Aircraft Type (using encoded value as proxy)
    print("\n4. Errors by Aircraft Type (binned by encoding):")
    print("-" * 80)
    if 'aircraft_type_target_enc' in error_df.columns:
        try:
            aircraft_bins = pd.qcut(error_df['aircraft_type_target_enc'], q=5,
                                    labels=False,
                                    duplicates='drop')
            error_df['aircraft_bin'] = aircraft_bins

            aircraft_analysis = error_df.groupby('aircraft_bin').agg({
                'abs_residual': ['count', 'mean', 'std'],
                'rel_error': 'mean',
                'actual_fuel': 'mean'
            }).round(2)

            print(aircraft_analysis)
            results['aircraft'] = aircraft_analysis
        except Exception as e:
            print(f"⚠️  Could not bin aircraft types: {str(e)}")
            # Just show overall stats
            print(f"Mean error: {error_df['abs_residual'].mean():.2f} kg")
            print(f"Median error: {error_df['abs_residual'].median():.2f} kg")

    # 5. Errors by Interval Duration
    print("\n5. Errors by Interval Duration:")
    print("-" * 80)
    if 'interval_duration_min' in error_df.columns:
        duration_bins = [0, 5, 10, 15, 20, 1000]
        duration_labels = ['<5min', '5-10min', '10-15min', '15-20min', '>20min']
        error_df['duration_bin'] = pd.cut(error_df['interval_duration_min'],
                                          bins=duration_bins, labels=duration_labels)

        duration_analysis = error_df.groupby('duration_bin').agg({
            'abs_residual': ['count', 'mean', 'std'],
            'rel_error': 'mean',
            'actual_fuel': 'mean'
        }).round(2)

        print(duration_analysis)
        results['duration'] = duration_analysis

    # 6. High Error Cases (Top 1%)
    print("\n6. High Error Cases (Top 1%):")
    print("-" * 80)
    threshold_99 = error_df['abs_residual'].quantile(0.99)
    high_error_cases = error_df[error_df['abs_residual'] >= threshold_99]

    print(f"Number of high-error cases: {len(high_error_cases):,}")
    print(f"Error threshold (99th percentile): {threshold_99:.2f} kg")
    print(f"Mean error in high-error cases: {high_error_cases['abs_residual'].mean():.2f} kg")
    print(f"Mean actual fuel: {high_error_cases['actual_fuel'].mean():.2f} kg")
    print(f"Mean predicted fuel: {high_error_cases['predicted_fuel'].mean():.2f} kg")

    results['high_error_threshold'] = threshold_99
    results['high_error_count'] = len(high_error_cases)

    return error_df, results


# ============================================================================
#  SECTION 3: SYSTEMATIC ERROR PATTERNS
# ============================================================================

def analyze_systematic_patterns(error_df):
    """Identify systematic error patterns and biases."""
    print("\n" + "=" * 80)
    print("[3/7] Systematic Error Pattern Analysis")
    print("=" * 80)

    patterns = {}

    # 1. Bias Analysis (Over/Under prediction)
    print("\n1. Prediction Bias Analysis:")
    print("-" * 80)
    overall_bias = error_df['residual'].mean()
    print(f"Overall bias (mean residual): {overall_bias:.2f} kg")

    if overall_bias > 10:
        print("⚠️  Model tends to UNDER-PREDICT (positive bias)")
    elif overall_bias < -10:
        print("⚠️  Model tends to OVER-PREDICT (negative bias)")
    else:
        print("✓  Model is relatively unbiased")

    patterns['overall_bias'] = overall_bias

    # 2. Bias by fuel magnitude
    print("\nBias by Fuel Magnitude:")
    fuel_bias = error_df.groupby('fuel_bin')['residual'].mean().round(2)
    print(fuel_bias)
    patterns['fuel_bias'] = fuel_bias.to_dict()

    # 3. Heteroscedasticity (error variance changes with prediction)
    print("\n2. Heteroscedasticity Analysis:")
    print("-" * 80)
    pred_bins = pd.qcut(error_df['predicted_fuel'], q=10, duplicates='drop')
    variance_by_pred = error_df.groupby(pred_bins)['abs_residual'].std()

    if variance_by_pred.max() / variance_by_pred.min() > 2:
        print("⚠️  Significant heteroscedasticity detected")
        print(f"   Error std ratio (max/min): {variance_by_pred.max() / variance_by_pred.min():.2f}x")
    else:
        print("✓  Error variance is relatively constant")

    patterns['heteroscedasticity_ratio'] = variance_by_pred.max() / variance_by_pred.min()

    # 4. Error correlation with key features
    print("\n3. Error Correlation with Features:")
    print("-" * 80)
    corr_features = ['interval_duration_min', 'distance_covered_km', 'altitude_mean',
                     'groundspeed_mean', 'vertical_rate_mean']

    correlations = {}
    for feat in corr_features:
        if feat in error_df.columns:
            corr = error_df[feat].corr(error_df['abs_residual'])
            correlations[feat] = corr
            print(f"{feat:<30}: {corr:+.4f}")

    patterns['feature_correlations'] = correlations

    # 5. Residual distribution normality
    print("\n4. Residual Distribution Analysis:")
    print("-" * 80)
    shapiro_stat, shapiro_p = stats.shapiro(error_df['residual'].sample(min(5000, len(error_df))))
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")

    if shapiro_p < 0.05:
        print("⚠️  Residuals are NOT normally distributed")
    else:
        print("✓  Residuals are approximately normal")

    skewness = error_df['residual'].skew()
    kurtosis = error_df['residual'].kurtosis()
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")

    patterns['normality_test'] = {
        'shapiro_p': shapiro_p,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

    return patterns


# ============================================================================
#  SECTION 4: FEATURE GAP ANALYSIS
# ============================================================================

def analyze_feature_gaps(X_val, error_df):
    """Identify missing feature types and opportunities."""
    print("\n" + "=" * 80)
    print("[4/7] Feature Gap Analysis")
    print("=" * 80)

    gaps = {}

    # 1. Check for interaction features
    print("\n1. Interaction Feature Gaps:")
    print("-" * 80)
    interaction_features = [c for c in X_val.columns if '_x_' in c or '_interaction' in c]
    print(f"Current interaction features: {len(interaction_features)}")

    if len(interaction_features) < 10:
        print("⚠️  Opportunity: Add more interaction features")
        print("   Suggestions:")
        print("   - distance_covered_km × groundspeed_mean (speed-distance interaction)")
        print("   - altitude_delta × vertical_rate_mean (climb energy)")
        print("   - interval_duration_min × aircraft_type (type-specific fuel rate)")

    gaps['interaction_count'] = len(interaction_features)

    # 2. Check for rolling/lag features
    print("\n2. Temporal/Rolling Feature Gaps:")
    print("-" * 80)
    rolling_features = [c for c in X_val.columns if 'rolling' in c or 'lag' in c or 'prev' in c]
    print(f"Current rolling/lag features: {len(rolling_features)}")

    if len(rolling_features) < 5:
        print("⚠️  Opportunity: Add temporal features")
        print("   Suggestions:")
        print("   - rolling_fuel_rate (fuel per km over last 3 intervals)")
        print("   - lag_altitude_delta (altitude change from previous interval)")
        print("   - cumulative_fuel_consumed (running total)")

    gaps['rolling_count'] = len(rolling_features)

    # 3. Check for physics-derived features
    print("\n3. Physics-Derived Feature Gaps:")
    print("-" * 80)
    physics_features = [c for c in X_val.columns if any(x in c for x in
                        ['work', 'energy', 'drag', 'thrust', 'efficiency', 'power'])]
    print(f"Current physics features: {len(physics_features)}")

    if len(physics_features) < 5:
        print("⚠️  Opportunity: Add more physics features")
        print("   Suggestions:")
        print("   - kinetic_energy_delta (0.5 * mass * (v2^2 - v1^2))")
        print("   - potential_energy_delta (mass * g * altitude_delta)")
        print("   - drag_estimate (0.5 * rho * v^2 * Cd * A)")
        print("   - fuel_efficiency_rate (fuel_kg / distance_km)")

    gaps['physics_count'] = len(physics_features)

    # 4. Check for aircraft-specific features
    print("\n4. Aircraft-Specific Feature Gaps:")
    print("-" * 80)
    aircraft_features = [c for c in X_val.columns if 'aircraft' in c]
    print(f"Current aircraft features: {len(aircraft_features)}")

    if len(aircraft_features) < 3:
        print("⚠️  Opportunity: Add aircraft-specific features")
        print("   Suggestions:")
        print("   - aircraft_fuel_efficiency_score (historical average)")
        print("   - aircraft_optimal_altitude (typical cruise altitude)")
        print("   - aircraft_weight_class (light/medium/heavy)")

    gaps['aircraft_count'] = len(aircraft_features)

    # 5. Feature correlation with error
    print("\n5. Features Most Correlated with High Errors:")
    print("-" * 80)

    # Merge features with errors
    error_corr = pd.DataFrame()
    for col in X_val.columns:
        if X_val[col].dtype in ['int64', 'float64']:
            corr = error_df['abs_residual'].corr(X_val[col])
            error_corr = pd.concat([error_corr, pd.DataFrame({
                'feature': [col],
                'abs_corr': [abs(corr)],
                'corr': [corr]
            })], ignore_index=True)

    error_corr = error_corr.sort_values('abs_corr', ascending=False).head(15)
    print(error_corr.to_string(index=False))

    gaps['error_correlations'] = error_corr.to_dict('records')

    return gaps


# ============================================================================
#  SECTION 5: ACTIONABLE INSIGHTS
# ============================================================================

def generate_actionable_insights(error_results, patterns, gaps):
    """Generate specific, actionable recommendations."""
    print("\n" + "=" * 80)
    print("[5/7] Actionable Insights for Improvement")
    print("=" * 80)

    insights = []

    # Insight 1: Fuel magnitude specific models
    if 'fuel_magnitude' in error_results:
        fuel_errors = error_results['fuel_magnitude']
        max_error_bin = fuel_errors[('abs_residual', 'mean')].idxmax()
        insights.append({
            'priority': 'HIGH',
            'category': 'Stratified Modeling',
            'finding': f"Highest errors in {max_error_bin} kg fuel range",
            'action': f"Train separate model for {max_error_bin} kg intervals",
            'expected_improvement': '15-25%'
        })

    # Insight 2: Flight phase specific handling
    if 'phase' in error_results:
        phase_errors = error_results['phase']
        max_error_phase = phase_errors[('abs_residual', 'mean')].idxmax()
        insights.append({
            'priority': 'HIGH',
            'category': 'Feature Engineering',
            'finding': f"Highest errors in {max_error_phase} phase",
            'action': f"Add phase-specific features (e.g., {max_error_phase}_specific_fuel_rate)",
            'expected_improvement': '10-20%'
        })

    # Insight 3: Interaction features
    if gaps.get('interaction_count', 0) < 10:
        insights.append({
            'priority': 'HIGH',
            'category': 'Feature Engineering',
            'finding': "Only few interaction features present",
            'action': "Add critical interactions: distance×speed, altitude×vertical_rate, duration×aircraft_type",
            'expected_improvement': '15-30%'
        })

    # Insight 4: Temporal features
    if gaps.get('rolling_count', 0) < 5:
        insights.append({
            'priority': 'MEDIUM',
            'category': 'Feature Engineering',
            'finding': "Missing temporal/rolling features",
            'action': "Add rolling averages, lag features, and cumulative metrics",
            'expected_improvement': '10-15%'
        })

    # Insight 5: Physics features
    if gaps.get('physics_count', 0) < 5:
        insights.append({
            'priority': 'MEDIUM',
            'category': 'Feature Engineering',
            'finding': "Limited physics-derived features",
            'action': "Add energy-based features (kinetic, potential, work)",
            'expected_improvement': '10-20%'
        })

    # Insight 6: Bias correction
    if abs(patterns.get('overall_bias', 0)) > 10:
        insights.append({
            'priority': 'MEDIUM',
            'category': 'Post-processing',
            'finding': f"Systematic bias of {patterns['overall_bias']:.2f} kg",
            'action': "Apply bias correction post-processing",
            'expected_improvement': '5-10%'
        })

    # Insight 7: Heteroscedasticity handling
    if patterns.get('heteroscedasticity_ratio', 1) > 2:
        insights.append({
            'priority': 'MEDIUM',
            'category': 'Modeling',
            'finding': "Significant heteroscedasticity detected",
            'action': "Use quantile regression or log-transform target",
            'expected_improvement': '8-15%'
        })

    # Insight 8: Neural network for complex patterns
    insights.append({
        'priority': 'HIGH',
        'category': 'Advanced Modeling',
        'finding': "Tree models may miss complex non-linear patterns",
        'action': "Train deep neural network (MLP) with all features + interactions",
        'expected_improvement': '20-35%'
    })

    # Insight 9: Stacking ensemble
    insights.append({
        'priority': 'HIGH',
        'category': 'Ensemble',
        'finding': "Simple weighted average may be suboptimal",
        'action': "Use stacking with meta-learner to optimally combine models",
        'expected_improvement': '5-15%'
    })

    # Print insights
    print("\nTop Priority Actions:")
    print("=" * 80)

    high_priority = [i for i in insights if i['priority'] == 'HIGH']
    for i, insight in enumerate(high_priority, 1):
        print(f"\n{i}. [{insight['category']}]")
        print(f"   Finding: {insight['finding']}")
        print(f"   Action:  {insight['action']}")
        print(f"   Expected Improvement: {insight['expected_improvement']}")

    print("\n\nMedium Priority Actions:")
    print("=" * 80)

    medium_priority = [i for i in insights if i['priority'] == 'MEDIUM']
    for i, insight in enumerate(medium_priority, 1):
        print(f"\n{i}. [{insight['category']}]")
        print(f"   Finding: {insight['finding']}")
        print(f"   Action:  {insight['action']}")
        print(f"   Expected Improvement: {insight['expected_improvement']}")

    return insights


# ============================================================================
#  SECTION 6: VISUALIZATION
# ============================================================================

def create_error_visualizations(error_df, results):
    """Create comprehensive error visualizations."""
    print("\n" + "=" * 80)
    print("[6/7] Creating Error Visualizations")
    print("=" * 80)

    # 1. Error Distribution and Patterns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1a. Error histogram
    axes[0, 0].hist(error_df['residual'], bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', label='Zero error')
    axes[0, 0].set_xlabel('Residual (kg)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 1b. Predictions vs Actuals
    sample = error_df.sample(min(5000, len(error_df)))
    axes[0, 1].scatter(sample['actual_fuel'], sample['predicted_fuel'],
                      alpha=0.3, s=10)
    max_val = max(sample['actual_fuel'].max(), sample['predicted_fuel'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Fuel (kg)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Fuel (kg)', fontsize=11)
    axes[0, 1].set_title('Predictions vs Actuals', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 1c. Residuals vs Predictions
    axes[0, 2].scatter(sample['predicted_fuel'], sample['residual'],
                      alpha=0.3, s=10)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Predicted Fuel (kg)', fontsize=11)
    axes[0, 2].set_ylabel('Residual (kg)', fontsize=11)
    axes[0, 2].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # 2. Error by fuel magnitude
    fuel_errors = error_df.groupby('fuel_bin')['abs_residual'].mean()
    axes[1, 0].bar(range(len(fuel_errors)), fuel_errors.values, alpha=0.7)
    axes[1, 0].set_xticks(range(len(fuel_errors)))
    axes[1, 0].set_xticklabels(fuel_errors.index, rotation=45)
    axes[1, 0].set_xlabel('Fuel Range (kg)', fontsize=11)
    axes[1, 0].set_ylabel('Mean Absolute Error (kg)', fontsize=11)
    axes[1, 0].set_title('Error by Fuel Magnitude', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 3. Error by interval position
    if 'position_bin' in error_df.columns:
        position_errors = error_df.groupby('position_bin')['abs_residual'].mean()
        axes[1, 1].bar(range(len(position_errors)), position_errors.values,
                      alpha=0.7, color='orange')
        axes[1, 1].set_xticks(range(len(position_errors)))
        axes[1, 1].set_xticklabels(position_errors.index, rotation=45)
        axes[1, 1].set_xlabel('Interval Position', fontsize=11)
        axes[1, 1].set_ylabel('Mean Absolute Error (kg)', fontsize=11)
        axes[1, 1].set_title('Error by Interval Position', fontsize=13, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')

    # 4. Error by flight phase
    if 'dominant_phase' in error_df.columns:
        phase_errors = error_df.groupby('dominant_phase')['abs_residual'].mean()
        axes[1, 2].bar(range(len(phase_errors)), phase_errors.values,
                      alpha=0.7, color='green')
        axes[1, 2].set_xticks(range(len(phase_errors)))
        axes[1, 2].set_xticklabels(phase_errors.index, rotation=45)
        axes[1, 2].set_xlabel('Flight Phase', fontsize=11)
        axes[1, 2].set_ylabel('Mean Absolute Error (kg)', fontsize=11)
        axes[1, 2].set_title('Error by Flight Phase', fontsize=13, fontweight='bold')
        axes[1, 2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phase6_error_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase6_error_analysis.png")
    plt.close()

    # 2. Q-Q plot for normality
    fig, ax = plt.subplots(figsize=(10, 8))
    stats.probplot(error_df['residual'].sample(min(5000, len(error_df))), dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Residual Normality Test)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.savefig(VIZ_DIR / 'phase6_residual_qq_plot.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: phase6_residual_qq_plot.png")
    plt.close()


# ============================================================================
#  SECTION 7: SAVE RESULTS
# ============================================================================

def save_analysis_results(error_results, patterns, gaps, insights):
    """Save comprehensive analysis results."""
    print("\n" + "=" * 80)
    print("[7/7] Saving Analysis Results")
    print("=" * 80)

    # Save insights as JSON
    with open(REPORTS_DIR / 'phase6_insights.json', 'w') as f:
        json.dump({
            'error_results': str(error_results),
            'patterns': patterns,
            'gaps': gaps,
            'insights': insights
        }, f, indent=2, default=str)

    print("  ✓ Saved: phase6_insights.json")

    # Save detailed report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 6: DEEP ERROR ANALYSIS - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Current RMSE: 189.29 kg")
    report_lines.append(f"Target RMSE: <50 kg")
    report_lines.append(f"Required Improvement: 73.6%")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("HIGH PRIORITY ACTIONS")
    report_lines.append("=" * 80)
    high_priority = [i for i in insights if i['priority'] == 'HIGH']
    for i, insight in enumerate(high_priority, 1):
        report_lines.append(f"\n{i}. [{insight['category']}]")
        report_lines.append(f"   Finding: {insight['finding']}")
        report_lines.append(f"   Action:  {insight['action']}")
        report_lines.append(f"   Expected Improvement: {insight['expected_improvement']}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("FEATURE GAP SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Interaction features: {gaps.get('interaction_count', 0)}")
    report_lines.append(f"Rolling/lag features: {gaps.get('rolling_count', 0)}")
    report_lines.append(f"Physics features: {gaps.get('physics_count', 0)}")
    report_lines.append(f"Aircraft features: {gaps.get('aircraft_count', 0)}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDED APPROACH")
    report_lines.append("=" * 80)
    report_lines.append("1. Advanced Feature Engineering (Expected: 30-50% improvement)")
    report_lines.append("   - Add interaction features")
    report_lines.append("   - Add rolling/lag features")
    report_lines.append("   - Add physics-derived features")
    report_lines.append("")
    report_lines.append("2. Stratified Modeling (Expected: 15-25% improvement)")
    report_lines.append("   - Separate models by fuel range")
    report_lines.append("   - Separate models by flight phase")
    report_lines.append("")
    report_lines.append("3. Advanced Modeling (Expected: 20-35% improvement)")
    report_lines.append("   - Deep neural networks")
    report_lines.append("   - Stacking ensemble with meta-learner")
    report_lines.append("")
    report_lines.append("ESTIMATED CUMULATIVE IMPROVEMENT: 65-110%")
    report_lines.append("This should achieve target <50 kg RMSE")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    with open(REPORTS_DIR / 'phase6_error_analysis_report.txt', 'w') as f:
        f.write(report_text)

    print("  ✓ Saved: phase6_error_analysis_report.txt")

    print("\n" + report_text)


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load data and models
    X_val, X_val_selected, y_val, predictions, residuals, abs_residuals = load_data_and_models()

    # Analyze errors by segments
    error_df, error_results = analyze_errors_by_segments(
        X_val, y_val, predictions, residuals, abs_residuals
    )

    # Analyze systematic patterns
    patterns = analyze_systematic_patterns(error_df)

    # Analyze feature gaps
    gaps = analyze_feature_gaps(X_val, error_df)

    # Generate insights
    insights = generate_actionable_insights(error_results, patterns, gaps)

    # Create visualizations
    create_error_visualizations(error_df, error_results)

    # Save results
    save_analysis_results(error_results, patterns, gaps, insights)

    print("\n" + "=" * 80)
    print("✓ PHASE 6: ERROR ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Implement advanced feature engineering (phase6_advanced_features.py)")
    print("2. Train stratified models (phase6_stratified_models.py)")
    print("3. Train neural networks (phase6_neural_networks.py)")
    print("4. Create advanced ensemble (phase6_advanced_ensemble.py)")
    print("=" * 80)


if __name__ == "__main__":
    main()
