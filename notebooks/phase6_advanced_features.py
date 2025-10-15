#!/usr/bin/env python3
"""
Phase 6: Advanced Feature Engineering - Aircraft Fuel Consumption Prediction
=============================================================================

This script creates advanced features based on Phase 6 error analysis:
1. Interaction features (distance×speed, altitude×vertical_rate, etc.)
2. Rolling/lag features (temporal patterns, previous interval info)
3. Advanced physics features (energy, drag, efficiency)
4. Aircraft-specific features (historical patterns, type-specific metrics)

Goal: Add 100+ high-quality features to improve from 189.29 kg to <100 kg RMSE

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

from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

# Configuration
BASE_DIR = Path(".")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
#  SECTION 1: LOAD BASE DATA
# ============================================================================

def load_base_features():
    """Load existing feature sets."""
    print("=" * 80)
    print("PHASE 6: ADVANCED FEATURE ENGINEERING")
    print("=" * 80)
    print("\n[1/8] Loading base features...")

    # Load processed data
    X_train = pd.read_parquet(CACHE_DIR / "X_train.parquet")
    y_train = pd.read_parquet(CACHE_DIR / "y_train.parquet")['fuel_kg']
    X_val = pd.read_parquet(CACHE_DIR / "X_val.parquet")
    y_val = pd.read_parquet(CACHE_DIR / "y_val.parquet")['fuel_kg']

    print(f"✓ Loaded training: {X_train.shape}")
    print(f"✓ Loaded validation: {X_val.shape}")
    print(f"\nBase feature count: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val


# ============================================================================
#  SECTION 2: INTERACTION FEATURES
# ============================================================================

def create_interaction_features(X):
    """Create interaction features based on physics and domain knowledge."""
    print("\n[2/8] Creating Interaction Features...")

    X_new = X.copy()
    new_features = []

    # 1. Distance-Speed Interactions (drag-related)
    if 'distance_covered_km' in X.columns and 'groundspeed_mean' in X.columns:
        # Speed-distance efficiency
        X_new['distance_speed_interaction'] = X['distance_covered_km'] * X['groundspeed_mean']
        X_new['distance_per_speed'] = X['distance_covered_km'] / (X['groundspeed_mean'] + 0.1)
        new_features.extend(['distance_speed_interaction', 'distance_per_speed'])

    # 2. Altitude-Vertical Rate Interactions (climb/descent energy)
    if 'altitude_delta' in X.columns and 'vertical_rate_mean' in X.columns:
        X_new['climb_energy_proxy'] = X['altitude_delta'] * np.abs(X['vertical_rate_mean'])
        X_new['altitude_vr_interaction'] = X['altitude_delta'] * X['vertical_rate_mean']
        new_features.extend(['climb_energy_proxy', 'altitude_vr_interaction'])

    # 3. Duration-Distance Interactions (speed efficiency)
    if 'interval_duration_min' in X.columns and 'distance_covered_km' in X.columns:
        X_new['duration_distance_ratio'] = X['interval_duration_min'] / (X['distance_covered_km'] + 0.1)
        X_new['distance_per_minute'] = X['distance_covered_km'] / (X['interval_duration_min'] + 0.1)
        new_features.extend(['duration_distance_ratio', 'distance_per_minute'])

    # 4. Aircraft-Duration Interactions (type-specific fuel rates)
    if 'aircraft_type_target_enc' in X.columns and 'interval_duration_min' in X.columns:
        X_new['aircraft_duration_interaction'] = X['aircraft_type_target_enc'] * X['interval_duration_min']
        X_new['aircraft_distance_interaction'] = X['aircraft_type_target_enc'] * X.get('distance_covered_km', 0)
        new_features.extend(['aircraft_duration_interaction', 'aircraft_distance_interaction'])

    # 5. Speed-Altitude Interactions (air density effects)
    if 'groundspeed_mean' in X.columns and 'altitude_mean' in X.columns:
        X_new['speed_altitude_interaction'] = X['groundspeed_mean'] * (X['altitude_mean'] / 1000)
        X_new['speed_squared'] = X['groundspeed_mean'] ** 2  # Drag proportional to v²
        new_features.extend(['speed_altitude_interaction', 'speed_squared'])

    # 6. Distance-Altitude Interactions
    if 'distance_covered_km' in X.columns and 'altitude_mean' in X.columns:
        X_new['distance_altitude_interaction'] = X['distance_covered_km'] * (X['altitude_mean'] / 1000)
        new_features.append('distance_altitude_interaction')

    # 7. Position-Duration Interactions (phase effects)
    if 'interval_position_pct' in X.columns and 'interval_duration_min' in X.columns:
        X_new['position_duration_interaction'] = X['interval_position_pct'] * X['interval_duration_min']
        new_features.append('position_duration_interaction')

    # 8. Triple Interactions (most important)
    if all(c in X.columns for c in ['distance_covered_km', 'groundspeed_mean', 'altitude_mean']):
        X_new['distance_speed_altitude'] = (X['distance_covered_km'] *
                                             X['groundspeed_mean'] *
                                             (X['altitude_mean'] / 10000))
        new_features.append('distance_speed_altitude')

    # 9. Ratio Features (efficiency metrics)
    if 'great_circle_distance_km' in X.columns and 'distance_covered_km' in X.columns:
        X_new['path_efficiency'] = X['distance_covered_km'] / (X['great_circle_distance_km'] + 0.1)
        new_features.append('path_efficiency')

    print(f"✓ Created {len(new_features)} interaction features")

    return X_new, new_features


# ============================================================================
#  SECTION 3: ROLLING/LAG FEATURES
# ============================================================================

def create_temporal_features(X_train, y_train, X_val):
    """Create rolling, lag, and cumulative features."""
    print("\n[3/8] Creating Temporal/Rolling Features...")

    # Note: This is simplified since we don't have flight-level data loaded
    # In production, you'd group by flight_id and create these features

    X_train_new = X_train.copy()
    X_val_new = X_val.copy()
    new_features = []

    # 1. Create fuel rate features (from target in training)
    if 'interval_duration_min' in X_train.columns:
        # Fuel per minute
        X_train_new['target_fuel_per_min'] = y_train / (X_train['interval_duration_min'] + 0.1)
        # For validation, we'll impute with mean from training
        mean_fuel_per_min = X_train_new['target_fuel_per_min'].mean()
        X_val_new['target_fuel_per_min'] = mean_fuel_per_min
        new_features.append('target_fuel_per_min')

    if 'distance_covered_km' in X_train.columns:
        # Fuel per km
        X_train_new['target_fuel_per_km'] = y_train / (X_train['distance_covered_km'] + 0.01)
        mean_fuel_per_km = X_train_new['target_fuel_per_km'].mean()
        X_val_new['target_fuel_per_km'] = mean_fuel_per_km
        new_features.append('target_fuel_per_km')

    # 2. Create positional context features
    if 'interval_position_pct' in X_train.columns:
        # Distance from start/end
        X_train_new['distance_from_start'] = X_train['interval_position_pct']
        X_train_new['distance_from_end'] = 100 - X_train['interval_position_pct']

        X_val_new['distance_from_start'] = X_val['interval_position_pct']
        X_val_new['distance_from_end'] = 100 - X_val['interval_position_pct']

        new_features.extend(['distance_from_start', 'distance_from_end'])

    # 3. Create exponential transformations for skewed features
    for feat in ['interval_duration_min', 'distance_covered_km']:
        if feat in X_train.columns:
            # Log transform
            X_train_new[f'{feat}_log'] = np.log1p(X_train[feat])
            X_val_new[f'{feat}_log'] = np.log1p(X_val[feat])

            # Square root transform
            X_train_new[f'{feat}_sqrt'] = np.sqrt(X_train[feat])
            X_val_new[f'{feat}_sqrt'] = np.sqrt(X_val[feat])

            new_features.extend([f'{feat}_log', f'{feat}_sqrt'])

    print(f"✓ Created {len(new_features)} temporal features")

    return X_train_new, X_val_new, new_features


# ============================================================================
#  SECTION 4: ADVANCED PHYSICS FEATURES
# ============================================================================

def create_physics_features(X):
    """Create advanced physics-derived features."""
    print("\n[4/8] Creating Advanced Physics Features...")

    X_new = X.copy()
    new_features = []

    # Physical constants
    G = 9.81  # Gravity (m/s²)
    AVG_AIRCRAFT_MASS = 70000  # kg (approximate for commercial aircraft)

    # 1. Potential Energy Change (altitude work)
    if 'altitude_delta' in X.columns:
        # Convert altitude delta from feet to meters, then calculate energy
        altitude_delta_m = X['altitude_delta'] * 0.3048
        X_new['potential_energy_delta_mj'] = (AVG_AIRCRAFT_MASS * G * altitude_delta_m) / 1e6
        new_features.append('potential_energy_delta_mj')

    # 2. Kinetic Energy Change (speed work)
    if 'groundspeed_delta' in X.columns:
        # Convert knots to m/s and calculate energy
        speed_delta_ms = X['groundspeed_delta'] * 0.5144
        X_new['kinetic_energy_delta_mj'] = (0.5 * AVG_AIRCRAFT_MASS * (speed_delta_ms ** 2)) / 1e6
        new_features.append('kinetic_energy_delta_mj')

    # 3. Total Energy Change
    if 'potential_energy_delta_mj' in X_new.columns and 'kinetic_energy_delta_mj' in X_new.columns:
        X_new['total_energy_delta_mj'] = (X_new['potential_energy_delta_mj'] +
                                          X_new['kinetic_energy_delta_mj'])
        new_features.append('total_energy_delta_mj')

    # 4. Drag Estimate (proportional to v² and air density)
    if 'groundspeed_mean' in X.columns and 'altitude_mean' in X.columns:
        # Air density decreases with altitude (simplified model)
        altitude_km = X['altitude_mean'] * 0.0003048  # feet to km
        air_density_ratio = np.exp(-altitude_km / 8.5)  # Scale height ~8.5 km

        # Drag proxy: ρ * v²
        speed_ms = X['groundspeed_mean'] * 0.5144
        X_new['drag_proxy'] = air_density_ratio * (speed_ms ** 2)
        X_new['air_density_ratio'] = air_density_ratio

        new_features.extend(['drag_proxy', 'air_density_ratio'])

    # 5. Climb Rate Work (power during climb)
    if 'vertical_rate_mean' in X.columns and 'interval_duration_min' in X.columns:
        # Power = force × velocity = m × g × vertical_speed
        vr_ms = X['vertical_rate_mean'] * 0.00508  # ft/min to m/s
        climb_power_mw = (AVG_AIRCRAFT_MASS * G * vr_ms) / 1e6  # Megawatts

        # Work = Power × time
        duration_s = X['interval_duration_min'] * 60
        X_new['climb_work_from_power_mj'] = climb_power_mw * duration_s / 60

        new_features.append('climb_work_from_power_mj')

    # 6. Efficiency Metrics
    if 'distance_covered_km' in X.columns and 'altitude_delta' in X.columns:
        # Distance per unit altitude change (efficiency of climb/descent)
        X_new['distance_per_alt_change'] = X['distance_covered_km'] / (np.abs(X['altitude_delta']) + 100)
        new_features.append('distance_per_alt_change')

    # 7. Specific Range (distance per unit fuel) - proxy using duration
    if 'distance_covered_km' in X.columns and 'interval_duration_min' in X.columns:
        X_new['specific_range_proxy'] = X['distance_covered_km'] / (X['interval_duration_min'] + 0.1)
        new_features.append('specific_range_proxy')

    # 8. Acceleration Features
    if 'groundspeed_delta' in X.columns and 'interval_duration_min' in X.columns:
        # Acceleration = Δv / Δt
        speed_delta_ms = X['groundspeed_delta'] * 0.5144
        duration_s = X['interval_duration_min'] * 60
        X_new['acceleration_ms2'] = speed_delta_ms / (duration_s + 1)
        X_new['acceleration_absolute'] = np.abs(X_new['acceleration_ms2'])

        new_features.extend(['acceleration_ms2', 'acceleration_absolute'])

    # 9. Thrust-to-weight ratio proxy
    if 'groundspeed_delta' in X.columns and 'altitude_delta' in X.columns:
        # Combined acceleration (horizontal + vertical)
        X_new['combined_acceleration_proxy'] = np.sqrt(
            X.get('acceleration_ms2', 0)**2 +
            (X['altitude_delta'] * 0.3048)**2  # Simplified
        )
        new_features.append('combined_acceleration_proxy')

    print(f"✓ Created {len(new_features)} physics features")

    return X_new, new_features


# ============================================================================
#  SECTION 5: AIRCRAFT-SPECIFIC FEATURES
# ============================================================================

def create_aircraft_features(X_train, y_train, X_val):
    """Create aircraft-specific statistical features."""
    print("\n[5/8] Creating Aircraft-Specific Features...")

    X_train_new = X_train.copy()
    X_val_new = X_val.copy()
    new_features = []

    # 1. Aircraft-specific fuel statistics (from training data)
    if 'aircraft_type_target_enc' in X_train.columns:
        # Group by aircraft and calculate statistics
        aircraft_stats = pd.DataFrame({
            'aircraft_type_target_enc': X_train['aircraft_type_target_enc'],
            'fuel_kg': y_train
        }).groupby('aircraft_type_target_enc').agg({
            'fuel_kg': ['mean', 'std', 'median', 'min', 'max']
        })

        aircraft_stats.columns = ['aircraft_fuel_mean', 'aircraft_fuel_std',
                                  'aircraft_fuel_median', 'aircraft_fuel_min',
                                  'aircraft_fuel_max']

        # Merge back
        X_train_new = X_train_new.merge(
            aircraft_stats,
            left_on='aircraft_type_target_enc',
            right_index=True,
            how='left'
        )

        X_val_new = X_val_new.merge(
            aircraft_stats,
            left_on='aircraft_type_target_enc',
            right_index=True,
            how='left'
        )

        # Fill missing values
        for col in aircraft_stats.columns:
            X_train_new[col].fillna(X_train_new[col].mean(), inplace=True)
            X_val_new[col].fillna(X_val_new[col].mean(), inplace=True)

        new_features.extend(aircraft_stats.columns.tolist())

    # 2. Aircraft-specific efficiency metrics
    if 'aircraft_type_target_enc' in X_train.columns and 'distance_covered_km' in X_train.columns:
        # Fuel per km by aircraft type
        aircraft_efficiency = pd.DataFrame({
            'aircraft_type_target_enc': X_train['aircraft_type_target_enc'],
            'fuel_per_km': y_train / (X_train['distance_covered_km'] + 0.01)
        }).groupby('aircraft_type_target_enc')['fuel_per_km'].agg(['mean', 'std'])

        aircraft_efficiency.columns = ['aircraft_fuel_per_km_mean', 'aircraft_fuel_per_km_std']

        X_train_new = X_train_new.merge(
            aircraft_efficiency,
            left_on='aircraft_type_target_enc',
            right_index=True,
            how='left'
        )

        X_val_new = X_val_new.merge(
            aircraft_efficiency,
            left_on='aircraft_type_target_enc',
            right_index=True,
            how='left'
        )

        for col in aircraft_efficiency.columns:
            X_train_new[col].fillna(X_train_new[col].mean(), inplace=True)
            X_val_new[col].fillna(X_val_new[col].mean(), inplace=True)

        new_features.extend(aircraft_efficiency.columns.tolist())

    # 3. Aircraft type deviation features
    if 'aircraft_fuel_mean' in X_train_new.columns and 'distance_covered_km' in X_train.columns:
        # How does this interval compare to aircraft average?
        X_train_new['fuel_deviation_from_aircraft_avg'] = (
            (y_train - X_train_new['aircraft_fuel_mean']) /
            (X_train_new['aircraft_fuel_std'] + 1)
        )

        # For validation, use overall mean as proxy
        X_val_new['fuel_deviation_from_aircraft_avg'] = 0  # Will be updated in training

        new_features.append('fuel_deviation_from_aircraft_avg')

    print(f"✓ Created {len(new_features)} aircraft-specific features")

    return X_train_new, X_val_new, new_features


# ============================================================================
#  SECTION 6: STATISTICAL AGGREGATION FEATURES
# ============================================================================

def create_statistical_features(X):
    """Create statistical aggregation features."""
    print("\n[6/8] Creating Statistical Aggregation Features...")

    X_new = X.copy()
    new_features = []

    # 1. Ratios of statistics
    if 'altitude_mean' in X.columns and 'altitude_std' in X.columns:
        X_new['altitude_cv'] = X['altitude_std'] / (X['altitude_mean'] + 1)  # Coefficient of variation
        new_features.append('altitude_cv')

    if 'groundspeed_mean' in X.columns and 'groundspeed_std' in X.columns:
        X_new['groundspeed_cv'] = X['groundspeed_std'] / (X['groundspeed_mean'] + 1)
        new_features.append('groundspeed_cv')

    # 2. Range features
    if 'altitude_max' in X.columns and 'altitude_min' in X.columns:
        X_new['altitude_range'] = X['altitude_max'] - X['altitude_min']
        new_features.append('altitude_range')

    if 'groundspeed_max' in X.columns and 'groundspeed_min' in X.columns:
        X_new['groundspeed_range'] = X['groundspeed_max'] - X['groundspeed_min']
        new_features.append('groundspeed_range')

    # 3. Percentile features
    percentile_cols = [c for c in X.columns if any(p in c for p in ['_p25', '_p50', '_p75'])]
    if percentile_cols:
        # Calculate inter-quartile range
        for base_feat in ['altitude', 'groundspeed']:
            if f'{base_feat}_p75' in X.columns and f'{base_feat}_p25' in X.columns:
                X_new[f'{base_feat}_iqr'] = X[f'{base_feat}_p75'] - X[f'{base_feat}_p25']
                new_features.append(f'{base_feat}_iqr')

    # 4. Density features (points per unit)
    if 'traj_point_count' in X.columns and 'interval_duration_min' in X.columns:
        X_new['point_density_per_min'] = X['traj_point_count'] / (X['interval_duration_min'] + 0.1)
        new_features.append('point_density_per_min')

    if 'traj_point_count' in X.columns and 'distance_covered_km' in X.columns:
        X_new['point_density_per_km'] = X['traj_point_count'] / (X['distance_covered_km'] + 0.1)
        new_features.append('point_density_per_km')

    print(f"✓ Created {len(new_features)} statistical features")

    return X_new, new_features


# ============================================================================
#  SECTION 7: COMBINE AND VALIDATE
# ============================================================================

def combine_all_features(X_train, y_train, X_val):
    """Apply all feature engineering steps."""
    print("\n[7/8] Combining All Feature Engineering Steps...")

    # Track all new features
    all_new_features = []

    # Step 1: Interaction features
    X_train, interaction_feats = create_interaction_features(X_train)
    X_val, _ = create_interaction_features(X_val)
    all_new_features.extend(interaction_feats)

    # Step 2: Temporal features
    X_train, X_val, temporal_feats = create_temporal_features(X_train, y_train, X_val)
    all_new_features.extend(temporal_feats)

    # Step 3: Physics features
    X_train, physics_feats = create_physics_features(X_train)
    X_val, _ = create_physics_features(X_val)
    all_new_features.extend(physics_feats)

    # Step 4: Aircraft-specific features
    X_train, X_val, aircraft_feats = create_aircraft_features(X_train, y_train, X_val)
    all_new_features.extend(aircraft_feats)

    # Step 5: Statistical features
    X_train, stats_feats = create_statistical_features(X_train)
    X_val, _ = create_statistical_features(X_val)
    all_new_features.extend(stats_feats)

    # Validate
    print(f"\nFeature Engineering Summary:")
    print(f"  Original features: {len(X_train.columns) - len(all_new_features)}")
    print(f"  New features added: {len(all_new_features)}")
    print(f"  Total features: {len(X_train.columns)}")

    # Check for infinite/NaN values
    inf_cols_train = X_train.columns[np.isinf(X_train).any()].tolist()
    inf_cols_val = X_val.columns[np.isinf(X_val).any()].tolist()

    if inf_cols_train or inf_cols_val:
        print(f"\n⚠️  Warning: Infinite values found in {len(set(inf_cols_train + inf_cols_val))} columns")
        for col in set(inf_cols_train + inf_cols_val):
            X_train[col].replace([np.inf, -np.inf], X_train[col][~np.isinf(X_train[col])].median(), inplace=True)
            X_val[col].replace([np.inf, -np.inf], X_val[col][~np.isinf(X_val[col])].median(), inplace=True)
        print("✓ Replaced infinite values with median")

    # Fill remaining NaNs
    X_train.fillna(X_train.median(), inplace=True)
    X_val.fillna(X_train.median(), inplace=True)  # Use training medians for validation

    return X_train, X_val, all_new_features


# ============================================================================
#  SECTION 8: SAVE RESULTS
# ============================================================================

def save_enhanced_features(X_train, X_val, y_train, y_val, new_features):
    """Save enhanced feature sets."""
    print("\n[8/8] Saving Enhanced Feature Sets...")

    # Save enhanced datasets
    X_train.to_parquet(CACHE_DIR / 'X_train_enhanced.parquet', index=False)
    X_val.to_parquet(CACHE_DIR / 'X_val_enhanced.parquet', index=False)

    print(f"✓ Saved: X_train_enhanced.parquet ({X_train.shape})")
    print(f"✓ Saved: X_val_enhanced.parquet ({X_val.shape})")

    # Save feature names
    feature_info = {
        'original_count': X_train.shape[1] - len(new_features),
        'new_count': len(new_features),
        'total_count': X_train.shape[1],
        'new_feature_names': new_features,
        'all_feature_names': X_train.columns.tolist()
    }

    with open(MODELS_DIR / 'phase6_enhanced_features_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"✓ Saved: phase6_enhanced_features_info.json")

    # Create summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 6: ADVANCED FEATURE ENGINEERING - SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("FEATURE COUNTS:")
    report_lines.append(f"  Original features:  {feature_info['original_count']}")
    report_lines.append(f"  New features:       {feature_info['new_count']}")
    report_lines.append(f"  Total features:     {feature_info['total_count']}")
    report_lines.append(f"  Increase:           +{(feature_info['new_count']/feature_info['original_count']*100):.1f}%")
    report_lines.append("")

    report_lines.append("FEATURE CATEGORIES:")
    interaction_count = len([f for f in new_features if 'interaction' in f or 'ratio' in f])
    physics_count = len([f for f in new_features if any(x in f for x in ['energy', 'drag', 'power', 'work'])])
    aircraft_count = len([f for f in new_features if 'aircraft' in f])
    temporal_count = len([f for f in new_features if any(x in f for x in ['log', 'sqrt', 'rate', 'per_'])])

    report_lines.append(f"  Interaction features:  {interaction_count}")
    report_lines.append(f"  Physics features:      {physics_count}")
    report_lines.append(f"  Aircraft features:     {aircraft_count}")
    report_lines.append(f"  Temporal features:     {temporal_count}")
    report_lines.append(f"  Other features:        {len(new_features) - interaction_count - physics_count - aircraft_count - temporal_count}")
    report_lines.append("")

    report_lines.append("TOP 20 NEW FEATURES:")
    for i, feat in enumerate(new_features[:20], 1):
        report_lines.append(f"  {i:2d}. {feat}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("NEXT STEPS:")
    report_lines.append("1. Train models with enhanced features")
    report_lines.append("2. Compare performance vs baseline (189.29 kg RMSE)")
    report_lines.append("3. Expected improvement: 30-50%")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    with open(REPORTS_DIR / 'phase6_feature_engineering_report.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Load base data
    X_train, y_train, X_val, y_val = load_base_features()

    # Apply all feature engineering
    X_train_enhanced, X_val_enhanced, new_features = combine_all_features(
        X_train, y_train, X_val
    )

    # Save results
    save_enhanced_features(X_train_enhanced, X_val_enhanced, y_train, y_val, new_features)

    print("\n" + "=" * 80)
    print("✓ PHASE 6: ADVANCED FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"\nEnhanced Dataset: {X_train_enhanced.shape}")
    print(f"New Features: {len(new_features)}")
    print("\nReady for model training with enhanced features!")
    print("=" * 80)


if __name__ == "__main__":
    main()
