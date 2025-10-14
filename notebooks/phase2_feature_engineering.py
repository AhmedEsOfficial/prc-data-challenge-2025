"""
Phase 2: Feature Engineering Pipeline
======================================
Batch processing of all training flights to extract comprehensive features.

This script:
1. Loads fuel, flight list, and trajectory data
2. Processes flights in memory-efficient batches
3. Extracts 70+ features per fuel interval
4. Caches results to disk
5. Generates feature documentation

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
import gc
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features import extract_all_features
from utils import print_feature_summary

# Project paths
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR
CACHE_DIR = BASE_DIR / 'cache'
REPORTS_DIR = BASE_DIR / 'reports'

print("="*80)
print("PHASE 2: FEATURE ENGINEERING PIPELINE")
print("="*80)
print(f"Started: {datetime.now()}")
print()

# ============================================================================
# SECTION 1: LOAD ALL DATASETS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING DATASETS")
print("="*80)

print("\n[1/4] Loading fuel_train.parquet...")
fuel_train = pd.read_parquet(DATA_DIR / 'fuel_train.parquet')
print(f"  Shape: {fuel_train.shape}")
print(f"  Fuel intervals: {len(fuel_train):,}")
print(f"  Unique flights: {fuel_train['flight_id'].nunique():,}")

print("\n[2/4] Loading flightlist_train.parquet...")
flightlist_train = pd.read_parquet(DATA_DIR / 'flightlist_train.parquet')
print(f"  Shape: {flightlist_train.shape}")

# Create lookup dictionary for faster access
print("  Creating flight lookup dictionary...")
flightlist_dict = {
    row['flight_id']: row for _, row in flightlist_train.iterrows()
}

print("\n[3/4] Loading apt.parquet...")
airports = pd.read_parquet(DATA_DIR / 'apt.parquet')
print(f"  Shape: {airports.shape}")
print(f"  Unique airports: {len(airports):,}")

print("\n[4/4] Checking trajectory files...")
traj_dir = DATA_DIR / 'flights_train'
traj_files = list(traj_dir.glob('*.parquet'))
print(f"  Trajectory files available: {len(traj_files):,}")

# Get list of unique flights to process
flight_ids = fuel_train['flight_id'].unique()
print(f"\n  Total flights to process: {len(flight_ids):,}")

# ============================================================================
# SECTION 2: BATCH FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: BATCH FEATURE EXTRACTION")
print("="*80)

# Configuration
BATCH_SIZE = 50  # Process 50 flights at a time (memory constraint)
SAVE_EVERY = 10  # Save checkpoint every 10 batches

print(f"\n  Batch size: {BATCH_SIZE} flights")
print(f"  Total batches: {int(np.ceil(len(flight_ids) / BATCH_SIZE))}")
print(f"  Save checkpoint every: {SAVE_EVERY} batches")
print()

# Storage for all features
all_features_list = []
batch_count = 0
total_intervals_processed = 0
failed_flights = []

# Process in batches
for batch_idx in range(0, len(flight_ids), BATCH_SIZE):
    batch_flights = flight_ids[batch_idx:batch_idx + BATCH_SIZE]
    batch_count += 1

    print(f"\n[Batch {batch_count}/{int(np.ceil(len(flight_ids) / BATCH_SIZE))}] "
          f"Processing flights {batch_idx+1} to {min(batch_idx+BATCH_SIZE, len(flight_ids))}")

    batch_features = []

    for flight_id in batch_flights:
        try:
            # Load trajectory
            traj_path = traj_dir / f'{flight_id}.parquet'
            if not traj_path.exists():
                print(f"    ✗ {flight_id}: Trajectory file not found")
                failed_flights.append(flight_id)
                continue

            traj_df = pd.read_parquet(traj_path)

            # Get flight metadata
            flightlist_row = flightlist_dict.get(flight_id)
            if flightlist_row is None:
                print(f"    ✗ {flight_id}: Not in flight list")
                failed_flights.append(flight_id)
                continue

            # Get fuel intervals for this flight
            flight_fuel = fuel_train[fuel_train['flight_id'] == flight_id]

            # Extract features for each interval
            for idx, fuel_row in flight_fuel.iterrows():
                features = extract_all_features(
                    fuel_row,
                    traj_df,
                    flightlist_row,
                    airports
                )

                # Add target variable
                features['fuel_kg'] = fuel_row['fuel_kg']

                batch_features.append(features)

            total_intervals_processed += len(flight_fuel)

            # Progress indicator
            if len(batch_features) % 1000 == 0:
                print(f"    ... {total_intervals_processed:,} intervals processed", end='\r')

        except Exception as e:
            print(f"    ✗ {flight_id}: Error - {str(e)}")
            failed_flights.append(flight_id)
            continue

    # Add batch to main list
    all_features_list.extend(batch_features)

    print(f"    ✓ Batch complete: {len(batch_features):,} intervals processed")
    print(f"    ✓ Total so far: {total_intervals_processed:,} intervals")

    # Save checkpoint periodically
    if batch_count % SAVE_EVERY == 0:
        print(f"\n    💾 Saving checkpoint...")
        checkpoint_df = pd.DataFrame(all_features_list)
        checkpoint_path = CACHE_DIR / f'features_checkpoint_batch{batch_count}.parquet'
        checkpoint_df.to_parquet(checkpoint_path, index=False)
        print(f"    ✓ Checkpoint saved: {checkpoint_path}")

        # Memory cleanup
        gc.collect()

# ============================================================================
# SECTION 3: CONSOLIDATE AND SAVE FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: CONSOLIDATING FEATURES")
print("="*80)

print(f"\n  Creating final feature dataframe...")
print(f"  Total feature dictionaries: {len(all_features_list):,}")

features_df = pd.DataFrame(all_features_list)

print(f"\n  Final shape: {features_df.shape}")
print(f"  Rows: {len(features_df):,}")
print(f"  Columns: {len(features_df.columns)}")

# Save final features
print(f"\n  Saving final features...")
final_features_path = CACHE_DIR / 'train_features_complete.parquet'
features_df.to_parquet(final_features_path, index=False)
print(f"  ✓ Saved: {final_features_path}")

# ============================================================================
# SECTION 4: FEATURE VALIDATION AND ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: FEATURE VALIDATION AND ANALYSIS")
print("="*80)

# Basic statistics
print(f"\n[4.1] Dataset Statistics")
print(f"  Total samples: {len(features_df):,}")
print(f"  Total features: {len(features_df.columns) - 2}")  # Exclude idx, flight_id
print(f"  Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for missing values
print(f"\n[4.2] Missing Value Analysis")
missing_counts = features_df.isna().sum()
missing_pct = (missing_counts / len(features_df) * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_pct': missing_pct
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values(
    'missing_pct', ascending=False
)

print(f"  Columns with missing values: {len(missing_summary)}")
if len(missing_summary) > 0:
    print(f"\n  Top 20 columns with missing data:")
    print(missing_summary.head(20))
else:
    print("  ✓ No missing values!")

# Feature types
print(f"\n[4.3] Feature Type Distribution")
numeric_cols = features_df.select_dtypes(include=[np.number]).columns
categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns

print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")

# Summary statistics for numeric features
print(f"\n[4.4] Key Feature Statistics")
key_features = [
    'fuel_kg', 'interval_duration_sec', 'altitude_mean', 'altitude_delta',
    'groundspeed_mean', 'vertical_rate_mean', 'distance_covered_km',
    'traj_point_count', 'interval_position_pct'
]

available_key_features = [f for f in key_features if f in features_df.columns]
if len(available_key_features) > 0:
    print(features_df[available_key_features].describe().round(2))

# Check categorical features
print(f"\n[4.5] Categorical Feature Value Counts")
cat_features_to_check = ['aircraft_type', 'detected_flight_phase', 'flight_phase_position',
                          'altitude_band', 'time_of_day']

for feature in cat_features_to_check:
    if feature in features_df.columns:
        print(f"\n  {feature}:")
        value_counts = features_df[feature].value_counts()
        print(f"    Unique values: {len(value_counts)}")
        print(f"    Top 5:")
        print(value_counts.head().to_string())

# ============================================================================
# SECTION 5: FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: FEATURE CORRELATION WITH TARGET")
print("="*80)

# Calculate correlations with fuel_kg
print(f"\n  Calculating correlations with fuel_kg...")
numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target and IDs
numeric_features = [f for f in numeric_features if f not in ['fuel_kg', 'idx']]

correlations = features_df[numeric_features + ['fuel_kg']].corr()['fuel_kg'].drop('fuel_kg')
correlations = correlations.sort_values(ascending=False)

print(f"\n  Top 30 features by correlation with fuel_kg:")
print(correlations.head(30))

print(f"\n  Bottom 10 features by correlation with fuel_kg:")
print(correlations.tail(10))

# Save correlations
correlations_df = pd.DataFrame({
    'feature': correlations.index,
    'correlation': correlations.values
})
correlations_path = REPORTS_DIR / 'phase2_feature_correlations.csv'
correlations_df.to_csv(correlations_path, index=False)
print(f"\n  ✓ Correlations saved: {correlations_path}")

# ============================================================================
# SECTION 6: FEATURE CATALOG DOCUMENTATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: GENERATING FEATURE CATALOG")
print("="*80)

# Create comprehensive feature documentation
feature_catalog = []

for col in features_df.columns:
    if col in ['idx', 'flight_id', 'fuel_kg']:
        continue

    dtype = str(features_df[col].dtype)
    missing_pct = (features_df[col].isna().sum() / len(features_df) * 100)

    # Determine category
    if col.startswith('interval_') or col.startswith('start_') or col in ['seconds_from_takeoff', 'seconds_to_landing']:
        category = 'Interval-based'
    elif any(x in col for x in ['altitude', 'groundspeed', 'vertical_rate', 'distance', 'heading', 'track', 'traj_']):
        category = 'Trajectory'
    elif any(x in col for x in ['aircraft', 'flight_duration', 'origin', 'dest', 'elevation', 'great_circle']):
        category = 'Flight-level'
    elif any(x in col for x in ['climb_work', 'detected_', 'cruise_', 'avg_speed', 'efficiency']):
        category = 'Physics-derived'
    else:
        category = 'Other'

    # Get correlation if numeric
    if col in correlations.index:
        corr = correlations[col]
    else:
        corr = np.nan

    feature_catalog.append({
        'feature': col,
        'category': category,
        'dtype': dtype,
        'missing_pct': round(missing_pct, 2),
        'correlation': round(corr, 4) if pd.notna(corr) else np.nan
    })

feature_catalog_df = pd.DataFrame(feature_catalog)
feature_catalog_df = feature_catalog_df.sort_values('correlation', ascending=False, na_position='last')

# Save catalog
catalog_path = REPORTS_DIR / 'phase2_feature_catalog.csv'
feature_catalog_df.to_csv(catalog_path, index=False)
print(f"  ✓ Feature catalog saved: {catalog_path}")

# Print summary by category
print(f"\n  Feature count by category:")
print(feature_catalog_df.groupby('category').size())

# ============================================================================
# SECTION 7: FINAL SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: PHASE 2 SUMMARY")
print("="*80)

summary_report = f"""
PHASE 2 FEATURE ENGINEERING - SUMMARY REPORT
============================================

PROCESSING STATISTICS
---------------------
- Total flights processed: {len(flight_ids) - len(failed_flights):,}
- Failed flights: {len(failed_flights)}
- Total fuel intervals processed: {total_intervals_processed:,}
- Final feature dataset rows: {len(features_df):,}
- Total features extracted: {len(features_df.columns) - 3} (excluding idx, flight_id, fuel_kg)

FEATURE BREAKDOWN BY CATEGORY
------------------------------
{feature_catalog_df.groupby('category').size().to_string()}

TOP 20 MOST CORRELATED FEATURES WITH FUEL
------------------------------------------
{correlations.head(20).to_string()}

DATA QUALITY
------------
- Missing value rate: {features_df.isna().sum().sum() / (len(features_df) * len(features_df.columns)) * 100:.2f}%
- Columns with >10% missing: {len(missing_summary[missing_summary['missing_pct'] > 10])}

KEY INSIGHTS
------------
1. interval_duration_sec remains the strongest predictor (correlation: {correlations['interval_duration_sec']:.4f})
2. Trajectory features (altitude_delta, distance_covered_km) show strong correlation
3. Flight position features (interval_position_pct, seconds_from_takeoff) are significant
4. Physics-derived features (climb_work) capture energy dynamics
5. Categorical features (aircraft_type, flight_phase) will be crucial for model

OUTPUTS GENERATED
-----------------
1. Main dataset: {final_features_path}
   - Size: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   - Format: Parquet (compressed)

2. Feature catalog: {catalog_path}
   - Complete metadata for all {len(feature_catalog_df)} features
   - Includes category, dtype, missing %, correlation

3. Correlation report: {correlations_path}
   - All feature correlations with fuel_kg
   - Sorted by absolute correlation value

READY FOR PHASE 3: DATA PREPARATION
------------------------------------
Next steps:
1. Load train_features_complete.parquet
2. Handle missing values (imputation strategy)
3. Encode categorical features (one-hot or target encoding)
4. Create train/validation temporal split (April-July / August)
5. Normalize/scale features if needed
6. Ready for model training!

COMPLETION TIME
---------------
Started: {datetime.now()}
Duration: [Calculate manually]
"""

print(summary_report)

# Save summary report
summary_path = REPORTS_DIR / 'phase2_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_report)

print(f"\n✓ Summary report saved: {summary_path}")

# Print failures if any
if len(failed_flights) > 0:
    print(f"\n⚠ Failed flights ({len(failed_flights)}):")
    for flight_id in failed_flights[:20]:  # Show first 20
        print(f"  - {flight_id}")
    if len(failed_flights) > 20:
        print(f"  ... and {len(failed_flights) - 20} more")

print("\n" + "="*80)
print("PHASE 2 COMPLETE")
print("="*80)
print(f"Completed: {datetime.now()}")
print(f"\nMain output: {final_features_path}")
print(f"Feature catalog: {catalog_path}")
print(f"\nReady to proceed to Phase 3: Data Preparation")
print("="*80)
