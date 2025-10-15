"""
Phase 3: Data Preparation Pipeline
===================================
Complete data preprocessing pipeline for model training.

This script:
1. Loads extracted features from Phase 2
2. Cleans data (removes constant/duplicate features)
3. Handles missing values (indicators + imputation)
4. Encodes categorical features
5. Creates temporal train/val split
6. Scales features
7. Saves processed datasets
8. Generates documentation

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    FeatureCleaner,
    MissingValueHandler,
    CategoricalEncoder,
    FeatureScaler,
    create_temporal_split,
    validate_no_leakage,
    get_feature_names,
    print_preprocessing_summary
)

# Project paths
BASE_DIR = Path('.')
CACHE_DIR = BASE_DIR / 'cache'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)

print("="*80)
print("PHASE 3: DATA PREPARATION PIPELINE")
print("="*80)
print(f"Started: {datetime.now()}")
print()

# ============================================================================
# SECTION 1: LOAD FEATURES FROM PHASE 2
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING FEATURES")
print("="*80)

features_path = CACHE_DIR / 'train_features_complete.parquet'
print(f"\nLoading: {features_path}")

df = pd.read_parquet(features_path)
print(f"  ✓ Loaded: {df.shape}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nInitial data summary:")
print(f"  Total samples: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Features: {len(df.columns) - 3}")  # Exclude idx, flight_id, fuel_kg

# Separate identifiers, features, and target
identifiers = df[['idx', 'flight_id']].copy()
target = df['fuel_kg'].copy()
features = df.drop(columns=['idx', 'flight_id', 'fuel_kg']).copy()

print(f"\n  Identifiers: {identifiers.shape}")
print(f"  Features: {features.shape}")
print(f"  Target: {target.shape}")

# ============================================================================
# SECTION 2: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DATA CLEANING")
print("="*80)

# Initialize and apply cleaner
cleaner = FeatureCleaner(
    remove_constant=True,
    remove_duplicates=True,
    verbose=True
)

features_clean = cleaner.fit_transform(features)

print(f"\nFeatures after cleaning: {features_clean.shape}")
print(f"  Removed: {features.shape[1] - features_clean.shape[1]} features")

# ============================================================================
# SECTION 3: TEMPORAL TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: TEMPORAL TRAIN/VALIDATION SPLIT")
print("="*80)

# Combine for splitting
df_for_split = pd.concat([identifiers, features_clean, target], axis=1)

# Create temporal split (April-July train, August validation)
train_df, val_df = create_temporal_split(
    df_for_split,
    split_month=8,
    date_column='start_month'
)

# Validate no leakage
is_valid = validate_no_leakage(train_df, val_df, identifier_col='flight_id')

if not is_valid:
    raise ValueError("Data leakage detected! Aborting.")

# Separate back into components
train_ids = train_df[['idx', 'flight_id']].copy()
val_ids = val_df[['idx', 'flight_id']].copy()

X_train_raw = train_df.drop(columns=['idx', 'flight_id', 'fuel_kg']).copy()
X_val_raw = val_df.drop(columns=['idx', 'flight_id', 'fuel_kg']).copy()

y_train = train_df['fuel_kg'].copy()
y_val = val_df['fuel_kg'].copy()

print(f"\nSplit summary:")
print(f"  Train: {len(X_train_raw):,} samples")
print(f"  Val: {len(X_val_raw):,} samples")

# ============================================================================
# SECTION 4: MISSING VALUE HANDLING
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: MISSING VALUE HANDLING")
print("="*80)

# Initialize handler
missing_handler = MissingValueHandler(
    create_indicators=True,
    indicator_threshold=0.10,
    group_by_aircraft=True,
    verbose=True
)

# Fit on training data only
missing_handler.fit(X_train_raw, y_train)

# Transform both sets
X_train_imputed = missing_handler.transform(X_train_raw)
X_val_imputed = missing_handler.transform(X_val_raw)

print(f"\nFeatures after missing value handling:")
print(f"  Train: {X_train_imputed.shape}")
print(f"  Val: {X_val_imputed.shape}")

# ============================================================================
# SECTION 5: CATEGORICAL ENCODING
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: CATEGORICAL ENCODING")
print("="*80)

# Initialize encoder
encoder = CategoricalEncoder(
    target_encode_cols=['aircraft_type'],
    onehot_cols=['detected_flight_phase', 'altitude_band', 'time_of_day', 'flight_phase_position'],
    frequency_encode_cols=['origin_icao', 'destination_icao'],
    verbose=True
)

# Fit on training data with target
y_train_named = y_train.copy()
y_train_named.name = 'fuel_kg'

encoder.fit(X_train_imputed, y_train_named)

# Transform both sets
X_train_encoded = encoder.transform(X_train_imputed)
X_val_encoded = encoder.transform(X_val_imputed)

print(f"\nFeatures after encoding:")
print(f"  Train: {X_train_encoded.shape}")
print(f"  Val: {X_val_encoded.shape}")

# Check for column mismatch
if X_train_encoded.shape[1] != X_val_encoded.shape[1]:
    print(f"\n⚠ WARNING: Train and val have different number of columns!")
    print(f"  Train: {X_train_encoded.shape[1]}")
    print(f"  Val: {X_val_encoded.shape[1]}")

    # Align columns
    train_cols = set(X_train_encoded.columns)
    val_cols = set(X_val_encoded.columns)

    missing_in_val = train_cols - val_cols
    missing_in_train = val_cols - train_cols

    if len(missing_in_val) > 0:
        print(f"  Columns in train but not val: {missing_in_val}")
        for col in missing_in_val:
            X_val_encoded[col] = 0

    if len(missing_in_train) > 0:
        print(f"  Columns in val but not train: {missing_in_train}")
        X_val_encoded = X_val_encoded.drop(columns=list(missing_in_train))

    # Reorder to match
    X_val_encoded = X_val_encoded[X_train_encoded.columns]

    print(f"  ✓ Columns aligned: {X_train_encoded.shape}")

# ============================================================================
# SECTION 6: FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: FEATURE SCALING")
print("="*80)

# Initialize scaler
scaler = FeatureScaler(
    exclude_binary=True,
    verbose=True
)

# Fit on training data
scaler.fit(X_train_encoded)

# Transform both sets
X_train_final = scaler.transform(X_train_encoded)
X_val_final = scaler.transform(X_val_encoded)

print(f"\nFinal feature shapes:")
print(f"  Train: {X_train_final.shape}")
print(f"  Val: {X_val_final.shape}")

# ============================================================================
# SECTION 7: FINAL VALIDATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: FINAL VALIDATION")
print("="*80)

# Check for any remaining missing values
train_missing = X_train_final.isna().sum().sum()
val_missing = X_val_final.isna().sum().sum()

print(f"\nMissing value check:")
print(f"  Train: {train_missing} missing values")
print(f"  Val: {val_missing} missing values")

if train_missing > 0 or val_missing > 0:
    print(f"\n⚠ WARNING: Remaining missing values detected!")

    # Show which columns
    if train_missing > 0:
        missing_cols = X_train_final.columns[X_train_final.isna().any()].tolist()
        print(f"  Train columns with missing: {missing_cols}")

    if val_missing > 0:
        missing_cols = X_val_final.columns[X_val_final.isna().any()].tolist()
        print(f"  Val columns with missing: {missing_cols}")

    # Fill any remaining with 0
    print(f"\n  Filling remaining with 0...")
    X_train_final = X_train_final.fillna(0)
    X_val_final = X_val_final.fillna(0)

    print(f"  ✓ All missing values filled")

# Check for infinite values
train_inf = np.isinf(X_train_final.select_dtypes(include=[np.number])).sum().sum()
val_inf = np.isinf(X_val_final.select_dtypes(include=[np.number])).sum().sum()

print(f"\nInfinite value check:")
print(f"  Train: {train_inf} infinite values")
print(f"  Val: {val_inf} infinite values")

if train_inf > 0 or val_inf > 0:
    print(f"\n⚠ WARNING: Infinite values detected!")
    # Replace with large finite values
    X_train_final = X_train_final.replace([np.inf, -np.inf], 0)
    X_val_final = X_val_final.replace([np.inf, -np.inf], 0)
    print(f"  ✓ Infinite values replaced with 0")

# Final summary
print_preprocessing_summary(X_train_final, X_val_final, y_train, y_val)

# ============================================================================
# SECTION 8: SAVE PROCESSED DATASETS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: SAVING PROCESSED DATASETS")
print("="*80)

# Save train set
train_X_path = CACHE_DIR / 'X_train.parquet'
train_y_path = CACHE_DIR / 'y_train.parquet'
train_ids_path = CACHE_DIR / 'train_ids.parquet'

X_train_final.to_parquet(train_X_path, index=False)
y_train.to_frame('fuel_kg').to_parquet(train_y_path, index=False)
train_ids.to_parquet(train_ids_path, index=False)

print(f"\n✓ Train set saved:")
print(f"  X_train: {train_X_path}")
print(f"  y_train: {train_y_path}")
print(f"  train_ids: {train_ids_path}")

# Save validation set
val_X_path = CACHE_DIR / 'X_val.parquet'
val_y_path = CACHE_DIR / 'y_val.parquet'
val_ids_path = CACHE_DIR / 'val_ids.parquet'

X_val_final.to_parquet(val_X_path, index=False)
y_val.to_frame('fuel_kg').to_parquet(val_y_path, index=False)
val_ids.to_parquet(val_ids_path, index=False)

print(f"\n✓ Validation set saved:")
print(f"  X_val: {val_X_path}")
print(f"  y_val: {val_y_path}")
print(f"  val_ids: {val_ids_path}")

# Save preprocessing pipeline
pipeline_path = MODELS_DIR / 'preprocessing_pipeline.pkl'

pipeline = {
    'cleaner': cleaner,
    'missing_handler': missing_handler,
    'encoder': encoder,
    'scaler': scaler
}

with open(pipeline_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"\n✓ Preprocessing pipeline saved: {pipeline_path}")

# Save feature names
feature_names_path = MODELS_DIR / 'feature_names.json'
feature_names = {
    'features': get_feature_names(X_train_final),
    'n_features': len(X_train_final.columns)
}

with open(feature_names_path, 'w') as f:
    json.dump(feature_names, f, indent=2)

print(f"✓ Feature names saved: {feature_names_path}")

# ============================================================================
# SECTION 9: GENERATE TRANSFORMATION REPORT
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: TRANSFORMATION REPORT")
print("="*80)

transformation_report = f"""
PHASE 3 DATA PREPARATION - TRANSFORMATION REPORT
=================================================

EXECUTION SUMMARY
-----------------
Started: {datetime.now()}
Status: SUCCESS

INPUT DATA
----------
Source: {features_path}
Initial shape: {df.shape}
Initial features: {features.shape[1]}

STEP 1: DATA CLEANING
----------------------
Features removed: {features.shape[1] - features_clean.shape[1]}
  - Constant features: {len(cleaner.constant_features_)}
    {cleaner.constant_features_}
  - Duplicate features: {len(cleaner.duplicate_features_)}
    {cleaner.duplicate_features_}

Features after cleaning: {features_clean.shape[1]}

STEP 2: TEMPORAL SPLIT
-----------------------
Split criterion: start_month < 8 (train), start_month == 8 (val)

Train set:
  Samples: {len(X_train_raw):,}
  Percentage: {len(X_train_raw) / len(df) * 100:.1f}%

Validation set:
  Samples: {len(X_val_raw):,}
  Percentage: {len(X_val_raw) / len(df) * 100:.1f}%

Leakage check: PASSED ✓

STEP 3: MISSING VALUE HANDLING
-------------------------------
Strategy: Missing indicators + median imputation by aircraft type

Missing value indicators created: {len(missing_handler.indicator_features_)}

Top features with missing values:
{pd.Series(missing_handler.missing_features_).head(10).to_string() if len(missing_handler.missing_features_) > 0 else "None"}

Imputation method:
  - Aircraft-specific median: Yes
  - Global median fallback: Yes

Final missing values:
  Train: 0
  Val: 0

STEP 4: CATEGORICAL ENCODING
-----------------------------
Target encoding (mean target by category):
  - aircraft_type ({len(encoder.target_encodings_.get('aircraft_type', {}))  if 'aircraft_type' in encoder.target_encodings_ else 0} categories)

Frequency encoding (proportion of occurrences):
  - origin_icao ({len(encoder.frequency_encodings_.get('origin_icao', {})) if 'origin_icao' in encoder.frequency_encodings_ else 0} categories)
  - destination_icao ({len(encoder.frequency_encodings_.get('destination_icao', {})) if 'destination_icao' in encoder.frequency_encodings_ else 0} categories)

One-hot encoding:
  - detected_flight_phase ({len(encoder.onehot_categories_.get('detected_flight_phase', []))  if 'detected_flight_phase' in encoder.onehot_categories_ else 0} categories)
  - altitude_band ({len(encoder.onehot_categories_.get('altitude_band', []))  if 'altitude_band' in encoder.onehot_categories_ else 0} categories)
  - time_of_day ({len(encoder.onehot_categories_.get('time_of_day', []))  if 'time_of_day' in encoder.onehot_categories_ else 0} categories)
  - flight_phase_position ({len(encoder.onehot_categories_.get('flight_phase_position', []))  if 'flight_phase_position' in encoder.onehot_categories_ else 0} categories)

Features after encoding: {X_train_encoded.shape[1]}

STEP 5: FEATURE SCALING
------------------------
Scaling method: StandardScaler (mean=0, std=1)

Features scaled: {len(scaler.features_to_scale_)}
Binary features excluded: {len(scaler.binary_features_)}

FINAL DATASET
-------------
Training set:
  X_train: {X_train_final.shape}
  y_train: {y_train.shape}
  Memory: {X_train_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Validation set:
  X_val: {X_val_final.shape}
  y_val: {y_val.shape}
  Memory: {X_val_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Feature summary:
  Total features: {X_train_final.shape[1]}
  Numeric features: {len(X_train_final.select_dtypes(include=[np.number]).columns)}
  Feature names saved: {len(feature_names['features'])}

TARGET DISTRIBUTION
-------------------
Training:
  Mean: {y_train.mean():.2f} kg
  Std: {y_train.std():.2f} kg
  Min: {y_train.min():.2f} kg
  Max: {y_train.max():.2f} kg
  Median: {y_train.median():.2f} kg

Validation:
  Mean: {y_val.mean():.2f} kg
  Std: {y_val.std():.2f} kg
  Min: {y_val.min():.2f} kg
  Max: {y_val.max():.2f} kg
  Median: {y_val.median():.2f} kg

Distribution comparison:
  Mean ratio (val/train): {y_val.mean() / y_train.mean():.3f}
  Std ratio (val/train): {y_val.std() / y_train.std():.3f}

QUALITY CHECKS
--------------
✓ No missing values in final datasets
✓ No infinite values in final datasets
✓ No data leakage between train and val
✓ Consistent feature columns in train and val
✓ All preprocessing transformations saved

OUTPUT FILES
------------
Training data:
  - {train_X_path}
  - {train_y_path}
  - {train_ids_path}

Validation data:
  - {val_X_path}
  - {val_y_path}
  - {val_ids_path}

Preprocessing artifacts:
  - {pipeline_path}
  - {feature_names_path}

NEXT STEPS: PHASE 4 - MODEL TRAINING
-------------------------------------
1. Load X_train.parquet and y_train.parquet
2. Train baseline models (Linear, Random Forest)
3. Train primary model (LightGBM)
4. Evaluate on X_val.parquet and y_val.parquet
5. Track experiments and hyperparameters
6. Select best model for Phase 5 optimization

READY FOR MODEL TRAINING!
==========================
All data preprocessing complete.
Datasets are clean, encoded, scaled, and ready for training.
"""

print(transformation_report)

# Save report
report_path = REPORTS_DIR / 'phase3_transformation_report.txt'
with open(report_path, 'w') as f:
    f.write(transformation_report)

print(f"\n✓ Transformation report saved: {report_path}")

print("\n" + "="*80)
print("PHASE 3 COMPLETE")
print("="*80)
print(f"Completed: {datetime.now()}")
print(f"\nProcessed datasets ready for training:")
print(f"  Train: {len(X_train_final):,} samples × {X_train_final.shape[1]} features")
print(f"  Val: {len(X_val_final):,} samples × {X_val_final.shape[1]} features")
print(f"\nReady to proceed to Phase 4: Model Training")
print("="*80)
