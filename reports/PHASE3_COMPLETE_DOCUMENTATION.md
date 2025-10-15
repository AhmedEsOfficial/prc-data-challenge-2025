# Phase 3: Data Preparation - Complete Documentation

**Date:** 2025-10-15
**Status:** ✅ READY FOR EXECUTION
**Prerequisites:** Phase 2 complete (train_features_complete.parquet available)

---

## Executive Summary

Phase 3 implements a comprehensive, production-ready data preprocessing pipeline that transforms raw features into model-ready datasets. The pipeline handles missing values intelligently, encodes categorical variables appropriately, creates a temporal train/validation split, and ensures data quality for robust model training.

### Pipeline Architecture

```
Input: train_features_complete.parquet (131,530 × 82)
  ↓
┌─────────────────────────────────────────┐
│ STEP 1: Data Cleaning                  │
│ - Remove constant features              │
│ - Remove duplicate features             │
│ Output: ~77 features                    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STEP 2: Temporal Split                 │
│ - Train: April-July (months 4-7)       │
│ - Val: August (month 8)                 │
│ - Leakage validation                    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STEP 3: Missing Value Handling         │
│ - Create missing indicators (>10%)     │
│ - Median imputation by aircraft type   │
│ - Global median fallback                │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STEP 4: Categorical Encoding           │
│ - aircraft_type: Target encoding        │
│ - airports: Frequency encoding          │
│ - phases/bands: One-hot encoding        │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STEP 5: Feature Scaling                │
│ - StandardScaler (mean=0, std=1)        │
│ - Exclude binary features               │
└─────────────────────────────────────────┘
  ↓
Output: X_train, y_train, X_val, y_val
        preprocessing_pipeline.pkl
        feature_names.json
```

---

## Implementation Details

### Module 1: FeatureCleaner

**Purpose:** Remove features that add no predictive value.

**What it removes:**
1. **Constant features** - Features with only 1 unique value
   - Example: `avg_speed_over_ground_kt` (all zeros)

2. **Duplicate features** - Same information in different units
   - `interval_duration_sec` vs `interval_duration_min` → Keep `_min`
   - `flight_duration_hours` vs `flight_duration_min` → Keep `_hours`

**Implementation:**
```python
cleaner = FeatureCleaner(
    remove_constant=True,
    remove_duplicates=True,
    verbose=True
)

features_clean = cleaner.fit_transform(features)
```

**Expected removal:** 2-3 features

---

### Module 2: Temporal Split

**Purpose:** Create time-based train/validation split to simulate production deployment.

**Split Strategy:**
- **Train:** Months 4-7 (April, May, June, July)
- **Validation:** Month 8 (August)
- **Rationale:** Ranking data is September, so August validates generalization

**Critical:** No data leakage validation
```python
train_df, val_df = create_temporal_split(df, split_month=8)
validate_no_leakage(train_df, val_df, identifier_col='flight_id')
```

**Expected split:**
- Train: ~75% of data (98,000+ samples)
- Val: ~25% of data (32,000+ samples)

---

### Module 3: MissingValueHandler

**Purpose:** Handle missing values intelligently while preserving information about missingness.

**Strategy:**

**Step 1: Create Missing Indicators**
- For features with >10% missing values
- Binary flag: 1 = missing, 0 = present
- Example: `groundspeed_mean_is_missing`

**Step 2: Impute Missing Values**

For high-importance features (groundspeed, vertical_rate):
```python
# By aircraft type (preferred)
median_by_aircraft = train.groupby('aircraft_type')['groundspeed_mean'].median()

# Global median (fallback)
global_median = train['groundspeed_mean'].median()
```

For other features (<10% missing):
- Simple median imputation

**Implementation:**
```python
missing_handler = MissingValueHandler(
    create_indicators=True,
    indicator_threshold=0.10,  # 10%
    group_by_aircraft=True,
    verbose=True
)

missing_handler.fit(X_train, y_train)
X_train_imputed = missing_handler.transform(X_train)
X_val_imputed = missing_handler.transform(X_val)
```

**Expected indicators created:** ~13 features

**Why this works:**
- Model learns that missingness itself is informative
- Aircraft-specific imputation preserves type-specific patterns
- No data leakage (fit only on train)

---

### Module 4: CategoricalEncoder

**Purpose:** Convert categorical features to numeric representations.

**Encoding Strategies:**

#### 1. Target Encoding (for high-cardinality)
**Features:** `aircraft_type` (26 categories)

**Method:**
```python
# For each aircraft type, calculate mean fuel consumption
encoding_map = {
    'A20N': 458.2,  # Mean fuel for A20N
    'B77L': 6111.8, # Mean fuel for B77L
    ...
}
```

**Advantages:**
- Captures relationship with target
- Only 1 column instead of 26
- Handles unseen categories (use global mean)

#### 2. Frequency Encoding (for airports)
**Features:** `origin_icao`, `destination_icao` (100+ categories each)

**Method:**
```python
# Encode as proportion of occurrences
frequency_map = {
    'KJFK': 0.025,  # 2.5% of flights
    'LFPG': 0.018,  # 1.8% of flights
    ...
}
```

**Advantages:**
- Captures airport popularity
- Compact representation
- Prevents explosion of features

#### 3. One-Hot Encoding (for low-cardinality)
**Features:**
- `detected_flight_phase` (7 categories)
- `altitude_band` (7 categories)
- `time_of_day` (4 categories)
- `flight_phase_position` (6 categories)

**Method:**
```python
# Creates binary columns for each category
detected_flight_phase_cruise: [1, 0, 0, ...]
detected_flight_phase_climb: [0, 1, 0, ...]
detected_flight_phase_descent: [0, 0, 1, ...]
```

**Total columns created:** ~24 new columns

**Implementation:**
```python
encoder = CategoricalEncoder(
    target_encode_cols=['aircraft_type'],
    onehot_cols=['detected_flight_phase', 'altitude_band', 'time_of_day', 'flight_phase_position'],
    frequency_encode_cols=['origin_icao', 'destination_icao'],
    verbose=True
)

encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_val_encoded = encoder.transform(X_val)
```

**Important:** Drop original categorical columns after encoding.

---

### Module 5: FeatureScaler

**Purpose:** Standardize numeric features to mean=0, std=1.

**Why scale:**
- Different features have vastly different ranges:
  - `distance_covered_km`: 0 - 10,000+
  - `altitude_mean`: 0 - 40,000+
  - `interval_duration_min`: 0 - 60
- Improves numerical stability
- Required for some models (neural networks, linear models)
- Less critical for tree-based models but doesn't hurt

**Strategy:**
```python
# StandardScaler: (X - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Exclusions:**
- Binary features (0/1) - already in good range
- Missing indicators - already 0/1
- One-hot encoded features - already 0/1

**Implementation:**
```python
scaler = FeatureScaler(
    exclude_binary=True,  # Don't scale binary features
    verbose=True
)

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

**Expected features scaled:** ~50-60 numeric features

---

## Expected Output

### Processed Datasets

| File | Shape | Size | Description |
|------|-------|------|-------------|
| `X_train.parquet` | ~98,000 × 100+ | ~80 MB | Training features |
| `y_train.parquet` | ~98,000 × 1 | ~1 MB | Training target |
| `train_ids.parquet` | ~98,000 × 2 | ~2 MB | Train identifiers (idx, flight_id) |
| `X_val.parquet` | ~32,000 × 100+ | ~25 MB | Validation features |
| `y_val.parquet` | ~32,000 × 1 | ~0.5 MB | Validation target |
| `val_ids.parquet` | ~32,000 × 2 | ~1 MB | Val identifiers |

### Preprocessing Artifacts

| File | Description |
|------|-------------|
| `preprocessing_pipeline.pkl` | Fitted transformers (cleaner, imputer, encoder, scaler) |
| `feature_names.json` | List of final feature names and count |

### Expected Final Features

**Starting features:** 79 (from Phase 2)

**After cleaning:** ~77 features
- Remove 2-3 constant/duplicate features

**After missing value handling:** ~90 features
- Add ~13 missing indicators

**After categorical encoding:** ~105-110 features
- Remove 7 original categorical columns
- Add ~24 one-hot encoded columns
- Replace aircraft_type with target encoding (1 column)
- Replace airports with frequency encoding (2 columns)

**Final feature count:** ~105-110 features

---

## Data Quality Checks

### Pre-Processing Validation

1. **Check input data integrity**
   ```python
   assert df['idx'].is_unique
   assert df['fuel_kg'].min() > 0
   assert df['fuel_kg'].isna().sum() == 0
   ```

2. **Verify no data leakage**
   ```python
   train_flights = set(train_df['flight_id'].unique())
   val_flights = set(val_df['flight_id'].unique())
   assert len(train_flights.intersection(val_flights)) == 0
   ```

### Post-Processing Validation

1. **No missing values**
   ```python
   assert X_train.isna().sum().sum() == 0
   assert X_val.isna().sum().sum() == 0
   ```

2. **No infinite values**
   ```python
   assert np.isinf(X_train).sum().sum() == 0
   assert np.isinf(X_val).sum().sum() == 0
   ```

3. **Consistent columns**
   ```python
   assert list(X_train.columns) == list(X_val.columns)
   assert X_train.shape[1] == X_val.shape[1]
   ```

4. **Target distribution sanity**
   ```python
   # Val should be similar to train (within reason)
   assert 0.8 < y_val.mean() / y_train.mean() < 1.2
   assert 0.8 < y_val.std() / y_train.std() < 1.2
   ```

---

## Usage Instructions

### Running Phase 3

**Execute the pipeline:**
```bash
python notebooks/phase3_data_preparation.py
```

**Expected runtime:** 2-5 minutes

**Console output:**
- Step-by-step progress
- Features removed/added at each step
- Data quality checks
- Final dataset summary

### Using Processed Data (Phase 4)

**Load training data:**
```python
import pandas as pd

X_train = pd.read_parquet('cache/X_train.parquet')
y_train = pd.read_parquet('cache/y_train.parquet')['fuel_kg']

print(f"Training data: {X_train.shape}")
```

**Load validation data:**
```python
X_val = pd.read_parquet('cache/X_val.parquet')
y_val = pd.read_parquet('cache/y_val.parquet')['fuel_kg']
```

**Load preprocessing pipeline:**
```python
import pickle

with open('models/preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Access individual transformers
cleaner = pipeline['cleaner']
missing_handler = pipeline['missing_handler']
encoder = pipeline['encoder']
scaler = pipeline['scaler']
```

**Apply to ranking data (Phase 7):**
```python
# Load ranking features (from Phase 2 feature extraction)
X_rank = pd.read_parquet('cache/rank_features_complete.parquet')

# Apply pipeline
X_rank_clean = cleaner.transform(X_rank)
X_rank_imputed = missing_handler.transform(X_rank_clean)
X_rank_encoded = encoder.transform(X_rank_imputed)
X_rank_final = scaler.transform(X_rank_encoded)

# Ready for prediction
predictions = model.predict(X_rank_final)
```

---

## Troubleshooting

### Issue 1: Column Mismatch Between Train and Val

**Symptom:**
```
⚠ WARNING: Train and val have different number of columns!
```

**Cause:** One-hot encoding creates different columns if categories differ.

**Solution:** Already handled in pipeline:
```python
# Add missing columns to val (with zeros)
for col in train_cols - val_cols:
    X_val[col] = 0

# Reorder to match train
X_val = X_val[X_train.columns]
```

### Issue 2: Remaining Missing Values

**Symptom:**
```
⚠ WARNING: Remaining missing values detected!
```

**Cause:** New categories in validation set, edge cases.

**Solution:** Fill with 0 (safe default):
```python
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
```

### Issue 3: Memory Error

**Symptom:** Process killed due to out of memory.

**Solution:**
- Process in smaller batches
- Reduce dtype precision (float64 → float32)
- Free memory between steps: `del X_train_raw; gc.collect()`

### Issue 4: Infinite Values After Scaling

**Symptom:**
```
RuntimeWarning: divide by zero encountered in true_divide
```

**Cause:** Feature with zero standard deviation.

**Solution:** Replace inf with 0:
```python
X_train = X_train.replace([np.inf, -np.inf], 0)
```

---

## Key Design Decisions

### Decision 1: Why Temporal Split?

**Alternative:** Random K-fold cross-validation

**Chosen:** Temporal split (April-July train, August val)

**Rationale:**
- Ranking data is September (future)
- Must validate that model generalizes forward in time
- Simulates production deployment scenario
- Prevents lookahead bias

### Decision 2: Why Missing Indicators?

**Alternative:** Just impute, don't create indicators

**Chosen:** Create indicators + impute

**Rationale:**
- Missingness itself is informative (ADS-B coverage patterns)
- Model can learn different behavior for flights with/without data
- Adds minimal features (~13) for significant information gain
- Standard practice in production ML

### Decision 3: Why Target Encoding for Aircraft Type?

**Alternative:** One-hot encoding (26 columns)

**Chosen:** Target encoding (1 column)

**Rationale:**
- 26 categories → too many for one-hot
- Aircraft type strongly correlates with fuel (different sizes)
- Target encoding captures this relationship directly
- More sample-efficient

**Risk:** Target leakage → Mitigated by fitting only on train

### Decision 4: Why StandardScaler Instead of MinMaxScaler?

**Alternative:** MinMaxScaler (scale to [0, 1])

**Chosen:** StandardScaler (mean=0, std=1)

**Rationale:**
- Tree-based models (our primary model) don't care about scale
- StandardScaler preserves outliers better
- More interpretable (units are standard deviations)
- Robust to new extreme values in validation/test

---

## Performance Benchmarks

**Expected runtime** (on laptop):
- Data loading: 10-20 seconds
- Feature cleaning: 5-10 seconds
- Train/val split: 5-10 seconds
- Missing value handling: 20-30 seconds
- Categorical encoding: 30-40 seconds
- Feature scaling: 10-20 seconds
- Saving datasets: 30-40 seconds

**Total:** 2-5 minutes

**Memory usage:**
- Peak: ~3-4 GB
- Output files: ~110 MB total

---

## Next Steps: Phase 4 - Model Training

### Preparation

Before starting Phase 4:
1. ✓ Verify all output files exist
2. ✓ Check file sizes are reasonable
3. ✓ Load X_train/y_train to confirm accessibility
4. ✓ Review transformation report for anomalies

### Phase 4 Overview

**Objective:** Train and evaluate machine learning models

**Models to train:**
1. **Baseline:** Linear Regression (Ridge)
2. **Baseline:** Random Forest
3. **Primary:** LightGBM (laptop-optimized)
4. **Alternative:** CatBoost (if time permits)

**Evaluation metrics:**
- Primary: RMSE (Root Mean Squared Error)
- Secondary: MAE, R², MAPE

**Validation strategy:**
- Train on X_train, evaluate on X_val
- Track overfitting (train vs val RMSE)
- Feature importance analysis

**Expected best model performance:**
- Baseline (Linear): RMSE ~600-700 kg
- Random Forest: RMSE ~450-550 kg
- **LightGBM: RMSE ~350-450 kg** ← Target

---

## Files Created

### Code Modules
```
src/
├── preprocessing.py (NEW)
│   ├── FeatureCleaner
│   ├── MissingValueHandler
│   ├── CategoricalEncoder
│   ├── FeatureScaler
│   └── Utility functions
```

### Scripts
```
notebooks/
└── phase3_data_preparation.py (NEW)
    └── Main pipeline orchestration
```

### Documentation
```
reports/
└── PHASE3_COMPLETE_DOCUMENTATION.md (THIS FILE)
```

---

## Checklist for Team

Before running Phase 3:
- [ ] Phase 2 complete (train_features_complete.parquet exists)
- [ ] At least 4 GB free memory
- [ ] At least 500 MB free disk space

After running Phase 3:
- [ ] All 6 output files created in cache/
- [ ] preprocessing_pipeline.pkl created in models/
- [ ] feature_names.json created in models/
- [ ] transformation_report.txt created in reports/
- [ ] No errors in console output
- [ ] X_train and X_val have same number of columns
- [ ] No missing or infinite values in final datasets

Ready for Phase 4:
- [ ] Can load X_train.parquet without errors
- [ ] Can load y_train.parquet without errors
- [ ] Feature count matches expectations (~105-110)
- [ ] Train/val split is ~75/25%

---

## Conclusion

**Phase 3 Status:** ✅ READY FOR EXECUTION

This comprehensive data preparation pipeline transforms raw features into clean, model-ready datasets. The pipeline is:

- **Robust:** Handles missing values, edge cases, unseen categories
- **Reproducible:** All transformations saved and reusable
- **Validated:** Multiple quality checks at each step
- **Documented:** Complete transformation report generated
- **Production-ready:** Can be applied to ranking data in Phase 7

**Team can now execute Phase 3 pipeline and proceed to Phase 4: Model Training**

---

**Document Version:** 1.0
**Last Updated:** 2025-10-15
**Author:** ML Team
**Status:** Final - Ready for Execution
