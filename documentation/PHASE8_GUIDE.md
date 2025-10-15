# Phase 8: Ultimate Training Pipeline - Complete Guide

**Date**: 2025-10-15
**Status**: ✅ READY FOR EXECUTION
**Target**: < 130 kg RMSE (from 144.19 kg Phase 7 baseline)

---

## Overview

Phase 8 represents the **ultimate combination** of:
- **Advanced Preprocessing** (9 sophisticated modules)
- **6-Stage Stratification** (proven best approach from Phase 7)
- **Per-Stage Data Augmentation** (addresses imbalanced ranges)
- **Multi-Model Ensemble** (LightGBM + CatBoost)

### Expected Improvements

| Component | Contribution |
|-----------|--------------|
| Advanced Missing Value Handling | +5-8 kg |
| Outlier Clipping | +2-4 kg |
| Polynomial Features | +3-6 kg |
| Ratio Features | +2-3 kg |
| Leave-One-Out Encoding | +2-4 kg |
| Advanced Scaling | +1-3 kg |
| Feature Selection | +2-4 kg |
| Data Augmentation | +5-10 kg |
| **Total Expected** | **22-42 kg improvement** |

**Target RMSE**: **120-135 kg** (vs 144.19 kg baseline)

---

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install imbalanced-learn scikit-learn>=1.0 lightgbm catboost
```

### Step 1: Prepare Features

First, ensure you have extracted features for both training and ranking data:

```python
# Your feature extraction script should create:
# - cache/processed_features.parquet (training features + fuel_kg target)
# - cache/ranking_features.parquet (ranking features)

# These files should include all engineered features from Phase 7:
# - Altitude features
# - Speed features
# - Vertical rate features
# - Distance features
# - Flight phase features
# - Aircraft type
# - Origin/destination
# - Temporal features
# etc.
```

### Step 2: Train Models

```bash
python train_phase8_ultimate.py
```

This will:
1. Load training data
2. Apply temporal split (April-July train, August validation)
3. Apply advanced preprocessing pipeline
4. Train 6-stage stratified models with data augmentation
5. Create ensemble predictions
6. Save models and results

**Expected Runtime**: 2-4 hours (depending on hardware)

### Step 3: Generate Predictions

```bash
python predict_phase8_ranking.py
```

This will:
1. Load trained models
2. Load ranking features
3. Apply same preprocessing
4. Generate stratified predictions
5. Create submission file

**Output**: `submissions/gentle-orange_phase8_v1_TIMESTAMP.parquet`

---

## Detailed Workflow

### Phase 8 Training Pipeline

#### Data Flow

```
Raw Features
    ↓
Temporal Split (train/val)
    ↓
Advanced Preprocessing
    ├─ Missing Value Handling (Hybrid: KNN/Iterative/Median)
    ├─ Outlier Clipping (Percentile: 0.5%-99.5%)
    ├─ Ratio Features (distance/duration, altitude/speed, etc.)
    ├─ LOO Encoding (aircraft_type, origin_icao, destination_icao)
    ├─ Polynomial Features (2nd degree interactions)
    ├─ Advanced Scaling (Hybrid: Robust/Quantile/Power)
    └─ Feature Selection (Variance + Correlation + MI)
    ↓
6-Stage Stratification
    ├─ Very Low (0-200 kg): LightGBM
    ├─ Low (200-500 kg): LightGBM
    ├─ Medium (500-1000 kg): LightGBM + Augmentation
    ├─ Medium-High (1000-1500 kg): LightGBM + Augmentation
    ├─ High (1500-2500 kg): LightGBM + CatBoost + Augmentation
    └─ Very High (2500+ kg): LightGBM + CatBoost + Augmentation
    ↓
Ensemble Predictions
    ↓
Evaluation & Model Saving
```

#### Configuration

The pipeline uses `PHASE8_CONFIG` dictionary in `train_phase8_ultimate.py`:

```python
PHASE8_CONFIG = {
    # Stratification thresholds
    'stratification_thresholds': [200, 500, 1000, 1500, 2500],

    # Preprocessing
    'preprocessing': {
        'missing_strategy': 'hybrid',  # KNN + Iterative + Median
        'outlier_method': 'percentile',
        'outlier_percentile_range': (0.5, 99.5),
        'scaling_strategy': 'hybrid',  # Robust + Quantile + Power
        'polynomial_degree': 2,
        'enable_polynomial': True,
        'enable_ratios': True,
        'enable_feature_selection': True,
        # ...
    },

    # Augmentation multipliers per stage
    'augmentation': {
        'very_low': None,
        'low': None,
        'medium': {'multiplier': 1.3, 'strategy': 'noise'},
        'medium_high': {'multiplier': 1.5, 'strategy': 'noise'},
        'high': {'multiplier': 2.0, 'strategy': 'interpolate'},
        'very_high': {'multiplier': 4.0, 'strategy': 'interpolate'}
    },

    # Model hyperparameters per stage
    'models': {
        'very_low': { 'lgb': {...} },
        'low': { 'lgb': {...} },
        'medium': { 'lgb': {...} },
        'medium_high': { 'lgb': {...} },
        'high': { 'lgb': {...}, 'cat': {...} },
        'very_high': { 'lgb': {...}, 'cat': {...} }
    },

    # Ensemble weights
    'ensemble_weights': {
        'very_low': {'lgb': 1.0},
        'low': {'lgb': 1.0},
        'medium': {'lgb': 1.0},
        'medium_high': {'lgb': 1.0},
        'high': {'lgb': 0.5, 'cat': 0.5},
        'very_high': {'lgb': 0.4, 'cat': 0.6}
    }
}
```

#### Output Files

**Models** (saved to `models/`):
- `phase8_preprocessing_pipeline.pkl` - Complete preprocessing pipeline
- `phase8_very_low_lgb.txt` - Very low fuel LightGBM
- `phase8_low_lgb.txt` - Low fuel LightGBM
- `phase8_medium_lgb.txt` - Medium fuel LightGBM
- `phase8_medium_high_lgb.txt` - Medium-high fuel LightGBM
- `phase8_high_lgb.txt` - High fuel LightGBM
- `phase8_high_cat.cbm` - High fuel CatBoost
- `phase8_very_high_lgb.txt` - Very high fuel LightGBM
- `phase8_very_high_cat.cbm` - Very high fuel CatBoost

**Reports** (saved to `reports/`):
- `phase8_results_TIMESTAMP.json` - Detailed metrics per stage

---

## Advanced Usage

### Custom Configuration

You can modify `PHASE8_CONFIG` to customize the pipeline:

#### Speed-Optimized Configuration

```python
# Faster training, slightly lower accuracy
PHASE8_CONFIG['preprocessing']['missing_strategy'] = 'median'
PHASE8_CONFIG['preprocessing']['enable_polynomial'] = False
PHASE8_CONFIG['augmentation']['high']['multiplier'] = 1.5
PHASE8_CONFIG['augmentation']['very_high']['multiplier'] = 2.0
PHASE8_CONFIG['training']['num_boost_round'] = 1000
```

#### Accuracy-Optimized Configuration

```python
# Slower training, best accuracy
PHASE8_CONFIG['preprocessing']['missing_strategy'] = 'iterative'
PHASE8_CONFIG['preprocessing']['polynomial_degree'] = 3
PHASE8_CONFIG['preprocessing']['mutual_info_percentile'] = 90
PHASE8_CONFIG['augmentation']['very_high']['multiplier'] = 5.0
PHASE8_CONFIG['training']['num_boost_round'] = 3000
```

### Per-Stage Tuning

If you want to focus on specific stages:

```python
# Focus on very high fuel range (main bottleneck)
PHASE8_CONFIG['augmentation']['very_high']['multiplier'] = 6.0
PHASE8_CONFIG['models']['very_high']['cat']['iterations'] = 3000
PHASE8_CONFIG['models']['very_high']['cat']['l2_leaf_reg'] = 15.0
PHASE8_CONFIG['ensemble_weights']['very_high'] = {'lgb': 0.3, 'cat': 0.7}
```

---

## Inference Pipeline

### Prediction Routing Methods

The inference script supports two routing methods:

#### 1. Heuristic Routing (Fast)

```python
predictions = pipeline.predict(X_rank_proc, method='heuristic')
```

- **Speed**: Fast, single-pass
- **Accuracy**: Good
- **How it works**: Estimates fuel range from `interval_duration_min` and `distance_covered_km`

#### 2. Iterative Routing (Accurate)

```python
predictions = pipeline.predict(X_rank_proc, method='iterative')
```

- **Speed**: Slower, two-pass
- **Accuracy**: Better
- **How it works**:
  1. First pass with heuristic routing
  2. Refine routing based on initial predictions
  3. Second pass with refined routing

**Recommended**: Use iterative method for final submission.

### Submission Validation

```python
# Load and validate submission
submission = pd.read_parquet('submissions/gentle-orange_phase8_v1_TIMESTAMP.parquet')

# Check format
assert submission.shape[0] == 24289, "Wrong number of rows"
assert list(submission.columns) == ['idx', 'flight_id', 'start', 'end', 'fuel_kg'], "Wrong columns"
assert submission['fuel_kg'].notna().all(), "Null values present"
assert (submission['fuel_kg'] >= 0).all(), "Negative values present"

# Check statistics
print(f"Fuel range: {submission['fuel_kg'].min():.2f} - {submission['fuel_kg'].max():.2f} kg")
print(f"Fuel mean: {submission['fuel_kg'].mean():.2f} kg")
print(f"Fuel median: {submission['fuel_kg'].median():.2f} kg")

# Compare with Phase 7
phase7_submission = pd.read_parquet('submissions/phase7_submission.parquet')
difference = submission['fuel_kg'] - phase7_submission['fuel_kg']
print(f"\nDifference from Phase 7:")
print(f"  Mean: {difference.mean():.2f} kg")
print(f"  Std: {difference.std():.2f} kg")
```

---

## Expected Results

### Per-Stage Performance

| Stage | Phase 7 RMSE | Phase 8 Target | Expected Improvement |
|-------|--------------|----------------|----------------------|
| **Very Low (0-200 kg)** | ~40 kg | **30-35 kg** | 12-25% |
| **Low (200-500 kg)** | ~45 kg | **35-40 kg** | 11-22% |
| **Medium (500-1000 kg)** | 90.67 kg | **70-75 kg** | 17-23% |
| **Medium-High (1000-1500 kg)** | 89.61 kg | **68-73 kg** | 19-24% |
| **High (1500-2500 kg)** | 134.38 kg | **100-110 kg** | 18-26% |
| **Very High (2500+ kg)** | 861-903 kg | **400-500 kg** | 45-55% |

### Overall Performance

| Metric | Phase 7 | Phase 8 Target | Improvement |
|--------|---------|----------------|-------------|
| **RMSE** | 144.19 kg | **120-135 kg** | **9-24 kg (6-17%)** |
| **MAE** | 44.73 kg | **35-40 kg** | **5-10 kg (11-22%)** |
| **R²** | 0.9620 | **0.967-0.970** | **+0.5-0.8%** |

---

## Troubleshooting

### Memory Issues

```python
# Reduce memory usage
PHASE8_CONFIG['preprocessing']['enable_polynomial'] = False
PHASE8_CONFIG['augmentation'] = {stage: None for stage in PHASE8_CONFIG['augmentation']}
```

### Training Too Slow

```python
# Speed up training
PHASE8_CONFIG['training']['num_boost_round'] = 1000
PHASE8_CONFIG['training']['verbose_eval'] = 200
PHASE8_CONFIG['preprocessing']['mutual_info_percentile'] = 90  # Keep more features
```

### Poor Performance on Specific Stage

```python
# Increase augmentation for that stage
PHASE8_CONFIG['augmentation']['very_high']['multiplier'] = 6.0

# Tune hyperparameters
PHASE8_CONFIG['models']['very_high']['lgb']['learning_rate'] = 0.01
PHASE8_CONFIG['models']['very_high']['lgb']['num_leaves'] = 30

# Adjust ensemble weights
PHASE8_CONFIG['ensemble_weights']['very_high'] = {'lgb': 0.2, 'cat': 0.8}
```

### Feature Extraction Issues

If you don't have `cache/processed_features.parquet`:

1. **Run your feature extraction pipeline** from Phase 7
2. Ensure it includes all engineered features
3. Save to `cache/processed_features.parquet` with:
   - All feature columns
   - `fuel_kg` target column
   - `flight_id` identifier
   - `start_month` for temporal split

---

## Performance Monitoring

### During Training

The script prints detailed statistics:

```
STAGE: VERY_HIGH
================================================================================
Samples: 5,325
Features: 156
Fuel range: 2500.10 - 32205.03 kg
Fuel mean: 4567.23 kg
Fuel std: 2345.67 kg

[VERY_HIGH] Applying augmentation...
  Strategy: interpolate
  Multiplier: 4.0x
  Original samples: 5,325
  Augmented samples: 21,300
  Added: 15,975

[VERY_HIGH] Training LightGBM...
  LightGBM - Train RMSE: 234.56 kg, Val RMSE: 456.78 kg

[VERY_HIGH] Training CatBoost...
  CatBoost - Train RMSE: 223.45 kg, Val RMSE: 445.67 kg

[VERY_HIGH] ENSEMBLE RESULTS:
  RMSE: 420.12 kg
  MAE: 189.34 kg
  R²: 0.9234
```

### After Training

Check `reports/phase8_results_TIMESTAMP.json`:

```json
{
  "overall": {
    "rmse": 125.34,
    "mae": 37.89,
    "r2": 0.9687
  },
  "very_low": {
    "metrics": {
      "rmse": 32.45,
      "mae": 18.23,
      "r2": 0.9876
    },
    "range": [0, 200],
    "train_samples": 10234,
    "train_samples_augmented": 10234,
    "val_samples": 5096
  },
  // ... other stages
}
```

---

## Validation Checklist

Before submitting:

- [ ] Training completed without errors
- [ ] All 6 stages trained successfully
- [ ] Overall RMSE < 144.19 kg (better than Phase 7)
- [ ] Per-stage RMSE reasonable (no extreme outliers)
- [ ] Models saved to `models/` directory
- [ ] Preprocessing pipeline saved
- [ ] Inference script runs without errors
- [ ] Submission file has correct format (24,289 rows)
- [ ] No null or negative predictions
- [ ] Fuel range is reasonable (0.45 - 32,205 kg)
- [ ] Submission mean/median similar to training data

---

## Next Steps After Phase 8

If Phase 8 achieves **120-135 kg RMSE**:

### Further Improvements (Diminishing Returns)

1. **External Data** (+10-20 kg potential):
   - Weather data (wind, temperature, pressure)
   - Aircraft performance databases
   - Historical routing patterns

2. **Advanced Models** (+5-10 kg potential):
   - LSTM/GRU for sequence modeling
   - Attention mechanisms
   - Physics-informed neural networks

3. **Ensemble Refinement** (+3-5 kg potential):
   - Stacking with meta-learner
   - Multiple random seeds
   - Cross-validation ensembles

### Practical Limit

With current data, **115-120 kg RMSE** appears to be the practical limit. To go below this requires external data sources not available in the competition.

---

## Summary

Phase 8 represents the **state-of-the-art** approach combining:
- ✅ Advanced preprocessing (9 modules)
- ✅ Proven 6-stage stratification
- ✅ Per-stage data augmentation
- ✅ Multi-model ensemble
- ✅ Comprehensive validation

**Expected Outcome**: **120-135 kg RMSE** (9-24 kg improvement from 144.19 kg)

**Status**: Ready for execution

---

**Team**: ML Engineering
**Date**: 2025-10-15
**Version**: Phase 8 Ultimate
