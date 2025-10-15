# Phase 3: Data Preparation - Quick Reference Guide

## ⚡ Quick Start

### Run Phase 3 Pipeline

```bash
cd .
python notebooks/phase3_data_preparation.py
```

**Runtime:** 2-5 minutes
**Memory:** ~4 GB peak

---

## 📊 What Phase 3 Does

```
Raw Features (79)
    ↓
[1] Clean: Remove 2-3 features
    ↓
[2] Split: Train (75%) / Val (25%) by time
    ↓
[3] Impute: Handle 35% missing values
    ↓
[4] Encode: Convert 7 categorical features
    ↓
[5] Scale: Standardize ~60 numeric features
    ↓
Final Datasets (~105 features)
```

---

## 📁 Output Files

### Data Files (cache/)
- `X_train.parquet` - Training features (~98K samples)
- `y_train.parquet` - Training target
- `train_ids.parquet` - Train identifiers
- `X_val.parquet` - Validation features (~32K samples)
- `y_val.parquet` - Validation target
- `val_ids.parquet` - Val identifiers

### Model Files (models/)
- `preprocessing_pipeline.pkl` - Fitted transformers
- `feature_names.json` - Final feature list

### Reports (reports/)
- `phase3_transformation_report.txt` - Detailed summary

---

## 🔍 Key Transformations

### 1. Features Removed
- `avg_speed_over_ground_kt` (constant)
- `interval_duration_sec` (duplicate of _min)
- Total removed: 2-3 features

### 2. Missing Values Handled
- **Created indicators:** ~13 features (e.g., `groundspeed_mean_is_missing`)
- **Imputation:** Median by aircraft type + global fallback
- **Final missing:** 0

### 3. Categorical Encoding

| Feature | Method | Output |
|---------|--------|--------|
| aircraft_type | Target encoding | 1 column |
| origin_icao, destination_icao | Frequency encoding | 2 columns |
| detected_flight_phase | One-hot | 7 columns |
| altitude_band | One-hot | 7 columns |
| time_of_day | One-hot | 4 columns |
| flight_phase_position | One-hot | 6 columns |

**Total:** Remove 7 original, add ~27 new columns

### 4. Feature Scaling
- **Method:** StandardScaler (mean=0, std=1)
- **Scaled:** ~60 numeric features
- **Excluded:** Binary features, indicators

---

## ✅ Quality Checks

After running, verify:

```python
import pandas as pd

# Load data
X_train = pd.read_parquet('cache/X_train.parquet')
X_val = pd.read_parquet('cache/X_val.parquet')
y_train = pd.read_parquet('cache/y_train.parquet')['fuel_kg']
y_val = pd.read_parquet('cache/y_val.parquet')['fuel_kg']

# Check 1: No missing values
assert X_train.isna().sum().sum() == 0
assert X_val.isna().sum().sum() == 0

# Check 2: Same columns
assert X_train.shape[1] == X_val.shape[1]
assert list(X_train.columns) == list(X_val.columns)

# Check 3: Reasonable split
assert len(X_train) > len(X_val)  # Train > Val
assert len(X_train) / (len(X_train) + len(X_val)) > 0.7  # ~75/25 split

# Check 4: Target distribution
print(f"Train fuel: {y_train.mean():.2f} ± {y_train.std():.2f} kg")
print(f"Val fuel: {y_val.mean():.2f} ± {y_val.std():.2f} kg")
# Should be similar

print("✅ All checks passed!")
```

---

## 🚀 Using the Data (Phase 4)

### Load for Training

```python
import pandas as pd

# Load train data
X_train = pd.read_parquet('cache/X_train.parquet')
y_train = pd.read_parquet('cache/y_train.parquet')['fuel_kg']

# Train model
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)
```

### Evaluate on Validation

```python
# Load validation data
X_val = pd.read_parquet('cache/X_val.parquet')
y_val = pd.read_parquet('cache/y_val.parquet')['fuel_kg']

# Predict
y_pred = model.predict(X_val)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_val, y_pred, squared=False)
mae = mean_absolute_error(y_val, y_pred)

print(f"Validation RMSE: {rmse:.2f} kg")
print(f"Validation MAE: {mae:.2f} kg")
```

### Apply to Ranking Data

```python
import pickle

# Load preprocessing pipeline
with open('models/preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load ranking features (must run Phase 2 on ranking data first)
X_rank_raw = pd.read_parquet('cache/rank_features_complete.parquet')

# Apply transformations
X_rank_clean = pipeline['cleaner'].transform(X_rank_raw)
X_rank_imputed = pipeline['missing_handler'].transform(X_rank_clean)
X_rank_encoded = pipeline['encoder'].transform(X_rank_imputed)
X_rank_final = pipeline['scaler'].transform(X_rank_encoded)

# Predict
predictions = model.predict(X_rank_final)
```

---

## 🐛 Common Issues

### Issue 1: Out of Memory

**Solution:**
```bash
# Reduce batch size in script or use swap
sudo sysctl vm.swappiness=60
```

### Issue 2: Column Mismatch

**Symptom:** `ValueError: Shape mismatch`

**Solution:** Pipeline automatically aligns columns, but if error persists:
```python
# Ensure same columns
X_val = X_val[X_train.columns]
```

### Issue 3: Can't Load Pipeline

**Symptom:** `FileNotFoundError: preprocessing_pipeline.pkl`

**Solution:** Run Phase 3 first to create the file

---

## 📈 Expected Results

### Train/Val Split
- **Train:** ~98,000 samples (April-July)
- **Val:** ~32,000 samples (August)
- **Ratio:** ~75/25%

### Feature Count
- **Before:** 79 features
- **After:** 105-110 features

### Target Distribution
- **Train mean:** ~555 kg
- **Val mean:** ~560 kg (should be close)

### Data Quality
- **Missing values:** 0
- **Infinite values:** 0
- **Duplicates:** Removed
- **Leakage:** None

---

## 📝 Next Steps

After Phase 3 completes successfully:

1. ✅ Verify all output files exist
2. ✅ Run quality checks
3. ✅ Review transformation report
4. ➡️ **Proceed to Phase 4: Model Training**

---

## 📚 Full Documentation

For detailed information, see:
- `reports/PHASE3_COMPLETE_DOCUMENTATION.md` - Complete guide
- `reports/phase3_transformation_report.txt` - Execution report
- `src/preprocessing.py` - Module documentation

---

## 🎯 Success Criteria

Phase 3 is successful if:

- ✅ Pipeline runs without errors
- ✅ All 8 output files created
- ✅ No missing/infinite values in final data
- ✅ Train and val have same feature columns
- ✅ Split is approximately 75/25%
- ✅ Feature count is 105-110
- ✅ Target distributions are similar

**If all criteria met → Ready for Phase 4!**

---

**Last Updated:** 2025-10-15
