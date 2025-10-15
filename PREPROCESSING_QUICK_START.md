# Advanced Preprocessing Quick Start Guide

## Installation

```bash
pip install imbalanced-learn scikit-learn>=1.0
```

## Quick Usage

### Option 1: Complete Automated Workflow (Recommended)

```python
from src.preprocessing_integration_guide import complete_preprocessing_workflow

# Load your features DataFrame with 'fuel_kg' target
df = pd.read_parquet('cache/processed_features.parquet')

# Run complete workflow
X_train, X_val, y_train, y_val, pipeline = complete_preprocessing_workflow(
    df, split_month=8
)

# Train model
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03},
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50)]
)

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"RMSE: {rmse:.2f} kg")
```

### Option 2: Manual Step-by-Step

```python
from src.preprocessing_integration_guide import (
    preprocess_data_advanced,
    augment_training_data,
    PREPROCESSING_CONFIG
)

# 1. Preprocess
X_train_proc, X_val_proc, pipeline = preprocess_data_advanced(
    X_train, X_val, y_train, config=PREPROCESSING_CONFIG
)

# 2. Augment (optional but recommended for high fuel ranges)
X_train_aug, y_train_aug = augment_training_data(
    X_train_proc, y_train, config=PREPROCESSING_CONFIG
)

# 3. Train your model...
```

### Option 3: Custom Configuration

```python
# Speed-optimized config
FAST_CONFIG = {
    'missing_strategy': 'median',
    'outlier_method': 'percentile',
    'scaling_strategy': 'robust',
    'enable_polynomial': False,
    'enable_augmentation': False
}

# Accuracy-optimized config
ACCURACY_CONFIG = {
    'missing_strategy': 'hybrid',
    'outlier_method': 'percentile',
    'scaling_strategy': 'hybrid',
    'polynomial_degree': 2,
    'enable_polynomial': True,
    'enable_augmentation': True,
    'augmentation_strategy': 'stratified'
}

X_train, X_val, pipeline = preprocess_data_advanced(
    X_train, X_val, y_train, config=ACCURACY_CONFIG
)
```

## Individual Module Usage

### Advanced Missing Value Handler
```python
from src.preprocessing_advanced import AdvancedMissingValueHandler

handler = AdvancedMissingValueHandler(strategy='hybrid')
handler.fit(X_train)
X_train = handler.transform(X_train)
```

### Outlier Handler
```python
from src.preprocessing_advanced import OutlierHandler

handler = OutlierHandler(method='percentile', percentile_range=(0.5, 99.5))
handler.fit(X_train)
X_train = handler.transform(X_train)
```

### Polynomial Features
```python
from src.preprocessing_advanced import PolynomialFeatureGenerator

generator = PolynomialFeatureGenerator(
    degree=2,
    include_features=['altitude_mean', 'groundspeed_mean']
)
generator.fit(X_train)
X_train = generator.transform(X_train)
```

### Data Augmentation
```python
from src.data_augmentation import FuelRangeAugmenter

augmenter = FuelRangeAugmenter(
    augment_ranges=[(2500, np.inf, 4.0)],
    strategy='interpolate'
)
X_aug, y_aug = augmenter.fit_resample(X_train, y_train)
```

## Expected Results

| Current (Phase 7) | Expected (Advanced) | Improvement |
|-------------------|---------------------|-------------|
| 144.19 kg RMSE | **120-135 kg RMSE** | **9-24 kg** |
| 44.73 kg MAE | **35-40 kg MAE** | **5-10 kg** |
| 0.9620 R² | **0.967-0.970 R²** | **+0.5-0.8%** |

## Key Improvements

1. **Hybrid Missing Value Handling**: KNN, Iterative, Median strategies
2. **Outlier Clipping**: Percentile-based robust handling
3. **Polynomial Features**: Intelligent interaction features
4. **Ratio Features**: Domain-informed physics ratios
5. **Leave-One-Out Encoding**: Robust categorical encoding
6. **Advanced Scaling**: Hybrid (Robust, Quantile, Power)
7. **Feature Selection**: Variance, Correlation, Mutual Information
8. **Data Augmentation**: SMOTE, Interpolation for extreme ranges

## Troubleshooting

**Error: "module not found"**
```bash
pip install imbalanced-learn scikit-learn>=1.0
```

**Memory issues**
```python
# Disable augmentation or polynomial features
config = PREPROCESSING_CONFIG.copy()
config['enable_augmentation'] = False
config['enable_polynomial'] = False
```

**Too slow**
```python
# Use fast config
config = {
    'missing_strategy': 'median',
    'outlier_method': 'none',
    'enable_feature_selection': False
}
```

## Documentation

- **Full guide**: `documentation/PREPROCESSING_REVAMP.md`
- **Integration examples**: `src/preprocessing_integration_guide.py`
- **Module reference**: `src/preprocessing_advanced.py`, `src/data_augmentation.py`

## Next Steps

1. Test on small subset first
2. Compare performance vs baseline
3. Iterate on configuration
4. Apply to full training data
5. Retrain stratified models
6. Generate submission

---

**Target**: 120-135 kg RMSE (from 144.19 kg baseline)
**Status**: Ready for testing and integration
