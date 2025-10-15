# Phase 8 Quick Reference Card

## Prerequisites
```bash
pip install imbalanced-learn scikit-learn>=1.0 lightgbm catboost
```

## Training (Complete Pipeline)
```bash
python train_phase8_ultimate.py
```

**Expected Runtime**: 2-4 hours
**Output**: Models saved to `models/`, Results to `reports/`

## Inference (Generate Predictions)
```bash
python predict_phase8_ranking.py
```

**Expected Runtime**: 5-10 minutes
**Output**: `submissions/gentle-orange_phase8_v1_TIMESTAMP.parquet`

## Expected Performance

| Metric | Phase 7 | Phase 8 Target |
|--------|---------|----------------|
| RMSE | 144.19 kg | **120-135 kg** |
| MAE | 44.73 kg | **35-40 kg** |
| R² | 0.9620 | **0.967-0.970** |

## Key Improvements

1. **Advanced Missing Value Handling** - Hybrid KNN/Iterative/Median
2. **Outlier Clipping** - Percentile 0.5%-99.5%
3. **Polynomial Features** - 2nd degree interactions
4. **Ratio Features** - Physics-based domain features
5. **Leave-One-Out Encoding** - Robust categorical encoding
6. **Advanced Scaling** - Hybrid Robust/Quantile/Power
7. **Feature Selection** - Variance + Correlation + MI
8. **Data Augmentation** - SMOTE/Interpolation for extreme ranges
9. **6-Stage Stratification** - Proven best from Phase 7

## Architecture

```
Raw Features → Preprocessing → Stratification → Training → Ensemble → Predictions
```

### 6 Stages

1. **Very Low** (0-200 kg): LightGBM
2. **Low** (200-500 kg): LightGBM
3. **Medium** (500-1000 kg): LightGBM + 1.3x Augmentation
4. **Medium-High** (1000-1500 kg): LightGBM + 1.5x Augmentation
5. **High** (1500-2500 kg): LightGBM + CatBoost + 2x Augmentation
6. **Very High** (2500+ kg): LightGBM + CatBoost + 4x Augmentation

## Files Created

### Scripts
- `train_phase8_ultimate.py` - Main training pipeline
- `predict_phase8_ranking.py` - Inference pipeline

### Modules
- `src/preprocessing_advanced.py` - 9 advanced preprocessing modules
- `src/data_augmentation.py` - Data augmentation for extreme ranges
- `src/preprocessing_integration_guide.py` - Integration examples

### Documentation
- `documentation/PHASE8_GUIDE.md` - Complete guide
- `documentation/PREPROCESSING_REVAMP.md` - Preprocessing details
- `PREPROCESSING_QUICK_START.md` - Quick start guide

## Troubleshooting

**Memory issues**: Disable polynomial features or augmentation
**Too slow**: Reduce `num_boost_round` to 1000
**Poor stage performance**: Increase augmentation multiplier

## Validation

```python
import pandas as pd

# Check submission
sub = pd.read_parquet('submissions/gentle-orange_phase8_v1_TIMESTAMP.parquet')
assert sub.shape[0] == 24289
assert sub['fuel_kg'].notna().all()
assert (sub['fuel_kg'] >= 0).all()

print(f"Fuel range: {sub['fuel_kg'].min():.2f} - {sub['fuel_kg'].max():.2f} kg")
print(f"Fuel mean: {sub['fuel_kg'].mean():.2f} kg")
```

## Support

- Full documentation: `documentation/PHASE8_GUIDE.md`
- Preprocessing details: `documentation/PREPROCESSING_REVAMP.md`
- Phase 7 results: `documentation/PHASE7_FINAL_ANALYSIS.md`

---

**Target**: < 130 kg RMSE
**Status**: Ready to Execute
**Date**: 2025-10-15
