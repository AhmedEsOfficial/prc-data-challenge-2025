# Comprehensive ML Plan: Aircraft Fuel Consumption Prediction

## Challenge Overview
**Objective**: Predict `fuel_kg` burnt during specific time intervals of flights using trajectory data
**Evaluation Metric**: RMSE (Root Mean Squared Error)
**Constraint**: Laptop-only training (memory & compute limited)
**Timeline**: 17 days

## Data Summary
- **Training**: 11,037 flights, 131,530 fuel intervals (April-August 2025)
- **Ranking**: 1,888 flights, 24,289 fuel intervals (September 2025)
- **Fuel Range**: 0.45 - 32,205 kg per interval (avg ~558 kg, std ~884 kg)
- **Aircraft Types**: 26 in training, 19 in ranking
- **Trajectory Completeness**: 29-32% missing for speed/track/vertical_rate, 99%+ missing for mach/TAS/CAS

---

## Phase 1: Deep Data Understanding (Days 1-2)

### 1.1 Load & Profile All Datasets
- Use **Polars** for memory-efficient parquet reading
- Analyze 5-10 representative trajectory files in detail
- Map fuel intervals to trajectory timestamps
- Understand temporal alignment between fuel intervals and trajectory points

### 1.2 Exploratory Data Analysis
**Fuel Distribution Analysis**:
- Fuel consumption by aircraft type
- Fuel consumption by flight duration
- Fuel consumption by interval position in flight (early/middle/late)
- Identify outliers and edge cases

**Trajectory Analysis**:
- Identify flight phases (takeoff, climb, cruise, descent, landing) from altitude/speed profiles
- Correlation analysis: altitude changes vs fuel, speed vs fuel, vertical rate vs fuel
- Missing data patterns in trajectories (gaps, sensor failures)
- Interval duration analysis (are they uniform or variable?)

**Cross-Dataset Relationships**:
- Origin/destination airport elevations impact on fuel
- Great circle distance vs actual flight path
- Time of day effects on fuel consumption

---

## Phase 2: Feature Engineering Pipeline (Days 3-5)

### 2.1 Core Features (Memory-Efficient Extraction)

**Interval-Based Features**:
- `interval_duration_seconds`: Length of the fuel measurement interval
- `interval_position_pct`: Position in flight (0-100%, where interval starts)
- `time_of_day`: Hour of day (may correlate with air traffic, weather)
- `seconds_from_takeoff`: Time elapsed since takeoff
- `seconds_to_landing`: Time remaining until landing

**Trajectory Aggregations (Within Each Interval)**:

*Altitude Features*:
- `altitude_mean`, `altitude_min`, `altitude_max`, `altitude_std`
- `altitude_delta`: Net altitude change during interval
- `altitude_range`: max - min altitude
- `time_above_30k_ft`: Seconds spent above 30,000 ft (cruise indicator)
- `time_below_10k_ft`: Seconds spent below 10,000 ft (takeoff/landing indicator)

*Speed Features*:
- `groundspeed_mean`, `groundspeed_max`, `groundspeed_std`
- `groundspeed_delta`: Speed change during interval
- `speed_acceleration`: Rate of speed change

*Vertical Rate Features*:
- `vertical_rate_mean`: Average climb/descent rate
- `vertical_rate_std`: Variance in vertical movement
- `climb_time_pct`: Percentage of interval spent climbing
- `descent_time_pct`: Percentage of interval spent descending

*Position & Navigation*:
- `distance_covered_km`: Haversine distance from first to last position in interval
- `heading_changes`: Number of significant track angle changes
- `trajectory_point_count`: Number of ADS-B/ACARS points in interval (data quality indicator)

**Flight-Level Features**:
- `aircraft_type`: Encoded categorically (one-hot or target encoding)
- `total_flight_duration_hours`: Planned flight duration
- `great_circle_distance_km`: Direct distance origin to destination
- `origin_elevation_ft`: Elevation of departure airport
- `dest_elevation_ft`: Elevation of arrival airport
- `elevation_delta_ft`: dest_elevation - origin_elevation

**Physics-Derived Features**:
- `climb_work_proxy`: altitude_delta × 1000 (proxy for energy expenditure)
- `flight_phase`: Categorical (takeoff/climb/cruise/descent/landing) detected from altitude/speed
- `cruise_deviation`: Difference between current altitude and typical cruise altitude for aircraft type
- `phase_transition`: Binary flag if interval contains phase change

**Missingness Indicators** (for robustness):
- `missing_groundspeed_pct`: Percentage of missing groundspeed data in interval
- `missing_vertical_rate_pct`: Percentage of missing vertical rate data in interval

### 2.2 Implementation Strategy
```python
# Batch processing to manage memory
BATCH_SIZE = 100

def process_trajectory_batch(flight_ids):
    features = []
    for flight_id in flight_ids:
        traj_df = pl.read_parquet(f'flights_train/{flight_id}.parquet')
        fuel_df = fuel_train[fuel_train['flight_id'] == flight_id]

        for _, fuel_row in fuel_df.iter_rows(named=True):
            interval_features = extract_interval_features(
                traj_df, fuel_row['start'], fuel_row['end']
            )
            features.append(interval_features)
    return pl.DataFrame(features)

# Cache processed features
for i in range(0, len(flight_ids), BATCH_SIZE):
    batch = flight_ids[i:i+BATCH_SIZE]
    features = process_trajectory_batch(batch)
    features.write_parquet(f'cache/features_batch_{i}.parquet')
```

---

## Phase 3: Data Preparation (Day 6)

### 3.1 Merge Strategy
- Join fuel intervals with trajectory features by `flight_id` and time windows
- Handle trajectories with gaps:
  - If interval has < 3 trajectory points: flag as low-quality, use forward-fill from previous interval
  - Use interpolation for isolated missing values

### 3.2 Missing Value Handling
- **Groundspeed/track/vertical_rate**: Use available data, create missingness percentage features
- **Mach/TAS/CAS**: Ignore entirely (99%+ missing)
- **Flight metadata**: Should be complete, validate no nulls

### 3.3 Train/Validation Split
**Primary Strategy: Temporal Split**
- Training: April - July 2025 (75% of data)
- Validation: August 2025 (25% of data)
- **Rationale**: Ensures model generalizes to future months (September ranking data)

**Alternative: Stratified K-Fold**
- 3-5 folds stratified by aircraft type
- Use if temporal split shows poor validation scores

### 3.4 Target Variable Analysis
- Check for extreme outliers in `fuel_kg`
- Consider log-transform if distribution is highly skewed
- Validate no negative or zero values (invalid)

---

## Phase 4: Model Development (Days 7-10)

### 4.1 Baseline Models (Quick Iteration)
**Purpose**: Establish performance floor, identify feature issues early

1. **Linear Regression with Regularization**
   ```python
   from sklearn.linear_model import Ridge, Lasso
   ridge = Ridge(alpha=1.0)
   ridge.fit(X_train, y_train)
   baseline_rmse = mean_squared_error(y_val, ridge.predict(X_val), squared=False)
   ```

2. **Lightweight Random Forest**
   ```python
   from sklearn.ensemble import RandomForestRegressor
   rf = RandomForestRegressor(
       n_estimators=100,
       max_depth=10,
       min_samples_split=20,
       n_jobs=-1,
       random_state=42
   )
   rf.fit(X_train, y_train)
   ```

### 4.2 Primary Model: LightGBM
**Why LightGBM**:
- Fastest training among gradient boosting libraries
- Lowest memory footprint (leaf-wise growth)
- Excellent performance on tabular data
- Native categorical feature support

**Configuration**:
```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,  # Feature subsampling
    'bagging_fraction': 0.8,  # Row subsampling
    'bagging_freq': 5,
    'min_child_samples': 20,  # Prevent overfitting
    'reg_alpha': 0.0,  # L1 regularization
    'reg_lambda': 0.0,  # L2 regularization
    'verbose': -1,
    'n_jobs': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    early_stopping_rounds=50,
    verbose_eval=50
)
```

### 4.3 Alternative Model: CatBoost
**Use if**: Categorical features (aircraft_type, origin/dest airports) are highly predictive

```python
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function='RMSE',
    early_stopping_rounds=50,
    verbose=50,
    task_type='CPU',
    random_seed=42
)

cat_features = ['aircraft_type', 'origin_icao', 'destination_icao', 'flight_phase']
cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)
```

### 4.4 Training Monitoring
- Track validation RMSE per aircraft type (identify weak spots)
- Plot learning curves (train vs val loss)
- Monitor feature importance (ensure no data leakage)
- Track training time and memory usage

### 4.5 Cross-Validation (If Needed)
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kfold.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = train_model(X_tr, y_tr, X_val, y_val)
    val_rmse = evaluate_model(model, X_val, y_val)
    cv_scores.append(val_rmse)

print(f"CV RMSE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

---

## Phase 5: Optimization (Days 11-13)

### 5.1 Feature Selection
**Goal**: Reduce dimensionality, improve generalization, speed up inference

**Method 1: SHAP Values**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

# Keep top 80% of cumulative importance
cumsum = importance_df['importance'].cumsum() / importance_df['importance'].sum()
selected_features = importance_df[cumsum <= 0.95]['feature'].tolist()
```

**Method 2: Native Feature Importance**
```python
importance = model.feature_importance(importance_type='gain')
selected_features = [f for f, imp in zip(feature_names, importance) if imp > threshold]
```

**Retrain** with selected features and validate performance doesn't degrade.

### 5.2 Hyperparameter Tuning
**Tool**: Optuna (Bayesian optimization)

**Budget**: 50-100 trials (laptop constraint)

```python
import optuna

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'verbose': -1
    }

    # Use 2-fold CV for speed
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=500,
        nfold=2,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    return cv_results['valid rmse-mean'][-1]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=7200)  # 2 hours max

print(f"Best RMSE: {study.best_value}")
print(f"Best params: {study.best_params}")
```

**Focus Parameters**:
- `num_leaves`: Tree complexity (higher = more complex)
- `learning_rate`: Step size (lower = more robust but slower)
- `min_child_samples`: Minimum samples per leaf (regularization)
- `reg_alpha`, `reg_lambda`: L1/L2 regularization

### 5.3 Ensemble Strategy
**Simple Weighted Average**:
```python
# Train multiple models
lgb_pred = lgb_model.predict(X_val)
cat_pred = cat_model.predict(X_val)
rf_pred = rf_model.predict(X_val)

# Grid search weights
best_rmse = float('inf')
best_weights = None

for w1 in np.linspace(0.3, 0.7, 5):
    for w2 in np.linspace(0.2, 0.5, 4):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue

        ensemble_pred = w1 * lgb_pred + w2 * cat_pred + w3 * rf_pred
        rmse = mean_squared_error(y_val, ensemble_pred, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)

print(f"Ensemble RMSE: {best_rmse}, Weights: {best_weights}")
```

**Validation**: Ensure ensemble doesn't overfit on validation set.

---

## Phase 6: Error Analysis & Refinement (Days 14-15)

### 6.1 Residual Analysis
```python
predictions = model.predict(X_val)
residuals = y_val - predictions

# Scatter plot: actual vs predicted
plt.scatter(y_val, predictions, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Fuel (kg)')
plt.ylabel('Predicted Fuel (kg)')
plt.title('Predictions vs Actuals')
plt.show()

# Residual distribution
plt.hist(residuals, bins=50)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()
```

### 6.2 Error Segmentation
**Identify High-Error Segments**:
```python
# Add predictions and residuals to validation set
val_df['prediction'] = predictions
val_df['residual'] = residuals
val_df['abs_residual'] = np.abs(residuals)

# Analyze by aircraft type
error_by_aircraft = val_df.groupby('aircraft_type')['abs_residual'].mean().sort_values(ascending=False)
print("High-error aircraft types:")
print(error_by_aircraft.head(10))

# Analyze by flight phase
error_by_phase = val_df.groupby('flight_phase')['abs_residual'].mean().sort_values(ascending=False)
print("\nHigh-error flight phases:")
print(error_by_phase)

# Analyze by interval position
error_by_position = val_df.groupby(pd.cut(val_df['interval_position_pct'], bins=10))['abs_residual'].mean()
print("\nError by interval position:")
print(error_by_position)
```

### 6.3 Targeted Improvements
**Based on error analysis**:
- If specific aircraft types have high error: Add aircraft-specific features or train separate models
- If takeoff/landing have high error: Add more granular phase detection or altitude-based features
- If early/late intervals have high error: Add cumulative fuel features or trajectory history

### 6.4 Validation on August Data
**Final Check**:
- Train on April-July, validate on August
- RMSE should be within 5-10% of cross-validation scores
- If validation RMSE >> CV RMSE: Model is overfitting, increase regularization

---

## Phase 7: Production Pipeline (Days 16-17)

### 7.1 Feature Extraction for Ranking Data
```python
# Process 1,888 ranking flights
ranking_flight_ids = flightlist_rank['flight_id'].unique()

ranking_features = []
for i in range(0, len(ranking_flight_ids), BATCH_SIZE):
    batch = ranking_flight_ids[i:i+BATCH_SIZE]
    features = process_trajectory_batch(batch, data_dir='flights_rank/')
    ranking_features.append(features)

X_rank = pl.concat(ranking_features)
```

### 7.2 Prediction Generation
```python
# Load submission template
submission = pl.read_parquet('fuel_rank_submission.parquet')

# Ensure feature alignment
X_rank = X_rank[selected_features]  # Use same features as training

# Generate predictions
predictions = model.predict(X_rank)

# Validate predictions are reasonable
assert len(predictions) == 24289, "Prediction count mismatch"
assert predictions.min() > 0, "Negative fuel predictions detected"
assert predictions.max() < 50000, "Unrealistic fuel predictions detected"
```

### 7.3 Submission File Creation
```python
# Replace fuel_kg column
submission = submission.with_columns(
    pl.Series('fuel_kg', predictions)
)

# Final validation
assert submission.shape == (24289, 5), "Submission shape mismatch"
assert submission['idx'].to_list() == list(range(24289)), "idx column altered"
assert submission['fuel_kg'].null_count() == 0, "Null predictions present"

# Save submission
team_name = "gentle-orange"
version = 1
submission.write_parquet(f'{team_name}_v{version}.parquet')

print(f"Submission saved: {team_name}_v{version}.parquet")
print(f"Predicted fuel range: {predictions.min():.2f} - {predictions.max():.2f} kg")
print(f"Predicted fuel mean: {predictions.mean():.2f} kg")
```

### 7.4 Submission Validation
```python
# Load and verify
check = pl.read_parquet(f'{team_name}_v{version}.parquet')

# Match original template exactly
template = pl.read_parquet('fuel_rank_submission.parquet')

assert check['idx'].to_list() == template['idx'].to_list(), "idx mismatch"
assert check['flight_id'].to_list() == template['flight_id'].to_list(), "flight_id mismatch"
assert check['start'].to_list() == template['start'].to_list(), "start mismatch"
assert check['end'].to_list() == template['end'].to_list(), "end mismatch"

print("✅ Submission validation passed")
```

### 7.5 Documentation
Create `README.md` with:
1. **Model Description**: LightGBM regression with X features
2. **Feature Engineering**: List of key features and rationale
3. **Training Procedure**: Data split, hyperparameters, validation strategy
4. **Performance**: Validation RMSE, cross-validation scores
5. **External Data**: None used (or document if any open data was added)
6. **Reproducibility**: Random seeds, library versions, training script

---

## Technical Implementation Guide

### Memory Management
```python
# Efficient parquet reading with Polars
import polars as pl

# Read only necessary columns
df = pl.read_parquet('fuel_train.parquet', columns=['idx', 'flight_id', 'start', 'end', 'fuel_kg'])

# Process in chunks to avoid memory overflow
CHUNK_SIZE = 100
for i in range(0, len(flight_ids), CHUNK_SIZE):
    chunk_features = process_chunk(flight_ids[i:i+CHUNK_SIZE])
    chunk_features.write_parquet(f'cache/features_{i}.parquet')
    del chunk_features  # Explicit memory cleanup

# Load cached features
all_features = pl.read_parquet('cache/features_*.parquet')
```

### Efficient Feature Extraction
```python
def extract_interval_features(trajectory_df, start_time, end_time):
    """Extract features from trajectory data for a single fuel interval."""

    # Filter trajectory to interval window
    mask = (trajectory_df['timestamp'] >= start_time) & (trajectory_df['timestamp'] <= end_time)
    interval_traj = trajectory_df.filter(mask)

    # Handle empty intervals (trajectory gap)
    if len(interval_traj) == 0:
        return default_features_with_missing_flag()

    features = {}

    # Interval metadata
    features['interval_duration'] = (end_time - start_time).total_seconds()
    features['point_count'] = len(interval_traj)

    # Altitude features
    alt = interval_traj['altitude']
    features['altitude_mean'] = alt.mean()
    features['altitude_std'] = alt.std()
    features['altitude_min'] = alt.min()
    features['altitude_max'] = alt.max()
    features['altitude_delta'] = alt[-1] - alt[0]

    # Speed features (handle missing values)
    if interval_traj['groundspeed'].null_count() < len(interval_traj):
        speed = interval_traj['groundspeed'].drop_nulls()
        features['groundspeed_mean'] = speed.mean()
        features['groundspeed_max'] = speed.max()
        features['groundspeed_std'] = speed.std()
    else:
        features['groundspeed_mean'] = None
        features['groundspeed_max'] = None
        features['groundspeed_std'] = None

    # Vertical rate features
    if interval_traj['vertical_rate'].null_count() < len(interval_traj):
        vr = interval_traj['vertical_rate'].drop_nulls()
        features['vertical_rate_mean'] = vr.mean()
        features['climb_pct'] = (vr > 100).sum() / len(vr)
        features['descent_pct'] = (vr < -100).sum() / len(vr)
    else:
        features['vertical_rate_mean'] = None
        features['climb_pct'] = None
        features['descent_pct'] = None

    # Distance covered
    if len(interval_traj) >= 2:
        lat1, lon1 = interval_traj['latitude'][0], interval_traj['longitude'][0]
        lat2, lon2 = interval_traj['latitude'][-1], interval_traj['longitude'][-1]
        features['distance_km'] = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        features['distance_km'] = 0

    return features
```

### Model Training Pipeline
```python
def train_final_model(X_train, y_train, X_val, y_val, params):
    """Train LightGBM model with early stopping."""

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = mean_squared_error(y_train, train_pred, squared=False)
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)

    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Best iteration: {model.best_iteration}")

    return model
```

---

## Expected Performance Targets

### Model Performance
- **Target Validation RMSE**: < 400 kg
  - Baseline (simple features + Ridge): ~600-800 kg
  - LightGBM (basic features): ~450-550 kg
  - LightGBM (optimized features + tuning): ~350-450 kg
  - Ensemble (if implemented): ~300-400 kg

### Computational Efficiency
- **Feature extraction**: 1-2 hours for 11,037 flights (with caching)
- **Model training**: 15-30 minutes per model (LightGBM)
- **Hyperparameter tuning**: 1-2 hours (100 trials)
- **Inference**: < 5 minutes for 1,888 flights

### Memory Usage
- **Peak RAM**: < 8 GB (with batch processing)
- **Disk cache**: ~2-3 GB for processed features

---

## Risk Mitigation Strategies

### 1. Trajectory Data Quality Issues
**Risk**: Gaps in ADS-B data, missing intervals
**Mitigation**:
- Robust aggregation functions that handle sparse data
- Missingness indicators as features
- Forward-fill/interpolation for small gaps
- Flag low-quality intervals (< 3 data points)

### 2. Overfitting
**Risk**: Model memorizes training data, poor generalization
**Mitigation**:
- Temporal validation split (April-July train, August validation)
- Conservative hyperparameters (high `min_child_samples`, regularization)
- Early stopping based on validation loss
- Feature selection to reduce complexity
- Monitor train vs validation RMSE gap

### 3. Memory Constraints
**Risk**: Cannot load all trajectory data simultaneously
**Mitigation**:
- Batch processing (100 flights at a time)
- Cache processed features to disk
- Use Polars for efficient parquet I/O
- Lazy loading with `.scan_parquet()`

### 4. Time Constraints (Laptop Speed)
**Risk**: Training takes too long
**Mitigation**:
- Focus on LightGBM (fastest gradient boosting)
- Limit hyperparameter search budget (50-100 trials)
- Use 3-fold CV instead of 10-fold
- Skip deep learning models (too slow for laptop)

### 5. Poor Generalization to September Data
**Risk**: Model performs well on validation but poorly on ranking data
**Mitigation**:
- Ensure temporal split mirrors train→rank transition
- Validate on August data specifically
- Monitor aircraft type distribution (ensure all types seen in training)
- Avoid overfitting to specific routes/airports

---

## Success Criteria

### Model Quality
- ✅ Validation RMSE < 400 kg (< 72% of fuel std)
- ✅ Cross-validation std < 50 kg (stable across folds)
- ✅ Feature importance aligns with aviation physics (altitude, speed, phase)
- ✅ No data leakage detected (future information in features)

### Generalization
- ✅ August validation RMSE within 10% of cross-validation RMSE
- ✅ Residuals are normally distributed (no systematic bias)
- ✅ Error rates consistent across aircraft types (< 2x difference)

### Efficiency
- ✅ Total training time < 4 hours on laptop
- ✅ Inference time < 10 minutes for full ranking dataset
- ✅ Peak memory usage < 8 GB

### Deliverables
- ✅ Submission file: `<team-name>_v1.parquet` with exact format match
- ✅ Documentation: README with model description, features, training procedure
- ✅ Code: Reproducible training script with random seeds
- ✅ Validation report: Performance metrics, error analysis

---

## Tools & Libraries

### Required
```bash
pip install polars pandas numpy scipy
pip install scikit-learn lightgbm catboost
pip install optuna shap
pip install matplotlib seaborn plotly
```

### Optional (for advanced features)
```bash
pip install geopandas pyproj  # Geospatial calculations
pip install scikit-mobility  # Trajectory analysis
pip install mlflow  # Experiment tracking
```

---

## Timeline Overview

| Phase | Days | Key Deliverable |
|-------|------|----------------|
| 1. Data Understanding | 1-2 | EDA report, data quality assessment |
| 2. Feature Engineering | 3-5 | Feature pipeline, cached features |
| 3. Data Preparation | 6 | Train/val split, cleaned dataset |
| 4. Model Development | 7-10 | Baseline + LightGBM models trained |
| 5. Optimization | 11-13 | Tuned hyperparameters, feature selection |
| 6. Error Analysis | 14-15 | Refinement based on error patterns |
| 7. Production Pipeline | 16-17 | Submission file + documentation |

---

## Next Steps

1. **Start Phase 1**: Load data, perform EDA, understand fuel patterns
2. **Create project structure**:
   ```
   /Competitions
   ├── data/              # Raw parquet files
   ├── cache/             # Processed features
   ├── models/            # Trained models
   ├── notebooks/         # EDA and experiments
   ├── src/               # Production code
   │   ├── features.py    # Feature extraction
   │   ├── train.py       # Model training
   │   ├── predict.py     # Inference pipeline
   │   └── utils.py       # Helper functions
   ├── submissions/       # Submission files
   └── README.md          # Documentation
   ```
3. **Initialize git repository** and commit initial structure
4. **Run first EDA notebook** to validate data loading and understand patterns

---

**Let's build the ultimate fuel prediction model! 🚀**
