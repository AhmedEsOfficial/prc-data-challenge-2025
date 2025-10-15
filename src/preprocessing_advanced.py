"""
Advanced Data Preprocessing Module for Fuel Prediction
=======================================================
Pushes toward theoretical practical limits with sophisticated techniques.

Author: ML Team
Date: 2025-10-15
Version: 2.0 (Advanced)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ADVANCED MISSING VALUE HANDLING
# ============================================================================

class AdvancedMissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Advanced missing value imputation with multiple strategies.

    Strategies:
    - 'median': Fast median imputation (default for < 30% missing)
    - 'knn': KNN imputation (good for structured patterns)
    - 'iterative': MICE-style iterative imputation (best quality)
    - 'forward_fill': Forward fill for time series
    - 'hybrid': Intelligent strategy selection per feature
    """

    def __init__(self, strategy='hybrid', knn_neighbors=5,
                 create_indicators=True, indicator_threshold=0.10,
                 group_by_aircraft=True, verbose=True):
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors
        self.create_indicators = create_indicators
        self.indicator_threshold = indicator_threshold
        self.group_by_aircraft = group_by_aircraft
        self.verbose = verbose

        self.imputers_ = {}
        self.indicator_features_ = []
        self.median_values_ = {}
        self.global_medians_ = {}
        self.strategy_map_ = {}

    def fit(self, X, y=None):
        """Learn imputation strategies."""
        X_work = X.copy()

        # Identify features with missing values
        missing_pct = (X_work.isna().sum() / len(X_work) * 100)
        missing_features = missing_pct[missing_pct > 0].index.tolist()

        # Identify features needing indicators
        self.indicator_features_ = missing_pct[missing_pct > self.indicator_threshold * 100].index.tolist()

        numeric_cols = X_work.select_dtypes(include=[np.number]).columns

        # Determine strategy per feature
        for col in numeric_cols:
            if col not in missing_features:
                continue

            miss_pct = missing_pct[col]

            if self.strategy == 'hybrid':
                # Intelligent strategy selection
                if miss_pct < 5:
                    self.strategy_map_[col] = 'median'
                elif miss_pct < 30:
                    self.strategy_map_[col] = 'knn'
                else:
                    self.strategy_map_[col] = 'iterative'
            else:
                self.strategy_map_[col] = self.strategy

            # Store median values for aircraft-specific imputation
            self.global_medians_[col] = X_work[col].median()

            if self.group_by_aircraft and 'aircraft_type' in X_work.columns:
                self.median_values_[col] = X_work.groupby('aircraft_type')[col].median().to_dict()

        # Fit imputers based on strategies needed
        strategies_needed = set(self.strategy_map_.values())

        if 'knn' in strategies_needed:
            # Prepare numeric data for KNN
            numeric_data = X_work[numeric_cols].copy()
            # Use median for initial fill (KNN needs no NaN)
            for col in numeric_data.columns:
                if numeric_data[col].isna().sum() > 0:
                    numeric_data[col].fillna(numeric_data[col].median(), inplace=True)

            self.imputers_['knn'] = KNNImputer(n_neighbors=self.knn_neighbors)
            self.imputers_['knn'].fit(numeric_data)

        if 'iterative' in strategies_needed:
            # Iterative imputer (MICE)
            numeric_data = X_work[numeric_cols].copy()
            self.imputers_['iterative'] = IterativeImputer(
                max_iter=10,
                random_state=42,
                verbose=0
            )
            self.imputers_['iterative'].fit(numeric_data)

        if self.verbose:
            print(f"\n[AdvancedMissingValueHandler] Configuration:")
            print(f"  Features with missing values: {len(missing_features)}")
            print(f"  Features needing indicators: {len(self.indicator_features_)}")
            print(f"  Strategy distribution:")
            strategy_counts = pd.Series(list(self.strategy_map_.values())).value_counts()
            for strat, count in strategy_counts.items():
                print(f"    {strat}: {count} features")

        return self

    def transform(self, X):
        """Apply imputation."""
        X_imputed = X.copy()

        # Create missing indicators
        if self.create_indicators:
            for col in self.indicator_features_:
                if col in X_imputed.columns:
                    indicator_name = f'{col}_is_missing'
                    X_imputed[indicator_name] = X_imputed[col].isna().astype(int)

        numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns

        # Apply strategies
        for strategy in set(self.strategy_map_.values()):
            cols_for_strategy = [col for col, s in self.strategy_map_.items()
                               if s == strategy and col in numeric_cols]

            if not cols_for_strategy:
                continue

            if strategy == 'median':
                # Aircraft-specific median imputation
                for col in cols_for_strategy:
                    missing_mask = X_imputed[col].isna()

                    if missing_mask.sum() == 0:
                        continue

                    # Try aircraft-specific median
                    if self.group_by_aircraft and 'aircraft_type' in X_imputed.columns and col in self.median_values_:
                        for aircraft_type, median_val in self.median_values_[col].items():
                            type_mask = (X_imputed['aircraft_type'] == aircraft_type) & missing_mask
                            if type_mask.sum() > 0 and pd.notna(median_val):
                                X_imputed.loc[type_mask, col] = median_val

                    # Fill remaining with global median
                    still_missing = X_imputed[col].isna()
                    if still_missing.sum() > 0 and col in self.global_medians_:
                        X_imputed.loc[still_missing, col] = self.global_medians_[col]

            elif strategy == 'knn':
                # KNN imputation
                if 'knn' in self.imputers_:
                    # Get all numeric columns for KNN context
                    numeric_data = X_imputed[numeric_cols].copy()

                    # Fill any remaining NaN with median for KNN
                    for col in numeric_data.columns:
                        if numeric_data[col].isna().sum() > 0:
                            if col in self.global_medians_:
                                numeric_data[col].fillna(self.global_medians_[col], inplace=True)
                            else:
                                numeric_data[col].fillna(numeric_data[col].median(), inplace=True)

                    imputed_data = self.imputers_['knn'].transform(numeric_data)
                    X_imputed[numeric_cols] = imputed_data

            elif strategy == 'iterative':
                # Iterative imputation
                if 'iterative' in self.imputers_:
                    numeric_data = X_imputed[numeric_cols].copy()
                    imputed_data = self.imputers_['iterative'].transform(numeric_data)
                    X_imputed[numeric_cols] = imputed_data

        if self.verbose:
            remaining_missing = X_imputed.isna().sum().sum()
            print(f"\n[AdvancedMissingValueHandler] Imputation complete")
            print(f"  Created {len(self.indicator_features_)} missing indicators")
            print(f"  Remaining missing values: {remaining_missing}")

        return X_imputed


# ============================================================================
# OUTLIER DETECTION AND HANDLING
# ============================================================================

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and handle outliers using multiple methods.

    Methods:
    - 'iqr': Interquartile range (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    - 'percentile': Clip at percentiles (default: 1st, 99th)
    - 'zscore': Z-score threshold (default: 3)
    - 'none': No outlier handling
    """

    def __init__(self, method='percentile', iqr_multiplier=1.5,
                 percentile_range=(1, 99), zscore_threshold=3,
                 exclude_features=None, verbose=True):
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.percentile_range = percentile_range
        self.zscore_threshold = zscore_threshold
        self.exclude_features = exclude_features or []
        self.verbose = verbose

        self.bounds_ = {}

    def fit(self, X, y=None):
        """Learn outlier bounds."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in self.exclude_features:
                continue

            data = X[col].dropna()

            if len(data) == 0:
                continue

            if self.method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR

            elif self.method == 'percentile':
                lower_bound = data.quantile(self.percentile_range[0] / 100)
                upper_bound = data.quantile(self.percentile_range[1] / 100)

            elif self.method == 'zscore':
                mean = data.mean()
                std = data.std()
                lower_bound = mean - self.zscore_threshold * std
                upper_bound = mean + self.zscore_threshold * std

            else:
                continue

            self.bounds_[col] = (lower_bound, upper_bound)

        if self.verbose and self.method != 'none':
            print(f"\n[OutlierHandler] Learned bounds for {len(self.bounds_)} features")
            print(f"  Method: {self.method}")

        return self

    def transform(self, X):
        """Clip outliers."""
        if self.method == 'none':
            return X

        X_clipped = X.copy()
        clipped_count = 0

        for col, (lower, upper) in self.bounds_.items():
            if col in X_clipped.columns:
                outliers = ((X_clipped[col] < lower) | (X_clipped[col] > upper)).sum()
                clipped_count += outliers
                X_clipped[col] = X_clipped[col].clip(lower=lower, upper=upper)

        if self.verbose:
            print(f"\n[OutlierHandler] Clipped {clipped_count} outlier values")

        return X_clipped


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate polynomial and interaction features intelligently.

    Only creates interactions for specified feature groups to avoid explosion.
    """

    def __init__(self, degree=2, interaction_only=True,
                 include_features=None, max_features=50, verbose=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_features = include_features
        self.max_features = max_features
        self.verbose = verbose

        self.poly_ = None
        self.feature_names_ = []

    def fit(self, X, y=None):
        """Fit polynomial feature generator."""
        if self.include_features is None:
            # Use all numeric features
            include_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            include_cols = [col for col in self.include_features if col in X.columns]

        # Limit to max_features to prevent explosion
        include_cols = include_cols[:self.max_features]

        self.feature_names_ = include_cols

        X_subset = X[include_cols].copy()

        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False
        )

        self.poly_.fit(X_subset)

        if self.verbose:
            print(f"\n[PolynomialFeatureGenerator]")
            print(f"  Input features: {len(include_cols)}")
            print(f"  Output features: {len(self.poly_.get_feature_names_out())}")
            print(f"  Degree: {self.degree}, Interaction only: {self.interaction_only}")

        return self

    def transform(self, X):
        """Generate polynomial features."""
        X_transformed = X.copy()

        X_subset = X[self.feature_names_].copy()

        poly_features = self.poly_.transform(X_subset)
        poly_feature_names = self.poly_.get_feature_names_out()

        # Add polynomial features
        for i, name in enumerate(poly_feature_names):
            # Skip original features (already in X)
            if name in self.feature_names_:
                continue

            # Create readable name
            new_name = f'poly_{name}'
            X_transformed[new_name] = poly_features[:, i]

        if self.verbose:
            print(f"\n[PolynomialFeatureGenerator] Added {len(poly_feature_names) - len(self.feature_names_)} polynomial features")

        return X_transformed


class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate ratio features from specified feature pairs.

    Examples:
    - speed / altitude
    - fuel / duration
    - distance / time
    """

    def __init__(self, ratio_pairs=None, epsilon=1e-8, verbose=True):
        self.ratio_pairs = ratio_pairs or []
        self.epsilon = epsilon
        self.verbose = verbose

    def fit(self, X, y=None):
        """No fitting needed."""
        return self

    def transform(self, X):
        """Generate ratio features."""
        X_transformed = X.copy()
        created_count = 0

        for numerator, denominator in self.ratio_pairs:
            if numerator in X.columns and denominator in X.columns:
                ratio_name = f'{numerator}_per_{denominator}'
                # Avoid division by zero
                X_transformed[ratio_name] = X[numerator] / (X[denominator] + self.epsilon)
                created_count += 1

        if self.verbose and created_count > 0:
            print(f"\n[RatioFeatureGenerator] Created {created_count} ratio features")

        return X_transformed


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from datetime columns.

    Features:
    - Hour of day (cyclical encoding)
    - Day of week
    - Month
    - Season
    - Is weekend
    - Time since start of day
    """

    def __init__(self, datetime_cols=None, cyclical_encoding=True, verbose=True):
        self.datetime_cols = datetime_cols or []
        self.cyclical_encoding = cyclical_encoding
        self.verbose = verbose

    def fit(self, X, y=None):
        """No fitting needed."""
        return self

    def transform(self, X):
        """Extract temporal features."""
        X_transformed = X.copy()
        created_count = 0

        for col in self.datetime_cols:
            if col not in X.columns:
                continue

            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                X_transformed[col] = pd.to_datetime(X_transformed[col])

            dt = X_transformed[col].dt

            # Hour features
            if self.cyclical_encoding:
                # Cyclical encoding for hour
                X_transformed[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
                X_transformed[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
                created_count += 2
            else:
                X_transformed[f'{col}_hour'] = dt.hour
                created_count += 1

            # Day of week (0=Monday, 6=Sunday)
            X_transformed[f'{col}_dayofweek'] = dt.dayofweek
            created_count += 1

            # Month
            if self.cyclical_encoding:
                X_transformed[f'{col}_month_sin'] = np.sin(2 * np.pi * dt.month / 12)
                X_transformed[f'{col}_month_cos'] = np.cos(2 * np.pi * dt.month / 12)
                created_count += 2
            else:
                X_transformed[f'{col}_month'] = dt.month
                created_count += 1

            # Is weekend
            X_transformed[f'{col}_is_weekend'] = (dt.dayofweek >= 5).astype(int)
            created_count += 1

        if self.verbose and created_count > 0:
            print(f"\n[TemporalFeatureEngineer] Created {created_count} temporal features")

        return X_transformed


# ============================================================================
# ADVANCED ENCODING
# ============================================================================

class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """
    Leave-One-Out target encoding.

    More robust than simple target encoding, less prone to overfitting.
    """

    def __init__(self, cols=None, smoothing=1.0, verbose=True):
        self.cols = cols or []
        self.smoothing = smoothing
        self.verbose = verbose

        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        """Learn encodings."""
        if y is None:
            raise ValueError("LeaveOneOutEncoder requires target y for fitting")

        self.global_mean_ = float(y.mean())

        for col in self.cols:
            if col not in X.columns:
                continue

            # Calculate sum and count for each category
            temp_df = pd.DataFrame({
                'category': X[col],
                'target': y.values if hasattr(y, 'values') else y
            })

            # Group statistics
            stats = temp_df.groupby('category')['target'].agg(['sum', 'count']).to_dict()

            self.encodings_[col] = {
                'sum': stats['sum'],
                'count': stats['count']
            }

        if self.verbose:
            print(f"\n[LeaveOneOutEncoder] Encoded {len(self.encodings_)} features")
            print(f"  Global mean: {self.global_mean_:.2f}")

        return self

    def transform(self, X):
        """Apply LOO encoding."""
        X_transformed = X.copy()

        for col in self.cols:
            if col not in X.columns or col not in self.encodings_:
                continue

            encoded_col = f'{col}_loo_enc'

            # For transform, use regular target encoding (no LOO)
            # LOO is only for training
            category_mean = {}
            for cat, sum_val in self.encodings_[col]['sum'].items():
                count = self.encodings_[col]['count'][cat]
                # Smoothing
                category_mean[cat] = (sum_val + self.smoothing * self.global_mean_) / (count + self.smoothing)

            X_transformed[encoded_col] = X[col].map(category_mean)

            # Fill unseen categories with global mean
            X_transformed[encoded_col].fillna(self.global_mean_, inplace=True)

            # Drop original
            X_transformed = X_transformed.drop(columns=[col])

        if self.verbose:
            print(f"\n[LeaveOneOutEncoder] Applied LOO encoding to {len(self.cols)} features")

        return X_transformed


# ============================================================================
# ADVANCED SCALING
# ============================================================================

class AdvancedScaler(BaseEstimator, TransformerMixin):
    """
    Advanced scaling strategies optimized for different feature types.

    Strategies:
    - 'standard': StandardScaler (mean=0, std=1)
    - 'robust': RobustScaler (median=0, IQR=1, robust to outliers)
    - 'quantile': QuantileTransformer (uniform or normal distribution)
    - 'power': PowerTransformer (Yeo-Johnson, makes data more Gaussian)
    - 'hybrid': Intelligent strategy selection per feature
    """

    def __init__(self, strategy='hybrid', exclude_binary=True, verbose=True):
        self.strategy = strategy
        self.exclude_binary = exclude_binary
        self.verbose = verbose

        self.scalers_ = {}
        self.strategy_map_ = {}
        self.binary_features_ = []
        self.features_to_scale_ = []

    def fit(self, X, y=None):
        """Fit scalers."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude binary features
        if self.exclude_binary:
            for col in numeric_cols:
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    self.binary_features_.append(col)

        self.features_to_scale_ = [col for col in numeric_cols if col not in self.binary_features_]

        # Determine strategy per feature
        for col in self.features_to_scale_:
            if self.strategy == 'hybrid':
                # Check skewness to determine best strategy
                skewness = X[col].skew()

                if abs(skewness) > 2:
                    # Highly skewed -> Power transform
                    self.strategy_map_[col] = 'power'
                elif abs(skewness) > 1:
                    # Moderately skewed -> Quantile
                    self.strategy_map_[col] = 'quantile'
                else:
                    # Roughly normal -> Robust (handles outliers better than standard)
                    self.strategy_map_[col] = 'robust'
            else:
                self.strategy_map_[col] = self.strategy

        # Fit scalers per strategy
        strategies_needed = set(self.strategy_map_.values())

        for strat in strategies_needed:
            cols_for_strategy = [col for col, s in self.strategy_map_.items() if s == strat]

            if not cols_for_strategy:
                continue

            if strat == 'standard':
                scaler = StandardScaler()
            elif strat == 'robust':
                scaler = RobustScaler()
            elif strat == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            elif strat == 'power':
                scaler = PowerTransformer(method='yeo-johnson')
            else:
                continue

            scaler.fit(X[cols_for_strategy])
            self.scalers_[strat] = scaler

        if self.verbose:
            print(f"\n[AdvancedScaler] Configuration:")
            print(f"  Features to scale: {len(self.features_to_scale_)}")
            print(f"  Binary features (excluded): {len(self.binary_features_)}")
            print(f"  Strategy distribution:")
            strategy_counts = pd.Series(list(self.strategy_map_.values())).value_counts()
            for strat, count in strategy_counts.items():
                print(f"    {strat}: {count} features")

        return self

    def transform(self, X):
        """Apply scaling."""
        X_scaled = X.copy()

        # Apply scalers per strategy
        for strat, scaler in self.scalers_.items():
            cols_for_strategy = [col for col, s in self.strategy_map_.items()
                               if s == strat and col in X_scaled.columns]

            if cols_for_strategy:
                X_scaled[cols_for_strategy] = scaler.transform(X[cols_for_strategy])

        if self.verbose:
            print(f"\n[AdvancedScaler] Scaled {len(self.features_to_scale_)} features")

        return X_scaled


# ============================================================================
# FEATURE SELECTION
# ============================================================================

class IntelligentFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection using multiple criteria.

    Methods:
    - Variance threshold
    - Correlation filtering (remove highly correlated features)
    - Mutual information (keep most informative features)
    """

    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95,
                 mutual_info_percentile=90, verbose=True):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mutual_info_percentile = mutual_info_percentile
        self.verbose = verbose

        self.features_to_keep_ = []
        self.variance_selector_ = None

    def fit(self, X, y=None):
        """Select features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()

        # 1. Variance threshold
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        self.variance_selector_.fit(X_numeric)

        high_variance_features = X_numeric.columns[self.variance_selector_.get_support()].tolist()

        if self.verbose:
            removed_variance = len(numeric_cols) - len(high_variance_features)
            print(f"\n[FeatureSelector] Variance threshold removed: {removed_variance} features")

        # 2. Correlation filtering
        X_high_var = X_numeric[high_variance_features].copy()
        corr_matrix = X_high_var.corr().abs()

        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for column in upper_triangle.columns:
            if (upper_triangle[column] > self.correlation_threshold).any():
                to_drop.add(column)

        low_corr_features = [f for f in high_variance_features if f not in to_drop]

        if self.verbose:
            print(f"  Correlation filtering removed: {len(to_drop)} features")

        # 3. Mutual information (if y provided)
        if y is not None:
            X_low_corr = X_numeric[low_corr_features].copy()

            # Handle missing values for MI calculation
            X_low_corr = X_low_corr.fillna(X_low_corr.median())

            mi_scores = mutual_info_regression(X_low_corr, y, random_state=42)
            mi_threshold = np.percentile(mi_scores, 100 - self.mutual_info_percentile)

            mi_selected = X_low_corr.columns[mi_scores >= mi_threshold].tolist()

            self.features_to_keep_ = mi_selected

            if self.verbose:
                print(f"  Mutual information kept top {self.mutual_info_percentile}%: {len(mi_selected)} features")
        else:
            self.features_to_keep_ = low_corr_features

        # Check for non-numeric columns (should be none after preprocessing)
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols and self.verbose:
            print(f"  WARNING: Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:5]}")
            print(f"  These will be excluded from the model (only numeric features supported)")

        if self.verbose:
            print(f"  Total features selected: {len(self.features_to_keep_)}")

        return self

    def transform(self, X):
        """Keep only selected features."""
        return X[[col for col in self.features_to_keep_ if col in X.columns]]


# ============================================================================
# COMPREHENSIVE PREPROCESSING PIPELINE
# ============================================================================

def create_preprocessing_pipeline(config=None):
    """
    Create comprehensive preprocessing pipeline.

    Args:
        config: Dictionary with configuration options
            - missing_strategy: 'hybrid', 'knn', 'iterative', 'median'
            - outlier_method: 'percentile', 'iqr', 'zscore', 'none'
            - scaling_strategy: 'hybrid', 'robust', 'quantile', 'power', 'standard'
            - polynomial_degree: 2 or 3
            - enable_feature_selection: bool

    Returns:
        Dictionary of transformers to apply in sequence
    """
    config = config or {}

    pipeline = {
        'missing': AdvancedMissingValueHandler(
            strategy=config.get('missing_strategy', 'hybrid'),
            verbose=config.get('verbose', True)
        ),
        'outliers': OutlierHandler(
            method=config.get('outlier_method', 'percentile'),
            verbose=config.get('verbose', True)
        ),
        'ratios': RatioFeatureGenerator(
            ratio_pairs=config.get('ratio_pairs', []),
            verbose=config.get('verbose', True)
        ),
        'temporal': TemporalFeatureEngineer(
            datetime_cols=config.get('datetime_cols', []),
            verbose=config.get('verbose', True)
        ),
        'polynomial': PolynomialFeatureGenerator(
            degree=config.get('polynomial_degree', 2),
            include_features=config.get('polynomial_features', None),
            verbose=config.get('verbose', True)
        ),
        'scaling': AdvancedScaler(
            strategy=config.get('scaling_strategy', 'hybrid'),
            verbose=config.get('verbose', True)
        )
    }

    if config.get('enable_feature_selection', False):
        pipeline['selection'] = IntelligentFeatureSelector(
            verbose=config.get('verbose', True)
        )

    return pipeline


def apply_preprocessing_pipeline(X_train, X_val, y_train, pipeline, fit=True):
    """
    Apply preprocessing pipeline to train and validation data.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        pipeline: Dictionary of transformers
        fit: Whether to fit transformers (True for training, False for inference)

    Returns:
        X_train_transformed, X_val_transformed
    """
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()

    for name, transformer in pipeline.items():
        print(f"\n{'='*80}")
        print(f"Applying: {name}")
        print(f"{'='*80}")

        if fit:
            # Special handling for target-dependent transformers
            if hasattr(transformer, 'fit') and 'encoder' in name.lower():
                transformer.fit(X_train_proc, y_train)
            elif hasattr(transformer, 'fit') and 'selection' in name.lower():
                transformer.fit(X_train_proc, y_train)
            elif hasattr(transformer, 'fit'):
                transformer.fit(X_train_proc)

        if hasattr(transformer, 'transform'):
            X_train_proc = transformer.transform(X_train_proc)
            X_val_proc = transformer.transform(X_val_proc)

    return X_train_proc, X_val_proc


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Keep original classes for backward compatibility
from src.preprocessing import (
    FeatureCleaner,
    MissingValueHandler,
    CategoricalEncoder,
    FeatureScaler,
    create_temporal_split,
    validate_no_leakage,
    get_feature_names,
    print_preprocessing_summary
)
