"""
Data Preprocessing Module for Fuel Prediction
==============================================
Handles data cleaning, missing values, encoding, and scaling.

Author: ML Team
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """
    Remove constant, duplicate, and low-value features.
    """

    def __init__(self, remove_constant=True, remove_duplicates=True, verbose=True):
        self.remove_constant = remove_constant
        self.remove_duplicates = remove_duplicates
        self.verbose = verbose
        self.constant_features_ = []
        self.duplicate_features_ = []
        self.features_to_drop_ = []

    def fit(self, X, y=None):
        """Identify features to remove."""
        self.constant_features_ = []
        self.duplicate_features_ = []

        # Identify constant features
        if self.remove_constant:
            for col in X.columns:
                if X[col].dtype in [np.number]:
                    if X[col].nunique() == 1:
                        self.constant_features_.append(col)

        # Identify duplicate features (keep one from each pair)
        if self.remove_duplicates:
            duplicate_pairs = [
                ('interval_duration_sec', 'interval_duration_min'),
                ('flight_duration_hours', 'flight_duration_min'),
            ]

            for col1, col2 in duplicate_pairs:
                if col1 in X.columns and col2 in X.columns:
                    # Keep the second one (more interpretable units)
                    self.duplicate_features_.append(col1)

        # Combine all features to drop
        self.features_to_drop_ = list(set(self.constant_features_ + self.duplicate_features_))

        if self.verbose:
            print(f"\n[FeatureCleaner] Identified features to remove:")
            print(f"  Constant features: {len(self.constant_features_)}")
            if len(self.constant_features_) > 0:
                print(f"    {self.constant_features_}")
            print(f"  Duplicate features: {len(self.duplicate_features_)}")
            if len(self.duplicate_features_) > 0:
                print(f"    {self.duplicate_features_}")
            print(f"  Total to drop: {len(self.features_to_drop_)}")

        return self

    def transform(self, X):
        """Remove identified features."""
        X_clean = X.copy()

        # Drop features
        features_to_drop_present = [f for f in self.features_to_drop_ if f in X_clean.columns]
        if len(features_to_drop_present) > 0:
            X_clean = X_clean.drop(columns=features_to_drop_present)

            if self.verbose:
                print(f"\n[FeatureCleaner] Dropped {len(features_to_drop_present)} features")

        return X_clean


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values with indicators + median imputation.

    Strategy:
    - Create binary indicators for features with >10% missing
    - Impute with median (by aircraft type for critical features)
    """

    def __init__(self, create_indicators=True, indicator_threshold=0.10,
                 group_by_aircraft=True, verbose=True):
        self.create_indicators = create_indicators
        self.indicator_threshold = indicator_threshold
        self.group_by_aircraft = group_by_aircraft
        self.verbose = verbose

        self.missing_features_ = []
        self.indicator_features_ = []
        self.median_values_ = {}
        self.global_medians_ = {}

    def fit(self, X, y=None):
        """Learn median values for imputation."""

        # Identify features with missing values
        missing_pct = (X.isna().sum() / len(X) * 100)
        self.missing_features_ = missing_pct[missing_pct > 0].index.tolist()

        # Identify features needing indicators
        self.indicator_features_ = missing_pct[missing_pct > self.indicator_threshold * 100].index.tolist()

        # Calculate medians for imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if X[col].isna().sum() > 0:
                # Global median as fallback
                self.global_medians_[col] = X[col].median()

                # Median by aircraft type if available
                if self.group_by_aircraft and 'aircraft_type' in X.columns:
                    self.median_values_[col] = X.groupby('aircraft_type')[col].median().to_dict()
                else:
                    self.median_values_[col] = {}

        if self.verbose:
            print(f"\n[MissingValueHandler] Missing value analysis:")
            print(f"  Features with missing values: {len(self.missing_features_)}")
            print(f"  Features needing indicators (>{self.indicator_threshold*100}%): {len(self.indicator_features_)}")
            if len(self.indicator_features_) > 0:
                print(f"    {self.indicator_features_[:10]}")  # Show first 10

        return self

    def transform(self, X):
        """Create indicators and impute missing values."""
        X_imputed = X.copy()

        # Create missing indicators
        if self.create_indicators:
            for col in self.indicator_features_:
                if col in X_imputed.columns:
                    indicator_name = f'{col}_is_missing'
                    X_imputed[indicator_name] = X_imputed[col].isna().astype(int)

        # Impute missing values
        for col in self.missing_features_:
            if col not in X_imputed.columns:
                continue

            missing_mask = X_imputed[col].isna()

            if missing_mask.sum() == 0:
                continue

            # Try aircraft-specific median first
            if self.group_by_aircraft and 'aircraft_type' in X_imputed.columns and col in self.median_values_:
                for aircraft_type, median_val in self.median_values_[col].items():
                    type_mask = (X_imputed['aircraft_type'] == aircraft_type) & missing_mask
                    if type_mask.sum() > 0 and pd.notna(median_val):
                        X_imputed.loc[type_mask, col] = median_val

            # Fill remaining with global median
            still_missing = X_imputed[col].isna()
            if still_missing.sum() > 0 and col in self.global_medians_:
                X_imputed.loc[still_missing, col] = self.global_medians_[col]

        if self.verbose:
            remaining_missing = X_imputed.isna().sum().sum()
            print(f"\n[MissingValueHandler] Imputation complete")
            print(f"  Created {len(self.indicator_features_)} missing indicators")
            print(f"  Remaining missing values: {remaining_missing}")

        return X_imputed


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features.

    Strategy:
    - aircraft_type: Target encoding (26 values)
    - detected_flight_phase, altitude_band, time_of_day: One-hot encoding
    - origin_icao, destination_icao: Frequency encoding (too many for one-hot)
    """

    def __init__(self, target_encode_cols=None, onehot_cols=None,
                 frequency_encode_cols=None, verbose=True):
        self.target_encode_cols = target_encode_cols or ['aircraft_type']
        self.onehot_cols = onehot_cols or ['detected_flight_phase', 'altitude_band',
                                            'time_of_day', 'flight_phase_position']
        self.frequency_encode_cols = frequency_encode_cols or ['origin_icao', 'destination_icao']
        self.verbose = verbose

        self.target_encodings_ = {}
        self.global_means_ = {}
        self.frequency_encodings_ = {}
        self.onehot_categories_ = {}

    def fit(self, X, y=None):
        """Learn encodings."""

        # Target encoding
        if y is not None:
            for col in self.target_encode_cols:
                if col in X.columns:
                    # Calculate mean target by category
                    # Create temporary dataframe with category and target
                    temp_df = pd.DataFrame({
                        'category': X[col],
                        'target': y.values if hasattr(y, 'values') else y
                    })
                    encoding_map = temp_df.groupby('category')['target'].mean().to_dict()

                    self.target_encodings_[col] = encoding_map
                    self.global_means_[col] = float(y.mean())

        # Frequency encoding
        for col in self.frequency_encode_cols:
            if col in X.columns:
                freq = X[col].value_counts(normalize=True).to_dict()
                self.frequency_encodings_[col] = freq

        # One-hot encoding - store categories
        for col in self.onehot_cols:
            if col in X.columns:
                self.onehot_categories_[col] = X[col].unique().tolist()

        if self.verbose:
            print(f"\n[CategoricalEncoder] Encoding summary:")
            print(f"  Target encoded: {len(self.target_encodings_)} features")
            print(f"  Frequency encoded: {len(self.frequency_encodings_)} features")
            print(f"  One-hot encoded: {len(self.onehot_categories_)} features")

            total_onehot_cols = sum(len(cats) for cats in self.onehot_categories_.values())
            print(f"  Total one-hot columns to create: {total_onehot_cols}")

        return self

    def transform(self, X):
        """Apply encodings."""
        X_encoded = X.copy()

        # Target encoding
        for col in self.target_encode_cols:
            if col in X_encoded.columns and col in self.target_encodings_:
                encoded_col = f'{col}_target_enc'
                X_encoded[encoded_col] = X_encoded[col].map(self.target_encodings_[col])

                # Fill unseen categories with global mean
                unseen_mask = X_encoded[encoded_col].isna()
                if unseen_mask.sum() > 0:
                    X_encoded.loc[unseen_mask, encoded_col] = self.global_means_.get(col, 0)

                # Drop original
                X_encoded = X_encoded.drop(columns=[col])

        # Frequency encoding
        for col in self.frequency_encode_cols:
            if col in X_encoded.columns and col in self.frequency_encodings_:
                encoded_col = f'{col}_freq_enc'
                X_encoded[encoded_col] = X_encoded[col].map(self.frequency_encodings_[col])

                # Fill unseen with 0
                X_encoded[encoded_col] = X_encoded[encoded_col].fillna(0)

                # Drop original
                X_encoded = X_encoded.drop(columns=[col])

        # One-hot encoding
        for col in self.onehot_cols:
            if col in X_encoded.columns:
                # Create dummies
                dummies = pd.get_dummies(X_encoded[col], prefix=col, dtype=int)

                # Ensure all expected categories are present (for consistency)
                if col in self.onehot_categories_:
                    for cat in self.onehot_categories_[col]:
                        dummy_col = f'{col}_{cat}'
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0

                # Add to dataframe
                X_encoded = pd.concat([X_encoded, dummies], axis=1)

                # Drop original
                X_encoded = X_encoded.drop(columns=[col])

        if self.verbose:
            print(f"\n[CategoricalEncoder] Encoding applied")
            print(f"  Shape after encoding: {X_encoded.shape}")

        return X_encoded


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scale numeric features using StandardScaler.

    Note: Less critical for tree-based models but important for neural networks.
    """

    def __init__(self, exclude_binary=True, verbose=True):
        self.exclude_binary = exclude_binary
        self.verbose = verbose
        self.scaler_ = StandardScaler()
        self.features_to_scale_ = []
        self.binary_features_ = []

    def fit(self, X, y=None):
        """Fit scaler on numeric features."""

        # Identify numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude binary features if requested
        if self.exclude_binary:
            for col in numeric_cols:
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    self.binary_features_.append(col)

            self.features_to_scale_ = [col for col in numeric_cols
                                        if col not in self.binary_features_]
        else:
            self.features_to_scale_ = numeric_cols

        # Fit scaler
        if len(self.features_to_scale_) > 0:
            self.scaler_.fit(X[self.features_to_scale_])

        if self.verbose:
            print(f"\n[FeatureScaler] Scaling summary:")
            print(f"  Features to scale: {len(self.features_to_scale_)}")
            print(f"  Binary features (excluded): {len(self.binary_features_)}")

        return self

    def transform(self, X):
        """Scale features."""
        X_scaled = X.copy()

        if len(self.features_to_scale_) > 0:
            X_scaled[self.features_to_scale_] = self.scaler_.transform(X[self.features_to_scale_])

            if self.verbose:
                print(f"\n[FeatureScaler] Scaled {len(self.features_to_scale_)} features")

        return X_scaled


def create_temporal_split(df, split_month=8, date_column='start_month', flight_id_col='flight_id'):
    """
    Create temporal train/validation split at FLIGHT level to prevent leakage.

    Flights are assigned to train/val based on majority of their intervals.
    This prevents the same flight from appearing in both sets.

    Args:
        df: Dataframe with features and target
        split_month: Month to use for validation (default: August = 8)
        date_column: Column containing month information
        flight_id_col: Column identifying unique flights

    Returns:
        train_df, val_df
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    if flight_id_col not in df.columns:
        raise ValueError(f"Column '{flight_id_col}' not found in dataframe")

    # Determine train/val assignment for each flight based on majority of intervals
    flight_months = df.groupby(flight_id_col)[date_column].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])

    train_flights = flight_months[flight_months < split_month].index.tolist()
    val_flights = flight_months[flight_months == split_month].index.tolist()

    # Assign ALL intervals from each flight to train or val
    train_mask = df[flight_id_col].isin(train_flights)
    val_mask = df[flight_id_col].isin(val_flights)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"\n[Temporal Split] Split by flight-level month assignment:")
    print(f"  Train flights: {len(train_flights):,} (months < {split_month})")
    print(f"  Val flights: {len(val_flights):,} (month = {split_month})")
    print(f"  Train intervals: {len(train_df):,} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"  Val intervals: {len(val_df):,} ({len(val_df) / len(df) * 100:.1f}%)")

    return train_df, val_df


def validate_no_leakage(train_df, val_df, identifier_col='flight_id'):
    """
    Validate no data leakage between train and val sets.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        identifier_col: Column to check for overlap

    Returns:
        Boolean indicating if validation passed
    """
    train_ids = set(train_df[identifier_col].unique())
    val_ids = set(val_df[identifier_col].unique())

    overlap = train_ids.intersection(val_ids)

    if len(overlap) > 0:
        print(f"\n⚠ LEAKAGE DETECTED!")
        print(f"  {len(overlap)} flights appear in both train and val")
        return False
    else:
        print(f"\n✓ No leakage detected")
        print(f"  Train: {len(train_ids)} unique flights")
        print(f"  Val: {len(val_ids)} unique flights")
        return True


def get_feature_names(X):
    """
    Get list of feature names from dataframe.

    Args:
        X: Feature dataframe

    Returns:
        List of feature names
    """
    return X.columns.tolist()


def print_preprocessing_summary(X_train, X_val, y_train, y_val):
    """
    Print summary of preprocessed datasets.

    Args:
        X_train, X_val: Feature dataframes
        y_train, y_val: Target series
    """
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)

    print(f"\nTrain Set:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Samples: {X_train.shape[0]:,}")
    print(f"  Target mean: {y_train.mean():.2f}")
    print(f"  Target std: {y_train.std():.2f}")
    print(f"  Missing values: {X_train.isna().sum().sum()}")

    print(f"\nValidation Set:")
    print(f"  Shape: {X_val.shape}")
    print(f"  Features: {X_val.shape[1]}")
    print(f"  Samples: {X_val.shape[0]:,}")
    print(f"  Target mean: {y_val.mean():.2f}")
    print(f"  Target std: {y_val.std():.2f}")
    print(f"  Missing values: {X_val.isna().sum().sum()}")

    print(f"\nFeature Types:")
    print(f"  Numeric: {len(X_train.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical: {len(X_train.select_dtypes(include=['object', 'category']).columns)}")

    print("\n" + "="*80)
