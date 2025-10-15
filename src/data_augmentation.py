"""
Data Augmentation Module for Extreme Fuel Ranges
=================================================
Addresses the major bottleneck: very high fuel range (2500+ kg) with 860+ kg RMSE

Author: ML Team
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class FuelRangeAugmenter(BaseEstimator, TransformerMixin):
    """
    Augment training data for rare high-fuel cases using intelligent oversampling.

    Strategies:
    - SMOTE: Synthetic Minority Over-sampling Technique
    - ADASYN: Adaptive Synthetic Sampling
    - Gaussian Noise: Add controlled noise to existing samples
    - Interpolation: Create samples between existing points
    """

    def __init__(self, target_col='fuel_kg', augment_ranges=None,
                 strategy='smote', target_ratio=0.5, noise_level=0.05,
                 random_state=42, verbose=True):
        """
        Args:
            target_col: Name of target column
            augment_ranges: List of (min, max, multiplier) tuples for augmentation
                           e.g., [(2500, np.inf, 3)] means augment very high fuel 3x
            strategy: 'smote', 'adasyn', 'noise', 'interpolate'
            target_ratio: Target ratio for oversampling (0-1)
            noise_level: Std dev of Gaussian noise (fraction of feature std)
            random_state: Random seed
            verbose: Print statistics
        """
        self.target_col = target_col
        self.augment_ranges = augment_ranges or [(2500, np.inf, 3)]
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.noise_level = noise_level
        self.random_state = random_state
        self.verbose = verbose
        self.numeric_columns_ = None

    def fit(self, X, y=None):
        """No fitting required."""
        return self

    def transform(self, X, y=None):
        """Not used - use fit_resample instead."""
        return X

    def fit_resample(self, X, y):
        """
        Augment data for specified fuel ranges.

        Returns:
            X_augmented, y_augmented
        """
        if y is None:
            raise ValueError("y (target) is required for augmentation")

        X_aug = X.copy()
        y_aug = y.copy()

        # Store numeric columns from the input data
        if isinstance(X, pd.DataFrame):
            self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()

            # Check for non-numeric columns
            non_numeric_cols = [col for col in X.columns if col not in self.numeric_columns_]
            if non_numeric_cols and self.verbose:
                print(f"[FuelRangeAugmenter] Warning: Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:5]}")
                print(f"  These columns will be excluded from augmentation")

            # Work only with numeric data for augmentation
            X_aug = X_aug[self.numeric_columns_].copy()

        original_size = len(X)
        augmented_samples = 0

        for min_fuel, max_fuel, multiplier in self.augment_ranges:
            # Find samples in this range
            if isinstance(y, pd.Series):
                mask = (y >= min_fuel) & (y < max_fuel)
            else:
                mask = (y >= min_fuel) & (y < max_fuel)

            if mask.sum() == 0:
                if self.verbose:
                    print(f"[FuelRangeAugmenter] No samples in range [{min_fuel}, {max_fuel})")
                continue

            X_range = X[mask].copy()
            y_range = y[mask].copy() if isinstance(y, pd.Series) else y[mask]

            n_samples = len(X_range)
            n_to_generate = int(n_samples * (multiplier - 1))

            if n_to_generate <= 0:
                continue

            if self.verbose:
                print(f"\n[FuelRangeAugmenter] Range [{min_fuel:.0f}, {max_fuel})")
                print(f"  Original samples: {n_samples}")
                print(f"  Generating: {n_to_generate} synthetic samples (multiplier={multiplier})")

            # Apply strategy
            if self.strategy == 'smote' and n_samples >= 6:
                X_syn, y_syn = self._smote_augment(X_range, y_range, n_to_generate)

            elif self.strategy == 'adasyn' and n_samples >= 6:
                X_syn, y_syn = self._adasyn_augment(X_range, y_range, n_to_generate)

            elif self.strategy == 'noise':
                X_syn, y_syn = self._noise_augment(X_range, y_range, n_to_generate)

            elif self.strategy == 'interpolate':
                X_syn, y_syn = self._interpolate_augment(X_range, y_range, n_to_generate)

            else:
                # Default to noise if SMOTE/ADASYN not applicable
                X_syn, y_syn = self._noise_augment(X_range, y_range, n_to_generate)

            # Append synthetic samples
            if isinstance(X_aug, pd.DataFrame):
                # Use only numeric columns for synthetic data
                X_syn_df = pd.DataFrame(X_syn, columns=self.numeric_columns_)
                X_aug = pd.concat([X_aug, X_syn_df], ignore_index=True)
            else:
                X_aug = np.vstack([X_aug, X_syn])

            if isinstance(y_aug, pd.Series):
                y_syn_series = pd.Series(y_syn, name=y_aug.name)
                y_aug = pd.concat([y_aug, y_syn_series], ignore_index=True)
            else:
                y_aug = np.hstack([y_aug, y_syn])

            augmented_samples += n_to_generate

        if self.verbose:
            print(f"\n[FuelRangeAugmenter] Augmentation complete")
            print(f"  Original size: {original_size:,}")
            print(f"  Augmented size: {len(X_aug):,}")
            print(f"  Added samples: {augmented_samples:,} ({augmented_samples/original_size*100:.1f}%)")

        return X_aug, y_aug

    def _smote_augment(self, X, y, n_samples):
        """SMOTE augmentation."""
        # Prepare data
        X_numeric = self._prepare_numeric(X)

        # Calculate number of neighbors (must be < n_samples)
        k_neighbors = min(5, len(X) - 1)

        try:
            # Create synthetic samples
            smote = SMOTE(
                sampling_strategy={1: len(X) + n_samples},  # Oversample minority class
                k_neighbors=k_neighbors,
                random_state=self.random_state
            )

            # Create binary labels for SMOTE (all samples are "minority")
            y_binary = np.ones(len(X))

            # Add one "majority" sample to enable SMOTE
            X_with_dummy = np.vstack([X_numeric, X_numeric[0]])
            y_with_dummy = np.hstack([y_binary, 0])

            X_resampled, _ = smote.fit_resample(X_with_dummy, y_with_dummy)

            # Extract synthetic samples (after original samples)
            X_syn = X_resampled[len(X):-1]  # Exclude dummy

            # Generate target values using KNN interpolation
            y_syn = self._interpolate_targets(X_numeric, y, X_syn)

            return X_syn[:n_samples], y_syn[:n_samples]

        except Exception as e:
            if self.verbose:
                print(f"    SMOTE failed: {e}, falling back to noise augmentation")
            return self._noise_augment(X, y, n_samples)

    def _adasyn_augment(self, X, y, n_samples):
        """ADASYN augmentation."""
        X_numeric = self._prepare_numeric(X)
        k_neighbors = min(5, len(X) - 1)

        try:
            adasyn = ADASYN(
                sampling_strategy={1: len(X) + n_samples},
                n_neighbors=k_neighbors,
                random_state=self.random_state
            )

            # Same approach as SMOTE
            y_binary = np.ones(len(X))
            X_with_dummy = np.vstack([X_numeric, X_numeric[0]])
            y_with_dummy = np.hstack([y_binary, 0])

            X_resampled, _ = adasyn.fit_resample(X_with_dummy, y_with_dummy)
            X_syn = X_resampled[len(X):-1]

            y_syn = self._interpolate_targets(X_numeric, y, X_syn)

            return X_syn[:n_samples], y_syn[:n_samples]

        except Exception as e:
            if self.verbose:
                print(f"    ADASYN failed: {e}, falling back to noise augmentation")
            return self._noise_augment(X, y, n_samples)

    def _noise_augment(self, X, y, n_samples):
        """Gaussian noise augmentation."""
        X_numeric = self._prepare_numeric(X)

        # Randomly select samples to augment
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), size=n_samples, replace=True)

        X_syn = X_numeric[indices].copy()

        # Add Gaussian noise
        for col_idx in range(X_syn.shape[1]):
            col_std = X_numeric[:, col_idx].std()
            noise = np.random.normal(0, col_std * self.noise_level, size=n_samples)
            X_syn[:, col_idx] += noise

        # Get corresponding target values with small noise
        if isinstance(y, pd.Series):
            y_base = y.iloc[indices].values
        else:
            y_base = y[indices]

        y_std = y_base.std()
        y_noise = np.random.normal(0, y_std * self.noise_level, size=n_samples)
        y_syn = y_base + y_noise

        # Ensure positive fuel values
        y_syn = np.maximum(y_syn, 0)

        return X_syn, y_syn

    def _interpolate_augment(self, X, y, n_samples):
        """Interpolation-based augmentation."""
        X_numeric = self._prepare_numeric(X)

        # Randomly select pairs of samples
        np.random.seed(self.random_state)
        n_orig = len(X)

        X_syn = []
        y_syn = []

        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(n_orig, size=2, replace=True)

            # Interpolation weight (random between 0 and 1)
            alpha = np.random.uniform(0, 1)

            # Interpolate features
            x_new = alpha * X_numeric[idx1] + (1 - alpha) * X_numeric[idx2]
            X_syn.append(x_new)

            # Interpolate target
            if isinstance(y, pd.Series):
                y1, y2 = y.iloc[idx1], y.iloc[idx2]
            else:
                y1, y2 = y[idx1], y[idx2]

            y_new = alpha * y1 + (1 - alpha) * y2
            y_syn.append(y_new)

        return np.array(X_syn), np.array(y_syn)

    def _interpolate_targets(self, X_train, y_train, X_syn):
        """Interpolate target values for synthetic samples using KNN."""
        # Use KNN to predict targets for synthetic samples
        k_neighbors = min(5, len(X_train))

        knn = NearestNeighbors(n_neighbors=k_neighbors)
        knn.fit(X_train)

        distances, indices = knn.kneighbors(X_syn)

        # Weight by inverse distance
        weights = 1 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Get target values
        if isinstance(y_train, pd.Series):
            y_neighbors = y_train.iloc[indices].values
        else:
            y_neighbors = y_train[indices]

        # Weighted average
        y_syn = (y_neighbors * weights).sum(axis=1)

        return y_syn

    def _prepare_numeric(self, X, store_columns=False):
        """Convert to numeric numpy array."""
        if isinstance(X, pd.DataFrame):
            # Keep only numeric columns
            numeric_df = X.select_dtypes(include=[np.number])

            # Store column names on first call
            if store_columns or self.numeric_columns_ is None:
                self.numeric_columns_ = numeric_df.columns.tolist()

            X_numeric = numeric_df.values
        else:
            X_numeric = X

        # Handle any remaining NaN
        X_numeric = np.nan_to_num(X_numeric, nan=0.0)

        return X_numeric


class StratifiedAugmenter:
    """
    Apply different augmentation strategies for different fuel ranges.
    """

    def __init__(self, augmentation_config=None, verbose=True):
        """
        Args:
            augmentation_config: List of dicts with keys:
                - range: (min, max)
                - strategy: 'smote', 'adasyn', 'noise', 'interpolate'
                - multiplier: float (augmentation factor)
        """
        self.augmentation_config = augmentation_config or [
            {'range': (1500, 2500), 'strategy': 'noise', 'multiplier': 2.0},
            {'range': (2500, np.inf), 'strategy': 'interpolate', 'multiplier': 4.0}
        ]
        self.verbose = verbose

    def fit_resample(self, X, y):
        """Apply stratified augmentation."""
        X_aug = X.copy()
        y_aug = y.copy()

        for config in self.augmentation_config:
            min_fuel, max_fuel = config['range']
            strategy = config['strategy']
            multiplier = config['multiplier']

            augmenter = FuelRangeAugmenter(
                augment_ranges=[(min_fuel, max_fuel, multiplier)],
                strategy=strategy,
                random_state=42,
                verbose=self.verbose
            )

            X_aug, y_aug = augmenter.fit_resample(X_aug, y_aug)

        return X_aug, y_aug


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_augmentation_to_stratified_data(X_train, y_train, stratification_thresholds):
    """
    Apply augmentation within each stratification range.

    Args:
        X_train: Training features
        y_train: Training target
        stratification_thresholds: List of threshold values (e.g., [200, 500, 1000, 1500, 2500])

    Returns:
        X_train_aug, y_train_aug (augmented datasets)
    """
    # Define augmentation strategy per range
    ranges = []
    for i in range(len(stratification_thresholds)):
        if i == 0:
            continue  # Skip low ranges, no augmentation needed

        min_fuel = stratification_thresholds[i-1] if i > 0 else 0
        max_fuel = stratification_thresholds[i] if i < len(stratification_thresholds) else np.inf

        # Determine multiplier based on range
        if min_fuel >= 2500:
            multiplier = 4.0  # Very high fuel
            strategy = 'interpolate'
        elif min_fuel >= 1500:
            multiplier = 2.0  # High fuel
            strategy = 'noise'
        else:
            multiplier = 1.2  # Medium fuel
            strategy = 'noise'

        ranges.append((min_fuel, max_fuel, multiplier))

    augmenter = FuelRangeAugmenter(
        augment_ranges=ranges,
        strategy='hybrid',
        verbose=True
    )

    X_train_aug, y_train_aug = augmenter.fit_resample(X_train, y_train)

    return X_train_aug, y_train_aug


def validate_augmentation(X_original, y_original, X_augmented, y_augmented):
    """
    Validate augmentation results.

    Prints statistics about augmentation quality.
    """
    print("\n" + "="*80)
    print("AUGMENTATION VALIDATION")
    print("="*80)

    print(f"\nSize:")
    print(f"  Original: {len(X_original):,} samples")
    print(f"  Augmented: {len(X_augmented):,} samples")
    print(f"  Added: {len(X_augmented) - len(X_original):,} samples ({(len(X_augmented)/len(X_original)-1)*100:.1f}%)")

    print(f"\nTarget distribution (original):")
    if isinstance(y_original, pd.Series):
        print(f"  Mean: {y_original.mean():.2f} kg")
        print(f"  Std: {y_original.std():.2f} kg")
        print(f"  Min: {y_original.min():.2f} kg")
        print(f"  Max: {y_original.max():.2f} kg")
    else:
        print(f"  Mean: {y_original.mean():.2f} kg")
        print(f"  Std: {y_original.std():.2f} kg")
        print(f"  Min: {y_original.min():.2f} kg")
        print(f"  Max: {y_original.max():.2f} kg")

    print(f"\nTarget distribution (augmented):")
    if isinstance(y_augmented, pd.Series):
        print(f"  Mean: {y_augmented.mean():.2f} kg")
        print(f"  Std: {y_augmented.std():.2f} kg")
        print(f"  Min: {y_augmented.min():.2f} kg")
        print(f"  Max: {y_augmented.max():.2f} kg")
    else:
        print(f"  Mean: {y_augmented.mean():.2f} kg")
        print(f"  Std: {y_augmented.std():.2f} kg")
        print(f"  Min: {y_augmented.min():.2f} kg")
        print(f"  Max: {y_augmented.max():.2f} kg")

    print("\n" + "="*80)
