"""
Phase 8: Inference Script for Ranking Data
===========================================
Generates predictions using trained Phase 8 models.

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import pickle
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    # Stratification thresholds (same as training)
    'stratification_thresholds': [200, 500, 1000, 1500, 2500],

    # Ensemble weights (same as training)
    'ensemble_weights': {
        'very_low': {'lgb': 1.0},
        'low': {'lgb': 1.0},
        'medium': {'lgb': 1.0},
        'medium_high': {'lgb': 1.0},
        'high': {'lgb': 0.5, 'cat': 0.5},
        'very_high': {'lgb': 0.4, 'cat': 0.6}
    },

    # Paths
    'paths': {
        'models': 'models',
        'submissions': 'submissions',
        'cache': 'cache'
    },

    # Team name for submission
    'team_name': 'gentle-orange'
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_stage_name_from_heuristic(X):
    """
    Estimate fuel range stage from features for routing.

    Since we don't know true fuel values during inference, we use heuristics:
    - interval_duration_min: Longer intervals typically use more fuel
    - altitude_mean: Higher altitudes typically cruise (medium fuel)
    - distance_covered_km: Longer distances use more fuel

    Returns array of stage names for each sample.
    """
    # Simple heuristic based on interval duration and distance
    duration = X['interval_duration_min'].values if 'interval_duration_min' in X.columns else np.ones(len(X)) * 10
    distance = X['distance_covered_km'].values if 'distance_covered_km' in X.columns else np.ones(len(X)) * 50

    # Rough fuel estimate: distance * 5 + duration * 20
    # (These are arbitrary weights - can be tuned based on training data analysis)
    fuel_estimate = distance * 5 + duration * 20

    stages = np.empty(len(X), dtype=object)

    stages[fuel_estimate < 200] = 'very_low'
    stages[(fuel_estimate >= 200) & (fuel_estimate < 500)] = 'low'
    stages[(fuel_estimate >= 500) & (fuel_estimate < 1000)] = 'medium'
    stages[(fuel_estimate >= 1000) & (fuel_estimate < 1500)] = 'medium_high'
    stages[(fuel_estimate >= 1500) & (fuel_estimate < 2500)] = 'high'
    stages[fuel_estimate >= 2500] = 'very_high'

    return stages


def get_stage_name_iterative(X, initial_predictions, thresholds):
    """
    Iteratively refine stage assignments based on initial predictions.

    More accurate than heuristics but requires two passes.
    """
    stages = np.empty(len(X), dtype=object)

    stages[initial_predictions < thresholds[0]] = 'very_low'
    stages[(initial_predictions >= thresholds[0]) & (initial_predictions < thresholds[1])] = 'low'
    stages[(initial_predictions >= thresholds[1]) & (initial_predictions < thresholds[2])] = 'medium'
    stages[(initial_predictions >= thresholds[2]) & (initial_predictions < thresholds[3])] = 'medium_high'
    stages[(initial_predictions >= thresholds[3]) & (initial_predictions < thresholds[4])] = 'high'
    stages[initial_predictions >= thresholds[4]] = 'very_high'

    return stages


# ============================================================================
# PHASE 8 INFERENCE PIPELINE
# ============================================================================

class Phase8InferencePipeline:
    """Inference pipeline for Phase 8 models."""

    def __init__(self, config):
        self.config = config
        self.preprocessing_pipeline = None
        self.models = {}
        self.loaded = False

    def load_models(self):
        """Load all trained models and preprocessing pipeline."""
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80)

        models_dir = Path(self.config['paths']['models'])

        # Load preprocessing pipeline
        preprocessing_path = models_dir / 'phase8_preprocessing_pipeline.pkl'
        if not preprocessing_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline not found: {preprocessing_path}")

        with open(preprocessing_path, 'rb') as f:
            self.preprocessing_pipeline = pickle.load(f)
        print(f"✓ Loaded preprocessing pipeline")

        # Load stage models
        stages = ['very_low', 'low', 'medium', 'medium_high', 'high', 'very_high']

        for stage_name in stages:
            self.models[stage_name] = {}

            # Load LightGBM
            lgb_path = models_dir / f'phase8_{stage_name}_lgb.txt'
            if lgb_path.exists():
                self.models[stage_name]['lgb'] = lgb.Booster(model_file=str(lgb_path))
                print(f"✓ Loaded {stage_name} LightGBM")

            # Load CatBoost (for high and very_high stages)
            cat_path = models_dir / f'phase8_{stage_name}_cat.cbm'
            if cat_path.exists():
                self.models[stage_name]['cat'] = cb.CatBoostRegressor()
                self.models[stage_name]['cat'].load_model(str(cat_path))
                print(f"✓ Loaded {stage_name} CatBoost")

        self.loaded = True
        print("\n✓ All models loaded successfully")

    def preprocess(self, X):
        """Apply preprocessing pipeline."""
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        print("\n" + "="*80)
        print("PREPROCESSING")
        print("="*80)
        print(f"Input shape: {X.shape}")

        X_proc = self.preprocessing_pipeline.transform(X)

        print(f"Output shape: {X_proc.shape}")
        print("✓ Preprocessing complete")

        return X_proc

    def predict(self, X, method='iterative'):
        """
        Generate predictions for input data.

        Args:
            X: Input features (preprocessed)
            method: Routing method ('heuristic' or 'iterative')
                - heuristic: Fast, single-pass, based on feature values
                - iterative: Slower, two-pass, more accurate, uses initial predictions

        Returns:
            predictions: Array of fuel predictions
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        print("\n" + "="*80)
        print(f"GENERATING PREDICTIONS (method={method})")
        print("="*80)

        predictions = np.zeros(len(X))

        if method == 'heuristic':
            # Single-pass heuristic routing
            stages = get_stage_name_from_heuristic(X)

            for stage_name in np.unique(stages):
                stage_mask = stages == stage_name
                n_samples = stage_mask.sum()

                if n_samples == 0:
                    continue

                print(f"\n[{stage_name.upper()}] Predicting {n_samples:,} samples...")

                X_stage = X[stage_mask]
                stage_pred = self._predict_stage(stage_name, X_stage)
                predictions[stage_mask] = stage_pred

        elif method == 'iterative':
            # Two-pass iterative routing
            print("\nPass 1: Initial predictions with heuristic routing...")
            initial_predictions = self.predict(X, method='heuristic')

            print("\nPass 2: Refined predictions with iterative routing...")
            stages = get_stage_name_iterative(
                X, initial_predictions, self.config['stratification_thresholds']
            )

            for stage_name in np.unique(stages):
                stage_mask = stages == stage_name
                n_samples = stage_mask.sum()

                if n_samples == 0:
                    continue

                print(f"\n[{stage_name.upper()}] Predicting {n_samples:,} samples...")

                X_stage = X[stage_mask]
                stage_pred = self._predict_stage(stage_name, X_stage)
                predictions[stage_mask] = stage_pred

        else:
            raise ValueError(f"Unknown method: {method}")

        print("\n✓ Predictions complete")
        print(f"  Min: {predictions.min():.2f} kg")
        print(f"  Max: {predictions.max():.2f} kg")
        print(f"  Mean: {predictions.mean():.2f} kg")
        print(f"  Median: {np.median(predictions):.2f} kg")

        return predictions

    def _predict_stage(self, stage_name, X_stage):
        """Generate predictions for a specific stage."""
        stage_models = self.models.get(stage_name, {})
        weights = self.config['ensemble_weights'].get(stage_name, {'lgb': 1.0})

        stage_predictions = {}

        # Get predictions from each model
        for model_name, model in stage_models.items():
            if model_name == 'lgb':
                stage_predictions[model_name] = model.predict(X_stage)
            elif model_name == 'cat':
                stage_predictions[model_name] = model.predict(X_stage)

        # Ensemble
        ensemble_pred = np.zeros(len(X_stage))
        for model_name, weight in weights.items():
            if model_name in stage_predictions:
                ensemble_pred += weight * stage_predictions[model_name]

        return ensemble_pred

    def create_submission(self, predictions, submission_template_path, version=1):
        """
        Create submission file.

        Args:
            predictions: Array of predictions
            submission_template_path: Path to submission template parquet
            version: Submission version number
        """
        print("\n" + "="*80)
        print("CREATING SUBMISSION FILE")
        print("="*80)

        # Load template
        submission = pd.read_parquet(submission_template_path)
        print(f"✓ Loaded template: {submission.shape}")

        # Validate
        if len(predictions) != len(submission):
            raise ValueError(f"Prediction count mismatch: {len(predictions)} != {len(submission)}")

        # Replace fuel_kg
        submission['fuel_kg'] = predictions

        # Final validation
        assert submission['fuel_kg'].notna().all(), "Null predictions detected!"
        assert (submission['fuel_kg'] >= 0).all(), "Negative predictions detected!"

        # Save
        submissions_dir = Path(self.config['paths']['submissions'])
        submissions_dir.mkdir(exist_ok=True)

        team_name = self.config['team_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{team_name}_phase8_v{version}_{timestamp}.parquet"
        output_path = submissions_dir / filename

        submission.write_parquet(output_path)

        print(f"\n✓ Submission saved: {output_path}")
        print(f"\nSubmission statistics:")
        print(f"  Shape: {submission.shape}")
        print(f"  Fuel range: {submission['fuel_kg'].min():.2f} - {submission['fuel_kg'].max():.2f} kg")
        print(f"  Fuel mean: {submission['fuel_kg'].mean():.2f} kg")
        print(f"  Fuel median: {submission['fuel_kg'].median():.2f} kg")

        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main inference pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    PHASE 8: RANKING PREDICTIONS                          ║
    ║                                                                          ║
    ║  Using Advanced Preprocessing + 6-Stage Stratification + Ensemble        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Paths
    ranking_features_path = 'cache/ranking_features.parquet'  # TODO: Update to actual path
    submission_template_path = 'fuel_rank_submission.parquet'

    # Validate paths
    if not Path(ranking_features_path).exists():
        print(f"\n❌ ERROR: Ranking features not found at {ranking_features_path}")
        print("\nPlease ensure you have extracted features for ranking data.")
        print("Expected file: cache/ranking_features.parquet")
        return

    if not Path(submission_template_path).exists():
        print(f"\n❌ ERROR: Submission template not found at {submission_template_path}")
        return

    # Load ranking data
    print("\n" + "="*80)
    print("LOADING RANKING DATA")
    print("="*80)

    X_rank = pd.read_parquet(ranking_features_path)
    print(f"✓ Loaded ranking data: {X_rank.shape}")

    # Initialize pipeline
    pipeline = Phase8InferencePipeline(INFERENCE_CONFIG)

    # Load models
    pipeline.load_models()

    # Preprocess
    X_rank_proc = pipeline.preprocess(X_rank)

    # Generate predictions (using iterative method for better accuracy)
    predictions = pipeline.predict(X_rank_proc, method='iterative')

    # Create submission
    submission_path = pipeline.create_submission(
        predictions,
        submission_template_path,
        version=1
    )

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nSubmission file ready: {submission_path}")
    print("\nNext steps:")
    print("  1. Review submission statistics above")
    print("  2. Validate submission format")
    print("  3. Submit to competition platform")


if __name__ == "__main__":
    main()
