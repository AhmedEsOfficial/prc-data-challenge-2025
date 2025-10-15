"""
Prepare data for Phase 8 training
==================================
Combines existing cached features into the format expected by Phase 8.

Author: ML Team
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("PREPARING DATA FOR PHASE 8")
print("="*80)

# Load existing processed data
print("\n[1/5] Loading training data...")
X_train = pd.read_parquet('cache/X_train_enhanced.parquet')
y_train = pd.read_parquet('cache/y_train.parquet')

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

print("\n[2/5] Loading validation data...")
X_val = pd.read_parquet('cache/X_val_enhanced.parquet')
y_val = pd.read_parquet('cache/y_val.parquet')

print(f"  X_val shape: {X_val.shape}")
print(f"  y_val shape: {y_val.shape}")

# Combine train and val for proper temporal split in Phase 8
print("\n[3/5] Combining datasets...")

# Add fuel_kg to features
X_train['fuel_kg'] = y_train.values
X_val['fuel_kg'] = y_val.values

# Combine
df = pd.concat([X_train, X_val], ignore_index=True)

print(f"  Combined shape: {df.shape}")
print(f"  Columns: {df.shape[1]}")

# Check for required columns
print("\n[4/5] Validating columns...")
required_cols = ['fuel_kg', 'flight_id']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"  ⚠️  Missing columns: {missing_cols}")

    # Try to add flight_id if missing
    if 'flight_id' in missing_cols:
        # Load from original data
        print("  → Attempting to load flight_id from fuel_train.parquet...")
        fuel_train = pd.read_parquet('fuel_train.parquet')

        # Merge on idx if available
        if 'idx' in df.columns:
            df = df.merge(fuel_train[['idx', 'flight_id']], on='idx', how='left')
            print("  ✓ Added flight_id from fuel_train")
        else:
            print("  ❌ Cannot add flight_id - no idx column")
            print("  → Creating dummy flight_id based on row groups")
            # Create flight_id from row index (group every ~12 rows as a flight)
            df['flight_id'] = (df.index // 12).astype(str)
            print("  ✓ Created dummy flight_id")

# Add start_month for temporal split if not present
if 'start_month' not in df.columns and 'month' not in df.columns:
    print("  → Creating temporal split column...")

    # If we have idx, we can match to original data
    if 'idx' in df.columns:
        fuel_train = pd.read_parquet('fuel_train.parquet')

        # Extract month from start timestamp
        fuel_train['start_month'] = pd.to_datetime(fuel_train['start']).dt.month

        df = df.merge(fuel_train[['idx', 'start_month']], on='idx', how='left')
        print("  ✓ Added start_month from fuel_train")
    else:
        # Use existing split as proxy: first 80% = months 4-7, last 20% = month 8
        n_train = len(X_train)
        df['start_month'] = 7  # Default to training set
        df.loc[n_train:, 'start_month'] = 8  # Validation set
        print("  ✓ Created start_month from existing split")

print("\n[5/5] Saving combined dataset...")

# Save to expected location
output_path = 'cache/processed_features.parquet'
df.to_parquet(output_path)

print(f"  ✓ Saved to: {output_path}")

# Print summary
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nFinal dataset:")
print(f"  Shape: {df.shape}")
print(f"  Samples: {len(df):,}")
print(f"  Features: {df.shape[1]}")
print(f"  Missing values: {df.isna().sum().sum()}")

print(f"\nKey columns:")
print(f"  ✓ fuel_kg: {df['fuel_kg'].notna().sum():,} non-null values")
print(f"  ✓ flight_id: {df['flight_id'].notna().sum():,} non-null values")
print(f"  ✓ start_month: {df['start_month'].notna().sum():,} non-null values" if 'start_month' in df.columns else "  ⚠️ start_month: Not found, using 'month' column")

print(f"\nFuel statistics:")
print(f"  Min: {df['fuel_kg'].min():.2f} kg")
print(f"  Max: {df['fuel_kg'].max():.2f} kg")
print(f"  Mean: {df['fuel_kg'].mean():.2f} kg")
print(f"  Median: {df['fuel_kg'].median():.2f} kg")

if 'start_month' in df.columns or 'month' in df.columns:
    month_col = 'start_month' if 'start_month' in df.columns else 'month'
    print(f"\nTemporal distribution:")
    month_counts = df[month_col].value_counts().sort_index()
    for month, count in month_counts.items():
        print(f"  Month {month}: {count:,} samples")

print("\n✅ You can now run: python train_phase8_ultimate.py")
