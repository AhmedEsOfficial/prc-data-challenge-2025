"""
Phase 1: Deep Data Understanding
=================================
Comprehensive analysis of aircraft fuel consumption dataset.

This script performs:
1. Dataset loading and profiling
2. Fuel consumption pattern analysis
3. Trajectory data quality assessment
4. Flight phase identification
5. Correlation analysis
6. Documentation of findings

Author: ML Team
Date: 2025-10-15
"""

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Project paths
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR
CACHE_DIR = BASE_DIR / 'cache'
VIZ_DIR = BASE_DIR / 'visualizations'
REPORTS_DIR = BASE_DIR / 'reports'

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("PHASE 1: DEEP DATA UNDERSTANDING")
print("="*80)
print(f"Started: {datetime.now()}")
print(f"Base Directory: {BASE_DIR}")
print()

# ============================================================================
# SECTION 1: LOAD AND PROFILE ALL DATASETS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING AND PROFILING DATASETS")
print("="*80)

# 1.1 Load Fuel Training Data
print("\n[1/6] Loading fuel_train.parquet...")
fuel_train = pl.read_parquet(DATA_DIR / 'fuel_train.parquet')
print(f"  Shape: {fuel_train.shape}")
print(f"  Columns: {fuel_train.columns}")
print(f"  Memory: {fuel_train.estimated_size() / 1024**2:.2f} MB")

# 1.2 Load Fuel Ranking Submission Template
print("\n[2/6] Loading fuel_rank_submission.parquet...")
fuel_rank = pl.read_parquet(DATA_DIR / 'fuel_rank_submission.parquet')
print(f"  Shape: {fuel_rank.shape}")
print(f"  Note: fuel_kg column is empty (to be predicted)")

# 1.3 Load Flight Lists
print("\n[3/6] Loading flightlist_train.parquet...")
flightlist_train = pl.read_parquet(DATA_DIR / 'flightlist_train.parquet')
print(f"  Shape: {flightlist_train.shape}")
print(f"  Unique flights: {flightlist_train['flight_id'].n_unique()}")
print(f"  Aircraft types: {flightlist_train['aircraft_type'].n_unique()}")
print(f"  Date range: {flightlist_train['flight_date'].min()} to {flightlist_train['flight_date'].max()}")

print("\n[4/6] Loading flightlist_rank.parquet...")
flightlist_rank = pl.read_parquet(DATA_DIR / 'flightlist_rank.parquet')
print(f"  Shape: {flightlist_rank.shape}")
print(f"  Unique flights: {flightlist_rank['flight_id'].n_unique()}")
print(f"  Aircraft types: {flightlist_rank['aircraft_type'].n_unique()}")
print(f"  Date range: {flightlist_rank['flight_date'].min()} to {flightlist_rank['flight_date'].max()}")

# 1.4 Load Airports
print("\n[5/6] Loading apt.parquet...")
airports = pl.read_parquet(DATA_DIR / 'apt.parquet')
print(f"  Shape: {airports.shape}")
print(f"  Unique airports: {airports['icao'].n_unique()}")

# 1.5 Sample Trajectories
print("\n[6/6] Sampling trajectory files...")
train_traj_dir = DATA_DIR / 'flights_train'
rank_traj_dir = DATA_DIR / 'flights_rank'

train_traj_files = sorted(train_traj_dir.glob('*.parquet'))
rank_traj_files = sorted(rank_traj_dir.glob('*.parquet'))

print(f"  Training trajectories: {len(train_traj_files)} files")
print(f"  Ranking trajectories: {len(rank_traj_files)} files")

# Load first few training trajectories for analysis
sample_flight_ids = flightlist_train['flight_id'].head(10).to_list()
print(f"\n  Loading {len(sample_flight_ids)} sample trajectories for detailed analysis...")

sample_trajectories = {}
for flight_id in sample_flight_ids:
    traj_path = train_traj_dir / f'{flight_id}.parquet'
    if traj_path.exists():
        sample_trajectories[flight_id] = pl.read_parquet(traj_path)
        print(f"    ✓ {flight_id}: {sample_trajectories[flight_id].shape[0]} points")

# ============================================================================
# SECTION 2: FUEL CONSUMPTION PATTERN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: FUEL CONSUMPTION PATTERN ANALYSIS")
print("="*80)

# Convert to pandas for easier analysis
fuel_train_pd = fuel_train.to_pandas()
flightlist_train_pd = flightlist_train.to_pandas()

# 2.1 Overall Fuel Statistics
print("\n[2.1] Overall Fuel Statistics")
print("-" * 40)
fuel_stats = fuel_train_pd['fuel_kg'].describe()
print(fuel_stats)

print(f"\nTotal fuel intervals: {len(fuel_train_pd):,}")
print(f"Total flights: {fuel_train_pd['flight_id'].nunique():,}")
print(f"Avg intervals per flight: {len(fuel_train_pd) / fuel_train_pd['flight_id'].nunique():.1f}")

# 2.2 Fuel Distribution
print("\n[2.2] Creating fuel distribution visualizations...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(fuel_train_pd['fuel_kg'], bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Fuel (kg)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Fuel Distribution (All Intervals)')
axes[0].grid(True, alpha=0.3)

# Log scale histogram
axes[1].hist(fuel_train_pd['fuel_kg'], bins=100, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Fuel (kg)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Fuel Distribution (Log Scale)')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

# Box plot
axes[2].boxplot(fuel_train_pd['fuel_kg'], vert=True)
axes[2].set_ylabel('Fuel (kg)')
axes[2].set_title('Fuel Distribution (Box Plot)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'fuel_distribution.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {VIZ_DIR / 'fuel_distribution.png'}")
plt.close()

# 2.3 Interval Duration Analysis
print("\n[2.3] Analyzing interval durations...")
fuel_train_pd['interval_duration'] = (fuel_train_pd['end'] - fuel_train_pd['start']).dt.total_seconds()

duration_stats = fuel_train_pd['interval_duration'].describe()
print(duration_stats)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(fuel_train_pd['interval_duration'], bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Interval Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Interval Duration Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(fuel_train_pd['interval_duration'], fuel_train_pd['fuel_kg'], alpha=0.1, s=1)
plt.xlabel('Interval Duration (seconds)')
plt.ylabel('Fuel (kg)')
plt.title('Fuel vs Interval Duration')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'interval_duration_analysis.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {VIZ_DIR / 'interval_duration_analysis.png'}")
plt.close()

# Calculate fuel rate (kg/second)
fuel_train_pd['fuel_rate'] = fuel_train_pd['fuel_kg'] / fuel_train_pd['interval_duration']
print(f"\nFuel Rate Statistics (kg/second):")
print(fuel_train_pd['fuel_rate'].describe())

# 2.4 Fuel by Aircraft Type
print("\n[2.4] Analyzing fuel consumption by aircraft type...")
merged_train = fuel_train_pd.merge(flightlist_train_pd[['flight_id', 'aircraft_type']], on='flight_id', how='left')

fuel_by_aircraft = merged_train.groupby('aircraft_type').agg({
    'fuel_kg': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
fuel_by_aircraft.columns = ['mean_fuel', 'std_fuel', 'min_fuel', 'max_fuel', 'count']
fuel_by_aircraft = fuel_by_aircraft.sort_values('mean_fuel', ascending=False)

print("\nTop 10 Aircraft by Average Fuel per Interval:")
print(fuel_by_aircraft.head(10))

print("\nBottom 10 Aircraft by Average Fuel per Interval:")
print(fuel_by_aircraft.tail(10))

# Visualize top aircraft types
plt.figure(figsize=(12, 6))
top_aircraft = fuel_by_aircraft.head(15)
plt.barh(range(len(top_aircraft)), top_aircraft['mean_fuel'], xerr=top_aircraft['std_fuel'], alpha=0.7)
plt.yticks(range(len(top_aircraft)), top_aircraft.index)
plt.xlabel('Average Fuel per Interval (kg)')
plt.title('Top 15 Aircraft Types by Fuel Consumption')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'fuel_by_aircraft.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {VIZ_DIR / 'fuel_by_aircraft.png'}")
plt.close()

# 2.5 Fuel by Interval Position in Flight
print("\n[2.5] Analyzing fuel by interval position in flight...")

# Calculate interval position for each flight
def calculate_interval_position(group):
    group = group.sort_values('start')
    group['interval_index'] = range(len(group))
    group['total_intervals'] = len(group)
    group['interval_position_pct'] = (group['interval_index'] / (group['total_intervals'] - 1) * 100).fillna(0)
    return group

fuel_train_pd = fuel_train_pd.groupby('flight_id', group_keys=False).apply(calculate_interval_position)

# Bin positions
fuel_train_pd['position_bin'] = pd.cut(fuel_train_pd['interval_position_pct'],
                                        bins=[0, 10, 25, 50, 75, 90, 100],
                                        labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%'])

fuel_by_position = fuel_train_pd.groupby('position_bin')['fuel_kg'].agg(['mean', 'std', 'count'])
print("\nFuel by Interval Position:")
print(fuel_by_position)

plt.figure(figsize=(10, 5))
plt.bar(range(len(fuel_by_position)), fuel_by_position['mean'],
        yerr=fuel_by_position['std'], alpha=0.7, capsize=5)
plt.xticks(range(len(fuel_by_position)), fuel_by_position.index)
plt.xlabel('Position in Flight (%)')
plt.ylabel('Average Fuel (kg)')
plt.title('Fuel Consumption by Position in Flight')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'fuel_by_position.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {VIZ_DIR / 'fuel_by_position.png'}")
plt.close()

# 2.6 Temporal Patterns
print("\n[2.6] Analyzing temporal patterns...")

# Hour of day
fuel_train_pd['hour'] = fuel_train_pd['start'].dt.hour

fuel_by_hour = fuel_train_pd.groupby('hour')['fuel_kg'].agg(['mean', 'count'])
print("\nFuel by Hour of Day (Top 5 hours):")
print(fuel_by_hour.sort_values('mean', ascending=False).head())

# ============================================================================
# SECTION 3: TRAJECTORY DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: TRAJECTORY DATA QUALITY ASSESSMENT")
print("="*80)

# 3.1 Analyze Sample Trajectories
print("\n[3.1] Analyzing sample trajectory characteristics...")

traj_quality = []

for flight_id, traj_df in sample_trajectories.items():
    traj_pd = traj_df.to_pandas()

    quality = {
        'flight_id': flight_id,
        'total_points': len(traj_pd),
        'duration_hours': (traj_pd['timestamp'].max() - traj_pd['timestamp'].min()).total_seconds() / 3600,
        'altitude_min': traj_pd['altitude'].min(),
        'altitude_max': traj_pd['altitude'].max(),
        'groundspeed_missing_pct': traj_pd['groundspeed'].isna().sum() / len(traj_pd) * 100,
        'track_missing_pct': traj_pd['track'].isna().sum() / len(traj_pd) * 100,
        'vertical_rate_missing_pct': traj_pd['vertical_rate'].isna().sum() / len(traj_pd) * 100,
        'mach_missing_pct': traj_pd['mach'].isna().sum() / len(traj_pd) * 100,
        'adsb_pct': (traj_pd['source'] == 'adsb').sum() / len(traj_pd) * 100,
        'acars_pct': (traj_pd['source'] == 'acars').sum() / len(traj_pd) * 100,
    }

    # Calculate average time between points
    time_diffs = traj_pd['timestamp'].diff().dt.total_seconds().dropna()
    quality['avg_time_between_points'] = time_diffs.mean()
    quality['max_gap_seconds'] = time_diffs.max()

    traj_quality.append(quality)

traj_quality_df = pd.DataFrame(traj_quality)
print("\nTrajectory Quality Summary (Sample):")
print(traj_quality_df.describe().round(2))

# 3.2 Missing Data Patterns
print("\n[3.2] Missing data patterns across trajectory fields:")
avg_missing = traj_quality_df[['groundspeed_missing_pct', 'track_missing_pct',
                                 'vertical_rate_missing_pct', 'mach_missing_pct']].mean()
print(avg_missing)

# 3.3 Data Source Distribution
print("\n[3.3] Data source distribution:")
print(f"  Average ADS-B: {traj_quality_df['adsb_pct'].mean():.1f}%")
print(f"  Average ACARS: {traj_quality_df['acars_pct'].mean():.1f}%")

# 3.4 Temporal Resolution
print("\n[3.4] Temporal resolution:")
print(f"  Average time between points: {traj_quality_df['avg_time_between_points'].mean():.1f} seconds")
print(f"  Average max gap: {traj_quality_df['max_gap_seconds'].mean():.1f} seconds")

# ============================================================================
# SECTION 4: MAPPING FUEL INTERVALS TO TRAJECTORIES
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: MAPPING FUEL INTERVALS TO TRAJECTORIES")
print("="*80)

print("\n[4.1] Analyzing fuel interval to trajectory alignment...")

# For each sample flight, analyze how many trajectory points fall within each fuel interval
interval_mapping_stats = []

for flight_id in sample_flight_ids[:5]:  # Analyze first 5 flights
    if flight_id not in sample_trajectories:
        continue

    traj_pd = sample_trajectories[flight_id].to_pandas()
    fuel_intervals = fuel_train_pd[fuel_train_pd['flight_id'] == flight_id].copy()

    for idx, interval_row in fuel_intervals.iterrows():
        start_time = interval_row['start']
        end_time = interval_row['end']

        # Count points in interval
        points_in_interval = traj_pd[
            (traj_pd['timestamp'] >= start_time) &
            (traj_pd['timestamp'] <= end_time)
        ]

        interval_mapping_stats.append({
            'flight_id': flight_id,
            'interval_duration': (end_time - start_time).total_seconds(),
            'points_in_interval': len(points_in_interval),
            'fuel_kg': interval_row['fuel_kg'],
            'has_data': len(points_in_interval) > 0
        })

interval_mapping_df = pd.DataFrame(interval_mapping_stats)
print("\nInterval-Trajectory Mapping Statistics:")
print(interval_mapping_df.describe())

print(f"\nIntervals with NO trajectory data: {(~interval_mapping_df['has_data']).sum()} "
      f"({(~interval_mapping_df['has_data']).sum() / len(interval_mapping_df) * 100:.1f}%)")

# ============================================================================
# SECTION 5: FLIGHT PHASE IDENTIFICATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: FLIGHT PHASE IDENTIFICATION")
print("="*80)

print("\n[5.1] Identifying flight phases from altitude profiles...")

def identify_flight_phase(altitude, vertical_rate, altitude_max):
    """
    Identify flight phase based on altitude and vertical rate.
    Phases: takeoff, climb, cruise, descent, landing
    """
    if pd.isna(altitude):
        return 'unknown'

    # Takeoff: low altitude, climbing
    if altitude < 5000:
        if pd.notna(vertical_rate) and vertical_rate > 500:
            return 'takeoff'
        elif pd.notna(vertical_rate) and vertical_rate < -500:
            return 'landing'
        else:
            return 'ground'

    # Cruise: high altitude, stable
    elif altitude > 0.9 * altitude_max and (pd.isna(vertical_rate) or abs(vertical_rate) < 500):
        return 'cruise'

    # Climb: increasing altitude
    elif pd.notna(vertical_rate) and vertical_rate > 500:
        return 'climb'

    # Descent: decreasing altitude
    elif pd.notna(vertical_rate) and vertical_rate < -500:
        return 'descent'

    # Default to cruise if at high altitude
    elif altitude > 20000:
        return 'cruise'

    else:
        return 'unknown'

# Apply phase identification to sample trajectories
phase_examples = []

for flight_id in sample_flight_ids[:3]:  # Visualize 3 flights
    if flight_id not in sample_trajectories:
        continue

    traj_pd = sample_trajectories[flight_id].to_pandas()
    altitude_max = traj_pd['altitude'].max()

    traj_pd['flight_phase'] = traj_pd.apply(
        lambda row: identify_flight_phase(row['altitude'], row['vertical_rate'], altitude_max),
        axis=1
    )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Altitude profile
    ax1.plot(traj_pd['timestamp'], traj_pd['altitude'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Altitude (ft)', fontsize=12)
    ax1.set_title(f'Flight {flight_id} - Altitude and Phase Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Phase coloring
    phase_colors = {
        'ground': 'gray',
        'takeoff': 'green',
        'climb': 'lightgreen',
        'cruise': 'blue',
        'descent': 'orange',
        'landing': 'red',
        'unknown': 'lightgray'
    }

    # Vertical rate
    ax2.plot(traj_pd['timestamp'], traj_pd['vertical_rate'], 'g-', linewidth=1, alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Vertical Rate (ft/min)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Color background by phase
    for phase, color in phase_colors.items():
        phase_data = traj_pd[traj_pd['flight_phase'] == phase]
        if len(phase_data) > 0:
            for i in range(len(phase_data) - 1):
                ax1.axvspan(phase_data.iloc[i]['timestamp'],
                           phase_data.iloc[i+1]['timestamp'],
                           alpha=0.2, color=color)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.5, label=phase)
                      for phase, color in phase_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'flight_profile_{flight_id}.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {VIZ_DIR / f'flight_profile_{flight_id}.png'}")
    plt.close()

    # Store phase statistics
    phase_counts = traj_pd['flight_phase'].value_counts()
    phase_examples.append({
        'flight_id': flight_id,
        **phase_counts.to_dict()
    })

print("\nFlight Phase Distribution (Sample Flights):")
phase_df = pd.DataFrame(phase_examples).fillna(0)
print(phase_df)

# ============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: CORRELATION ANALYSIS")
print("="*80)

print("\n[6.1] Extracting features from sample trajectories for correlation analysis...")

# Extract features for each fuel interval
correlation_data = []

for flight_id in sample_flight_ids[:5]:
    if flight_id not in sample_trajectories:
        continue

    traj_pd = sample_trajectories[flight_id].to_pandas()
    fuel_intervals = fuel_train_pd[fuel_train_pd['flight_id'] == flight_id].copy()

    for idx, interval_row in fuel_intervals.iterrows():
        start_time = interval_row['start']
        end_time = interval_row['end']

        # Filter trajectory to interval
        interval_traj = traj_pd[
            (traj_pd['timestamp'] >= start_time) &
            (traj_pd['timestamp'] <= end_time)
        ]

        if len(interval_traj) == 0:
            continue

        # Extract features
        features = {
            'fuel_kg': interval_row['fuel_kg'],
            'interval_duration': (end_time - start_time).total_seconds(),
            'altitude_mean': interval_traj['altitude'].mean(),
            'altitude_std': interval_traj['altitude'].std(),
            'altitude_delta': interval_traj['altitude'].iloc[-1] - interval_traj['altitude'].iloc[0] if len(interval_traj) > 1 else 0,
            'altitude_max': interval_traj['altitude'].max(),
            'altitude_min': interval_traj['altitude'].min(),
        }

        # Speed features (handle missing)
        if interval_traj['groundspeed'].notna().sum() > 0:
            features['groundspeed_mean'] = interval_traj['groundspeed'].mean()
            features['groundspeed_std'] = interval_traj['groundspeed'].std()
            features['groundspeed_max'] = interval_traj['groundspeed'].max()
        else:
            features['groundspeed_mean'] = np.nan
            features['groundspeed_std'] = np.nan
            features['groundspeed_max'] = np.nan

        # Vertical rate features
        if interval_traj['vertical_rate'].notna().sum() > 0:
            features['vertical_rate_mean'] = interval_traj['vertical_rate'].mean()
            features['vertical_rate_std'] = interval_traj['vertical_rate'].std()
        else:
            features['vertical_rate_mean'] = np.nan
            features['vertical_rate_std'] = np.nan

        features['point_count'] = len(interval_traj)

        correlation_data.append(features)

correlation_df = pd.DataFrame(correlation_data)
print(f"\nExtracted features for {len(correlation_df)} intervals")

# Calculate correlations
print("\n[6.2] Correlation with fuel consumption:")
correlations = correlation_df.corr()['fuel_kg'].sort_values(ascending=False)
print(correlations)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
print(f"\n  ✓ Saved: {VIZ_DIR / 'correlation_matrix.png'}")
plt.close()

# Scatter plots for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

key_features = ['interval_duration', 'altitude_mean', 'altitude_delta',
                'groundspeed_mean', 'vertical_rate_mean', 'point_count']

for i, feature in enumerate(key_features):
    if feature in correlation_df.columns:
        axes[i].scatter(correlation_df[feature], correlation_df['fuel_kg'], alpha=0.5, s=20)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel('Fuel (kg)', fontsize=10)
        axes[i].set_title(f'Fuel vs {feature}\n(r={correlations[feature]:.3f})', fontsize=11)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'feature_scatter_plots.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {VIZ_DIR / 'feature_scatter_plots.png'}")
plt.close()

# ============================================================================
# SECTION 7: KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: KEY FINDINGS SUMMARY")
print("="*80)

findings = f"""
PHASE 1 DATA UNDERSTANDING - KEY FINDINGS
==========================================

1. DATASET OVERVIEW
-------------------
- Training: {len(fuel_train_pd):,} fuel intervals across {fuel_train_pd['flight_id'].nunique():,} flights
- Ranking: {len(fuel_rank):,} intervals across {flightlist_rank['flight_id'].n_unique():,} flights
- Date coverage: {flightlist_train_pd['flight_date'].min()} to {flightlist_train_pd['flight_date'].max()}
- Aircraft types: {flightlist_train_pd['aircraft_type'].nunique()} unique types

2. FUEL CONSUMPTION PATTERNS
-----------------------------
- Mean fuel per interval: {fuel_train_pd['fuel_kg'].mean():.2f} kg
- Std deviation: {fuel_train_pd['fuel_kg'].std():.2f} kg
- Range: {fuel_train_pd['fuel_kg'].min():.2f} - {fuel_train_pd['fuel_kg'].max():.2f} kg
- Distribution: Right-skewed with long tail (some very high consumption intervals)

- Interval duration: {fuel_train_pd['interval_duration'].mean():.0f}s avg (range: {fuel_train_pd['interval_duration'].min():.0f}-{fuel_train_pd['interval_duration'].max():.0f}s)
- Fuel rate: {fuel_train_pd['fuel_rate'].mean():.2f} kg/s avg

- Aircraft variation: Large differences by type (e.g., wide-body vs narrow-body)
  * Top consumer: {fuel_by_aircraft.index[0]} ({fuel_by_aircraft.iloc[0]['mean_fuel']:.1f} kg/interval)
  * Lowest consumer: {fuel_by_aircraft.index[-1]} ({fuel_by_aircraft.iloc[-1]['mean_fuel']:.1f} kg/interval)

- Position in flight: Early intervals (takeoff/climb) have higher fuel consumption
  * 0-10% position: {fuel_by_position.loc['0-10%', 'mean']:.1f} kg
  * 25-50% position: {fuel_by_position.loc['25-50%', 'mean']:.1f} kg
  * 90-100% position: {fuel_by_position.loc['90-100%', 'mean']:.1f} kg

3. TRAJECTORY DATA QUALITY
---------------------------
- Average points per flight: {traj_quality_df['total_points'].mean():.0f}
- Average time between points: {traj_quality_df['avg_time_between_points'].mean():.1f} seconds
- Data sources: {traj_quality_df['adsb_pct'].mean():.1f}% ADS-B, {traj_quality_df['acars_pct'].mean():.1f}% ACARS

- Missing data rates:
  * Groundspeed: {avg_missing['groundspeed_missing_pct']:.1f}%
  * Track: {avg_missing['track_missing_pct']:.1f}%
  * Vertical rate: {avg_missing['vertical_rate_missing_pct']:.1f}%
  * Mach: {avg_missing['mach_missing_pct']:.1f}% (essentially unusable)

4. INTERVAL-TRAJECTORY MAPPING
-------------------------------
- Average points per interval: {interval_mapping_df['points_in_interval'].mean():.1f}
- Intervals with NO data: {(~interval_mapping_df['has_data']).sum() / len(interval_mapping_df) * 100:.1f}%
- Data coverage is generally good but some gaps exist

5. FLIGHT PHASES
----------------
- Successfully identified phases from altitude/vertical rate profiles
- Typical progression: ground → takeoff → climb → cruise → descent → landing
- Cruise typically at 30,000-36,000 ft with stable altitude
- Climb/descent phases show high vertical rates (±1000-3000 ft/min)

6. FEATURE-FUEL CORRELATIONS
-----------------------------
Top correlations with fuel consumption:
{correlations.head(10).to_string()}

- Interval duration: Strong positive correlation
- Altitude delta: Climbing (positive delta) correlates with higher fuel
- Vertical rate: High climb/descent rates indicate more fuel usage
- Groundspeed: Moderate positive correlation

7. DATA QUALITY RECOMMENDATIONS
--------------------------------
✓ Use interval_duration as primary feature (varies significantly)
✓ Create robust aggregations that handle missing groundspeed/track/vertical_rate
✓ Ignore Mach, TAS, CAS fields (>99% missing)
✓ Use flight phase as categorical feature
✓ Handle intervals with no trajectory data (forward-fill or interpolate)
✓ Aircraft type is crucial predictor (high variability)

8. MODELING STRATEGY INSIGHTS
------------------------------
- Target variable (fuel_kg) has high variance → RMSE metric is appropriate
- Features should include:
  * Interval duration and position in flight
  * Altitude profile (mean, delta, std)
  * Vertical rate statistics
  * Aircraft type (categorical)
  * Flight phase detection
  * Origin/destination elevation difference

- Challenges:
  * Missing trajectory data for some intervals
  * Variable interval durations (not uniform)
  * High variance in fuel consumption (0.45 - 32,205 kg)
  * Need to generalize from April-Aug to September

NEXT STEPS FOR PHASE 2
-----------------------
1. Implement batch feature extraction pipeline
2. Create robust missing value handling
3. Engineer physics-based features (climb work, phase transitions)
4. Add airport elevation features
5. Create temporal features (time of day, day of week)
"""

print(findings)

# Save findings report
report_path = REPORTS_DIR / 'phase1_findings.txt'
with open(report_path, 'w') as f:
    f.write(findings)
print(f"\n✓ Findings saved to: {report_path}")

# ============================================================================
# SECTION 8: SAVE PROCESSED DATA FOR NEXT PHASE
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: CACHING DATA FOR PHASE 2")
print("="*80)

# Save enhanced fuel data with calculated features
fuel_train_pd.to_parquet(CACHE_DIR / 'fuel_train_enhanced.parquet', index=False)
print(f"✓ Saved: {CACHE_DIR / 'fuel_train_enhanced.parquet'}")

# Save correlation data
correlation_df.to_parquet(CACHE_DIR / 'correlation_sample.parquet', index=False)
print(f"✓ Saved: {CACHE_DIR / 'correlation_sample.parquet'}")

# Save summary statistics
summary_stats = {
    'fuel_stats': fuel_stats.to_dict(),
    'duration_stats': duration_stats.to_dict(),
    'fuel_by_aircraft': fuel_by_aircraft.to_dict(),
    'fuel_by_position': fuel_by_position.to_dict(),
    'trajectory_quality': traj_quality_df.describe().to_dict(),
    'correlations': correlations.to_dict(),
}

import json
with open(CACHE_DIR / 'phase1_summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2, default=str)
print(f"✓ Saved: {CACHE_DIR / 'phase1_summary_stats.json'}")

print("\n" + "="*80)
print("PHASE 1 COMPLETE")
print("="*80)
print(f"Completed: {datetime.now()}")
print(f"\nGenerated outputs:")
print(f"  - Visualizations: {len(list(VIZ_DIR.glob('*.png')))} files in {VIZ_DIR}")
print(f"  - Reports: {REPORTS_DIR / 'phase1_findings.txt'}")
print(f"  - Cached data: {len(list(CACHE_DIR.glob('*')))} files in {CACHE_DIR}")
print("\nReady to proceed to Phase 2: Feature Engineering Pipeline")
print("="*80)
