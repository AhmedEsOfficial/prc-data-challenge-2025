"""
Utility Functions for Fuel Prediction Model
============================================
Helper functions for feature engineering and data processing.

Author: ML Team
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (decimal degrees)
        lat2, lon2: Latitude and longitude of point 2 (decimal degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Earth radius in kilometers
    R = 6371.0
    distance = R * c

    return distance


def identify_flight_phase(altitude, vertical_rate, altitude_max, groundspeed=None):
    """
    Identify flight phase based on altitude and vertical rate.

    Phases: ground, takeoff, climb, cruise, descent, landing, unknown

    Args:
        altitude: Current altitude (ft)
        vertical_rate: Vertical rate (ft/min)
        altitude_max: Maximum altitude in flight (ft)
        groundspeed: Optional groundspeed (kt)

    Returns:
        Flight phase as string
    """
    if pd.isna(altitude):
        return 'unknown'

    # Ground phase: very low altitude with low/no speed
    if altitude < 500:
        if groundspeed is not None and not pd.isna(groundspeed) and groundspeed < 50:
            return 'ground'
        elif pd.notna(vertical_rate) and vertical_rate < 100:
            return 'ground'

    # Takeoff: low altitude, climbing rapidly
    if altitude < 5000:
        if pd.notna(vertical_rate) and vertical_rate > 500:
            return 'takeoff'
        elif pd.notna(vertical_rate) and vertical_rate < -500:
            return 'landing'
        else:
            return 'takeoff'  # Default for low altitude

    # Landing: low altitude, descending
    if altitude < 10000:
        if pd.notna(vertical_rate) and vertical_rate < -300:
            return 'landing'

    # Cruise: high altitude, stable (within 10% of max, low vertical rate)
    if altitude > 0.85 * altitude_max:
        if pd.isna(vertical_rate) or abs(vertical_rate) < 500:
            return 'cruise'

    # Climb: increasing altitude
    if pd.notna(vertical_rate) and vertical_rate > 500:
        return 'climb'

    # Descent: decreasing altitude
    if pd.notna(vertical_rate) and vertical_rate < -500:
        return 'descent'

    # Default based on altitude
    if altitude > 20000:
        return 'cruise'
    elif altitude > 10000:
        return 'climb'
    else:
        return 'unknown'


def calculate_great_circle_distance(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Calculate great circle distance between origin and destination airports.

    Args:
        origin_lat, origin_lon: Origin airport coordinates
        dest_lat, dest_lon: Destination airport coordinates

    Returns:
        Distance in kilometers
    """
    return haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)


def safe_mean(series):
    """Calculate mean, handling empty or all-NaN series."""
    if len(series) == 0 or series.isna().all():
        return np.nan
    return series.mean()


def safe_std(series):
    """Calculate std, handling empty or all-NaN series."""
    if len(series) == 0 or series.isna().all():
        return np.nan
    return series.std()


def safe_min(series):
    """Calculate min, handling empty or all-NaN series."""
    if len(series) == 0 or series.isna().all():
        return np.nan
    return series.min()


def safe_max(series):
    """Calculate max, handling empty or all-NaN series."""
    if len(series) == 0 or series.isna().all():
        return np.nan
    return series.max()


def safe_percentile(series, q):
    """Calculate percentile, handling empty or all-NaN series."""
    if len(series) == 0 or series.isna().all():
        return np.nan
    return series.quantile(q)


def detect_phase_transition(phases):
    """
    Detect if interval contains a phase transition.

    Args:
        phases: List or array of flight phases

    Returns:
        Boolean indicating if there's a transition
    """
    if len(phases) == 0:
        return False

    unique_phases = pd.Series(phases).unique()
    return len(unique_phases) > 1


def calculate_fuel_rate_percentile(aircraft_type, fuel_rate, fuel_train_df):
    """
    Calculate what percentile this fuel rate is for the aircraft type.

    Args:
        aircraft_type: Aircraft type code
        fuel_rate: Fuel rate (kg/s)
        fuel_train_df: Training dataframe with fuel rates by aircraft

    Returns:
        Percentile (0-100)
    """
    aircraft_fuel_rates = fuel_train_df[
        fuel_train_df['aircraft_type'] == aircraft_type
    ]['fuel_rate']

    if len(aircraft_fuel_rates) == 0 or pd.isna(fuel_rate):
        return 50.0  # Default to median

    percentile = (aircraft_fuel_rates < fuel_rate).sum() / len(aircraft_fuel_rates) * 100
    return percentile


def encode_time_cyclical(hour):
    """
    Encode hour of day as cyclical features (sin/cos).

    Args:
        hour: Hour (0-23)

    Returns:
        Tuple of (sin_hour, cos_hour)
    """
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour_sin, hour_cos


def calculate_climb_work(altitude_delta, estimated_mass_kg=200000):
    """
    Calculate approximate work done during altitude change.

    Work = mass × gravity × height

    Args:
        altitude_delta: Change in altitude (ft)
        estimated_mass_kg: Estimated aircraft mass (default: 200,000 kg)

    Returns:
        Work in megajoules (MJ)
    """
    # Convert altitude from feet to meters
    altitude_delta_m = altitude_delta * 0.3048

    # Gravity (m/s^2)
    g = 9.81

    # Work in joules
    work_j = estimated_mass_kg * g * altitude_delta_m

    # Convert to megajoules
    work_mj = work_j / 1e6

    return work_mj


def interpolate_missing_trajectory_data(traj_df):
    """
    Interpolate missing values in trajectory data.

    Args:
        traj_df: Trajectory dataframe

    Returns:
        Dataframe with interpolated values
    """
    traj_df = traj_df.copy()

    # Interpolate numeric columns
    numeric_cols = ['altitude', 'groundspeed', 'track', 'vertical_rate']

    for col in numeric_cols:
        if col in traj_df.columns:
            # Linear interpolation
            traj_df[col] = traj_df[col].interpolate(method='linear', limit=5)

            # Forward fill for remaining gaps
            traj_df[col] = traj_df[col].fillna(method='ffill', limit=3)

            # Backward fill for leading NaNs
            traj_df[col] = traj_df[col].fillna(method='bfill', limit=3)

    return traj_df


def calculate_trajectory_smoothness(latitudes, longitudes):
    """
    Calculate trajectory smoothness based on heading changes.

    Args:
        latitudes: Array of latitudes
        longitudes: Array of longitudes

    Returns:
        Smoothness score (lower = smoother)
    """
    if len(latitudes) < 3:
        return 0.0

    # Calculate bearing between consecutive points
    bearings = []
    for i in range(len(latitudes) - 1):
        lat1, lon1 = radians(latitudes[i]), radians(longitudes[i])
        lat2, lon2 = radians(latitudes[i+1]), radians(longitudes[i+1])

        dlon = lon2 - lon1
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(x, y)
        bearings.append(bearing)

    # Calculate changes in bearing
    bearing_changes = np.diff(bearings)

    # Normalize to [-pi, pi]
    bearing_changes = np.arctan2(np.sin(bearing_changes), np.cos(bearing_changes))

    # Return mean absolute bearing change
    smoothness = np.abs(bearing_changes).mean()

    return smoothness


def get_altitude_band(altitude):
    """
    Categorize altitude into bands.

    Args:
        altitude: Altitude in feet

    Returns:
        Altitude band as string
    """
    if pd.isna(altitude):
        return 'unknown'
    elif altitude < 5000:
        return '0-5k'
    elif altitude < 10000:
        return '5-10k'
    elif altitude < 20000:
        return '10-20k'
    elif altitude < 30000:
        return '20-30k'
    elif altitude < 40000:
        return '30-40k'
    else:
        return '40k+'


def calculate_speed_acceleration(groundspeeds, timestamps):
    """
    Calculate acceleration from groundspeed changes.

    Args:
        groundspeeds: Array of groundspeeds (kt)
        timestamps: Array of timestamps

    Returns:
        Mean acceleration (kt/s)
    """
    if len(groundspeeds) < 2:
        return 0.0

    # Remove NaN values
    valid_idx = ~pd.isna(groundspeeds)
    speeds = np.array(groundspeeds)[valid_idx]
    times = pd.Series(timestamps)[valid_idx]

    if len(speeds) < 2:
        return 0.0

    # Calculate time differences in seconds using pandas
    # Convert to pandas Timestamp if not already, then to seconds
    times_pd = pd.to_datetime(times)
    time_diffs_td = times_pd.diff().dropna()
    time_diffs = time_diffs_td.dt.total_seconds().values

    # Calculate speed differences (skip first element to match time_diffs length)
    speed_diffs = np.diff(speeds)

    # Ensure arrays are same length
    if len(speed_diffs) != len(time_diffs):
        return 0.0

    # Avoid division by zero
    time_diffs = time_diffs[time_diffs > 0]
    speed_diffs = speed_diffs[:len(time_diffs)]

    if len(time_diffs) == 0:
        return 0.0

    # Acceleration
    accelerations = speed_diffs / time_diffs

    return np.mean(accelerations)


def print_feature_summary(features_df):
    """
    Print summary statistics for extracted features.

    Args:
        features_df: DataFrame of features
    """
    print("\n" + "="*80)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total samples: {len(features_df):,}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"\nMemory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\nMissing values:")
    missing = features_df.isna().sum()
    missing_pct = (missing / len(features_df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    })
    missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )

    if len(missing_summary) > 0:
        print(missing_summary.head(20))
    else:
        print("  No missing values!")

    print("\n" + "="*80)
