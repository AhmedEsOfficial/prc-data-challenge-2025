"""
Feature Extraction Module for Fuel Prediction
==============================================
Comprehensive feature engineering from trajectory and flight data.

Features are organized into categories:
1. Interval-based features
2. Trajectory aggregation features
3. Flight-level features
4. Physics-derived features

Author: ML Team
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils import (
    haversine_distance, identify_flight_phase, safe_mean, safe_std,
    safe_min, safe_max, safe_percentile, detect_phase_transition,
    encode_time_cyclical, calculate_climb_work, interpolate_missing_trajectory_data,
    calculate_trajectory_smoothness, get_altitude_band, calculate_speed_acceleration
)


def extract_interval_features(fuel_row):
    """
    Extract features from the fuel interval itself (metadata).

    Args:
        fuel_row: Row from fuel dataframe with start, end, flight_id, etc.

    Returns:
        Dictionary of interval features
    """
    features = {}

    # Interval duration (CRITICAL FEATURE - 0.986 correlation)
    start_time = pd.to_datetime(fuel_row['start'])
    end_time = pd.to_datetime(fuel_row['end'])
    duration = (end_time - start_time).total_seconds()

    features['interval_duration_sec'] = duration
    features['interval_duration_min'] = duration / 60

    # Time-based features
    features['start_hour'] = start_time.hour
    features['start_day_of_week'] = start_time.dayofweek
    features['start_day_of_year'] = start_time.dayofyear
    features['start_month'] = start_time.month

    # Cyclical encoding of hour
    hour_sin, hour_cos = encode_time_cyclical(start_time.hour)
    features['start_hour_sin'] = hour_sin
    features['start_hour_cos'] = hour_cos

    # Time of day category
    hour = start_time.hour
    if 6 <= hour < 12:
        features['time_of_day'] = 'morning'
    elif 12 <= hour < 18:
        features['time_of_day'] = 'afternoon'
    elif 18 <= hour < 22:
        features['time_of_day'] = 'evening'
    else:
        features['time_of_day'] = 'night'

    return features


def extract_trajectory_features(traj_df, start_time, end_time):
    """
    Extract aggregated features from trajectory data within the interval.

    Args:
        traj_df: Full trajectory dataframe for the flight
        start_time: Interval start timestamp
        end_time: Interval end timestamp

    Returns:
        Dictionary of trajectory features
    """
    features = {}

    # Filter trajectory to interval
    mask = (traj_df['timestamp'] >= start_time) & (traj_df['timestamp'] <= end_time)
    interval_traj = traj_df[mask].copy()

    # Data quality features
    features['traj_point_count'] = len(interval_traj)
    features['traj_has_data'] = len(interval_traj) > 0

    # If no data, return with NaN features
    if len(interval_traj) == 0:
        return _fill_missing_trajectory_features(features)

    # Optionally interpolate missing values
    interval_traj = interpolate_missing_trajectory_data(interval_traj)

    # ===== ALTITUDE FEATURES =====
    altitude = interval_traj['altitude'].dropna()

    if len(altitude) > 0:
        features['altitude_mean'] = safe_mean(altitude)
        features['altitude_std'] = safe_std(altitude)
        features['altitude_min'] = safe_min(altitude)
        features['altitude_max'] = safe_max(altitude)
        features['altitude_range'] = safe_max(altitude) - safe_min(altitude)
        features['altitude_median'] = safe_percentile(altitude, 0.5)
        features['altitude_p25'] = safe_percentile(altitude, 0.25)
        features['altitude_p75'] = safe_percentile(altitude, 0.75)

        # Altitude change
        if len(altitude) > 1:
            features['altitude_delta'] = altitude.iloc[-1] - altitude.iloc[0]
            features['altitude_delta_abs'] = abs(features['altitude_delta'])
        else:
            features['altitude_delta'] = 0.0
            features['altitude_delta_abs'] = 0.0

        # Time spent in altitude bands
        total_points = len(altitude)
        features['pct_below_5k'] = (altitude < 5000).sum() / total_points * 100
        features['pct_5k_10k'] = ((altitude >= 5000) & (altitude < 10000)).sum() / total_points * 100
        features['pct_10k_20k'] = ((altitude >= 10000) & (altitude < 20000)).sum() / total_points * 100
        features['pct_20k_30k'] = ((altitude >= 20000) & (altitude < 30000)).sum() / total_points * 100
        features['pct_above_30k'] = (altitude >= 30000).sum() / total_points * 100

        # Dominant altitude band
        altitude_mean = features['altitude_mean']
        features['altitude_band'] = get_altitude_band(altitude_mean)
    else:
        features.update(_fill_missing_altitude_features())

    # ===== GROUNDSPEED FEATURES =====
    groundspeed = interval_traj['groundspeed'].dropna()

    if len(groundspeed) > 0:
        features['groundspeed_mean'] = safe_mean(groundspeed)
        features['groundspeed_std'] = safe_std(groundspeed)
        features['groundspeed_min'] = safe_min(groundspeed)
        features['groundspeed_max'] = safe_max(groundspeed)
        features['groundspeed_range'] = safe_max(groundspeed) - safe_min(groundspeed)
        features['groundspeed_median'] = safe_percentile(groundspeed, 0.5)

        # Speed change
        if len(groundspeed) > 1:
            features['groundspeed_delta'] = groundspeed.iloc[-1] - groundspeed.iloc[0]
            features['groundspeed_delta_abs'] = abs(features['groundspeed_delta'])

            # Acceleration
            features['groundspeed_acceleration'] = calculate_speed_acceleration(
                groundspeed.values,
                interval_traj.loc[groundspeed.index, 'timestamp'].values
            )
        else:
            features['groundspeed_delta'] = 0.0
            features['groundspeed_delta_abs'] = 0.0
            features['groundspeed_acceleration'] = 0.0

        # Missing data indicator
        features['groundspeed_missing_pct'] = (interval_traj['groundspeed'].isna().sum() /
                                                len(interval_traj) * 100)
    else:
        features.update(_fill_missing_groundspeed_features())
        features['groundspeed_missing_pct'] = 100.0

    # ===== VERTICAL RATE FEATURES =====
    vertical_rate = interval_traj['vertical_rate'].dropna()

    if len(vertical_rate) > 0:
        features['vertical_rate_mean'] = safe_mean(vertical_rate)
        features['vertical_rate_std'] = safe_std(vertical_rate)
        features['vertical_rate_min'] = safe_min(vertical_rate)
        features['vertical_rate_max'] = safe_max(vertical_rate)
        features['vertical_rate_range'] = safe_max(vertical_rate) - safe_min(vertical_rate)
        features['vertical_rate_abs_mean'] = safe_mean(vertical_rate.abs())

        # Time spent climbing vs descending
        total_vr_points = len(vertical_rate)
        features['pct_climbing'] = (vertical_rate > 500).sum() / total_vr_points * 100
        features['pct_descending'] = (vertical_rate < -500).sum() / total_vr_points * 100
        features['pct_level'] = ((vertical_rate >= -500) & (vertical_rate <= 500)).sum() / total_vr_points * 100

        # Climbing intensity
        climbing_vr = vertical_rate[vertical_rate > 0]
        descending_vr = vertical_rate[vertical_rate < 0]

        features['climb_rate_mean'] = safe_mean(climbing_vr) if len(climbing_vr) > 0 else 0.0
        features['descent_rate_mean'] = safe_mean(descending_vr) if len(descending_vr) > 0 else 0.0

        features['vertical_rate_missing_pct'] = (interval_traj['vertical_rate'].isna().sum() /
                                                  len(interval_traj) * 100)
    else:
        features.update(_fill_missing_vertical_rate_features())
        features['vertical_rate_missing_pct'] = 100.0

    # ===== POSITION/DISTANCE FEATURES =====
    if len(interval_traj) >= 2:
        # Distance covered (haversine)
        lat_start = interval_traj.iloc[0]['latitude']
        lon_start = interval_traj.iloc[0]['longitude']
        lat_end = interval_traj.iloc[-1]['latitude']
        lon_end = interval_traj.iloc[-1]['longitude']

        features['distance_covered_km'] = haversine_distance(lat_start, lon_start, lat_end, lon_end)

        # Trajectory smoothness
        features['trajectory_smoothness'] = calculate_trajectory_smoothness(
            interval_traj['latitude'].values,
            interval_traj['longitude'].values
        )
    else:
        features['distance_covered_km'] = 0.0
        features['trajectory_smoothness'] = 0.0

    # ===== TRACK/HEADING FEATURES =====
    track = interval_traj['track'].dropna()

    if len(track) > 1:
        # Heading change
        features['heading_change'] = abs(track.iloc[-1] - track.iloc[0])

        # Normalize to [0, 180]
        if features['heading_change'] > 180:
            features['heading_change'] = 360 - features['heading_change']

        # Heading variability
        features['heading_std'] = safe_std(track)

        # Number of significant heading changes (> 15 degrees)
        track_changes = track.diff().abs()
        features['num_heading_changes'] = (track_changes > 15).sum()
    else:
        features['heading_change'] = 0.0
        features['heading_std'] = 0.0
        features['num_heading_changes'] = 0

    return features


def extract_flight_level_features(fuel_row, flightlist_row, airports_df):
    """
    Extract flight-level features (aircraft type, route, etc.).

    Args:
        fuel_row: Row from fuel dataframe
        flightlist_row: Corresponding row from flight list
        airports_df: Airport reference data

    Returns:
        Dictionary of flight-level features
    """
    features = {}

    # Aircraft type
    features['aircraft_type'] = flightlist_row['aircraft_type']

    # Origin and destination
    features['origin_icao'] = flightlist_row['origin_icao']
    features['destination_icao'] = flightlist_row['destination_icao']

    # Flight duration
    takeoff = pd.to_datetime(flightlist_row['takeoff'])
    landed = pd.to_datetime(flightlist_row['landed'])
    flight_duration = (landed - takeoff).total_seconds()

    features['flight_duration_hours'] = flight_duration / 3600
    features['flight_duration_min'] = flight_duration / 60

    # Interval position in flight
    start_time = pd.to_datetime(fuel_row['start'])
    seconds_from_takeoff = (start_time - takeoff).total_seconds()
    seconds_to_landing = (landed - start_time).total_seconds()

    features['seconds_from_takeoff'] = seconds_from_takeoff
    features['seconds_to_landing'] = seconds_to_landing

    # Interval position percentage
    if flight_duration > 0:
        features['interval_position_pct'] = (seconds_from_takeoff / flight_duration) * 100
    else:
        features['interval_position_pct'] = 0.0

    # Position category
    pos_pct = features['interval_position_pct']
    if pos_pct < 10:
        features['flight_phase_position'] = 'early'
    elif pos_pct < 25:
        features['flight_phase_position'] = 'early-mid'
    elif pos_pct < 50:
        features['flight_phase_position'] = 'mid'
    elif pos_pct < 75:
        features['flight_phase_position'] = 'mid-late'
    elif pos_pct < 90:
        features['flight_phase_position'] = 'late'
    else:
        features['flight_phase_position'] = 'final'

    # Airport elevation data
    origin_airport = airports_df[airports_df['icao'] == flightlist_row['origin_icao']]
    dest_airport = airports_df[airports_df['icao'] == flightlist_row['destination_icao']]

    if len(origin_airport) > 0:
        features['origin_elevation_ft'] = origin_airport.iloc[0]['elevation']
        features['origin_lat'] = origin_airport.iloc[0]['latitude']
        features['origin_lon'] = origin_airport.iloc[0]['longitude']
    else:
        features['origin_elevation_ft'] = np.nan
        features['origin_lat'] = np.nan
        features['origin_lon'] = np.nan

    if len(dest_airport) > 0:
        features['dest_elevation_ft'] = dest_airport.iloc[0]['elevation']
        features['dest_lat'] = dest_airport.iloc[0]['latitude']
        features['dest_lon'] = dest_airport.iloc[0]['longitude']
    else:
        features['dest_elevation_ft'] = np.nan
        features['dest_lat'] = np.nan
        features['dest_lon'] = np.nan

    # Elevation difference
    if pd.notna(features['origin_elevation_ft']) and pd.notna(features['dest_elevation_ft']):
        features['elevation_delta_ft'] = features['dest_elevation_ft'] - features['origin_elevation_ft']
        features['elevation_delta_abs_ft'] = abs(features['elevation_delta_ft'])
    else:
        features['elevation_delta_ft'] = 0.0
        features['elevation_delta_abs_ft'] = 0.0

    # Great circle distance
    if (pd.notna(features['origin_lat']) and pd.notna(features['origin_lon']) and
        pd.notna(features['dest_lat']) and pd.notna(features['dest_lon'])):
        features['great_circle_distance_km'] = haversine_distance(
            features['origin_lat'], features['origin_lon'],
            features['dest_lat'], features['dest_lon']
        )
    else:
        features['great_circle_distance_km'] = np.nan

    return features


def extract_physics_features(traj_features, flight_features):
    """
    Extract physics-based derived features.

    Args:
        traj_features: Dictionary of trajectory features
        flight_features: Dictionary of flight-level features

    Returns:
        Dictionary of physics-derived features
    """
    features = {}

    # Climb work (energy expended in altitude change)
    if 'altitude_delta' in traj_features:
        features['climb_work_mj'] = calculate_climb_work(traj_features['altitude_delta'])
        features['climb_work_abs_mj'] = abs(features['climb_work_mj'])
    else:
        features['climb_work_mj'] = 0.0
        features['climb_work_abs_mj'] = 0.0

    # Average speed (distance / time)
    if ('distance_covered_km' in traj_features and
        'interval_duration_sec' in traj_features and
        traj_features['interval_duration_sec'] > 0):

        # Convert km to nm (nautical miles) and seconds to hours
        distance_nm = traj_features['distance_covered_km'] * 0.539957
        time_hours = traj_features['interval_duration_sec'] / 3600

        features['avg_speed_over_ground_kt'] = distance_nm / time_hours if time_hours > 0 else 0.0
    else:
        features['avg_speed_over_ground_kt'] = 0.0

    # Detect flight phase from altitude and vertical rate
    if ('altitude_mean' in traj_features and
        'vertical_rate_mean' in traj_features and
        'altitude_max' in flight_features):

        features['detected_flight_phase'] = identify_flight_phase(
            traj_features.get('altitude_mean'),
            traj_features.get('vertical_rate_mean'),
            flight_features.get('altitude_max', traj_features.get('altitude_mean', 30000))
        )
    else:
        features['detected_flight_phase'] = 'unknown'

    # Cruise altitude deviation (how far from typical cruise altitude)
    if 'altitude_mean' in traj_features:
        # Typical cruise altitude is 30,000-40,000 ft
        typical_cruise = 35000
        features['cruise_altitude_deviation'] = abs(traj_features['altitude_mean'] - typical_cruise)
    else:
        features['cruise_altitude_deviation'] = np.nan

    # Efficiency metric: distance per altitude change
    if ('distance_covered_km' in traj_features and
        'altitude_delta_abs' in traj_features and
        traj_features['altitude_delta_abs'] > 0):

        features['distance_per_alt_change'] = (traj_features['distance_covered_km'] /
                                                (traj_features['altitude_delta_abs'] / 1000))
    else:
        features['distance_per_alt_change'] = 0.0

    # Vertical activity index (how much vertical maneuvering)
    if 'vertical_rate_std' in traj_features:
        features['vertical_activity_index'] = traj_features.get('vertical_rate_std', 0.0)
    else:
        features['vertical_activity_index'] = 0.0

    return features


def extract_all_features(fuel_row, traj_df, flightlist_row, airports_df):
    """
    Extract all feature categories for a single fuel interval.

    Args:
        fuel_row: Row from fuel dataframe
        traj_df: Full trajectory dataframe for the flight
        flightlist_row: Row from flight list
        airports_df: Airport reference data

    Returns:
        Dictionary of all features
    """
    all_features = {}

    # 1. Interval-based features
    interval_features = extract_interval_features(fuel_row)
    all_features.update(interval_features)

    # 2. Trajectory features
    start_time = pd.to_datetime(fuel_row['start'])
    end_time = pd.to_datetime(fuel_row['end'])
    traj_features = extract_trajectory_features(traj_df, start_time, end_time)
    all_features.update(traj_features)

    # 3. Flight-level features
    flight_features = extract_flight_level_features(fuel_row, flightlist_row, airports_df)
    all_features.update(flight_features)

    # Store max altitude for physics features
    if 'altitude_max' in traj_features and pd.notna(traj_features['altitude_max']):
        flight_features['altitude_max'] = traj_features['altitude_max']

    # 4. Physics-derived features
    physics_features = extract_physics_features(traj_features, flight_features)
    all_features.update(physics_features)

    # Add identifiers
    all_features['idx'] = fuel_row['idx']
    all_features['flight_id'] = fuel_row['flight_id']

    return all_features


# Helper functions for missing data
def _fill_missing_trajectory_features(features):
    """Fill with NaN for missing trajectory data."""
    missing_features = {
        'altitude_mean': np.nan,
        'altitude_std': np.nan,
        'altitude_min': np.nan,
        'altitude_max': np.nan,
        'altitude_range': np.nan,
        'altitude_median': np.nan,
        'altitude_p25': np.nan,
        'altitude_p75': np.nan,
        'altitude_delta': 0.0,
        'altitude_delta_abs': 0.0,
        'pct_below_5k': 0.0,
        'pct_5k_10k': 0.0,
        'pct_10k_20k': 0.0,
        'pct_20k_30k': 0.0,
        'pct_above_30k': 0.0,
        'altitude_band': 'unknown',
        'groundspeed_mean': np.nan,
        'groundspeed_std': np.nan,
        'groundspeed_min': np.nan,
        'groundspeed_max': np.nan,
        'groundspeed_range': np.nan,
        'groundspeed_median': np.nan,
        'groundspeed_delta': 0.0,
        'groundspeed_delta_abs': 0.0,
        'groundspeed_acceleration': 0.0,
        'groundspeed_missing_pct': 100.0,
        'vertical_rate_mean': np.nan,
        'vertical_rate_std': np.nan,
        'vertical_rate_min': np.nan,
        'vertical_rate_max': np.nan,
        'vertical_rate_range': np.nan,
        'vertical_rate_abs_mean': np.nan,
        'pct_climbing': 0.0,
        'pct_descending': 0.0,
        'pct_level': 0.0,
        'climb_rate_mean': 0.0,
        'descent_rate_mean': 0.0,
        'vertical_rate_missing_pct': 100.0,
        'distance_covered_km': 0.0,
        'trajectory_smoothness': 0.0,
        'heading_change': 0.0,
        'heading_std': 0.0,
        'num_heading_changes': 0,
    }
    features.update(missing_features)
    return features


def _fill_missing_altitude_features():
    """Fill altitude features for missing data."""
    return {
        'altitude_mean': np.nan,
        'altitude_std': np.nan,
        'altitude_min': np.nan,
        'altitude_max': np.nan,
        'altitude_range': np.nan,
        'altitude_median': np.nan,
        'altitude_p25': np.nan,
        'altitude_p75': np.nan,
        'altitude_delta': 0.0,
        'altitude_delta_abs': 0.0,
        'pct_below_5k': 0.0,
        'pct_5k_10k': 0.0,
        'pct_10k_20k': 0.0,
        'pct_20k_30k': 0.0,
        'pct_above_30k': 0.0,
        'altitude_band': 'unknown',
    }


def _fill_missing_groundspeed_features():
    """Fill groundspeed features for missing data."""
    return {
        'groundspeed_mean': np.nan,
        'groundspeed_std': np.nan,
        'groundspeed_min': np.nan,
        'groundspeed_max': np.nan,
        'groundspeed_range': np.nan,
        'groundspeed_median': np.nan,
        'groundspeed_delta': 0.0,
        'groundspeed_delta_abs': 0.0,
        'groundspeed_acceleration': 0.0,
    }


def _fill_missing_vertical_rate_features():
    """Fill vertical rate features for missing data."""
    return {
        'vertical_rate_mean': np.nan,
        'vertical_rate_std': np.nan,
        'vertical_rate_min': np.nan,
        'vertical_rate_max': np.nan,
        'vertical_rate_range': np.nan,
        'vertical_rate_abs_mean': np.nan,
        'pct_climbing': 0.0,
        'pct_descending': 0.0,
        'pct_level': 0.0,
        'climb_rate_mean': 0.0,
        'descent_rate_mean': 0.0,
    }
