"""
Utility functions for loading and preparing traffic congestion data.

The goal is to keep this module beginner-friendly and easy to follow,
so each transformation step is broken out into its own function.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Reasonable bounds for quality checks (values outside these ranges are unusual for city traffic)
MIN_SPEED = 0.0
MAX_SPEED = 120.0  # km/h
MIN_VEHICLE_COUNT = 0


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the traffic dataset from disk.

    Parameters
    ----------
    csv_path : str
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw pandas DataFrame with parsed timestamps.
    """
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing or abnormal values.

    Steps:
      * Remove duplicate rows
      * Drop rows missing critical fields
      * Clip speeds and vehicle counts to realistic ranges
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    required_columns = ["latitude", "longitude", "speed(kmph)", "vehicle_count"]
    cleaned = cleaned.dropna(subset=required_columns)

    cleaned = cleaned[
        (cleaned["speed(kmph)"] >= MIN_SPEED)
        & (cleaned["speed(kmph)"] <= MAX_SPEED)
        & (cleaned["vehicle_count"] >= MIN_VEHICLE_COUNT)
    ]

    cleaned["speed(kmph)"] = cleaned["speed(kmph)"].clip(MIN_SPEED, MAX_SPEED)

    # Remove extremely large outliers in vehicle count (keep it simple with a percentile clip)
    upper_vehicle = cleaned["vehicle_count"].quantile(0.99)
    cleaned["vehicle_count"] = cleaned["vehicle_count"].clip(MIN_VEHICLE_COUNT, upper_vehicle)

    # Fill any missing timestamps with forward fill so feature engineering keeps an hour value
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = cleaned["timestamp"].ffill()

    return cleaned


def add_congestion_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the congestion_level label using the provided speed ranges.
    """
    labeled = df.copy()
    speeds = labeled["speed(kmph)"]

    conditions = [
        speeds > 40,
        (speeds >= 20) & (speeds <= 40),
        speeds < 20,
    ]
    choices = ["Low", "Medium", "High"]
    labeled["congestion_level"] = np.select(conditions, choices, default="Medium")

    return labeled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple time-based features that may help the model.
    """
    engineered = df.copy()
    if "timestamp" in engineered.columns:
        engineered["hour_of_day"] = engineered["timestamp"].dt.hour.fillna(0).astype(int)
    else:
        engineered["hour_of_day"] = 0

    return engineered


def prepare_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Load, clean, label, and prepare the dataset for modeling.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Congestion level labels.
    feature_columns : list[str]
        Ordered list of feature names used for modeling.
    processed_df : pd.DataFrame
        Final processed dataframe (useful for EDA or visualizations).
    """
    raw_df = load_data(csv_path)
    cleaned_df = clean_data(raw_df)
    labeled_df = add_congestion_label(cleaned_df)
    feature_df = engineer_features(labeled_df)

    feature_columns = ["latitude", "longitude", "speed(kmph)", "vehicle_count", "hour_of_day"]
    X = feature_df[feature_columns]
    y = feature_df["congestion_level"]

    return X, y, feature_columns, feature_df


def get_location_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Helper to extract min/max latitude and longitude for simulation.
    """
    return {
        "latitude": (df["latitude"].min(), df["latitude"].max()),
        "longitude": (df["longitude"].min(), df["longitude"].max()),
    }


