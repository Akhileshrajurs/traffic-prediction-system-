"""
Tools for simulating GPS input and predicting congestion levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional
import random

import joblib
import numpy as np
import pandas as pd


LABEL_TO_COLOR = {"Low": "green", "Medium": "orange", "High": "red"}


@dataclass
class ModelBundle:
    model: any
    label_encoder: any
    feature_columns: list[str]
    training_data_bounds: dict
    accuracy: float


def load_model(model_path: Path) -> ModelBundle:
    """
    Load the trained RandomForest artifacts.
    """
    artifacts = joblib.load(model_path)
    return ModelBundle(
        model=artifacts["model"],
        label_encoder=artifacts["label_encoder"],
        feature_columns=artifacts["feature_columns"],
        training_data_bounds=artifacts.get("training_data_bounds", {}),
        accuracy=artifacts.get("accuracy", 0.0),
    )


def simulate_gps_points(
    num_points: int,
    bounds: Optional[dict] = None,
    speed_range: tuple[float, float] | None = None,
    vehicle_range: tuple[int, int] | None = None,
) -> List[Dict[str, float]]:
    """
    Generate a batch of simulated GPS points near the training data bounds.
    """
    bounds = bounds or {
        "latitude": (12.96, 12.98),
        "longitude": (77.59, 77.61),
        "speed": (5.0, 55.0),
        "vehicle_count": (10, 45),
    }

    speed_min, speed_max = speed_range or bounds.get("speed", (5.0, 55.0))
    vehicle_min, vehicle_max = vehicle_range or bounds.get("vehicle_count", (10, 45))

    points = []
    for _ in range(num_points):
        lat = random.uniform(*bounds.get("latitude", (12.96, 12.98)))
        lon = random.uniform(*bounds.get("longitude", (77.59, 77.61)))
        speed = random.uniform(speed_min, speed_max)
        vehicle_count = random.randint(int(vehicle_min), int(vehicle_max))
        points.append(
            {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed(kmph)": round(speed, 2),
                "vehicle_count": int(vehicle_count),
            }
        )
    return points


def simulate_balanced_gps_points(
    num_per_level: int = 10,
    bounds: Optional[dict] = None,
) -> List[Dict[str, float]]:
    """
    Generate balanced GPS points: exactly num_per_level points for each congestion level.
    This ensures we get equal representation of Low, Medium, and High congestion.
    
    Parameters
    ----------
    num_per_level : int
        Number of points to generate for each congestion level (default: 10)
    bounds : Optional[dict]
        Geographic bounds for latitude/longitude
        
    Returns
    -------
    List[Dict[str, float]]
        List of GPS points with speeds designed to create balanced congestion levels
    """
    bounds = bounds or {
        "latitude": (12.96, 12.98),
        "longitude": (77.59, 77.61),
        "vehicle_count": (10, 50),
    }
    
    points = []
    
    # Generate Low congestion points (speed > 40 km/h)
    for _ in range(num_per_level):
        lat = random.uniform(*bounds.get("latitude", (12.96, 12.98)))
        lon = random.uniform(*bounds.get("longitude", (77.59, 77.61)))
        speed = random.uniform(41.0, 60.0)  # Low congestion: high speed
        vehicle_count = random.randint(10, 30)  # Lower vehicle count for free flow
        points.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed(kmph)": round(speed, 2),
            "vehicle_count": int(vehicle_count),
        })
    
    # Generate Medium congestion points (20-40 km/h)
    for _ in range(num_per_level):
        lat = random.uniform(*bounds.get("latitude", (12.96, 12.98)))
        lon = random.uniform(*bounds.get("longitude", (77.59, 77.61)))
        speed = random.uniform(20.0, 40.0)  # Medium congestion
        vehicle_count = random.randint(25, 45)  # Moderate vehicle count
        points.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed(kmph)": round(speed, 2),
            "vehicle_count": int(vehicle_count),
        })
    
    # Generate High congestion points (speed < 20 km/h)
    for _ in range(num_per_level):
        lat = random.uniform(*bounds.get("latitude", (12.96, 12.98)))
        lon = random.uniform(*bounds.get("longitude", (77.59, 77.61)))
        speed = random.uniform(5.0, 19.9)  # High congestion: low speed
        vehicle_count = random.randint(35, 55)  # Higher vehicle count
        points.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed(kmph)": round(speed, 2),
            "vehicle_count": int(vehicle_count),
        })
    
    # Shuffle to mix up the order
    random.shuffle(points)
    return points


def predict_congestion(
    model_bundle: ModelBundle,
    gps_points: Iterable[dict],
    hour_of_day: int | None = None,
    include_confidence: bool = True,
) -> List[Dict[str, any]]:
    """
    Use the trained model to predict congestion level for each GPS point.
    
    Parameters
    ----------
    model_bundle : ModelBundle
        Loaded model bundle
    gps_points : Iterable[dict]
        GPS points to predict
    hour_of_day : int | None
        Hour of day for predictions
    include_confidence : bool
        Whether to include prediction confidence scores
        
    Returns
    -------
    List[Dict[str, any]]
        List of predictions with congestion levels and optional confidence scores
    """
    features = []
    for point in gps_points:
        feature_row = {
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "speed(kmph)": point["speed(kmph)"],
            "vehicle_count": point["vehicle_count"],
            "hour_of_day": hour_of_day if hour_of_day is not None else 0,
        }
        features.append(feature_row)

    feature_df = pd.DataFrame(features, columns=model_bundle.feature_columns)

    encoded_predictions = model_bundle.model.predict(feature_df)
    labels = model_bundle.label_encoder.inverse_transform(encoded_predictions)
    
    # Get prediction probabilities for confidence scores
    if include_confidence:
        probabilities = model_bundle.model.predict_proba(feature_df)
        max_probs = probabilities.max(axis=1)
    else:
        max_probs = [None] * len(labels)

    results = []
    for point, label, confidence in zip(features, labels, max_probs):
        result = {
            **point,
            "congestion_level": label,
            "marker_color": LABEL_TO_COLOR.get(label, "blue"),
        }
        if include_confidence and confidence is not None:
            result["confidence"] = float(confidence)
        results.append(result)
    return results


