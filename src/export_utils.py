"""
Export utilities for saving predictions and analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import json
import csv

import pandas as pd


def export_predictions_to_csv(predictions: List[Dict], output_path: Path) -> None:
    """
    Export predictions to CSV file.
    
    Parameters
    ----------
    predictions : List[Dict]
        List of prediction dictionaries
    output_path : Path
        Path to save CSV file
    """
    df = pd.DataFrame(predictions)
    # Reorder columns for better readability
    column_order = ['latitude', 'longitude', 'speed(kmph)', 'vehicle_count', 
                   'congestion_level', 'hour_of_day']
    df = df[[col for col in column_order if col in df.columns]]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions exported to CSV: {output_path}")


def export_predictions_to_json(predictions: List[Dict], output_path: Path) -> None:
    """
    Export predictions to JSON file.
    
    Parameters
    ----------
    predictions : List[Dict]
        List of prediction dictionaries
    output_path : Path
        Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"Predictions exported to JSON: {output_path}")


def export_statistics_to_json(stats: Dict, output_path: Path) -> None:
    """
    Export statistics to JSON file.
    
    Parameters
    ----------
    stats : Dict
        Statistics dictionary
    output_path : Path
        Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Statistics exported to JSON: {output_path}")

