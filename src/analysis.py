"""
Time-based analysis and statistical utilities for traffic congestion data.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

import pandas as pd
import numpy as np


def analyze_peak_hours(df: pd.DataFrame) -> Dict:
    """
    Analyze peak traffic hours from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'hour_of_day' and 'congestion_level' columns
        
    Returns
    -------
    Dict
        Dictionary with peak hour analysis results
    """
    if 'hour_of_day' not in df.columns:
        return {"error": "hour_of_day column not found"}
    
    # Count congestion by hour
    hour_analysis = df.groupby(['hour_of_day', 'congestion_level']).size().unstack(fill_value=0)
    
    # Find peak hours for each congestion level
    peak_high = hour_analysis['High'].idxmax() if 'High' in hour_analysis.columns else None
    peak_medium = hour_analysis['Medium'].idxmax() if 'Medium' in hour_analysis.columns else None
    peak_low = hour_analysis['Low'].idxmax() if 'Low' in hour_analysis.columns else None
    
    # Calculate average congestion by hour
    congestion_scores = {'Low': 1, 'Medium': 2, 'High': 3}
    df['congestion_score'] = df['congestion_level'].map(congestion_scores)
    avg_congestion_by_hour = df.groupby('hour_of_day')['congestion_score'].mean()
    
    return {
        "peak_high_congestion_hour": int(peak_high) if peak_high is not None else None,
        "peak_medium_congestion_hour": int(peak_medium) if peak_medium is not None else None,
        "peak_low_congestion_hour": int(peak_low) if peak_low is not None else None,
        "average_congestion_by_hour": avg_congestion_by_hour.to_dict(),
        "hourly_breakdown": hour_analysis.to_dict()
    }


def generate_statistics(predictions: List[Dict], processed_df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive statistics about predictions and training data.
    
    Parameters
    ----------
    predictions : List[Dict]
        List of prediction results
    processed_df : pd.DataFrame
        Processed training data
        
    Returns
    -------
    Dict
        Dictionary with various statistics
    """
    pred_df = pd.DataFrame(predictions)
    
    stats = {
        "predictions": {
            "total_locations": len(predictions),
            "congestion_distribution": dict(Counter(pred_df['congestion_level'])),
            "average_speed": float(pred_df['speed(kmph)'].mean()),
            "average_vehicle_count": float(pred_df['vehicle_count'].mean()),
            "speed_range": {
                "min": float(pred_df['speed(kmph)'].min()),
                "max": float(pred_df['speed(kmph)'].max()),
                "std": float(pred_df['speed(kmph)'].std())
            },
            "vehicle_count_range": {
                "min": int(pred_df['vehicle_count'].min()),
                "max": int(pred_df['vehicle_count'].max()),
                "std": float(pred_df['vehicle_count'].std())
            }
        },
        "training_data": {
            "total_samples": len(processed_df),
            "congestion_distribution": dict(Counter(processed_df['congestion_level'])),
            "average_speed": float(processed_df['speed(kmph)'].mean()),
            "average_vehicle_count": float(processed_df['vehicle_count'].mean()),
            "date_range": {
                "earliest": str(processed_df['timestamp'].min()) if 'timestamp' in processed_df.columns else None,
                "latest": str(processed_df['timestamp'].max()) if 'timestamp' in processed_df.columns else None
            }
        }
    }
    
    # Add peak hour analysis if available
    if 'hour_of_day' in processed_df.columns:
        stats["peak_hours"] = analyze_peak_hours(processed_df)
    
    return stats


def print_statistics_report(stats: Dict) -> None:
    """
    Print a formatted statistics report to console.
    
    Parameters
    ----------
    stats : Dict
        Statistics dictionary from generate_statistics()
    """
    print("\n" + "="*60)
    print("TRAFFIC CONGESTION ANALYSIS REPORT")
    print("="*60)
    
    print("\n📊 PREDICTION STATISTICS")
    print("-" * 60)
    pred_stats = stats['predictions']
    print(f"Total Locations Analyzed: {pred_stats['total_locations']}")
    print(f"\nCongestion Distribution:")
    for level, count in pred_stats['congestion_distribution'].items():
        percentage = (count / pred_stats['total_locations']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    print(f"\nSpeed Statistics:")
    print(f"  Average: {pred_stats['average_speed']:.2f} km/h")
    print(f"  Range: {pred_stats['speed_range']['min']:.1f} - {pred_stats['speed_range']['max']:.1f} km/h")
    print(f"  Std Dev: {pred_stats['speed_range']['std']:.2f} km/h")
    
    print(f"\nVehicle Count Statistics:")
    print(f"  Average: {pred_stats['average_vehicle_count']:.1f} vehicles")
    print(f"  Range: {pred_stats['vehicle_count_range']['min']} - {pred_stats['vehicle_count_range']['max']} vehicles")
    
    print("\n📈 TRAINING DATA STATISTICS")
    print("-" * 60)
    train_stats = stats['training_data']
    print(f"Total Training Samples: {train_stats['total_samples']}")
    print(f"Average Speed: {train_stats['average_speed']:.2f} km/h")
    print(f"Average Vehicle Count: {train_stats['average_vehicle_count']:.1f} vehicles")
    
    if 'peak_hours' in stats:
        print("\n⏰ PEAK HOUR ANALYSIS")
        print("-" * 60)
        peak = stats['peak_hours']
        if peak.get('peak_high_congestion_hour') is not None:
            print(f"Peak High Congestion Hour: {peak['peak_high_congestion_hour']}:00")
        if peak.get('peak_medium_congestion_hour') is not None:
            print(f"Peak Medium Congestion Hour: {peak['peak_medium_congestion_hour']}:00")
        if peak.get('peak_low_congestion_hour') is not None:
            print(f"Peak Low Congestion Hour: {peak['peak_low_congestion_hour']}:00")
    
    print("\n" + "="*60 + "\n")

