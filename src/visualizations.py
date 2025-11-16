"""
Advanced visualization utilities for traffic congestion analysis.
Creates charts, feature importance plots, and statistical dashboards.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_feature_importance(model, feature_names: List[str], output_path: Path) -> None:
    """
    Plot feature importance from the trained RandomForest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    feature_names : List[str]
        List of feature names
    output_path : Path
        Where to save the plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance in Traffic Congestion Prediction", fontsize=14, fontweight='bold')
    bars = plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.7)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel("Importance Score", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {output_path}")


def create_statistical_dashboard(
    predictions: List[Dict],
    processed_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create a comprehensive statistical dashboard with multiple visualizations.
    
    Parameters
    ----------
    predictions : List[Dict]
        List of prediction results
    processed_df : pd.DataFrame
        Processed training data
    output_dir : Path
        Directory to save all plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Congestion Distribution Pie Chart
    pred_df = pd.DataFrame(predictions)
    congestion_counts = Counter(pred_df['congestion_level'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Traffic Congestion Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Pie chart
    labels = list(congestion_counts.keys())
    sizes = list(congestion_counts.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Congestion Level Distribution', fontweight='bold')
    
    # Speed Distribution Histogram
    axes[0, 1].hist(pred_df['speed(kmph)'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=40, color='green', linestyle='--', label='Low threshold (40 km/h)')
    axes[0, 1].axvline(x=20, color='red', linestyle='--', label='High threshold (20 km/h)')
    axes[0, 1].set_xlabel('Speed (km/h)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Speed Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Vehicle Count vs Speed Scatter
    for level in ['Low', 'Medium', 'High']:
        subset = pred_df[pred_df['congestion_level'] == level]
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        axes[1, 0].scatter(subset['speed(kmph)'], subset['vehicle_count'], 
                          label=level, alpha=0.6, color=color_map[level], s=50)
    axes[1, 0].set_xlabel('Speed (km/h)', fontsize=10)
    axes[1, 0].set_ylabel('Vehicle Count', fontsize=10)
    axes[1, 0].set_title('Speed vs Vehicle Count by Congestion Level', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Congestion by Hour (if available in processed_df)
    if 'hour_of_day' in processed_df.columns:
        hour_congestion = processed_df.groupby(['hour_of_day', 'congestion_level']).size().unstack(fill_value=0)
        hour_congestion.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                            color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        axes[1, 1].set_xlabel('Hour of Day', fontsize=10)
        axes[1, 1].set_ylabel('Count', fontsize=10)
        axes[1, 1].set_title('Congestion by Hour of Day', fontweight='bold')
        axes[1, 1].legend(title='Congestion Level')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        # Bar chart of congestion counts
        bars = axes[1, 1].bar(congestion_counts.keys(), congestion_counts.values(), 
                              color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Congestion Level', fontsize=10)
        axes[1, 1].set_ylabel('Count', fontsize=10)
        axes[1, 1].set_title('Congestion Level Counts', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    dashboard_path = output_dir / "statistical_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Statistical dashboard saved to: {dashboard_path}")


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot correlation matrix for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical features
    output_path : Path
        Where to save the plot
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to: {output_path}")

