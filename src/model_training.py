"""
Model training script for the traffic congestion project.

Run this module directly to train a RandomForest model and persist it to disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import prepare_dataset
from visualizations import plot_feature_importance, plot_correlation_matrix

DEFAULT_DATA_PATH = Path("data") / "traffic_data.csv"
DEFAULT_MODEL_PATH = Path("models") / "traffic_congestion_model.joblib"
DEFAULT_PLOT_PATH = Path("reports") / "speed_vs_vehicle.png"
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("reports") / "feature_importance.png"
DEFAULT_CORRELATION_PATH = Path("reports") / "correlation_matrix.png"


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
) -> dict:
    """
    Train the RandomForestClassifier and persist it with metadata.

    Returns
    -------
    dict
        Dictionary containing model, label encoder, feature order, and accuracy.
    """
    X, y, feature_columns, processed_df = prepare_dataset(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    clf.fit(X_train, y_train_encoded)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)

    print("Model Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _create_diagnostic_plot(processed_df, plot_path)
    print(f"Saved diagnostic plot to: {plot_path.resolve()}")
    
    # Create feature importance plot
    feature_importance_path = plot_path.parent / "feature_importance.png"
    plot_feature_importance(clf, feature_columns, feature_importance_path)
    
    # Create correlation matrix
    correlation_path = plot_path.parent / "correlation_matrix.png"
    plot_correlation_matrix(processed_df, correlation_path)

    model_artifacts = {
        "model": clf,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "training_data_bounds": {
            "latitude": (processed_df["latitude"].min(), processed_df["latitude"].max()),
            "longitude": (processed_df["longitude"].min(), processed_df["longitude"].max()),
            "speed": (processed_df["speed(kmph)"].min(), processed_df["speed(kmph)"].max()),
            "vehicle_count": (
                processed_df["vehicle_count"].min(),
                processed_df["vehicle_count"].max(),
            ),
        },
        "accuracy": accuracy,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifacts, model_path)
    print(f"\nModel saved to: {model_path.resolve()}")

    return model_artifacts


def _create_diagnostic_plot(processed_df, plot_path: Path) -> None:
    """
    Simple Matplotlib scatter plot to visualise speed vs vehicle count.
    """
    plt.figure(figsize=(8, 6))
    colors = {"Low": "green", "Medium": "orange", "High": "red"}
    for label, group in processed_df.groupby("congestion_level"):
        plt.scatter(
            group["speed(kmph)"],
            group["vehicle_count"],
            label=label,
            alpha=0.7,
            c=colors.get(label, "blue"),
        )
    plt.title("Speed vs Vehicle Count by Congestion Level")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train_model()

