"""Prediction, regret, classification, and confidence calibration metrics."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import ALGORITHMS
from src.features import FEATURE_NAMES


MOVEMENT_COLUMNS = {name: f"{name.lower().replace('-', '')}_movement" for name in ALGORITHMS}


def evaluate(model, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    probabilities = model.predict_proba(data[FEATURE_NAMES])
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    evaluated = data.copy()
    evaluated["phase"] = evaluated["phase"] if "phase" in evaluated.columns else "held_out"
    evaluated["true_best_scheduler"] = evaluated["best_scheduler"]
    evaluated["predicted_scheduler"] = predictions
    evaluated["confidence"] = confidence
    evaluated["correct"] = evaluated["predicted_scheduler"] == evaluated["best_scheduler"]
    evaluated["selected_movement"] = [
        row[MOVEMENT_COLUMNS[prediction]] for (_, row), prediction in zip(evaluated.iterrows(), predictions)
    ]
    movement_cols = list(MOVEMENT_COLUMNS.values())
    evaluated["oracle_movement"] = evaluated[movement_cols].min(axis=1)
    evaluated["regret"] = evaluated["selected_movement"] - evaluated["oracle_movement"]
    evaluated["relative_regret"] = evaluated["regret"] / evaluated["oracle_movement"].replace(0, np.nan)
    labels = list(ALGORITHMS)
    report = classification_report(evaluated["best_scheduler"], predictions, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(evaluated["best_scheduler"], predictions, labels=labels)
    one_hot = np.column_stack([(evaluated["best_scheduler"].to_numpy() == cls).astype(float) for cls in model.classes_])
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    metrics = {
        "accuracy": float(accuracy_score(evaluated["best_scheduler"], predictions)),
        "mean_confidence": float(confidence.mean()),
        "confidence_when_correct": float(evaluated.loc[evaluated["correct"], "confidence"].mean()),
        "confidence_when_incorrect": float(evaluated.loc[~evaluated["correct"], "confidence"].mean()),
        "mean_selected_movement": float(evaluated["selected_movement"].mean()),
        "mean_oracle_movement": float(evaluated["oracle_movement"].mean()),
        "mean_regret": float(evaluated["regret"].mean()),
        "multiclass_brier": multiclass_brier,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }
    return evaluated, metrics


def confidence_table(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0.0, 1.0, 11)
    bin_index = np.minimum((evaluated["confidence"] * 10).astype(int), 9)
    rows = []
    ece = 0.0
    total = len(evaluated)
    for index in range(10):
        subset = evaluated[bin_index == index]
        if subset.empty:
            continue
        avg_conf = float(subset["confidence"].mean())
        accuracy = float(subset["correct"].mean())
        count = len(subset)
        ece += count / total * abs(avg_conf - accuracy)
        rows.append({
            "confidence_bin": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
            "number_predictions": count,
            "average_confidence": avg_conf,
            "observed_accuracy": accuracy,
        })
    return pd.DataFrame(rows), float(ece)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
