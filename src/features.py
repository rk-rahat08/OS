"""Transparent window features that do not use scheduler outcomes."""

import numpy as np


FEATURE_NAMES = [
    "request_count", "mean_cylinder", "std_cylinder", "range_cylinder",
    "head_to_mean", "head_to_nearest", "fraction_above", "fraction_below",
    "mean_abs_jump", "std_abs_jump", "max_jump", "unique_ratio",
    "locality_ratio", "trend_slope", "direction_changes", "histogram_entropy",
]


def extract_features(requests: list[int], head: int, minimum: int = 0, maximum: int = 199, locality_threshold: int = 10) -> dict[str, float]:
    """Extract compact distribution, locality, and order features from one window."""
    if not requests:
        raise ValueError("feature extraction needs at least one request")
    values = np.asarray(requests, dtype=float)
    if np.any(values < minimum) or np.any(values > maximum):
        raise ValueError("a request is outside the disk range")
    jumps = np.abs(np.diff(values))
    signed = np.diff(values)
    changes = np.sum(np.sign(signed[1:]) != np.sign(signed[:-1])) if len(signed) > 1 else 0
    counts, _ = np.histogram(values, bins=min(10, maximum - minimum + 1), range=(minimum, maximum + 1))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
    return {
        "request_count": float(len(values)),
        "mean_cylinder": float(values.mean()),
        "std_cylinder": float(values.std()),
        "range_cylinder": float(values.max() - values.min()),
        "head_to_mean": float(abs(head - values.mean())),
        "head_to_nearest": float(np.min(np.abs(values - head))),
        "fraction_above": float(np.mean(values > head)),
        "fraction_below": float(np.mean(values < head)),
        "mean_abs_jump": float(jumps.mean()) if len(jumps) else 0.0,
        "std_abs_jump": float(jumps.std()) if len(jumps) else 0.0,
        "max_jump": float(jumps.max()) if len(jumps) else 0.0,
        "unique_ratio": float(len(np.unique(values)) / len(values)),
        "locality_ratio": float(np.mean(jumps <= locality_threshold)) if len(jumps) else 1.0,
        "trend_slope": slope,
        "direction_changes": float(changes),
        "histogram_entropy": entropy,
    }
