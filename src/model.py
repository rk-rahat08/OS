"""Training and persistence for the learned scheduler selector."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from config import ExperimentConfig
from src.features import FEATURE_NAMES


def train_model(data: pd.DataFrame, config: ExperimentConfig, model_path: Path):
    """Split by trace to prevent related windows crossing train/test boundaries."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
    train_idx, test_idx = next(splitter.split(data, data["best_scheduler"], groups=data["trace_id"]))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=config.random_seed,
    )
    model.fit(train[FEATURE_NAMES], train["best_scheduler"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "config": config}, model_path)
    return model, train, test
