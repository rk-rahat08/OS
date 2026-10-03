import pytest
from src.features import extract_features, FEATURE_NAMES


def test_feature_keys_and_basic_values():
    features = extract_features([10, 12, 14, 16], 10)
    assert list(features.keys()) == FEATURE_NAMES
    assert features["request_count"] == 4
    assert features["mean_abs_jump"] == 2
    assert features["locality_ratio"] == 1
    assert features["trend_slope"] > 0


def test_features_do_not_include_scheduler_outputs():
    forbidden = {"movement", "best", "winner", "scheduler"}
    assert not any(any(word in name for word in forbidden) for name in FEATURE_NAMES)


def test_empty_window_fails():
    with pytest.raises(ValueError):
        extract_features([], 100)
