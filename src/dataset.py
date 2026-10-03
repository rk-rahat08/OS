"""Supervised dataset construction from synthetic request windows."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import ExperimentConfig, ALGORITHMS, TIE_BREAK_ORDER
from src.features import extract_features, FEATURE_NAMES
from src.schedulers import run_all
from src.workload import generate


def label_best(movements: dict[str, int]) -> str:
    """Return the minimum-cost scheduler using a documented deterministic tie order."""
    best_value = min(movements.values())
    return next(name for name in TIE_BREAK_ORDER if movements[name] == best_value)


def build_dataset(config: ExperimentConfig, output_path: Path) -> pd.DataFrame:
    """Build windows from independent traces and save the research dataset."""
    rows: list[dict] = []
    trace_length = config.window_size * config.windows_per_trace
    workload_types = ("sequential", "random", "bursty")
    for type_index, workload_type in enumerate(workload_types):
        for trace_number in range(config.n_traces_per_type):
            seed = config.random_seed + type_index * 10000 + trace_number
            trace = generate(workload_type, trace_length, seed, config.min_cylinder, config.max_cylinder)
            head_rng = np.random.default_rng(seed + 50000)
            trace_id = f"{workload_type}-{seed}"
            for window_id in range(config.windows_per_trace):
                start = window_id * config.window_size
                window = trace[start:start + config.window_size]
                head = config.initial_head if window_id == 0 else int(head_rng.integers(config.min_cylinder, config.max_cylinder + 1))
                results = run_all(window, head, config.min_cylinder, config.max_cylinder, config.initial_direction)
                movements = {name: result.total_head_movement for name, result in results.items()}
                row = {
                    "comparison_id": f"{trace_id}-window-{window_id}",
                    "workload_type": workload_type,
                    "trace_id": trace_id,
                    "seed": seed,
                    "window_id": window_id,
                    "starting_head": head,
                    "request_sequence": json.dumps(window, separators=(",", ":")),
                    **extract_features(window, head, config.min_cylinder, config.max_cylinder),
                    **{f"{name.lower().replace('-', '')}_movement": movements[name] for name in ALGORITHMS},
                    "best_scheduler": label_best(movements),
                }
                rows.append(row)
    frame = pd.DataFrame(rows)
    expected = config.n_traces_per_type * len(workload_types) * config.windows_per_trace
    assert len(frame) == expected
    assert set(FEATURE_NAMES).issubset(frame.columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
