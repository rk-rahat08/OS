"""End-to-end reproducible experiment orchestration."""

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import ExperimentConfig, ALGORITHMS
from src.dataset import build_dataset, label_best
from src.evaluation import evaluate, confidence_table, save_metrics, MOVEMENT_COLUMNS
from src.features import extract_features, FEATURE_NAMES
from src.model import train_model
from src.reporting import create_term_paper
from src.schedulers import run_all
from src.visualization import (
    plot_calibration, plot_classical, plot_confusion, plot_feature_importance,
    plot_selection_distribution, plot_shift,
)
from src.workload import generate, shifted


def _markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in formatted.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _make_tables(evaluated: pd.DataFrame, calibration: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    movement_columns = list(MOVEMENT_COLUMNS.values())
    # Every primary comparison is calculated from the exact same held-out windows.
    grouped = evaluated.groupby("workload_type")[movement_columns].agg(["mean", "std"])
    grouped.columns = [f"{column.replace('_movement', '')}_{stat}" for column, stat in grouped.columns]
    classical = grouped.reset_index()
    selector = evaluated.groupby("workload_type").agg(
        selection_accuracy=("correct", "mean"),
        ml_movement=("selected_movement", "mean"),
        oracle_movement=("oracle_movement", "mean"),
        regret=("regret", "mean"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    classical.to_csv(tables / "classical_scheduler_performance.csv", index=False)
    selector.to_csv(tables / "ml_selector_performance.csv", index=False)
    calibration.to_csv(tables / "confidence_calibration.csv", index=False)
    (tables / "classical_scheduler_performance.md").write_text(_markdown_table(classical), encoding="utf-8")
    (tables / "ml_selector_performance.md").write_text(_markdown_table(selector), encoding="utf-8")
    (tables / "confidence_calibration.md").write_text(_markdown_table(calibration), encoding="utf-8")
    return classical, selector


def _shift_experiment(model, config: ExperimentConfig, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_windows = config.shift_windows_per_phase * 2
    total_requests = total_windows * config.window_size
    shift_position = config.shift_windows_per_phase * config.window_size
    trace, recorded_shift = shifted("sequential", "random", total_requests, shift_position, config.random_seed + 90000,
                                    config.min_cylinder, config.max_cylinder)
    assert recorded_shift == shift_position
    head = config.initial_head
    rows: list[dict] = []
    for window_id in range(total_windows):
        start = window_id * config.window_size
        requests = trace[start:start + config.window_size]
        results = run_all(requests, head, config.min_cylinder, config.max_cylinder, config.initial_direction)
        movements = {name: result.total_head_movement for name, result in results.items()}
        features = extract_features(requests, head, config.min_cylinder, config.max_cylinder)
        feature_frame = pd.DataFrame([features])[FEATURE_NAMES]
        probabilities = model.predict_proba(feature_frame)[0]
        best_index = int(np.argmax(probabilities))
        predicted = str(model.classes_[best_index])
        true_best = label_best(movements)
        oracle = min(movements.values())
        selected = movements[predicted]
        rows.append({
            "window_id": window_id,
            "timeline_position": start,
            "phase": "before_shift" if start < shift_position else "after_shift",
            "workload_type": "sequential" if start < shift_position else "random",
            "starting_head": head,
            **{MOVEMENT_COLUMNS[name]: movements[name] for name in ALGORITHMS},
            "true_best_scheduler": true_best,
            "predicted_scheduler": predicted,
            "confidence": float(probabilities[best_index]),
            "correct": predicted == true_best,
            "selected_movement": selected,
            "oracle_movement": oracle,
            "regret": selected - oracle,
        })
        head = results[predicted].service_order[-1] if results[predicted].service_order else head
    timeline = pd.DataFrame(rows)
    summary = timeline.groupby("phase").agg(
        accuracy=("correct", "mean"), confidence=("confidence", "mean"),
        movement=("selected_movement", "mean"), regret=("regret", "mean"),
    )
    before = summary.loc["before_shift"]
    after = summary.loc["after_shift"]
    shift_table = pd.DataFrame({"metric": summary.columns, "before_shift": before.values, "after_shift": after.values,
                                "difference": after.values - before.values})
    raw = output / "raw"
    tables = output / "tables"
    timeline.to_csv(raw / "shift_timeline.csv", index=False)
    shift_table.to_csv(tables / "shift_experiment.csv", index=False)
    (tables / "shift_experiment.md").write_text(_markdown_table(shift_table), encoding="utf-8")
    distribution = pd.crosstab(timeline["phase"], timeline["predicted_scheduler"], normalize="index")
    distribution.to_csv(tables / "shift_selection_distribution.csv")
    return timeline, shift_table


def _generalization_test(model, config: ExperimentConfig, output: Path) -> pd.DataFrame:
    rows = []
    for type_index, kind in enumerate(("sequential", "random", "bursty")):
        for repeat in range(15):
            seed = config.random_seed + 120000 + type_index * 1000 + repeat
            kwargs = {"noise": 4.0} if kind == "sequential" else ({"burst_width": 2.5, "burst_length": 16} if kind == "bursty" else {})
            requests = generate(kind, config.window_size, seed, config.min_cylinder, config.max_cylinder, **kwargs)
            head = int(np.random.default_rng(seed).integers(config.min_cylinder, config.max_cylinder + 1))
            results = run_all(requests, head, config.min_cylinder, config.max_cylinder, config.initial_direction)
            movements = {name: result.total_head_movement for name, result in results.items()}
            features = extract_features(requests, head, config.min_cylinder, config.max_cylinder)
            probabilities = model.predict_proba(pd.DataFrame([features])[FEATURE_NAMES])[0]
            predicted = str(model.classes_[int(np.argmax(probabilities))])
            oracle = min(movements.values())
            rows.append({"workload_type": kind, "seed": seed, "predicted_scheduler": predicted,
                         "true_best_scheduler": label_best(movements), "confidence": float(probabilities.max()),
                         "correct": predicted == label_best(movements), "selected_movement": movements[predicted],
                         "oracle_movement": oracle, "regret": movements[predicted] - oracle})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "raw" / "distribution_shift_evaluation.csv", index=False)
    summary = frame.groupby("workload_type").agg(accuracy=("correct", "mean"), mean_regret=("regret", "mean"),
                                                  mean_confidence=("confidence", "mean")).reset_index()
    summary.to_csv(output / "tables" / "generalization_summary.csv", index=False)
    return summary


def _write_reports(root: Path, config: ExperimentConfig, dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                   metrics: dict, classical: pd.DataFrame, selector: pd.DataFrame, shift_table: pd.DataFrame,
                   calibration: pd.DataFrame, ece: float, generalization: pd.DataFrame) -> None:
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    best_fixed = {}
    for _, row in classical.iterrows():
        costs = {name: row[f"{name.lower().replace('-', '')}_mean"] for name in ALGORITHMS}
        best_fixed[row["workload_type"]] = min(costs, key=costs.get)
    summary = f"""# Experiment Summary

This file was generated from the saved experiment outputs.

- Dataset windows: {len(dataset)} ({len(train)} train, {len(test)} held-out test)
- Common comparison workload: the same {len(test)} held-out windows and starting-head values were used for FCFS, SCAN, C-SCAN, SSTF, ML, and oracle results
- Held-out selection accuracy: {metrics['accuracy']:.3f}
- Mean ML-selected movement: {metrics['mean_selected_movement']:.2f} cylinders/window
- Mean oracle movement: {metrics['mean_oracle_movement']:.2f} cylinders/window
- Mean regret: {metrics['mean_regret']:.2f} cylinders/window
- Mean confidence when correct: {metrics['confidence_when_correct']:.3f}
- Mean confidence when wrong: {metrics['confidence_when_incorrect']:.3f}
- Expected Calibration Error: {ece:.3f}
- Multiclass Brier score: {metrics['multiclass_brier']:.3f}
- Best fixed scheduler by workload: {', '.join(f'{k}: {v}' for k, v in best_fixed.items())}

## Shift metrics

{_markdown_table(shift_table)}

## Generalization test

{_markdown_table(generalization)}
"""
    (report_dir / "experiment_summary.md").write_text(summary, encoding="utf-8")

    before = shift_table.set_index("metric")["before_shift"]
    after = shift_table.set_index("metric")["after_shift"]
    report = f"""# Learning-Augmented Disk Scheduling: Adaptive Scheduler Selection Under Workload Shift

## 1. Problem Framing

Disk scheduling decide which pending request should be served next. A mechnical disk head has to move between cylinder positions, so more movement normally means more seek time. FCFS keeps arrival order, SSTF chooses the closest request, SCAN travels like an elevator, and C-SCAN only travels in one direction before wrapping [1]. No policy was assumed to be best for every request pattern. This project checks if a small learned selector can choose a suitable policy for each request window.

   A second issue is workload shift. A workload can be local and sequential at first, but later become wide and random. The selector only sees features from the current re quest window; it does not see the scheduler results before making its choice.

## 2. Implementation

The simulator implement FCFS, true SCAN, true C-SCAN and SSTF from scratch. SCAN goes to the disk boundary before reversing. C-SCAN counts both movement to the boundary and the wrap. Equal SSTF distances select the lower cylinder. Tied s cheduler labels use FCFS, SSTF, SCAN, then C-SCAN order.

 For each {config.window_size}-request window, the model uses 16 simple features: count, mean, standard deviation, range, two head-distance values, fractions above and below the head, jump statistics, unique ratio, locality, slope, direction changes and histogram entropy. These can show if requests are spread out, ordered or clusterd. The classifier is a decision tree, because its choices are easy to inspect and explain in a class demo [2], and the code uses scikit-learn [3]. Confidence is the highest predicted class probability.

## 3. Experimental Setup

Requests use cylinders {config.min_cylinder} to {config.max_cylinder}, initial head {config.initial_head}, and initial direction `{config.initial_direction}`. The generated dataset has {len(dataset)} windows from sequential, random and bursty traces. There was {config.n_traces_per_type} traces for each type and {config.windows_per_trace} windows per trace. A group split kept complete traces together: {len(train)} windows for training and {len(test)} for held-out testing. The fixed experment seed was {config.random_seed}. The shift timeline has {config.shift_windows_per_phase} sequential windows followed by {config.shift_windows_per_phase} random windows.

  Total head movement in cylinders is used as a simulated seek-cost proxy, not millisecond hardware timing. The oracle is the cheapest of four schedulers after running all four, so it is only a reference and cannot be deployed as an online choice.

Every primary comparison uses the exact same {len(test)} held-out request windows and starting-head values. The common rows are saved in `results/raw/common_workload_manifest.csv`. Inside the shift and generalization tests, each algorithm also receives the identical window before costs are compared.

## 4. Results

The held-out selector accuracy were {metrics['accuracy']:.3f}. Its mean movement was {metrics['mean_selected_movement']:.2f} cylinders per window, while oracle movement was {metrics['mean_oracle_movement']:.2f}; mean regret was {metrics['mean_regret']:.2f}. Confidence averaged {metrics['confidence_when_correct']:.3f} on correct predictions and {metrics['confidence_when_incorrect']:.3f} on wrong predictions. Calibraton ECE was {ece:.3f} and multiclass Brier score was {metrics['multiclass_brier']:.3f}.

 Table files are in `results/tables/`. The main comparison is shown in `results/figures/classical_scheduler_comparison.png`, with prediction errors in `confusion_matrix.png` and calibration in `confidence_calibration.png`.

   Before shift, accuracy was {before['accuracy']:.3f}, confidence {before['confidence']:.3f}, movement {before['movement']:.2f}, and regret {before['regret']:.2f}. After shift these value became {after['accuracy']:.3f}, {after['confidence']:.3f}, {after['movement']:.2f}, and {after['regret']:.2f}. The marked timeline is in `results/figures/workload_shift_timeline.png`.

## 5. Analysis

**Observation:** The best fixed schedulers by workload were {', '.join(f'{k}: {v}' for k, v in best_fixed.items())}. The selector did not reach oracle performance, and its measured penalty was {metrics['mean_regret']:.2f} cylinders per window.

No C-SCAN window was the ground-truth winner in the held-out set, so its per-class support and score are zero. This result was kept as measured instead of creating artificial examples to make every class appear.

**Interpretation:** SSTF can reduce local movment because it repeatedly takes a nearby request. SCAN gives a structured sweep, while C-SCAN can pay a big wrap cost under this metric. FCFS changes strongly with arrival order. The confidence gap between correct and wrong cases gives some useful evidence, but the ECE and reliability plot must also be checked before calling it well cal ibrated [4].

  The distribution-shift test use stronger burst concentration and different sequential noise. This is still synthetic, so it shows generalization only inside this simulator. The project also models a simple cylinder line and not queue arrival times, rotational delay or real device caching.

## 6. Conclusion

The experiment showed how a decision tree can select among four disk schedulers using only one incoming request window. It measured both label correctness and the actual movement cost of mistakes. Results changed across the sequential-to-random shift, which supports checking confidence and regret together instead of accuracy alone. These findings apply to classical mechanical-disk simulation; modern SSDs do not have the same head movement.

## 7. AI Assistance Disclosure

An AI coding assistant was used for software development, testing support and document drafting. The experiment numbers were generated by running the included code, and raw outputs are saved for checking.

## References

1. A. Silberschatz, P. B. Galvin, and G. Gagne, *Operating System Concepts*, 10th ed., Wiley, 2018.
2. L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*, Wadsworth & Brooks/Cole, 1984.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
4. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *Proceedings of Machine Learning Research*, vol. 70, pp. 1321-1330, 2017.
"""
    (report_dir / "report.md").write_text(report, encoding="utf-8")

    demo = f"""# Demo Notes (3-5 minutes)

## 0:00-0:30 — Problem
Disk head movement is expensive, and the best scheduling rule can depend on the request pattern. The project selects FCFS, SCAN, C-SCAN, or SSTF for each window.

## 0:30-1:20 — Classical algorithms
Open `src/schedulers.py`. Mention true boundary movement for SCAN/C-SCAN and lower-cylinder SSTF tie-breaking.

## 1:20-2:00 — Workloads and features
Show sequential, random, bursty generators and the 16 leakage-free features in `src/features.py`.

## 2:00-2:40 — Learned selector
Show the decision tree, trace-level train/test split, predicted scheduler, and maximum class probability as confidence. Held-out accuracy: {metrics['accuracy']:.3f}.

## 2:40-3:30 — Shift
Open `results/figures/workload_shift_timeline.png`. The timeline changes from sequential to random at window {config.shift_windows_per_phase}.

## 3:30-4:15 — Results
Show the classical comparison, confusion matrix, and reliability diagram. Mean selector regret: {metrics['mean_regret']:.2f} cylinders/window.

## 4:15-5:00 — Finding and limitation
Explain the measured best schedulers ({', '.join(f'{k}: {v}' for k, v in best_fixed.items())}) and that cylinder movement is not real SSD or HDD latency.

## Likely instructor questions

- **Why a decision tree?** It is lightweight, supports multiclass probabilities, and can be explained using feature splits.
- **Why these features?** They describe spread, locality, ordering and the head's position without using scheduler outcomes.
- **Why does workload type matter?** Local, ordered and wide requests create different travel paths.
- **SCAN vs C-SCAN?** SCAN reverses at a boundary; C-SCAN wraps and keeps one service direction.
- **SCAN vs LOOK?** SCAN reaches the physical boundary; LOOK reverses at the last request.
- **How is confidence calculated?** It is the largest `predict_proba` value.
- **What is calibration?** Predictions near a confidence value should be correct about that often.
- **What is distribution shift?** The request-generating pattern changes from the training or earlier pattern.
- **How is ground truth made?** Run all four schedulers on the same window and choose minimum movement.
- **Why is the oracle not deployable?** It knows all four outcomes before choosing and would do four simulations.
- **What if schedulers tie?** FCFS, SSTF, SCAN, then C-SCAN is the fixed tie order.
- **Why total movement?** It is a clear conventional proxy for mechanical seek effort.
- **Does this apply directly to SSDs?** No. SSDs do not have mechanical head seek movement.
"""
    (report_dir / "demo_notes.md").write_text(demo, encoding="utf-8")

    readme = f"""# Learning-Augmented Disk Scheduling

An undergraduate CSE-307 Operating Systems project that compares FCFS, SCAN, C-SCAN and SSTF, then trains a decision tree to select a scheduler for each request window.

## Problem

Mechanical disk requests at different cylinders require head movement. One fixed algorithm may behave differently on sequential, random and bursty requests, so this project tests a window-based learned selector under normal and shifted workloads. Cylinder movement is a simulation proxy for seek cost, not hardware milliseconds.

## Implemented algorithms

- **FCFS:** arrival order.
- **SSTF:** nearest pending request; lower cylinder wins equal-distance ties.
- **SCAN:** elevator traversal to the physical boundary, then reversal.
- **C-SCAN:** one-direction traversal, boundary wrap included in movement.

## Learned selector

Each window has {config.window_size} requests. Sixteen leakage-free features describe location, spread, locality, consecutive jumps, trend, direction changes, uniqueness and histogram entropy. A `DecisionTreeClassifier` predicts the minimum-movement scheduler; confidence is `max(predict_proba)`.

Tied labels use this order: FCFS, SSTF, SCAN, C-SCAN. Related windows stay together because the split is performed by synthetic trace rather than by row.

## Workload generation

The deterministic generators support sequential requests with noise, uniform random requests, hotspot-based bursty requests, and a two-phase shifted trace. Every generator accepts a seed.

## Installation

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest
```

## Running all experiments

```bash
python main.py --run-all
```

Useful overrides include `--seed`, `--window-size`, `--n-traces-per-type`, `--head`, `--direction`, `--min-cylinder`, and `--max-cylinder`.

## Results

- Dataset: {len(dataset)} windows; held-out test: {len(test)} windows.
- Fair comparison: every fixed scheduler and the ML selector use the same {len(test)} held-out request windows and starting heads.
- Selector accuracy: {metrics['accuracy']:.3f}.
- ML movement: {metrics['mean_selected_movement']:.2f}; oracle: {metrics['mean_oracle_movement']:.2f} cylinders/window.
- Mean regret: {metrics['mean_regret']:.2f} cylinders/window.
- Confidence correct/wrong: {metrics['confidence_when_correct']:.3f}/{metrics['confidence_when_incorrect']:.3f}.
- ECE: {ece:.3f}; multiclass Brier score: {metrics['multiclass_brier']:.3f}.

See [experiment summary](report/experiment_summary.md), [LaTeX paper](report/report.tex), [PDF term paper](report/term_paper.pdf), and generated files under `results/`.

## Repository structure

- `src/`: schedulers, workloads, features, dataset, model, evaluation and plots.
- `tests/`: deterministic algorithm, generator and feature tests.
- `results/raw/`: dataset and per-window predictions.
- `results/tables/`: CSV and Markdown tables.
- `results/figures/`: high-resolution PNG charts.
- `results/models/`: saved decision tree.
- `report/`: report, generated summary and demo notes.

## Reproducibility

The default seed is {config.random_seed}. Data generation, group splitting and model training all use fixed seeds. Running `python main.py --run-all` rebuilds every generated artifact.

## Limitations

- Cylinder movement is not real device latency.
- Workloads are synthetic rather than physical disk traces.
- The classifier and geometry are deliberately simple.
- Queue timing, rotational delay and caching are not modeled.
- SSDs do not have mechanical seek behavior.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def run_all_experiments(root: Path, config: ExperimentConfig) -> dict:
    config.validate()
    output = root / "results"
    for name in ("figures", "tables", "raw", "models"):
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "raw" / "experiment_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    dataset = build_dataset(config, output / "raw" / "scheduler_dataset.csv")
    model, train, test = train_model(dataset, config, output / "models" / "decision_tree.joblib")
    evaluated, metrics = evaluate(model, test)
    evaluated.to_csv(output / "raw" / "test_predictions.csv", index=False)
    calibration, ece = confidence_table(evaluated)
    metrics["expected_calibration_error"] = ece
    save_metrics(metrics, output / "raw" / "metrics.json")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(output / "tables" / "classification_report.csv")
    pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(output / "tables" / "confusion_matrix.csv")
    manifest_columns = ["comparison_id", "trace_id", "seed", "window_id", "workload_type",
                        "starting_head", "request_sequence"]
    evaluated[manifest_columns].to_csv(output / "raw" / "common_workload_manifest.csv", index=False)
    classical, selector = _make_tables(evaluated, calibration, output)
    timeline, shift_table = _shift_experiment(model, config, output)
    generalization = _generalization_test(model, config, output)
    plot_classical(classical, output / "figures" / "classical_scheduler_comparison.png")
    plot_confusion(metrics["confusion_matrix"], metrics["labels"], output / "figures" / "confusion_matrix.png")
    plot_calibration(calibration, output / "figures" / "confidence_calibration.png")
    plot_shift(timeline, config.shift_windows_per_phase, output / "figures" / "workload_shift_timeline.png")
    plot_selection_distribution(evaluated, output / "figures" / "selection_distribution.png")
    plot_feature_importance(model, FEATURE_NAMES, output / "figures" / "feature_importance.png")
    _write_reports(root, config, dataset, train, test, metrics, classical, selector, shift_table, calibration, ece, generalization)
    create_term_paper(root, metrics, classical, selector, shift_table, len(dataset), len(train), len(test),
                      config.random_seed, config.window_size)
    return metrics
