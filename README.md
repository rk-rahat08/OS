# Learning-Augmented Disk Scheduling

An undergraduate CSE-307 Operating Systems project that compares FCFS, SCAN, C-SCAN and SSTF, then trains a decision tree to select a scheduler for each request window.

## Problem

Mechanical disk requests at different cylinders require head movement. One fixed algorithm may behave differently on sequential, random and bursty requests, so this project tests a window-based learned selector under normal and shifted workloads. Cylinder movement is a simulation proxy for seek cost, not hardware milliseconds.

## Implemented algorithms

- **FCFS:** arrival order.
- **SSTF:** nearest pending request; lower cylinder wins equal-distance ties.
- **SCAN:** elevator traversal to the physical boundary, then reversal.
- **C-SCAN:** one-direction traversal, boundary wrap included in movement.

## Learned selector

Each window has 20 requests. Sixteen leakage-free features describe location, spread, locality, consecutive jumps, trend, direction changes, uniqueness and histogram entropy. A `DecisionTreeClassifier` predicts the minimum-movement scheduler; confidence is `max(predict_proba)`.

Tied labels use this order: FCFS, SSTF, SCAN, C-SCAN. Related windows stay together because the split is performed by synthetic trace rather than by row.

## Workload generation

The deterministic generators support sequential requests with noise, uniform random requests, hotspot-based bursty requests, and a two-phase shifted trace. Every generator accepts a seed.

## Installation

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
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

- Dataset: 1800 windows; held-out test: 450 windows.
- Fair comparison: every fixed scheduler and the ML selector use the same 450 held-out request windows and starting heads.
- Selector accuracy: 0.858.
- ML movement: 186.22; oracle: 182.02 cylinders/window.
- Mean regret: 4.20 cylinders/window.
- Confidence correct/wrong: 0.945/0.887.
- ECE: 0.140; multiclass Brier score: 0.263.

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

The default seed is 307. Data generation, group splitting and model training all use fixed seeds. Running `python main.py --run-all` rebuilds every generated artifact.

## Limitations

- Cylinder movement is not real device latency.
- Workloads are synthetic rather than physical disk traces.
- The classifier and geometry are deliberately simple.
- Queue timing, rotational delay and caching are not modeled.
- SSDs do not have mechanical seek behavior.
