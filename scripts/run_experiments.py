"""Convenience wrapper that runs the complete default experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ExperimentConfig
from src.experiments import run_all_experiments


if __name__ == "__main__":
    run_all_experiments(ROOT, ExperimentConfig())
