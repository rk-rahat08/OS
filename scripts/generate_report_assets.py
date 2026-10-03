"""Rebuild all report assets from deterministic experiment inputs."""

from run_experiments import ROOT, ExperimentConfig, run_all_experiments


if __name__ == "__main__":
    run_all_experiments(ROOT, ExperimentConfig())
