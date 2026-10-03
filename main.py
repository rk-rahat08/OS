"""Command-line entry point for the complete project."""

import argparse
from pathlib import Path
from config import ExperimentConfig
from src.experiments import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning-augmented disk scheduling experiments")
    parser.add_argument("--run-all", action="store_true", help="generate, train, evaluate, plot, and report")
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--n-traces-per-type", type=int, default=60)
    parser.add_argument("--windows-per-trace", type=int, default=10)
    parser.add_argument("--head", type=int, default=100)
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    parser.add_argument("--min-cylinder", type=int, default=0)
    parser.add_argument("--max-cylinder", type=int, default=199)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_all:
        raise SystemExit("Use --run-all to execute the complete reproducible pipeline.")
    config = ExperimentConfig(
        min_cylinder=args.min_cylinder, max_cylinder=args.max_cylinder,
        initial_head=args.head, initial_direction=args.direction,
        window_size=args.window_size, n_traces_per_type=args.n_traces_per_type,
        windows_per_trace=args.windows_per_trace, random_seed=args.seed,
    )
    metrics = run_all_experiments(Path(__file__).resolve().parent, config)
    print(f"Complete. Accuracy={metrics['accuracy']:.3f}, mean regret={metrics['mean_regret']:.2f}")


if __name__ == "__main__":
    main()
