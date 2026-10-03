"""Publication-style plots for experiment outputs."""

from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALGORITHMS


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_classical(table: pd.DataFrame, path: Path) -> None:
    means = table.set_index("workload_type")[[f"{a.lower().replace('-', '')}_mean" for a in ALGORITHMS]]
    means.columns = ALGORITHMS
    ax = means.plot(kind="bar", figsize=(8, 4.5), color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
    ax.set_title("Classical scheduler movement by workload")
    ax.set_xlabel("Workload type")
    ax.set_ylabel("Mean head movement (cylinders per window)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Scheduler")
    _finish(path)


def plot_confusion(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted scheduler", ylabel="True best scheduler", title="Scheduler selector confusion matrix")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, values[row, col], ha="center", va="center",
                    color="white" if values[row, col] > values.max() / 2 else "black")
    _finish(path)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(table["average_confidence"], table["observed_accuracy"], marker="o", color="#4C78A8", label="Decision tree")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Average predicted confidence", ylabel="Observed accuracy",
           title="Confidence reliability diagram")
    ax.legend()
    ax.grid(alpha=0.25)
    _finish(path)


def plot_shift(timeline: pd.DataFrame, shift_window: int, path: Path) -> None:
    x = timeline["window_id"]
    rolling_correct = timeline["correct"].rolling(5, min_periods=1).mean()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, timeline["selected_movement"], label="ML selected", color="#4C78A8")
    axes[0].plot(x, timeline["oracle_movement"], label="Oracle", color="#54A24B", alpha=0.8)
    axes[0].set_ylabel("Movement")
    axes[0].legend()
    axes[1].plot(x, timeline["regret"], color="#E45756")
    axes[1].set_ylabel("Regret")
    axes[2].plot(x, timeline["confidence"], label="Confidence", color="#F58518")
    axes[2].plot(x, rolling_correct, label="5-window accuracy", color="#72B7B2")
    axes[2].set(xlabel="Window number", ylabel="Score", ylim=(0, 1.05))
    axes[2].legend()
    for ax in axes:
        ax.axvline(shift_window - 0.5, color="black", linestyle="--", label="Workload shift")
        ax.grid(alpha=0.2)
    fig.suptitle("Sequential to random workload shift")
    _finish(path)


def plot_selection_distribution(evaluated: pd.DataFrame, path: Path) -> None:
    counts = pd.crosstab(evaluated["workload_type"], evaluated["predicted_scheduler"], normalize="index")
    counts = counts.reindex(columns=ALGORITHMS, fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(8, 4.5), color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
    ax.set(xlabel="Workload type", ylabel="Fraction of predictions", title="ML scheduler-selection distribution")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Predicted scheduler", bbox_to_anchor=(1.02, 1), loc="upper left")
    _finish(path)


def plot_feature_importance(model, feature_names: list[str], path: Path) -> None:
    order = np.argsort(model.feature_importances_)[-10:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.asarray(feature_names)[order], model.feature_importances_[order], color="#4C78A8")
    ax.set(xlabel="Decision-tree importance", title="Ten most important selector features")
    _finish(path)
