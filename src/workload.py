"""Reproducible synthetic disk-request workload generators."""

from typing import Literal
import numpy as np


WorkloadType = Literal["sequential", "random", "bursty"]


def _clip(values: np.ndarray, minimum: int, maximum: int) -> list[int]:
    return np.clip(np.rint(values), minimum, maximum).astype(int).tolist()


def sequential(n: int, seed: int, minimum: int = 0, maximum: int = 199, step: float = 4.0, noise: float = 1.5) -> list[int]:
    """Generate mostly progressive requests with small local noise and wraparound."""
    rng = np.random.default_rng(seed)
    span = maximum - minimum + 1
    start = int(rng.integers(minimum, maximum + 1))
    increments = np.maximum(1, rng.normal(step, noise, n)).round().astype(int)
    values = minimum + ((start - minimum + np.cumsum(increments)) % span)
    return values.astype(int).tolist()


def random_workload(n: int, seed: int, minimum: int = 0, maximum: int = 199) -> list[int]:
    """Generate requests uniformly across the full disk range."""
    rng = np.random.default_rng(seed)
    return rng.integers(minimum, maximum + 1, size=n).astype(int).tolist()


def bursty(n: int, seed: int, minimum: int = 0, maximum: int = 199, burst_width: float = 7.0, burst_length: int = 10) -> list[int]:
    """Generate temporary hotspots that move after each burst."""
    rng = np.random.default_rng(seed)
    values: list[int] = []
    while len(values) < n:
        center = int(rng.integers(minimum, maximum + 1))
        size = min(burst_length, n - len(values))
        values.extend(_clip(rng.normal(center, burst_width, size), minimum, maximum))
    return values


def generate(kind: WorkloadType, n: int, seed: int, minimum: int = 0, maximum: int = 199, **kwargs) -> list[int]:
    """Dispatch to a named workload generator."""
    if n < 0:
        raise ValueError("n cannot be negative")
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if kind == "sequential":
        return sequential(n, seed, minimum, maximum, kwargs.get("step", 4.0), kwargs.get("noise", 1.5))
    if kind == "random":
        return random_workload(n, seed, minimum, maximum)
    if kind == "bursty":
        return bursty(n, seed, minimum, maximum, kwargs.get("burst_width", 7.0), kwargs.get("burst_length", 10))
    raise ValueError(f"unknown workload type: {kind}")


def shifted(first: WorkloadType, second: WorkloadType, n: int, shift_position: int, seed: int, minimum: int = 0, maximum: int = 199, **kwargs) -> tuple[list[int], int]:
    """Join two workload phases and return the exact shift position."""
    if not 0 < shift_position < n:
        raise ValueError("shift_position must be inside the trace")
    left = generate(first, shift_position, seed, minimum, maximum, **kwargs)
    right = generate(second, n - shift_position, seed + 1, minimum, maximum, **kwargs)
    return left + right, shift_position
