"""Central configuration for the disk scheduling simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    min_cylinder: int = 0
    max_cylinder: int = 199
    initial_head: int = 100
    initial_direction: str = "up"
    window_size: int = 20
    n_traces_per_type: int = 60
    windows_per_trace: int = 10
    random_seed: int = 307
    test_size: float = 0.25
    shift_windows_per_phase: int = 30

    def validate(self) -> None:
        if self.min_cylinder >= self.max_cylinder:
            raise ValueError("min_cylinder must be smaller than max_cylinder")
        if not self.min_cylinder <= self.initial_head <= self.max_cylinder:
            raise ValueError("initial_head is outside the disk range")
        if self.initial_direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.n_traces_per_type < 4 or self.windows_per_trace < 2:
            raise ValueError("experiment size is too small")


ALGORITHMS = ("FCFS", "SCAN", "C-SCAN", "SSTF")
TIE_BREAK_ORDER = ("FCFS", "SSTF", "SCAN", "C-SCAN")
