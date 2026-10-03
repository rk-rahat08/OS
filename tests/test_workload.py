import pytest
from src.workload import generate, shifted


@pytest.mark.parametrize("kind", ["sequential", "random", "bursty"])
def test_workloads_are_reproducible_and_in_bounds(kind):
    first = generate(kind, 100, 42, 0, 199)
    second = generate(kind, 100, 42, 0, 199)
    assert first == second
    assert len(first) == 100
    assert all(0 <= value <= 199 for value in first)


def test_shift_position_and_length():
    trace, position = shifted("sequential", "random", 80, 40, 9)
    assert len(trace) == 80
    assert position == 40


def test_bad_workload_type_fails():
    with pytest.raises(ValueError):
        generate("unknown", 10, 1)  # type: ignore[arg-type]
