"""Classical disk scheduling algorithms using cylinder movement as seek cost."""

from dataclasses import dataclass, asdict
from typing import Callable, Iterable


@dataclass
class ScheduleResult:
    algorithm: str
    request_order: list[int]
    service_order: list[int]
    movement_sequence: list[int]
    total_head_movement: int
    average_head_movement: float

    def as_dict(self) -> dict:
        return asdict(self)


def _validate(requests: Iterable[int], head: int, minimum: int, maximum: int) -> list[int]:
    values = [int(value) for value in requests]
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if not minimum <= head <= maximum:
        raise ValueError("head is outside the disk range")
    if any(value < minimum or value > maximum for value in values):
        raise ValueError("a request is outside the disk range")
    return values


def _result(name: str, requests: list[int], service: list[int], movement: list[int]) -> ScheduleResult:
    total = sum(abs(b - a) for a, b in zip(movement, movement[1:]))
    average = total / len(requests) if requests else 0.0
    return ScheduleResult(name, requests.copy(), service, movement, total, average)


def fcfs(requests: Iterable[int], head: int, minimum: int = 0, maximum: int = 199, direction: str = "up") -> ScheduleResult:
    """Service requests in their original arrival order."""
    values = _validate(requests, head, minimum, maximum)
    return _result("FCFS", values, values.copy(), [head, *values])


def sstf(requests: Iterable[int], head: int, minimum: int = 0, maximum: int = 199, direction: str = "up") -> ScheduleResult:
    """Service the nearest request; ties go to the lower cylinder."""
    pending = _validate(requests, head, minimum, maximum)
    original = pending.copy()
    service: list[int] = []
    current = head
    while pending:
        chosen_index = min(range(len(pending)), key=lambda i: (abs(pending[i] - current), pending[i], i))
        current = pending.pop(chosen_index)
        service.append(current)
    return _result("SSTF", original, service, [head, *service])


def scan(requests: Iterable[int], head: int, minimum: int = 0, maximum: int = 199, direction: str = "up") -> ScheduleResult:
    """True SCAN: travel to the physical boundary, then reverse direction."""
    values = _validate(requests, head, minimum, maximum)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if not values:
        return _result("SCAN", values, [], [head])
    below = sorted((x for x in values if x < head), reverse=True)
    equal = [x for x in values if x == head]
    above = sorted(x for x in values if x > head)
    if direction == "up":
        service = equal + above + below
        movement = [head, *equal, *above]
        if movement[-1] != maximum:
            movement.append(maximum)
        movement.extend(below)
    else:
        service = equal + below + above
        movement = [head, *equal, *below]
        if movement[-1] != minimum:
            movement.append(minimum)
        movement.extend(above)
    return _result("SCAN", values, service, movement)


def cscan(requests: Iterable[int], head: int, minimum: int = 0, maximum: int = 199, direction: str = "up") -> ScheduleResult:
    """True C-SCAN: move one way, reach the boundary, wrap, and continue."""
    values = _validate(requests, head, minimum, maximum)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    if not values:
        return _result("C-SCAN", values, [], [head])
    below_asc = sorted(x for x in values if x < head)
    equal = [x for x in values if x == head]
    above_asc = sorted(x for x in values if x > head)
    if direction == "up":
        service = equal + above_asc + below_asc
        movement = [head, *equal, *above_asc]
        if movement[-1] != maximum:
            movement.append(maximum)
        if below_asc:
            movement.append(minimum)
            movement.extend(below_asc)
    else:
        below_desc = list(reversed(below_asc))
        above_desc = list(reversed(above_asc))
        service = equal + below_desc + above_desc
        movement = [head, *equal, *below_desc]
        if movement[-1] != minimum:
            movement.append(minimum)
        if above_desc:
            movement.append(maximum)
            movement.extend(above_desc)
    return _result("C-SCAN", values, service, movement)


SCHEDULERS: dict[str, Callable[..., ScheduleResult]] = {
    "FCFS": fcfs,
    "SCAN": scan,
    "C-SCAN": cscan,
    "SSTF": sstf,
}


def run_all(requests: Iterable[int], head: int, minimum: int = 0, maximum: int = 199, direction: str = "up") -> dict[str, ScheduleResult]:
    """Run every scheduler on the same requests and starting state."""
    values = list(requests)
    return {
        name: scheduler(values, head, minimum, maximum, direction)
        for name, scheduler in SCHEDULERS.items()
    }
