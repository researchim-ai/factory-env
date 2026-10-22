"""Event types and the total order used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

# Absolute tolerance for comparing simulated times. Integer durations in
# float64 are exact; the tolerance covers accumulated binary error only.
TIME_TOLERANCE = 1e-9

# Sentinel passed to SimulationEngine.step. The Gymnasium adapter maps the
# last discrete index onto this value. It is not an operation index.
WAIT_ACTION = -1


class EventType(StrEnum):
    """Simulation events. Lower priority number is processed first."""

    OPERATION_FINISH = "operation_finish"
    JOB_RELEASE = "job_release"
    OPERATION_START = "operation_start"
    DECISION_REQUIRED = "decision_required"
    EPISODE_END = "episode_end"


# Same-timestamp order: free resources, release work, apply decisions already
# taken, then ask for the next decision, then close the episode.
EVENT_PRIORITY: dict[EventType, int] = {
    EventType.OPERATION_FINISH: 10,
    EventType.JOB_RELEASE: 20,
    EventType.OPERATION_START: 30,
    EventType.DECISION_REQUIRED: 40,
    EventType.EPISODE_END: 50,
}


class Phase(StrEnum):
    """Whether the simulator is waiting for an action or has stopped."""

    DECISION = "decision"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class Event:
    """One scheduled or logged event.

    Ordering key is ``(time, priority, sequence)``. ``sequence`` is unique,
    so the key never depends on dictionary order or thread scheduling.
    """

    time: float
    priority: int
    sequence: int
    event_id: str
    type: EventType
    payload: MappingProxyType[str, Any]

    @property
    def sort_key(self) -> tuple[float, int, int]:
        return (self.time, self.priority, self.sequence)
