"""Logical simulation clock. It never moves backwards."""

from __future__ import annotations

from factoryrl.core.events import TIME_TOLERANCE
from factoryrl.exceptions import InvariantViolationError


class SimulationClock:
    """Monotone logical time in the instance time unit."""

    def __init__(self) -> None:
        self.now = 0.0

    def reset(self) -> None:
        self.now = 0.0

    def advance_to(self, time: float) -> None:
        if time < self.now - TIME_TOLERANCE:
            raise InvariantViolationError(
                "Simulation time would move backwards.",
                context={"now": self.now, "target": time},
            )
        self.now = float(time)
