"""Binary heap of future simulation events."""

from __future__ import annotations

import heapq

from factoryrl.core.events import Event
from factoryrl.exceptions import InvariantViolationError


class EventQueue:
    """Min-heap ordered by ``(time, priority, sequence)``."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int]] = []
        self._events: dict[int, Event] = {}

    def push(self, event: Event) -> None:
        if event.sequence in self._events:
            raise InvariantViolationError(
                f"Duplicate event sequence {event.sequence}.",
                context={"event_id": event.event_id},
            )
        heapq.heappush(self._heap, event.sort_key)
        self._events[event.sequence] = event

    def peek(self) -> Event | None:
        if not self._heap:
            return None
        return self._events[self._heap[0][2]]

    def pop(self) -> Event:
        if not self._heap:
            raise InvariantViolationError("Cannot pop an empty event queue.")
        sequence = heapq.heappop(self._heap)[2]
        return self._events.pop(sequence)

    def __len__(self) -> int:
        return len(self._heap)

    def times(self) -> list[float]:
        """Queued event times, for invariant checks."""
        return [self._events[sequence].time for _, _, sequence in self._heap]
