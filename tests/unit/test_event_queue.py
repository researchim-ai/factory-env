"""Heap order is (time, priority, sequence), not insertion order."""

from __future__ import annotations

from types import MappingProxyType

from factoryrl.core.event_queue import EventQueue
from factoryrl.core.events import EVENT_PRIORITY, Event, EventType


def _event(event_type: EventType, time: float, sequence: int) -> Event:
    return Event(
        time=time,
        priority=EVENT_PRIORITY[event_type],
        sequence=sequence,
        event_id=f"e{sequence}",
        type=event_type,
        payload=MappingProxyType({}),
    )


def test_same_timestamp_uses_priority_then_sequence():
    queue = EventQueue()
    queue.push(_event(EventType.OPERATION_START, 5, 1))
    queue.push(_event(EventType.JOB_RELEASE, 5, 3))
    queue.push(_event(EventType.OPERATION_FINISH, 5, 2))
    queue.push(_event(EventType.OPERATION_FINISH, 5, 4))
    order = [queue.pop().type for _ in range(4)]
    assert order == [
        EventType.OPERATION_FINISH,
        EventType.OPERATION_FINISH,
        EventType.JOB_RELEASE,
        EventType.OPERATION_START,
    ]


def test_earlier_time_beats_higher_priority():
    queue = EventQueue()
    queue.push(_event(EventType.OPERATION_FINISH, 10, 1))
    queue.push(_event(EventType.OPERATION_START, 3, 2))
    assert queue.pop().time == 3
    assert queue.pop().time == 10
