"""Negative branches that the happy-path tests do not reach."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from factoryrl.actions.dispatch import decode_gym_action
from factoryrl.actions.masks import build_action_mask
from factoryrl.core.clock import SimulationClock
from factoryrl.core.event_queue import EventQueue
from factoryrl.core.events import EVENT_PRIORITY, Event, EventType
from factoryrl.core.invariants import assert_invariants
from factoryrl.core.replay import assert_replay
from factoryrl.core.rng import spawn_generators, validate_seed
from factoryrl.core.state import OperationStatus
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.exceptions import (
    ConfigurationError,
    InstanceValidationError,
    InvalidActionError,
    InvariantViolationError,
    ReplayMismatchError,
)
from factoryrl.scenarios.builtins import demo_instance
from factoryrl.scenarios.instance import instance, job, machine, operation


def _engine():
    env = JobShopEnv(instance=demo_instance())
    env.reset(seed=0)
    env.step(0)
    return env._engine


def test_decode_and_mask_reject_bad_shapes():
    with pytest.raises(InvalidActionError):
        decode_gym_action(2, n_operations=2, max_operations=4)
    with pytest.raises(ConfigurationError):
        build_action_mask(np.zeros(1, dtype=bool), n_operations=1, max_operations=0, wait_allowed=False)
    with pytest.raises(ConfigurationError):
        build_action_mask(np.zeros(2, dtype=bool), n_operations=1, max_operations=2, wait_allowed=False)


def test_clock_and_queue_guards():
    clock = SimulationClock()
    clock.advance_to(3)
    clock.reset()
    assert clock.now == 0
    clock.advance_to(1)
    with pytest.raises(InvariantViolationError):
        clock.advance_to(0)
    queue = EventQueue()
    event = Event(
        0, EVENT_PRIORITY[EventType.JOB_RELEASE], 1, "e1", EventType.JOB_RELEASE, MappingProxyType({})
    )
    queue.push(event)
    with pytest.raises(InvariantViolationError):
        queue.push(event)
    queue.pop()
    with pytest.raises(InvariantViolationError):
        queue.pop()


def test_rng_streams_and_bad_seed():
    streams = spawn_generators(4, ("env", "agent"))
    assert set(streams) == {"env", "agent"}
    with pytest.raises(ConfigurationError):
        validate_seed(-1)


def test_replay_detects_a_changed_action():
    env = JobShopEnv(instance=demo_instance())
    env.reset(seed=3)
    for action in (0, 2, 1, 3):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    trace = env.export_trace()
    with pytest.raises(ReplayMismatchError):
        assert_replay(
            demo_instance(),
            [2, 0, 1, 3],
            trace["events"],
            trace["metrics"],
            seed=3,
            horizon=1000,
            allow_voluntary_idle=False,
            max_events=1_000_000,
            max_decisions=1_000_000,
        )


def test_invariant_failures_are_reported():
    engine = _engine()
    compiled = engine.compiled
    state = engine.state
    with pytest.raises(InvariantViolationError):
        assert_invariants(compiled, state, -1, [])
    with pytest.raises(InvariantViolationError):
        assert_invariants(compiled, state, engine.clock.now, [-5])
    running = next(index for index, status in enumerate(state.status) if status is OperationStatus.RUNNING)
    saved = state.start[running]
    state.start[running] = None
    with pytest.raises(InvariantViolationError):
        assert_invariants(compiled, state, engine.clock.now, [])
    state.start[running] = saved
    state.operations_completed += 1
    with pytest.raises(InvariantViolationError):
        assert_invariants(compiled, state, engine.clock.now, [])


def test_self_predecessor_and_duplicate_metadata_types():
    with pytest.raises(InstanceValidationError):
        instance(
            "self",
            [machine("M1")],
            [job("J1", [operation("A", "M1", 1, predecessors=("A",))])],
        )
    with pytest.raises(InstanceValidationError):
        instance(
            "meta", [machine("M1")], [job("J1", [operation("A", "M1", 1)])], metadata={"n": float("nan")}
        )
