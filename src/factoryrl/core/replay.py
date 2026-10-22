"""Replay a recorded action list and compare the event trace."""

from __future__ import annotations

from typing import Any

from factoryrl.core.engine import SimulationEngine
from factoryrl.core.events import TIME_TOLERANCE
from factoryrl.exceptions import ReplayMismatchError
from factoryrl.models.factory import FactoryInstance
from factoryrl.version import __version__


def export_trace(
    engine: SimulationEngine,
    *,
    scenario_hash: str,
    config_hash: str,
    agent_id: str | None,
) -> dict[str, Any]:
    """Build a JSON-ready trace from an engine that recorded events."""
    return {
        "package_version": __version__,
        "schema_version": engine.instance.schema_version,
        "rng_algorithm": "numpy.random.PCG64",
        "seed": engine.seed,
        "scenario_hash": scenario_hash,
        "agent_id": agent_id,
        "config_hash": config_hash,
        "actions": list(engine.applied_actions),
        "events": list(engine.trace),
        "intervals": engine.public_intervals(),
        "metrics": engine.metrics(),
    }


def assert_replay(
    instance: FactoryInstance,
    actions: list[int],
    expected_events: list[dict[str, Any]],
    expected_metrics: dict[str, Any],
    *,
    seed: int | None,
    horizon: float | None,
    allow_voluntary_idle: bool,
    max_events: int,
    max_decisions: int,
    scenario_hash: str | None = None,
    expected_hash: str | None = None,
) -> SimulationEngine:
    """Re-execute ``actions`` and raise if events or schedule metrics differ.

    Event order must match. Floating event times may differ by at most
    ``TIME_TOLERANCE``. Wall-clock fields are not part of the contract.
    """
    if expected_hash is not None and scenario_hash is not None and expected_hash != scenario_hash:
        raise ReplayMismatchError(
            "Scenario hash does not match the instance being replayed.",
            context={"expected": expected_hash, "actual": scenario_hash},
        )
    engine = SimulationEngine(
        instance,
        horizon=horizon,
        allow_voluntary_idle=allow_voluntary_idle,
        max_events=max_events,
        max_decisions=max_decisions,
        check_invariants=True,
        record_events=True,
    )
    engine.reset(seed=seed)
    for action in actions:
        if engine.phase.value != "decision":
            break
        engine.step(action)
    if not events_match(engine.trace, expected_events):
        raise ReplayMismatchError(
            "Replayed events do not match the recorded trace.",
            context={"expected_length": len(expected_events), "actual_length": len(engine.trace)},
        )
    actual = engine.metrics()
    if not schedule_metrics_match(actual, expected_metrics):
        raise ReplayMismatchError(
            "Replayed schedule metrics do not match the recording.",
            context={
                "expected_makespan": expected_metrics.get("makespan"),
                "actual_makespan": actual.get("makespan"),
            },
        )
    return engine


def events_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """Compare traces by order, type, priority, payload, and time."""
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected, strict=True):
        if left["type"] != right["type"] or left["priority"] != right["priority"]:
            return False
        if abs(float(left["time"]) - float(right["time"])) > TIME_TOLERANCE:
            return False
        if left["payload"] != right["payload"]:
            return False
    return True


def schedule_metrics_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare production metrics that are functions of the schedule."""
    keys = (
        "makespan",
        "sim_time",
        "mean_flow_time",
        "total_tardiness",
        "weighted_tardiness",
        "jobs_completed",
        "operations_completed",
        "success",
        "termination_reason",
        "truncation_reason",
        "decision_count",
        "event_count",
    )
    for key in keys:
        if not _close(actual.get(key), expected.get(key)):
            return False
    if not _mapping_close(actual.get("machine_busy_time", {}), expected.get("machine_busy_time", {})):
        return False
    return True


def _mapping_close(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    return all(_close(actual[key], expected[key]) for key in actual)


def _close(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual == expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-6
    return bool(actual == expected)
