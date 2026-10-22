"""Discrete-event engine for non-preemptive job shops.

The agent is asked for an action only at a decision point: at least one
ready operation has an idle machine. ``step`` applies that action and
advances until the next decision or until the episode stops. Events that
share a timestamp are drained, in priority order, before the next decision.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from factoryrl.core.clock import SimulationClock
from factoryrl.core.event_queue import EventQueue
from factoryrl.core.events import (
    EVENT_PRIORITY,
    TIME_TOLERANCE,
    WAIT_ACTION,
    Event,
    EventType,
    Phase,
)
from factoryrl.core.invariants import assert_invariants
from factoryrl.core.metrics import production_metrics
from factoryrl.core.rng import validate_seed
from factoryrl.core.state import (
    RuntimeState,
    StateSnapshot,
    build_snapshot,
    initial_state,
)
from factoryrl.core.transition import (
    apply_job_release,
    apply_operation_finish,
    apply_operation_start,
)
from factoryrl.exceptions import ConfigurationError, InvalidActionError, InvariantViolationError
from factoryrl.models.factory import FactoryInstance
from factoryrl.models.topology import CompiledInstance, compile_instance


@dataclass(frozen=True, slots=True)
class EngineStep:
    """Result of applying one dispatch or WAIT action."""

    delta_sim_time: float
    terminated: bool
    truncated: bool
    termination_reason: str | None
    truncation_reason: str | None
    success: bool | None


class SimulationEngine:
    """Deterministic job-shop simulator.

    Actions are operation indexes, or :data:`WAIT` (``-1``). The Gymnasium
    adapter maps the padded discrete action onto this interface.
    """

    def __init__(
        self,
        instance: FactoryInstance,
        *,
        horizon: float | None = None,
        allow_voluntary_idle: bool = False,
        max_events: int = 1_000_000,
        max_decisions: int = 1_000_000,
        check_invariants: bool = True,
        record_events: bool = True,
    ) -> None:
        if max_events < 1 or max_decisions < 1:
            raise ConfigurationError("max_events and max_decisions must be at least 1.")
        if isinstance(allow_voluntary_idle, bool) is False:
            raise ConfigurationError("allow_voluntary_idle must be a bool.")
        self.compiled: CompiledInstance = compile_instance(instance)
        self.horizon = instance.horizon if horizon is None else horizon
        if self.horizon is not None and (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, (int, float))
            or not math.isfinite(float(self.horizon))
            or float(self.horizon) <= 0.0
        ):
            raise ConfigurationError("horizon must be a finite number > 0, or None.")
        if self.horizon is not None:
            self.horizon = float(self.horizon)
        self.allow_voluntary_idle = allow_voluntary_idle
        self.max_events = max_events
        self.max_decisions = max_decisions
        self.check_invariants = check_invariants
        self.record_events = record_events
        self.instance = instance
        self.seed: int | None = None
        self.state: RuntimeState = initial_state(self.compiled)
        self.queue = EventQueue()
        self.clock = SimulationClock()
        self.trace: list[dict[str, Any]] = []
        self.applied_actions: list[int] = []
        self.phase = Phase.DECISION
        self.sequence = 1
        self.event_count = 0
        self.decision_count = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.success: bool | None = None
        self.termination_reason: str | None = None
        self.truncation_reason: str | None = None
        self.diagnosis: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None) -> None:
        """Return the engine to the instance start and stop at the first decision."""
        if seed is not None:
            validate_seed(seed)
        self.seed = seed
        self.state = initial_state(self.compiled)
        self.queue = EventQueue()
        self.clock = SimulationClock()
        self.trace = []
        self.applied_actions = []
        self.phase = Phase.DECISION
        self.sequence = 1
        self.event_count = 0
        self.decision_count = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.success = None
        self.termination_reason = None
        self.truncation_reason = None
        self.diagnosis = {}
        for job_index in range(self.compiled.n_jobs):
            self._schedule(
                EventType.JOB_RELEASE,
                self.compiled.job_release[job_index],
                {"job_index": job_index},
            )
        self._run(must_advance=False)
        if self.phase is not Phase.DECISION:
            reason = self.termination_reason or self.truncation_reason
            raise ConfigurationError(
                "The episode ended during reset, before the first decision. "
                f"Reason: {reason}. Increase the event budget or fix the instance.",
                context={"reason": reason, **self.diagnosis},
            )

    def step(self, action: int) -> EngineStep:
        """Apply one legal action and advance to the next decision or stop."""
        if self.phase is not Phase.DECISION:
            raise InvalidActionError(
                "There is no active decision point. Call reset() before the next episode.",
                context={"phase": self.phase.value},
            )
        if isinstance(action, bool):
            self.invalid_action_count += 1
            raise InvalidActionError(
                "Action must be an integer operation index, not a boolean.",
                context={"action": action},
            )
        try:
            action = operator.index(action)
        except TypeError as exc:
            self.invalid_action_count += 1
            raise InvalidActionError(
                f"Action must be an integer operation index or WAIT ({WAIT_ACTION}).",
                context={"action": action},
            ) from exc
        started = self.clock.now
        try:
            self._prepare_action(action)
        except InvalidActionError:
            self.invalid_action_count += 1
            raise
        self.decision_count += 1
        self.valid_action_count += 1
        self.applied_actions.append(action)
        self._run(must_advance=action == WAIT_ACTION)
        return EngineStep(
            delta_sim_time=self.clock.now - started,
            terminated=self.termination_reason is not None,
            truncated=self.truncation_reason is not None,
            termination_reason=self.termination_reason,
            truncation_reason=self.truncation_reason,
            success=self.success,
        )

    def dispatchable_indices(self) -> list[int]:
        """Operation indexes the agent may start at the current decision."""
        if self.phase is not Phase.DECISION:
            return []
        return self._idle_ready_ops()

    def wait_allowed(self) -> bool:
        """WAIT is legal only in voluntary-idle mode when time can move forward."""
        if self.phase is not Phase.DECISION or not self.allow_voluntary_idle:
            return False
        nxt = self.queue.peek()
        return nxt is not None and nxt.time > self.clock.now + TIME_TOLERANCE

    def snapshot(self) -> StateSnapshot:
        """Read-only view of the current logical state."""
        return build_snapshot(self.compiled, self.state, self.clock.now)

    def metrics(self) -> dict[str, Any]:
        """Schedule metrics. Wall-clock time is intentionally absent."""
        return production_metrics(
            self.compiled,
            self.snapshot(),
            success=self.success,
            termination_reason=self.termination_reason,
            truncation_reason=self.truncation_reason,
            decision_count=self.decision_count,
            event_count=self.event_count,
            valid_action_count=self.valid_action_count,
            invalid_action_count=self.invalid_action_count,
        )

    def public_intervals(self) -> list[dict[str, Any]]:
        """Completed operation intervals, in the order they finished."""
        snapshot = self.snapshot()
        return [
            {
                "machine_id": interval.machine_id,
                "operation_id": interval.operation_id,
                "job_id": interval.job_id,
                "start": interval.start,
                "end": interval.end,
            }
            for interval in snapshot.intervals
        ]

    def _prepare_action(self, action: int) -> None:
        if action == WAIT_ACTION:
            if not self.allow_voluntary_idle:
                raise InvalidActionError(
                    "WAIT is disabled while dispatch actions exist (allow_voluntary_idle is false).",
                    context=self._action_context(action),
                )
            nxt = self.queue.peek()
            if nxt is None or nxt.time <= self.clock.now + TIME_TOLERANCE:
                raise InvalidActionError(
                    "WAIT requires an event strictly later than the current time.",
                    context=self._action_context(action),
                )
            return
        if action < 0 or action >= self.compiled.n_operations:
            raise InvalidActionError(
                "Action index is outside this instance.",
                context=self._action_context(action),
            )
        if action not in self._idle_ready_ops():
            raise InvalidActionError(
                f"Operation {self.compiled.operation_ids[action]} is not dispatchable.",
                context=self._action_context(action),
            )
        self._schedule(
            EventType.OPERATION_START,
            self.clock.now,
            {"operation_index": action},
        )

    def _action_context(self, action: int) -> dict[str, Any]:
        legal = [self.compiled.operation_ids[index] for index in self._idle_ready_ops()]
        return {
            "action": action,
            "now": self.clock.now,
            "dispatchable_operation_ids": legal,
            "wait_allowed": self.allow_voluntary_idle and self._has_future_event(),
        }

    def _run(self, must_advance: bool) -> None:
        started = self.clock.now
        guard_limit = self.max_events + self.compiled.n_operations + 8
        for _ in range(guard_limit):
            if not self._drain_now():
                self._stop_truncated("max_events")
                return
            if self._all_complete():
                self._stop_success()
                return
            if self.horizon is not None and self.clock.now >= self.horizon - TIME_TOLERANCE:
                self._stop_truncated("horizon")
                return
            advanced = self.clock.now > started + TIME_TOLERANCE
            if must_advance and not advanced:
                self._advance_one(require_future=True)
                if self.phase is Phase.TERMINAL:
                    return
                continue
            if self._idle_ready_ops():
                if self.decision_count >= self.max_decisions:
                    self._stop_truncated("max_decisions")
                    return
                self._enter_decision()
                return
            self._advance_one(require_future=False)
            if self.phase is Phase.TERMINAL:
                return
        raise InvariantViolationError("The event loop exceeded its safety guard. This is a simulator bug.")

    def _drain_now(self) -> bool:
        while True:
            nxt = self.queue.peek()
            if nxt is None or nxt.time > self.clock.now + TIME_TOLERANCE:
                return True
            if nxt.time < self.clock.now - TIME_TOLERANCE:
                raise InvariantViolationError(
                    "Refusing to process an event from the past.",
                    context={"now": self.clock.now, "event_time": nxt.time, "event_id": nxt.event_id},
                )
            if not self._process_next():
                return False

    def _advance_one(self, *, require_future: bool) -> None:
        nxt = self.queue.peek()
        if nxt is None or nxt.time <= self.clock.now + TIME_TOLERANCE:
            if require_future:
                raise InvalidActionError(
                    "WAIT cannot advance time because no future event remains.",
                    context={"now": self.clock.now},
                )
            self._stop_deadlock()
            return
        if self.horizon is not None and nxt.time > self.horizon + TIME_TOLERANCE:
            self.clock.advance_to(float(self.horizon))
            self._stop_truncated("horizon")
            return
        if not self._process_next():
            self._stop_truncated("max_events")

    def _process_next(self) -> bool:
        if self.event_count >= self.max_events:
            return False
        event = self.queue.pop()
        self.clock.advance_to(event.time)
        self.event_count += 1
        self._remember(event)
        payload = event.payload
        if event.type is EventType.JOB_RELEASE:
            apply_job_release(self.state, self.compiled, int(payload["job_index"]), self.clock.now)
        elif event.type is EventType.OPERATION_START:
            operation = int(payload["operation_index"])
            finish_at = apply_operation_start(self.state, self.compiled, operation, self.clock.now)
            self._schedule(EventType.OPERATION_FINISH, finish_at, {"operation_index": operation})
        elif event.type is EventType.OPERATION_FINISH:
            apply_operation_finish(
                self.state,
                self.compiled,
                int(payload["operation_index"]),
                self.clock.now,
            )
        else:
            raise InvariantViolationError(f"Unexpected queued event type {event.type.value}.")
        if self.check_invariants:
            self._check_invariants()
        return True

    def _enter_decision(self) -> None:
        self.phase = Phase.DECISION
        self.success = None
        self.termination_reason = None
        self.truncation_reason = None
        ids = [self.compiled.operation_ids[index] for index in self._idle_ready_ops()]
        self._log_marker(EventType.DECISION_REQUIRED, {"dispatchable_operation_ids": ids})

    def _stop_success(self) -> None:
        self._stop(terminated=True, truncated=False, reason="completed", success=True)

    def _stop_deadlock(self) -> None:
        self.diagnosis = self._deadlock_diagnosis()
        self._stop(terminated=True, truncated=False, reason="deadlock", success=False)

    def _stop_truncated(self, reason: str) -> None:
        self.diagnosis = {"budget": reason, "now": self.clock.now}
        self._stop(terminated=False, truncated=True, reason=reason, success=False)

    def _stop(self, *, terminated: bool, truncated: bool, reason: str, success: bool) -> None:
        self.phase = Phase.TERMINAL
        self.success = success
        self.termination_reason = reason if terminated else None
        self.truncation_reason = reason if truncated else None
        self._log_marker(EventType.EPISODE_END, {"reason": reason, "success": success})
        if self.check_invariants:
            self._check_invariants()

    def _all_complete(self) -> bool:
        return self.state.operations_completed == self.compiled.n_operations

    def _idle_ready_ops(self) -> list[int]:
        ready: list[int] = []
        for machine, status in enumerate(self.state.machine_status):
            if status.value == "idle":
                ready.extend(self.state.ready_by_machine[machine])
        ready.sort()
        return ready

    def _has_future_event(self) -> bool:
        nxt = self.queue.peek()
        return nxt is not None and nxt.time > self.clock.now + TIME_TOLERANCE

    def _schedule(self, event_type: EventType, time: float, payload: dict[str, Any]) -> Event:
        if time < self.clock.now - TIME_TOLERANCE:
            raise InvariantViolationError(
                "Cannot schedule an event in the past.",
                context={"now": self.clock.now, "time": time, "type": event_type.value},
            )
        event = self._make_event(event_type, time, payload)
        self.queue.push(event)
        return event

    def _log_marker(self, event_type: EventType, payload: dict[str, Any]) -> None:
        event = self._make_event(event_type, self.clock.now, payload)
        self._remember(event)

    def _make_event(self, event_type: EventType, time: float, payload: dict[str, Any]) -> Event:
        sequence = self.sequence
        self.sequence += 1
        frozen = {key: tuple(value) if isinstance(value, list) else value for key, value in payload.items()}
        return Event(
            time=float(time),
            priority=EVENT_PRIORITY[event_type],
            sequence=sequence,
            event_id=f"e{sequence}",
            type=event_type,
            payload=MappingProxyType(frozen),
        )

    def _remember(self, event: Event) -> None:
        if not self.record_events:
            return
        self.trace.append(
            {
                "time": event.time,
                "priority": event.priority,
                "sequence": event.sequence,
                "event_id": event.event_id,
                "type": event.type.value,
                "payload": self._public_payload(event),
            }
        )

    def _public_payload(self, event: Event) -> dict[str, Any]:
        payload = dict(event.payload)
        if event.type is EventType.JOB_RELEASE:
            job_index = int(payload["job_index"])
            return {"job_id": self.compiled.job_ids[job_index]}
        if event.type in (EventType.OPERATION_START, EventType.OPERATION_FINISH):
            operation = int(payload["operation_index"])
            job = self.compiled.operation_job[operation]
            machine = self.compiled.operation_machine[operation]
            return {
                "operation_id": self.compiled.operation_ids[operation],
                "job_id": self.compiled.job_ids[job],
                "machine_id": self.compiled.machine_ids[machine],
                "duration": self.compiled.operation_duration[operation],
            }
        if event.type is EventType.DECISION_REQUIRED:
            ids = payload["dispatchable_operation_ids"]
            return {"dispatchable_operation_ids": list(ids)}
        if event.type is EventType.EPISODE_END:
            return {"reason": payload["reason"], "success": bool(payload["success"])}
        raise InvariantViolationError(f"Cannot publish event type {event.type.value}.")

    def _check_invariants(self) -> None:
        assert_invariants(self.compiled, self.state, self.clock.now, self.queue.times())

    def _deadlock_diagnosis(self) -> dict[str, Any]:
        unfinished = [
            {
                "operation_id": self.compiled.operation_ids[index],
                "status": status.value,
                "machine_id": self.compiled.machine_ids[self.compiled.operation_machine[index]],
            }
            for index, status in enumerate(self.state.status)
            if status.value != "completed"
        ]
        machines = [
            {"machine_id": self.compiled.machine_ids[index], "status": status.value}
            for index, status in enumerate(self.state.machine_status)
        ]
        return {
            "now": self.clock.now,
            "queue_empty": len(self.queue) == 0,
            "unfinished_operations": unfinished,
            "machines": machines,
        }
