"""Runtime invariants. A violation is a simulator defect, not an agent mistake."""

from __future__ import annotations

from factoryrl.core.events import TIME_TOLERANCE
from factoryrl.core.state import MachineStatus, OperationStatus, RuntimeState
from factoryrl.exceptions import InvariantViolationError
from factoryrl.models.topology import CompiledInstance


def assert_invariants(
    compiled: CompiledInstance,
    state: RuntimeState,
    now: float,
    queued_times: list[float],
) -> None:
    """Check the v0.1 safety properties against the current runtime state."""
    if now < -TIME_TOLERANCE:
        _fail("Simulation time is negative.", now=now)
    for queued in queued_times:
        if queued < now - TIME_TOLERANCE:
            _fail("An event is queued in the past.", now=now, queued=queued)

    running_on = [0 for _ in range(compiled.n_machines)]
    for operation, status in enumerate(state.status):
        machine = compiled.operation_machine[operation]
        job = compiled.operation_job[operation]
        if not 0 <= machine < compiled.n_machines or not 0 <= job < compiled.n_jobs:
            _fail("Operation points at a missing machine or job.", operation=operation)
        start = state.start[operation]
        end = state.end[operation]
        remaining_preds = sum(
            1
            for pred in compiled.predecessors[operation]
            if state.status[pred] is not OperationStatus.COMPLETED
        )
        if remaining_preds != state.pred_remaining[operation]:
            _fail(
                "Predecessor counter drifted.",
                operation_id=compiled.operation_ids[operation],
                expected=remaining_preds,
                actual=state.pred_remaining[operation],
            )
        if status is OperationStatus.RUNNING:
            running_on[machine] += 1
            if start is None or end is not None or start > now + TIME_TOLERANCE:
                _fail(
                    "Running operation has an inconsistent interval.",
                    operation_id=compiled.operation_ids[operation],
                )
        elif status is OperationStatus.COMPLETED:
            if start is None or end is None:
                _fail(
                    "Completed operation is missing start or end.",
                    operation_id=compiled.operation_ids[operation],
                )
            elif (
                abs((end - start) - compiled.operation_duration[operation]) > TIME_TOLERANCE
                or end + TIME_TOLERANCE < start
            ):
                _fail(
                    "Completed operation duration mismatch.", operation_id=compiled.operation_ids[operation]
                )
            elif end > now + TIME_TOLERANCE:
                _fail(
                    "Completed operation ends in the future.", operation_id=compiled.operation_ids[operation]
                )
        else:
            if start is not None or end is not None:
                _fail(
                    "An operation that is not running has a schedule.",
                    operation_id=compiled.operation_ids[operation],
                )
        if status is not OperationStatus.NOT_READY and remaining_preds != 0:
            _fail(
                "An operation advanced before its predecessors completed.",
                operation_id=compiled.operation_ids[operation],
            )
        if state.job_released[job] and remaining_preds == 0 and status is OperationStatus.NOT_READY:
            _fail(
                "A released operation with finished predecessors is not ready.",
                operation_id=compiled.operation_ids[operation],
            )
        in_ready = operation in state.ready_by_machine[machine]
        if (status is OperationStatus.READY) != in_ready:
            _fail(
                "Ready set does not match operation status.", operation_id=compiled.operation_ids[operation]
            )
        if status in (OperationStatus.RUNNING, OperationStatus.COMPLETED):
            for pred in compiled.predecessors[operation]:
                pred_end = state.end[pred]
                if start is not None and pred_end is not None and pred_end > start + TIME_TOLERANCE:
                    _fail(
                        "Operation starts before a predecessor ends.",
                        operation_id=compiled.operation_ids[operation],
                    )

    if sum(1 for status in state.status if status is OperationStatus.COMPLETED) != state.operations_completed:
        _fail("operations_completed does not match statuses.")
    if len(state.intervals) != state.operations_completed:
        _fail("Interval log length does not match completed operations.")

    jobs_done = 0
    for job, ops in enumerate(compiled.job_operations):
        done = all(state.status[op] is OperationStatus.COMPLETED for op in ops)
        if done:
            jobs_done += 1
            if state.job_remaining[job] != 0 or state.job_completion[job] is None:
                _fail("A completed job is missing its completion time.", job_id=compiled.job_ids[job])
        elif state.job_remaining[job] == 0 or state.job_completion[job] is not None:
            _fail("An unfinished job is marked complete.", job_id=compiled.job_ids[job])
    if jobs_done != state.jobs_completed:
        _fail("jobs_completed does not match job statuses.")

    for machine, machine_state in enumerate(state.machine_status):
        current = state.machine_current[machine]
        if running_on[machine] > 1:
            _fail(
                "A unit-capacity machine is running more than one operation.",
                machine_id=compiled.machine_ids[machine],
            )
        if machine_state is MachineStatus.BUSY:
            if current is None or state.status[current] is not OperationStatus.RUNNING:
                _fail("A busy machine has no running operation.", machine_id=compiled.machine_ids[machine])
            elif compiled.operation_machine[current] != machine:
                _fail(
                    "A machine is running an operation assigned elsewhere.",
                    machine_id=compiled.machine_ids[machine],
                )
        elif machine_state is MachineStatus.IDLE:
            if current is not None or running_on[machine] != 0:
                _fail(
                    "An idle machine still has a running operation.", machine_id=compiled.machine_ids[machine]
                )
        else:
            _fail("Unknown machine status.", machine_id=compiled.machine_ids[machine])
        spans = sorted((start, end) for m_index, _op, start, end in state.intervals if m_index == machine)
        if current is not None and state.start[current] is not None:
            spans.append((state.start[current], now))  # type: ignore[arg-type]
        for previous, nxt in zip(spans, spans[1:], strict=False):
            if nxt[0] < previous[1] - TIME_TOLERANCE:
                _fail("Machine schedule overlaps.", machine_id=compiled.machine_ids[machine])


def _fail(message: str, **context: object) -> None:
    raise InvariantViolationError(message, context=context)
