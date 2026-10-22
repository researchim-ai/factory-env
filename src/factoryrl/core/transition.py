"""Validated state transitions. Callers schedule the follow-up events."""

from __future__ import annotations

from factoryrl.core.events import TIME_TOLERANCE
from factoryrl.core.state import MachineStatus, OperationStatus, RuntimeState, mark_ready
from factoryrl.exceptions import InvariantViolationError
from factoryrl.models.topology import CompiledInstance


def apply_job_release(
    state: RuntimeState,
    compiled: CompiledInstance,
    job_index: int,
    now: float,
) -> None:
    """Release a job and mark operations with no outstanding predecessors ready."""
    if state.job_released[job_index]:
        raise InvariantViolationError(
            f"Job {compiled.job_ids[job_index]} was released twice.",
            context={"job_id": compiled.job_ids[job_index]},
        )
    state.job_released[job_index] = True
    for operation in compiled.job_operations[job_index]:
        if state.pred_remaining[operation] == 0 and state.status[operation] is OperationStatus.NOT_READY:
            mark_ready(state, compiled, operation, now)


def apply_operation_start(
    state: RuntimeState,
    compiled: CompiledInstance,
    operation: int,
    now: float,
) -> float:
    """Start an operation on its idle machine. Returns the finish time."""
    op_id = compiled.operation_ids[operation]
    if state.status[operation] is not OperationStatus.READY:
        raise InvariantViolationError(
            f"Operation {op_id} cannot start from status {state.status[operation].value}.",
            context={"operation_id": op_id},
        )
    machine = compiled.operation_machine[operation]
    if state.machine_status[machine] is not MachineStatus.IDLE or state.machine_current[machine] is not None:
        raise InvariantViolationError(
            f"Machine {compiled.machine_ids[machine]} is not idle for {op_id}.",
            context={"operation_id": op_id, "machine_id": compiled.machine_ids[machine]},
        )
    if operation not in state.ready_by_machine[machine]:
        raise InvariantViolationError(f"Operation {op_id} is ready but missing from the machine ready set.")
    for predecessor in compiled.predecessors[operation]:
        if state.status[predecessor] is not OperationStatus.COMPLETED:
            raise InvariantViolationError(
                f"Operation {op_id} started before predecessor "
                f"{compiled.operation_ids[predecessor]} completed."
            )
        pred_end = state.end[predecessor]
        if pred_end is None or pred_end > now + TIME_TOLERANCE:
            raise InvariantViolationError(f"Operation {op_id} started before its predecessor finished.")
    state.ready_by_machine[machine].remove(operation)
    state.status[operation] = OperationStatus.RUNNING
    state.start[operation] = now
    state.machine_status[machine] = MachineStatus.BUSY
    state.machine_current[machine] = operation
    return now + compiled.operation_duration[operation]


def apply_operation_finish(
    state: RuntimeState,
    compiled: CompiledInstance,
    operation: int,
    now: float,
) -> None:
    """Complete a running operation and unlock the successor in its chain."""
    op_id = compiled.operation_ids[operation]
    if state.status[operation] is not OperationStatus.RUNNING:
        raise InvariantViolationError(
            f"Operation {op_id} received a finish event while {state.status[operation].value}.",
            context={"operation_id": op_id},
        )
    start = state.start[operation]
    if start is None:
        raise InvariantViolationError(f"Operation {op_id} finished without a start time.")
    duration = compiled.operation_duration[operation]
    if abs((now - start) - duration) > TIME_TOLERANCE or now + TIME_TOLERANCE < start:
        raise InvariantViolationError(
            f"Operation {op_id} finish time does not match its duration.",
            context={"start": start, "end": now, "duration": duration},
        )
    machine = compiled.operation_machine[operation]
    if state.machine_current[machine] != operation or state.machine_status[machine] is not MachineStatus.BUSY:
        raise InvariantViolationError(f"Machine {compiled.machine_ids[machine]} is not running {op_id}.")
    state.status[operation] = OperationStatus.COMPLETED
    state.end[operation] = now
    state.machine_status[machine] = MachineStatus.IDLE
    state.machine_current[machine] = None
    state.machine_busy[machine] += duration
    state.intervals.append((machine, operation, start, now))
    state.operations_completed += 1
    job = compiled.operation_job[operation]
    state.job_remaining[job] -= 1
    if state.job_remaining[job] < 0:
        raise InvariantViolationError(f"Job {compiled.job_ids[job]} completed more operations than it has.")
    if state.job_remaining[job] == 0:
        state.job_completion[job] = now
        state.jobs_completed += 1
    for successor in compiled.successors[operation]:
        state.pred_remaining[successor] -= 1
        if state.pred_remaining[successor] < 0:
            raise InvariantViolationError(
                f"Predecessor count of {compiled.operation_ids[successor]} became negative."
            )
        if (
            state.pred_remaining[successor] == 0
            and state.job_released[compiled.operation_job[successor]]
            and state.status[successor] is OperationStatus.NOT_READY
        ):
            mark_ready(state, compiled, successor, now)
