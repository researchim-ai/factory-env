"""Mutable runtime state and the read-only snapshot shown to callers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from factoryrl.models.topology import CompiledInstance


class OperationStatus(StrEnum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"


class MachineStatus(StrEnum):
    IDLE = "idle"
    BUSY = "busy"


@dataclass(slots=True)
class RuntimeState:
    """Arrays mutated only by the transition functions."""

    status: list[OperationStatus]
    start: list[float | None]
    end: list[float | None]
    ready_time: list[float | None]
    machine_status: list[MachineStatus]
    machine_current: list[int | None]
    machine_busy: list[float]
    pred_remaining: list[int]
    job_remaining: list[int]
    job_released: list[bool]
    job_completion: list[float | None]
    ready_by_machine: list[set[int]]
    intervals: list[tuple[int, int, float, float]]
    operations_completed: int
    jobs_completed: int


def initial_state(compiled: CompiledInstance) -> RuntimeState:
    """State before any release event. Nothing is ready yet."""
    n_ops = compiled.n_operations
    return RuntimeState(
        status=[OperationStatus.NOT_READY for _ in range(n_ops)],
        start=[None for _ in range(n_ops)],
        end=[None for _ in range(n_ops)],
        ready_time=[None for _ in range(n_ops)],
        machine_status=[MachineStatus.IDLE for _ in range(compiled.n_machines)],
        machine_current=[None for _ in range(compiled.n_machines)],
        machine_busy=[0.0 for _ in range(compiled.n_machines)],
        pred_remaining=[len(preds) for preds in compiled.predecessors],
        job_remaining=[len(ops) for ops in compiled.job_operations],
        job_released=[False for _ in range(compiled.n_jobs)],
        job_completion=[None for _ in range(compiled.n_jobs)],
        ready_by_machine=[set() for _ in range(compiled.n_machines)],
        intervals=[],
        operations_completed=0,
        jobs_completed=0,
    )


def mark_ready(state: RuntimeState, compiled: CompiledInstance, operation: int, now: float) -> None:
    """Move an operation to READY and index it on its machine."""
    if state.status[operation] is not OperationStatus.NOT_READY:
        from factoryrl.exceptions import InvariantViolationError

        raise InvariantViolationError(
            f"Operation {compiled.operation_ids[operation]} cannot become ready "
            f"from {state.status[operation].value}."
        )
    state.status[operation] = OperationStatus.READY
    state.ready_time[operation] = now
    machine = compiled.operation_machine[operation]
    state.ready_by_machine[machine].add(operation)


@dataclass(frozen=True, slots=True)
class IntervalView:
    machine_id: str
    operation_id: str
    job_id: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class OperationView:
    id: str
    index: int
    job_id: str
    machine_id: str
    status: OperationStatus
    duration: float
    start_time: float | None
    end_time: float | None
    ready_time: float | None
    remaining_time: float
    waiting_time: float
    due_date: float | None
    priority: float
    position: int
    job_length: int
    dispatchable: bool


@dataclass(frozen=True, slots=True)
class MachineView:
    id: str
    index: int
    status: MachineStatus
    current_operation_id: str | None
    remaining_time: float
    busy_time: float
    pending_load: float
    ready_count: int


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Read-only copy. Mutating it does not change the simulator."""

    now: float
    operations: tuple[OperationView, ...]
    machines: tuple[MachineView, ...]
    intervals: tuple[IntervalView, ...]
    jobs_completed: int
    operations_completed: int
    job_completion: tuple[float | None, ...]
    n_jobs: int
    n_operations: int
    n_machines: int


def _remaining(duration: float, status: OperationStatus, start: float | None, now: float) -> float:
    if status is OperationStatus.COMPLETED:
        return 0.0
    if status is OperationStatus.RUNNING and start is not None:
        return max(0.0, duration - (now - start))
    return duration


def build_snapshot(compiled: CompiledInstance, state: RuntimeState, now: float) -> StateSnapshot:
    """Copy runtime arrays into frozen views."""
    pending = [0.0 for _ in range(compiled.n_machines)]
    for operation, status in enumerate(state.status):
        if status is not OperationStatus.COMPLETED:
            pending[compiled.operation_machine[operation]] += compiled.operation_duration[operation]

    operations: list[OperationView] = []
    for index in range(compiled.n_operations):
        machine = compiled.operation_machine[index]
        job = compiled.operation_job[index]
        status = state.status[index]
        start = state.start[index]
        ready_time = state.ready_time[index]
        machine_idle = state.machine_status[machine] is MachineStatus.IDLE
        waiting = 0.0
        if status is OperationStatus.READY and ready_time is not None:
            waiting = max(0.0, now - ready_time)
        operations.append(
            OperationView(
                id=compiled.operation_ids[index],
                index=index,
                job_id=compiled.job_ids[job],
                machine_id=compiled.machine_ids[machine],
                status=status,
                duration=compiled.operation_duration[index],
                start_time=start,
                end_time=state.end[index],
                ready_time=ready_time,
                remaining_time=_remaining(compiled.operation_duration[index], status, start, now),
                waiting_time=waiting,
                due_date=compiled.job_due[job],
                priority=compiled.job_priority[job],
                position=compiled.operation_position[index],
                job_length=compiled.job_length[job],
                dispatchable=status is OperationStatus.READY and machine_idle,
            )
        )

    machines: list[MachineView] = []
    for index in range(compiled.n_machines):
        current = state.machine_current[index]
        remaining = 0.0
        busy = state.machine_busy[index]
        current_id = None
        if current is not None and state.start[current] is not None:
            current_id = compiled.operation_ids[current]
            started = state.start[current]
            assert started is not None
            remaining = _remaining(
                compiled.operation_duration[current],
                state.status[current],
                started,
                now,
            )
            busy += max(0.0, now - started)
        machines.append(
            MachineView(
                id=compiled.machine_ids[index],
                index=index,
                status=state.machine_status[index],
                current_operation_id=current_id,
                remaining_time=remaining,
                busy_time=busy,
                pending_load=pending[index],
                ready_count=len(state.ready_by_machine[index]),
            )
        )

    intervals: list[IntervalView] = []
    for machine, operation, start, end in state.intervals:
        job = compiled.operation_job[operation]
        intervals.append(
            IntervalView(
                machine_id=compiled.machine_ids[machine],
                operation_id=compiled.operation_ids[operation],
                job_id=compiled.job_ids[job],
                start=start,
                end=end,
            )
        )
    return StateSnapshot(
        now=now,
        operations=tuple(operations),
        machines=tuple(machines),
        intervals=tuple(intervals),
        jobs_completed=state.jobs_completed,
        operations_completed=state.operations_completed,
        job_completion=tuple(state.job_completion),
        n_jobs=compiled.n_jobs,
        n_operations=compiled.n_operations,
        n_machines=compiled.n_machines,
    )
