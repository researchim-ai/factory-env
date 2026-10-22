"""Versioned observation features for the v0.1 scheduling environments.

Schema ``obs_v1``. Every feature is known at the decision instant: machine
and operation status, static durations, due dates, and priorities. Future
random events are not part of v0.1 and must not be added here.

Times are divided by ``time_scale``. A missing due date or horizon is the
sentinel ``-1``. Padding rows are zeros and are excluded by the binary masks.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from factoryrl.core.state import StateSnapshot
from factoryrl.exceptions import ConfigurationError
from factoryrl.observations.normalization import scale_time

OBS_SCHEMA_VERSION = 1


class MachineFeature(IntEnum):
    IS_IDLE = 0
    IS_BUSY = 1
    REMAINING_TIME = 2
    BUSY_TIME = 3
    PENDING_LOAD = 4
    READY_COUNT = 5


class OperationFeature(IntEnum):
    IS_NOT_READY = 0
    IS_READY = 1
    IS_RUNNING = 2
    IS_COMPLETED = 3
    DURATION = 4
    REMAINING_TIME = 5
    WAITING_TIME = 6
    MACHINE_INDEX = 7
    JOB_INDEX = 8
    POSITION_INDEX = 9
    DUE_DATE = 10
    PRIORITY = 11
    DISPATCHABLE = 12


class GlobalFeature(IntEnum):
    NOW = 0
    HORIZON = 1
    FRACTION_OPERATIONS_COMPLETED = 2
    FRACTION_JOBS_COMPLETED = 3
    FRACTION_DISPATCHABLE = 4
    MEAN_UTILIZATION = 5


N_MACHINE_FEATURES = len(MachineFeature)
N_OPERATION_FEATURES = len(OperationFeature)
N_GLOBAL_FEATURES = len(GlobalFeature)


def build_observation(
    snapshot: StateSnapshot,
    *,
    max_machines: int,
    max_operations: int,
    time_scale: float,
    horizon: float | None,
    observation_bound: float,
) -> dict[str, np.ndarray]:
    """Build the ``obs_v1`` dictionary. Arrays are newly allocated."""
    if max_machines < snapshot.n_machines or max_operations < snapshot.n_operations:
        raise ConfigurationError("Observation capacity is smaller than the instance.")
    machines = np.zeros((max_machines, N_MACHINE_FEATURES), dtype=np.float32)
    operations = np.zeros((max_operations, N_OPERATION_FEATURES), dtype=np.float32)
    machine_mask = np.zeros(max_machines, dtype=np.int8)
    operation_mask = np.zeros(max_operations, dtype=np.int8)
    machine_mask[: snapshot.n_machines] = 1
    operation_mask[: snapshot.n_operations] = 1

    n_machines = max(snapshot.n_machines, 1)
    n_jobs = max(snapshot.n_jobs, 1)
    machine_norm = {machine.id: machine.index / n_machines for machine in snapshot.machines}
    for view in snapshot.machines:
        row = machines[view.index]
        row[MachineFeature.IS_IDLE] = 1.0 if view.status.value == "idle" else 0.0
        row[MachineFeature.IS_BUSY] = 1.0 if view.status.value == "busy" else 0.0
        row[MachineFeature.REMAINING_TIME] = scale_time(view.remaining_time, time_scale)
        row[MachineFeature.BUSY_TIME] = scale_time(view.busy_time, time_scale)
        row[MachineFeature.PENDING_LOAD] = scale_time(view.pending_load, time_scale)
        row[MachineFeature.READY_COUNT] = float(view.ready_count)

    dispatchable = 0
    job_of: dict[str, int] = {}
    for operation in snapshot.operations:
        if operation.job_id not in job_of:
            job_of[operation.job_id] = len(job_of)
        row = operations[operation.index]
        row[OperationFeature.IS_NOT_READY] = 1.0 if operation.status.value == "not_ready" else 0.0
        row[OperationFeature.IS_READY] = 1.0 if operation.status.value == "ready" else 0.0
        row[OperationFeature.IS_RUNNING] = 1.0 if operation.status.value == "running" else 0.0
        row[OperationFeature.IS_COMPLETED] = 1.0 if operation.status.value == "completed" else 0.0
        row[OperationFeature.DURATION] = scale_time(operation.duration, time_scale)
        row[OperationFeature.REMAINING_TIME] = scale_time(operation.remaining_time, time_scale)
        row[OperationFeature.WAITING_TIME] = scale_time(operation.waiting_time, time_scale)
        row[OperationFeature.MACHINE_INDEX] = machine_norm[operation.machine_id]
        row[OperationFeature.JOB_INDEX] = job_of[operation.job_id] / n_jobs
        row[OperationFeature.POSITION_INDEX] = operation.position / max(operation.job_length, 1)
        row[OperationFeature.DUE_DATE] = (
            -1.0 if operation.due_date is None else scale_time(operation.due_date, time_scale)
        )
        row[OperationFeature.PRIORITY] = float(operation.priority)
        row[OperationFeature.DISPATCHABLE] = 1.0 if operation.dispatchable else 0.0
        dispatchable += int(operation.dispatchable)

    global_row = np.zeros(N_GLOBAL_FEATURES, dtype=np.float32)
    global_row[GlobalFeature.NOW] = scale_time(snapshot.now, time_scale)
    global_row[GlobalFeature.HORIZON] = -1.0 if horizon is None else scale_time(horizon, time_scale)
    global_row[GlobalFeature.FRACTION_OPERATIONS_COMPLETED] = (
        snapshot.operations_completed / snapshot.n_operations
    )
    global_row[GlobalFeature.FRACTION_JOBS_COMPLETED] = snapshot.jobs_completed / snapshot.n_jobs
    global_row[GlobalFeature.FRACTION_DISPATCHABLE] = dispatchable / snapshot.n_operations
    if snapshot.now > 0.0 and snapshot.machines:
        global_row[GlobalFeature.MEAN_UTILIZATION] = float(
            np.mean([machine.busy_time / snapshot.now for machine in snapshot.machines])
        )

    _check_bounds(machines, observation_bound, "machines")
    _check_bounds(operations, observation_bound, "operations")
    _check_bounds(global_row, observation_bound, "global")
    return {
        "machines": machines,
        "operations": operations,
        "machine_mask": machine_mask,
        "operation_mask": operation_mask,
        "global": global_row,
    }


def _check_bounds(array: np.ndarray, bound: float, name: str) -> None:
    if not np.isfinite(array).all():
        raise ConfigurationError(f"Observation feature '{name}' is not finite.")
    if float(array.min(initial=0.0)) < -1.0 - 1e-6 or float(array.max(initial=0.0)) > bound + 1e-6:
        raise ConfigurationError(
            f"Observation feature '{name}' exceeds observation_bound={bound}. "
            "Increase observation_bound or time_scale.",
            context={"feature": name, "min": float(array.min()), "max": float(array.max())},
        )
