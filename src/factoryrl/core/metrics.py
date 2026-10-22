"""Production metrics derived from the schedule, not from the reward signal."""

from __future__ import annotations

from typing import Any

from factoryrl.core.state import StateSnapshot
from factoryrl.models.topology import CompiledInstance


def production_metrics(
    compiled: CompiledInstance,
    snapshot: StateSnapshot,
    *,
    success: bool | None,
    termination_reason: str | None,
    truncation_reason: str | None,
    decision_count: int,
    event_count: int,
    valid_action_count: int,
    invalid_action_count: int,
) -> dict[str, Any]:
    """Compute the v0.1 metric set.

    ``makespan`` is the maximum job completion time and is present only when
    every job has finished. ``due_date is None`` drops that job from both
    tardiness totals; if no job has a due date the tardiness fields are
    ``None`` rather than zero. Utilization uses makespan on a successful
    episode and elapsed time otherwise.
    """
    completed = [value for value in snapshot.job_completion if value is not None]
    all_done = snapshot.jobs_completed == compiled.n_jobs and len(completed) == compiled.n_jobs
    makespan = max(completed) if all_done else None
    if completed:
        flow = [
            float(value) - compiled.job_release[index]
            for index, value in enumerate(snapshot.job_completion)
            if value is not None
        ]
        mean_flow: float | None = sum(flow) / len(flow)
    else:
        mean_flow = None

    due_indexes = [index for index, due in enumerate(compiled.job_due) if due is not None]
    if not due_indexes:
        total_tardiness: float | None = None
        weighted_tardiness: float | None = None
    else:
        total = 0.0
        weighted = 0.0
        for index in due_indexes:
            completion = snapshot.job_completion[index]
            if completion is None:
                continue
            due = compiled.job_due[index]
            assert due is not None
            tardiness = max(0.0, float(completion) - due)
            total += tardiness
            weighted += compiled.job_priority[index] * tardiness
        total_tardiness = total
        weighted_tardiness = weighted

    busy = {machine.id: float(machine.busy_time) for machine in snapshot.machines}
    denominator: float | None
    if makespan is not None and makespan > 0.0:
        denominator = makespan
    elif snapshot.now > 0.0:
        denominator = snapshot.now
    else:
        denominator = None
    utilization: dict[str, float | None] = {}
    for machine_id, busy_time in busy.items():
        utilization[machine_id] = None if denominator is None else busy_time / denominator

    return {
        "makespan": None if makespan is None else float(makespan),
        "sim_time": float(snapshot.now),
        "mean_flow_time": None if mean_flow is None else float(mean_flow),
        "total_tardiness": total_tardiness,
        "weighted_tardiness": weighted_tardiness,
        "machine_busy_time": busy,
        "utilization": utilization,
        "jobs_completed": int(snapshot.jobs_completed),
        "jobs_total": int(compiled.n_jobs),
        "operations_completed": int(snapshot.operations_completed),
        "operations_total": int(compiled.n_operations),
        "valid_action_count": int(valid_action_count),
        "invalid_action_count": int(invalid_action_count),
        "decision_count": int(decision_count),
        "event_count": int(event_count),
        "success": success,
        "termination_reason": termination_reason,
        "truncation_reason": truncation_reason,
    }
