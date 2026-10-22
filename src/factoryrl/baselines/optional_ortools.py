"""Optional CP-SAT makespan solver. Importing FactoryRL does not import OR-Tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from factoryrl.exceptions import ConfigurationError
from factoryrl.models.factory import FactoryInstance
from factoryrl.models.route import resolved_predecessors


@dataclass(frozen=True, slots=True)
class SolverResult:
    """What the external solver reported. Status names come from OR-Tools."""

    solver_status: str
    objective: float | None
    best_bound: float | None
    elapsed_time: float
    starts: dict[str, int]


def solve_job_shop(instance: FactoryInstance, *, time_limit_seconds: float = 10.0) -> SolverResult:
    """Minimize makespan with CP-SAT. Durations must be positive integers.

    The solver sees the static instance only. It is an offline baseline, not
    a policy inside the Gymnasium step loop.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ConfigurationError(
            "OR-Tools is not installed. Install the solver extra: pip install 'factoryrl[solver]'."
        ) from exc
    if time_limit_seconds <= 0:
        raise ConfigurationError("time_limit_seconds must be > 0.")

    tasks: dict[str, tuple[Any, Any, Any]] = {}
    model = cp_model.CpModel()
    horizon = 0
    for job in instance.jobs:
        for op in job.operations:
            if float(op.duration) != int(op.duration):
                raise ConfigurationError(f"CP-SAT requires integer durations; {op.id} has {op.duration}.")
            horizon += int(op.duration)
    for job in instance.jobs:
        resolved = resolved_predecessors(job)
        for op in job.operations:
            start = model.new_int_var(0, horizon, f"start-{op.id}")
            end = model.new_int_var(0, horizon, f"end-{op.id}")
            interval = model.new_interval_var(start, int(op.duration), end, f"iv-{op.id}")
            tasks[op.id] = (start, end, interval)
        for op, preds in zip(job.operations, resolved, strict=True):
            for pred in preds:
                model.add(tasks[op.id][0] >= tasks[pred][1])

    by_machine: dict[str, list[Any]] = {machine.id: [] for machine in instance.machines}
    for job in instance.jobs:
        for op in job.operations:
            by_machine[op.machine_id].append(tasks[op.id][2])
    for intervals in by_machine.values():
        if intervals:
            model.add_no_overlap(intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [tasks[op.id][1] for job in instance.jobs for op in job.operations])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    started = time.perf_counter()
    status = solver.solve(model)
    elapsed = time.perf_counter() - started
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    starts = {op_id: int(solver.value(item[0])) for op_id, item in tasks.items()} if feasible else {}
    objective = float(solver.objective_value) if feasible else None
    return SolverResult(
        solver_status=solver.status_name(status),
        objective=objective,
        best_bound=float(solver.best_objective_bound),
        elapsed_time=elapsed,
        starts=starts,
    )
