"""Seeded instance generators. The same seed rebuilds the same instance."""

from __future__ import annotations

from typing import Any

import numpy as np

from factoryrl.core.rng import make_generator, validate_seed
from factoryrl.exceptions import ConfigurationError
from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.instance import instance, job, machine, operation


def generated_shape(config: dict[str, Any]) -> tuple[int, int]:
    """Return ``(n_machines, n_operations)`` implied by a generator config."""
    n_machines = _positive_int(config.get("n_machines"), what="n_machines")
    n_jobs = _positive_int(config.get("n_jobs"), what="n_jobs")
    kind = str(config.get("kind", "job_shop"))
    assignment = str(config.get("machine_assignment", "permutation"))
    if kind == "flow_shop" or assignment == "permutation":
        ops_per_job = n_machines
    else:
        ops_per_job = _positive_int(config.get("ops_per_job", n_machines), what="ops_per_job")
    return n_machines, n_jobs * ops_per_job


def generate_from_config(config: dict[str, Any], *, seed: int) -> FactoryInstance:
    """Dispatch to the job-shop or flow-shop generator."""
    kind = str(config.get("kind", "job_shop"))
    n_machines = _positive_int(config.get("n_machines"), what="n_machines")
    n_jobs = _positive_int(config.get("n_jobs"), what="n_jobs")
    distribution = str(config.get("processing_time_distribution", "uniform_int"))
    low = int(config.get("low", 1))
    high = int(config.get("high", 10))
    fixed_value = config.get("fixed_value")
    name = config.get("name")
    due_date_factor = config.get("due_date_factor")
    if kind == "job_shop":
        ops_per_job = config.get("ops_per_job")
        return generate_job_shop(
            n_machines,
            n_jobs,
            seed=seed,
            processing_time_distribution=distribution,
            low=low,
            high=high,
            fixed_value=None if fixed_value is None else float(fixed_value),
            machine_assignment=str(config.get("machine_assignment", "permutation")),
            ops_per_job=None if ops_per_job is None else int(ops_per_job),
            name=None if name is None else str(name),
            due_date_factor=None if due_date_factor is None else float(due_date_factor),
        )
    if kind == "flow_shop":
        return generate_flow_shop(
            n_machines,
            n_jobs,
            seed=seed,
            processing_time_distribution=distribution,
            low=low,
            high=high,
            fixed_value=None if fixed_value is None else float(fixed_value),
            name=None if name is None else str(name),
            due_date_factor=None if due_date_factor is None else float(due_date_factor),
        )
    raise ConfigurationError(f"Unknown generator kind {kind!r}.", context={"kind": kind})


def generate_job_shop(
    n_machines: int,
    n_jobs: int,
    *,
    seed: int,
    processing_time_distribution: str = "uniform_int",
    low: int = 1,
    high: int = 10,
    fixed_value: float | None = None,
    machine_assignment: str = "permutation",
    ops_per_job: int | None = None,
    name: str | None = None,
    due_date_factor: float | None = None,
) -> FactoryInstance:
    """Classic job shop. ``permutation`` gives one operation per machine per job."""
    validate_seed(seed)
    if n_machines < 1 or n_jobs < 1:
        raise ConfigurationError("n_machines and n_jobs must be >= 1.")
    rng = make_generator(seed)
    per_job = n_machines if machine_assignment == "permutation" else int(ops_per_job or n_machines)
    if per_job < 1:
        raise ConfigurationError("ops_per_job must be >= 1.")
    machines = [machine(f"M{index}") for index in range(1, n_machines + 1)]
    jobs = []
    for job_number in range(1, n_jobs + 1):
        if machine_assignment == "permutation":
            order = [int(value) for value in rng.permutation(n_machines)]
        elif machine_assignment == "random":
            order = [int(value) for value in rng.integers(0, n_machines, size=per_job)]
        else:
            raise ConfigurationError(
                f"Unknown machine_assignment {machine_assignment!r}.",
                context={"machine_assignment": machine_assignment},
            )
        operations = []
        for step, machine_offset in enumerate(order, start=1):
            duration = _duration(
                rng,
                distribution=processing_time_distribution,
                low=low,
                high=high,
                fixed_value=fixed_value,
            )
            operations.append(operation(f"J{job_number}-O{step}", f"M{machine_offset + 1}", duration))
        due = _due_date(due_date_factor, [op.duration for op in operations])
        jobs.append(job(f"J{job_number}", operations, due_date=due))
    label = name or f"jobshop-{n_machines}x{n_jobs}-seed{seed}"
    return instance(label, machines, jobs, metadata={"generator_seed": seed, "kind": "job_shop"})


def generate_flow_shop(
    n_machines: int,
    n_jobs: int,
    *,
    seed: int,
    processing_time_distribution: str = "uniform_int",
    low: int = 1,
    high: int = 10,
    fixed_value: float | None = None,
    name: str | None = None,
    due_date_factor: float | None = None,
) -> FactoryInstance:
    """Permutation flow shop: every job visits M1..Mn in that order."""
    validate_seed(seed)
    if n_machines < 1 or n_jobs < 1:
        raise ConfigurationError("n_machines and n_jobs must be >= 1.")
    rng = make_generator(seed)
    machines = [machine(f"M{index}") for index in range(1, n_machines + 1)]
    jobs = []
    for job_number in range(1, n_jobs + 1):
        operations = []
        for step, machine_number in enumerate(range(1, n_machines + 1), start=1):
            duration = _duration(
                rng,
                distribution=processing_time_distribution,
                low=low,
                high=high,
                fixed_value=fixed_value,
            )
            operations.append(operation(f"J{job_number}-O{step}", f"M{machine_number}", duration))
        due = _due_date(due_date_factor, [op.duration for op in operations])
        jobs.append(job(f"J{job_number}", operations, due_date=due))
    label = name or f"flowshop-{n_machines}x{n_jobs}-seed{seed}"
    return instance(label, machines, jobs, metadata={"generator_seed": seed, "kind": "flow_shop"})


def _duration(
    rng: np.random.Generator,
    *,
    distribution: str,
    low: int,
    high: int,
    fixed_value: float | None,
) -> float:
    if distribution == "fixed":
        if fixed_value is None or isinstance(fixed_value, bool) or float(fixed_value) <= 0:
            raise ConfigurationError("fixed processing times require fixed_value > 0.")
        return float(fixed_value)
    if distribution == "uniform_int":
        if isinstance(low, bool) or isinstance(high, bool) or low < 1 or high < low:
            raise ConfigurationError("uniform_int requires 1 <= low <= high.")
        return float(int(rng.integers(low, high + 1)))
    raise ConfigurationError(
        f"Unknown processing_time_distribution {distribution!r}.",
        context={"processing_time_distribution": distribution},
    )


def _due_date(factor: float | None, durations: list[float]) -> float | None:
    if factor is None:
        return None
    if isinstance(factor, bool) or float(factor) <= 0:
        raise ConfigurationError("due_date_factor must be > 0 when it is set.")
    return float(factor) * float(sum(durations))


def _positive_int(value: object, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{what} must be an integer >= 1.", context={what: value})
    return value
