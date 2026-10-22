"""Helpers for constructing a factory instance from plain Python values."""

from __future__ import annotations

from typing import Any

from factoryrl.limits import DEFAULT_LIMITS, ResourceLimits
from factoryrl.models.factory import FactoryInstance
from factoryrl.models.job import JobSpec
from factoryrl.models.machine import MachineSpec
from factoryrl.models.operation import OperationSpec


def operation(
    op_id: str,
    machine_id: str,
    duration: float,
    *,
    predecessors: tuple[str, ...] | list[str] = (),
    operation_type: str | None = None,
) -> OperationSpec:
    """Build one operation. Predecessors default to the job list order."""
    return OperationSpec(
        id=op_id,
        machine_id=machine_id,
        duration=duration,
        predecessor_ids=tuple(predecessors),
        operation_type=operation_type,
    )


def job(
    job_id: str,
    operations: list[OperationSpec] | tuple[OperationSpec, ...],
    *,
    release_time: float = 0.0,
    due_date: float | None = None,
    priority: float = 1.0,
) -> JobSpec:
    """Build one job."""
    return JobSpec(
        id=job_id,
        operations=tuple(operations),
        release_time=release_time,
        due_date=due_date,
        priority=priority,
    )


def machine(
    machine_id: str,
    *,
    capacity: int = 1,
    supported_operation_types: tuple[str, ...] | list[str] = (),
) -> MachineSpec:
    """Build one machine."""
    return MachineSpec(
        id=machine_id,
        capacity=capacity,
        supported_operation_types=tuple(supported_operation_types),
    )


def instance(
    name: str,
    machines: list[MachineSpec] | tuple[MachineSpec, ...],
    jobs: list[JobSpec] | tuple[JobSpec, ...],
    *,
    horizon: float | None = None,
    metadata: dict[str, Any] | None = None,
    limits: ResourceLimits = DEFAULT_LIMITS,
    time_unit: str = "minute",
    schema_version: int = 1,
) -> FactoryInstance:
    """Validate and return a factory instance."""
    return FactoryInstance(
        name=name,
        machines=tuple(machines),
        jobs=tuple(jobs),
        schema_version=schema_version,
        time_unit=time_unit,
        horizon=horizon,
        metadata=metadata or {},
        limits=limits,
    )
