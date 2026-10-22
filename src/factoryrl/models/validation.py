"""Static validation for :class:`FactoryInstance`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from factoryrl.exceptions import InstanceValidationError, SchemaVersionError
from factoryrl.limits import ResourceLimits
from factoryrl.models.ids import as_finite_number, check_id, check_name
from factoryrl.models.machine import MachineSpec
from factoryrl.models.operation import OperationSpec
from factoryrl.models.route import resolved_predecessors

if TYPE_CHECKING:
    from factoryrl.models.factory import FactoryInstance

SUPPORTED_SCHEMA_VERSION = 1
TIME_UNIT = "minute"


def validate_instance(instance: FactoryInstance) -> None:
    """Reject instances that cannot be simulated under the v0.1 contract."""
    limits = instance.limits
    if isinstance(instance.schema_version, bool) or not isinstance(instance.schema_version, int):
        raise SchemaVersionError("schema_version must be the integer 1.")
    if instance.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"schema_version {instance.schema_version} is not supported. "
            f"This release reads version {SUPPORTED_SCHEMA_VERSION}.",
            context={"schema_version": instance.schema_version},
        )
    check_name(instance.name, limits=limits)
    if instance.time_unit != TIME_UNIT:
        raise InstanceValidationError(
            "v0.1 time_unit must be 'minute'.",
            context={"time_unit": instance.time_unit},
        )
    if instance.horizon is not None:
        as_finite_number(
            instance.horizon,
            what="horizon",
            positive=True,
            maximum=limits.max_horizon,
        )
    _validate_metadata_size(instance)

    if not instance.machines:
        raise InstanceValidationError("At least one machine is required.")
    if len(instance.machines) > limits.max_machines:
        raise InstanceValidationError(
            f"Machine count {len(instance.machines)} exceeds the limit {limits.max_machines}."
        )
    if not instance.jobs:
        raise InstanceValidationError("At least one job is required.")
    if len(instance.jobs) > limits.max_jobs:
        raise InstanceValidationError(f"Job count {len(instance.jobs)} exceeds the limit {limits.max_jobs}.")

    machine_ids: set[str] = set()
    machine_by_id: dict[str, MachineSpec] = {}
    for machine in instance.machines:
        machine_id = check_id(machine.id, what="machine id", limits=limits)
        if machine_id in machine_ids:
            raise InstanceValidationError(f"Duplicate machine id {machine_id}.")
        machine_ids.add(machine_id)
        if isinstance(machine.capacity, bool) or machine.capacity != 1:
            raise InstanceValidationError(
                f"Machine {machine_id} must have capacity 1 in v0.1.",
                context={"machine_id": machine_id, "capacity": machine.capacity},
            )
        for operation_type in machine.supported_operation_types:
            check_id(operation_type, what=f"operation type on {machine_id}", limits=limits)
        if len(set(machine.supported_operation_types)) != len(machine.supported_operation_types):
            raise InstanceValidationError(f"Machine {machine_id} repeats a supported operation type.")
        machine_by_id[machine_id] = machine

    job_ids: set[str] = set()
    operation_ids: set[str] = set()
    total_operations = 0
    for job in instance.jobs:
        job_id = check_id(job.id, what="job id", limits=limits)
        if job_id in job_ids:
            raise InstanceValidationError(f"Duplicate job id {job_id}.")
        job_ids.add(job_id)
        release = as_finite_number(job.release_time, what=f"{job_id} release_time", nonnegative=True)
        if release != 0.0:
            raise InstanceValidationError(
                f"Job {job_id} has release_time {release}. v0.1 only supports release_time 0.",
                context={"job_id": job_id, "release_time": release},
            )
        if job.due_date is not None:
            as_finite_number(job.due_date, what=f"{job_id} due_date", nonnegative=True)
        as_finite_number(
            job.priority,
            what=f"{job_id} priority",
            positive=True,
            maximum=limits.max_priority,
        )
        if not job.operations:
            raise InstanceValidationError(f"Job {job_id} has no operations.")
        if len(job.operations) > limits.max_operations_per_job:
            raise InstanceValidationError(
                f"Job {job_id} exceeds the per-job operation limit ({limits.max_operations_per_job})."
            )
        for op in job.operations:
            _validate_operation(op, job_id=job_id, limits=limits, machine_by_id=machine_by_id)
            if op.id in operation_ids:
                raise InstanceValidationError(f"Duplicate operation id {op.id}.")
            operation_ids.add(op.id)
            total_operations += 1
        resolved_predecessors(job)

    if total_operations > limits.max_operations:
        raise InstanceValidationError(
            f"Operation count {total_operations} exceeds the limit {limits.max_operations}."
        )


def _validate_operation(
    op: OperationSpec,
    *,
    job_id: str,
    limits: ResourceLimits,
    machine_by_id: dict[str, MachineSpec],
) -> None:
    check_id(op.id, what=f"operation id in {job_id}", limits=limits)
    check_id(op.machine_id, what=f"machine reference on {op.id}", limits=limits)
    if op.machine_id not in machine_by_id:
        raise InstanceValidationError(
            f"Operation {op.id} references unknown machine {op.machine_id}.",
            context={"operation_id": op.id, "machine_id": op.machine_id},
        )
    as_finite_number(
        op.duration,
        what=f"duration of {op.id}",
        positive=True,
        maximum=limits.max_duration,
    )
    if op.operation_type is not None:
        check_id(op.operation_type, what=f"operation type of {op.id}", limits=limits)
    machine = machine_by_id[op.machine_id]
    if machine.supported_operation_types:
        if op.operation_type not in machine.supported_operation_types:
            raise InstanceValidationError(
                f"Operation {op.id} type {op.operation_type!r} is not supported by machine {machine.id}.",
                context={"operation_id": op.id, "machine_id": machine.id},
            )


def _validate_metadata_size(instance: FactoryInstance) -> None:
    encoded = json.dumps(thaw_metadata(instance), ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > instance.limits.max_metadata_bytes:
        raise InstanceValidationError("metadata exceeds the configured byte limit.")


def thaw_metadata(instance: FactoryInstance) -> dict[str, object]:
    """Return a plain copy of instance metadata."""
    from factoryrl.models.ids import thaw_json

    thawed = thaw_json(instance.metadata)
    if not isinstance(thawed, dict):
        raise InstanceValidationError("metadata must be a JSON object.")
    return thawed
