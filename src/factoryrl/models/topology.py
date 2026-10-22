"""Compiled, index-based view of a validated factory instance."""

from __future__ import annotations

from dataclasses import dataclass

from factoryrl.models.factory import FactoryInstance
from factoryrl.models.route import resolved_predecessors


@dataclass(slots=True)
class CompiledInstance:
    """Read-only indexes used by the simulator hot path.

    Operation index 0..n-1 follows job order, then the operation list order.
    That index is the Gymnasium dispatch action for an unpadded instance.
    """

    instance: FactoryInstance
    operation_ids: tuple[str, ...]
    operation_index: dict[str, int]
    machine_ids: tuple[str, ...]
    machine_index: dict[str, int]
    job_ids: tuple[str, ...]
    job_index: dict[str, int]
    operation_machine: list[int]
    operation_job: list[int]
    operation_duration: list[float]
    operation_position: list[int]
    job_length: list[int]
    job_due: list[float | None]
    job_priority: list[float]
    job_release: list[float]
    job_operations: list[tuple[int, ...]]
    predecessors: list[tuple[int, ...]]
    successors: list[tuple[int, ...]]

    @property
    def n_operations(self) -> int:
        return len(self.operation_ids)

    @property
    def n_machines(self) -> int:
        return len(self.machine_ids)

    @property
    def n_jobs(self) -> int:
        return len(self.job_ids)


def compile_instance(instance: FactoryInstance) -> CompiledInstance:
    """Index a validated instance. The instance constructor already validated it."""
    machine_ids = tuple(machine.id for machine in instance.machines)
    machine_index = {machine_id: index for index, machine_id in enumerate(machine_ids)}
    job_ids = tuple(job.id for job in instance.jobs)
    job_index = {job_id: index for index, job_id in enumerate(job_ids)}

    operation_ids: list[str] = []
    operation_index: dict[str, int] = {}
    operation_machine: list[int] = []
    operation_job: list[int] = []
    operation_duration: list[float] = []
    operation_position: list[int] = []
    job_length: list[int] = []
    job_due: list[float | None] = []
    job_priority: list[float] = []
    job_release: list[float] = []
    job_operations: list[tuple[int, ...]] = []
    predecessors: list[tuple[int, ...]] = []

    for job_pos, job in enumerate(instance.jobs):
        job_length.append(len(job.operations))
        job_due.append(None if job.due_date is None else float(job.due_date))
        job_priority.append(float(job.priority))
        job_release.append(float(job.release_time))
        resolved = resolved_predecessors(job)
        op_indexes: list[int] = []
        for position, (op, _preds) in enumerate(zip(job.operations, resolved, strict=True)):
            index = len(operation_ids)
            operation_ids.append(op.id)
            operation_index[op.id] = index
            operation_machine.append(machine_index[op.machine_id])
            operation_job.append(job_pos)
            operation_duration.append(float(op.duration))
            operation_position.append(position)
            op_indexes.append(index)
            predecessors.append(())
        base = op_indexes[0]
        for offset, predecessor_ids in enumerate(resolved):
            predecessors[base + offset] = tuple(operation_index[pred] for pred in predecessor_ids)
        job_operations.append(tuple(op_indexes))

    successors: list[list[int]] = [[] for _ in operation_ids]
    for index, predecessor_indexes in enumerate(predecessors):
        for predecessor in predecessor_indexes:
            successors[predecessor].append(index)

    return CompiledInstance(
        instance=instance,
        operation_ids=tuple(operation_ids),
        operation_index=operation_index,
        machine_ids=machine_ids,
        machine_index=machine_index,
        job_ids=job_ids,
        job_index=job_index,
        operation_machine=operation_machine,
        operation_job=operation_job,
        operation_duration=operation_duration,
        operation_position=operation_position,
        job_length=job_length,
        job_due=job_due,
        job_priority=job_priority,
        job_release=job_release,
        job_operations=job_operations,
        predecessors=predecessors,
        successors=[tuple(items) for items in successors],
    )
