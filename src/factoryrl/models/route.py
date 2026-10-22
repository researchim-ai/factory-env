"""Routing helpers: linear job chains and permutation flow shops."""

from __future__ import annotations

from typing import TYPE_CHECKING

from factoryrl.exceptions import InstanceValidationError
from factoryrl.models.job import JobSpec

if TYPE_CHECKING:
    from factoryrl.models.factory import FactoryInstance


def resolved_predecessors(job: JobSpec) -> tuple[tuple[str, ...], ...]:
    """Return the predecessor id of each operation in a single chain.

    When every operation omits predecessors, the list order is the chain.
    Explicit predecessors must also form one path that covers the job.
    """
    operations = job.operations
    if not operations:
        raise InstanceValidationError(f"Job {job.id} has no operations.")
    ids = [op.id for op in operations]
    if len(set(ids)) != len(ids):
        raise InstanceValidationError(f"Job {job.id} repeats an operation id.")
    explicit = any(op.predecessor_ids for op in operations)
    if not explicit:
        preds: list[tuple[str, ...]] = [() for _ in ids]
        for index in range(1, len(ids)):
            preds[index] = (ids[index - 1],)
        return tuple(preds)

    known = set(ids)
    raw: list[tuple[str, ...]] = []
    for op in operations:
        seen: set[str] = set()
        for predecessor in op.predecessor_ids:
            if predecessor == op.id or predecessor not in known or predecessor in seen:
                raise InstanceValidationError(
                    f"Operation {op.id} has an invalid predecessor {predecessor!r}. "
                    "Predecessors must be other operations of the same job, without duplicates.",
                    context={"operation_id": op.id, "predecessor": predecessor},
                )
            seen.add(predecessor)
        if len(op.predecessor_ids) > 1:
            raise InstanceValidationError(
                f"Operation {op.id} has multiple predecessors. v0.1 only allows a linear chain.",
                context={"operation_id": op.id},
            )
        raw.append(tuple(op.predecessor_ids))

    index_of = {op_id: index for index, op_id in enumerate(ids)}
    dependents: list[list[int]] = [[] for _ in ids]
    indegree = [0 for _ in ids]
    for index, predecessors in enumerate(raw):
        for predecessor in predecessors:
            indegree[index] += 1
            dependents[index_of[predecessor]].append(index)
    roots = [index for index, degree in enumerate(indegree) if degree == 0]
    if len(roots) != 1:
        raise InstanceValidationError(
            f"Job {job.id} must be one chain with a single root operation.",
            context={"job_id": job.id, "roots": len(roots)},
        )
    order: list[int] = []
    cursor = roots[0]
    seen_nodes: set[int] = set()
    while True:
        if cursor in seen_nodes:
            raise InstanceValidationError(f"Job {job.id} contains a precedence cycle.")
        seen_nodes.add(cursor)
        order.append(cursor)
        nxt = dependents[cursor]
        if len(nxt) > 1:
            raise InstanceValidationError(f"Job {job.id} branches. v0.1 only allows a linear chain.")
        if not nxt:
            break
        cursor = nxt[0]
    if len(order) != len(ids):
        raise InstanceValidationError(f"Job {job.id} precedence does not cover every operation exactly once.")
    return tuple(raw)


def assert_flow_shop(instance: FactoryInstance) -> tuple[str, ...]:
    """Return the shared machine sequence or reject the instance.

    A v0.1 flow shop sends every job through the same machines, each machine
    at most once, in the same order. Processing times may differ.
    """
    if not instance.jobs:
        raise InstanceValidationError("A flow shop needs at least one job.")
    sequences = [tuple(op.machine_id for op in job.operations) for job in instance.jobs]
    expected = sequences[0]
    if not expected:
        raise InstanceValidationError("A flow shop job needs at least one operation.")
    if len(set(expected)) != len(expected):
        raise InstanceValidationError(
            "A flow shop route cannot visit the same machine twice.",
            context={"sequence": expected},
        )
    for job, sequence in zip(instance.jobs, sequences, strict=True):
        if sequence != expected:
            raise InstanceValidationError(
                f"Job {job.id} visits {sequence}, but the flow shop route is {expected}.",
                context={"job_id": job.id, "sequence": sequence, "expected": expected},
            )
    return expected
