"""Immutable specification of one production operation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One non-preemptive operation assigned to a single machine.

    ``predecessor_ids`` is empty when the job list order is the chain.
    P0 accepts only a single linear chain per job; a DAG is rejected.
    """

    id: str
    machine_id: str
    duration: float
    predecessor_ids: tuple[str, ...] = ()
    operation_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessor_ids", tuple(self.predecessor_ids))
