"""Immutable specification of one job."""

from __future__ import annotations

from dataclasses import dataclass

from factoryrl.models.operation import OperationSpec


@dataclass(frozen=True, slots=True)
class JobSpec:
    """A job is an ordered chain of operations in v0.1.

    ``release_time`` must be 0. Non-zero releases are a later version.
    ``due_date=None`` excludes the job from tardiness totals.
    """

    id: str
    operations: tuple[OperationSpec, ...]
    release_time: float = 0.0
    due_date: float | None = None
    priority: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
