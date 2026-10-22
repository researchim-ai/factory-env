"""Immutable specification of one machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """A unit-capacity machine.

    ``supported_operation_types`` empty means every operation is allowed.
    Capacity other than 1 is rejected in v0.1.
    """

    id: str
    capacity: int = 1
    supported_operation_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_operation_types", tuple(self.supported_operation_types))
