"""Immutable factory instance: the static input of an episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from factoryrl.limits import DEFAULT_LIMITS, ResourceLimits
from factoryrl.models.ids import freeze_json, thaw_json
from factoryrl.models.job import JobSpec
from factoryrl.models.machine import MachineSpec


@dataclass(frozen=True, slots=True)
class FactoryInstance:
    """Fully defined static production instance.

    Objects of this type are immutable. Runtime status lives in the simulator,
    never on the instance. ``limits`` only affects validation and is excluded
    from equality and from the scenario hash.
    """

    name: str
    machines: tuple[MachineSpec, ...]
    jobs: tuple[JobSpec, ...]
    schema_version: int = 1
    time_unit: str = "minute"
    horizon: float | None = None
    metadata: MappingProxyType[str, Any] | dict[str, Any] = field(default_factory=dict)
    limits: ResourceLimits = field(default=DEFAULT_LIMITS, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "machines", tuple(self.machines))
        object.__setattr__(self, "jobs", tuple(self.jobs))
        object.__setattr__(
            self,
            "metadata",
            freeze_json(dict(self.metadata), what="metadata", max_depth=self.limits.max_yaml_depth),
        )
        from factoryrl.models.validation import validate_instance

        validate_instance(self)

    def to_dict(self) -> dict[str, Any]:
        """Canonical scenario document without the environment section."""
        return {
            "schema_version": self.schema_version,
            "factory": {
                "name": self.name,
                "time_unit": self.time_unit,
                "horizon": None if self.horizon is None else float(self.horizon),
                "metadata": thaw_json(self.metadata),
            },
            "machines": [
                {
                    "id": machine.id,
                    "capacity": machine.capacity,
                    "supported_operation_types": list(machine.supported_operation_types),
                }
                for machine in self.machines
            ],
            "jobs": [
                {
                    "id": job.id,
                    "release_time": float(job.release_time),
                    "due_date": None if job.due_date is None else float(job.due_date),
                    "priority": float(job.priority),
                    "operations": [
                        {
                            "id": op.id,
                            "machine_id": op.machine_id,
                            "duration": float(op.duration),
                            "predecessor_ids": list(op.predecessor_ids),
                            "operation_type": op.operation_type,
                        }
                        for op in job.operations
                    ],
                }
                for job in self.jobs
            ],
        }


def operation_count(instance: FactoryInstance) -> int:
    """Return the number of operations in stable job-major order."""
    return sum(len(job.operations) for job in instance.jobs)
