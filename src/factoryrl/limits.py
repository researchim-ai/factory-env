"""Resource ceilings applied when instances and scenario files are accepted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Hard caps against accidental or hostile scenario sizes.

    Raising a limit is an explicit constructor argument. Defaults are high
    enough for the documented stress size (50 machines, 1000 jobs, one
    operation per machine) and low enough to reject unbounded documents.
    """

    max_machines: int = 200
    max_jobs: int = 5_000
    max_operations: int = 200_000
    max_operations_per_job: int = 500
    max_id_length: int = 128
    max_name_length: int = 256
    max_metadata_bytes: int = 100_000
    max_file_bytes: int = 2_000_000
    max_yaml_nodes: int = 200_000
    max_yaml_depth: int = 40
    max_yaml_collection: int = 200_000
    max_horizon: float = 1.0e12
    max_duration: float = 1.0e9
    max_priority: float = 1.0e6


DEFAULT_LIMITS = ResourceLimits()
