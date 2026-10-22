"""Pydantic boundary for scenario documents, plus the reserved v1 to v2 migration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from factoryrl.exceptions import InstanceValidationError, SchemaVersionError, SerializationError
from factoryrl.limits import DEFAULT_LIMITS, ResourceLimits
from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.instance import instance, job, machine, operation

SUPPORTED_SCHEMA_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reject_bool(value: object, what: str) -> object:
    if isinstance(value, bool):
        raise ValueError(f"{what} must not be a boolean")
    return value


class OperationModel(_Strict):
    id: str
    machine_id: str
    duration: float
    predecessor_ids: list[str] = Field(default_factory=list)
    operation_type: str | None = None

    @field_validator("duration", mode="before")
    @classmethod
    def duration_number(cls, value: object) -> object:
        value = _reject_bool(value, "duration")
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError("duration must be finite")
        return value


class JobModel(_Strict):
    id: str
    release_time: float = 0
    due_date: float | None = None
    priority: float = 1
    operations: list[OperationModel]

    @field_validator("release_time", "priority", mode="before")
    @classmethod
    def numbers(cls, value: object) -> object:
        return _reject_bool(value, "numeric field")

    @field_validator("due_date", mode="before")
    @classmethod
    def due(cls, value: object) -> object:
        if value is None:
            return None
        return _reject_bool(value, "due_date")


class MachineModel(_Strict):
    id: str
    capacity: int = 1
    supported_operation_types: list[str] = Field(default_factory=list)

    @field_validator("capacity", mode="before")
    @classmethod
    def capacity_int(cls, value: object) -> object:
        return _reject_bool(value, "capacity")


class FactoryModel(_Strict):
    name: str
    time_unit: str = "minute"
    horizon: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("horizon", mode="before")
    @classmethod
    def horizon_number(cls, value: object) -> object:
        if value is None:
            return None
        return _reject_bool(value, "horizon")


class EnvironmentModel(_Strict):
    id: str = "FactoryRL/JobShop-v0"
    reward: str = "negative_elapsed_time"
    invalid_action_mode: str = "raise"
    allow_voluntary_idle: bool = False
    time_scale: float = 1.0
    failure_penalty: float = 0.0
    max_events_per_episode: int = 1_000_000
    max_decisions_per_episode: int = 1_000_000
    record_events: bool = True
    check_invariants: bool = True

    @field_validator("time_scale", "failure_penalty", mode="before")
    @classmethod
    def env_numbers(cls, value: object) -> object:
        return _reject_bool(value, "environment number")


class ScenarioModel(_Strict):
    schema_version: int
    factory: FactoryModel
    machines: list[MachineModel]
    jobs: list[JobModel]
    environment: EnvironmentModel = Field(default_factory=EnvironmentModel)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Environment section of a scenario file. It is not part of the instance hash."""

    id: str = "FactoryRL/JobShop-v0"
    reward: str = "negative_elapsed_time"
    invalid_action_mode: str = "raise"
    allow_voluntary_idle: bool = False
    time_scale: float = 1.0
    failure_penalty: float = 0.0
    max_events_per_episode: int = 1_000_000
    max_decisions_per_episode: int = 1_000_000
    record_events: bool = True
    check_invariants: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reward": self.reward,
            "invalid_action_mode": self.invalid_action_mode,
            "allow_voluntary_idle": self.allow_voluntary_idle,
            "time_scale": self.time_scale,
            "failure_penalty": self.failure_penalty,
            "max_events_per_episode": self.max_events_per_episode,
            "max_decisions_per_episode": self.max_decisions_per_episode,
            "record_events": self.record_events,
            "check_invariants": self.check_invariants,
        }


@dataclass(frozen=True, slots=True)
class Scenario:
    """A validated instance plus the environment options stored beside it."""

    instance: FactoryInstance
    environment: EnvironmentConfig

    def to_dict(self) -> dict[str, Any]:
        document = self.instance.to_dict()
        document["environment"] = self.environment.to_dict()
        return document


def scenario_from_document(document: object, *, limits: ResourceLimits = DEFAULT_LIMITS) -> Scenario:
    """Validate a parsed JSON-like document and build a scenario."""
    if not isinstance(document, dict):
        raise SerializationError("A scenario document must be a JSON/YAML object.")
    version = document.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Unsupported schema_version {version!r}. This release reads version {SUPPORTED_SCHEMA_VERSION}. "
            "migrate_v1_to_v2 is reserved and does not produce a loadable instance yet."
        )
    try:
        parsed = ScenarioModel.model_validate(document)
    except ValidationError as exc:
        raise SerializationError(f"Scenario document was rejected: {exc}") from exc
    try:
        built = instance(
            parsed.factory.name,
            [
                machine(
                    item.id,
                    capacity=item.capacity,
                    supported_operation_types=tuple(item.supported_operation_types),
                )
                for item in parsed.machines
            ],
            [
                job(
                    item.id,
                    [
                        operation(
                            op.id,
                            op.machine_id,
                            op.duration,
                            predecessors=tuple(op.predecessor_ids),
                            operation_type=op.operation_type,
                        )
                        for op in item.operations
                    ],
                    release_time=item.release_time,
                    due_date=item.due_date,
                    priority=item.priority,
                )
                for item in parsed.jobs
            ],
            horizon=parsed.factory.horizon,
            metadata=parsed.factory.metadata,
            limits=limits,
            time_unit=parsed.factory.time_unit,
            schema_version=parsed.schema_version,
        )
    except (InstanceValidationError, SchemaVersionError):
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise SerializationError(f"Scenario could not be constructed: {exc}") from exc
    env = parsed.environment
    return Scenario(
        instance=built,
        environment=EnvironmentConfig(
            id=env.id,
            reward=env.reward,
            invalid_action_mode=env.invalid_action_mode,
            allow_voluntary_idle=env.allow_voluntary_idle,
            time_scale=float(env.time_scale),
            failure_penalty=float(env.failure_penalty),
            max_events_per_episode=env.max_events_per_episode,
            max_decisions_per_episode=env.max_decisions_per_episode,
            record_events=env.record_events,
            check_invariants=env.check_invariants,
        ),
    )


def migrate_v1_to_v2(document: dict[str, Any]) -> dict[str, Any]:
    """Reserve the next schema version without making v2 loadable.

    The returned document is for migration tests and future code. The v0.1
    loader rejects ``schema_version: 2`` on purpose.
    """
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SchemaVersionError("migrate_v1_to_v2 expects a schema_version 1 document.")
    migrated = copy.deepcopy(document)
    migrated["schema_version"] = 2
    factory = migrated.setdefault("factory", {})
    if not isinstance(factory, dict):
        raise SchemaVersionError("factory must be an object in order to migrate it.")
    metadata = factory.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise SchemaVersionError("factory.metadata must be an object in order to migrate it.")
    metadata["migrated_from"] = 1
    return migrated


def apply_migrations(document: dict[str, Any], *, target: int = SUPPORTED_SCHEMA_VERSION) -> dict[str, Any]:
    """Move a document toward ``target``. Only version 1 is loadable today."""
    current = document.get("schema_version")
    if current == target:
        return copy.deepcopy(document)
    if current == 1 and target == 2:
        return migrate_v1_to_v2(document)
    raise SchemaVersionError(
        f"No migration path from schema {current!r} to {target}.",
        context={"schema_version": current, "target": target},
    )
