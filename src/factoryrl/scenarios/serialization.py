"""Safe JSON and YAML scenario IO. User files are data, never code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from factoryrl.exceptions import ResourceLimitError, SerializationError
from factoryrl.hashing import sha256_json
from factoryrl.limits import DEFAULT_LIMITS, ResourceLimits
from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.schema import Scenario, scenario_from_document


class _BoundedLoader(yaml.SafeLoader):
    """Safe YAML loader with node, depth, and collection ceilings."""

    def __init__(self, stream: str, limits: ResourceLimits) -> None:
        super().__init__(stream)
        self._limits = limits
        self._nodes = 0
        self._depth = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        self._nodes += 1
        self._depth += 1
        if self._nodes > self._limits.max_yaml_nodes:
            raise SerializationError("YAML document exceeds the node limit.")
        if self._depth > self._limits.max_yaml_depth:
            raise SerializationError("YAML document exceeds the nesting limit.")
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_sequence(self, node: yaml.nodes.SequenceNode, deep: bool = False) -> list[Any]:
        if len(node.value) > self._limits.max_yaml_collection:
            raise SerializationError("YAML sequence exceeds the length limit.")
        built = super().construct_sequence(node, deep)
        if not isinstance(built, list):
            raise SerializationError("YAML sequence did not decode to a list.")
        return built

    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
        if len(node.value) > self._limits.max_yaml_collection:
            raise SerializationError("YAML mapping exceeds the size limit.")
        built = super().construct_mapping(node, deep)
        if not isinstance(built, dict):
            raise SerializationError("YAML mapping did not decode to an object.")
        return built


def scenario_hash(instance: FactoryInstance) -> str:
    """Hash the canonical instance document."""
    return sha256_json(instance.to_dict())


def load_scenario_text(
    text: str,
    *,
    fmt: str,
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> Scenario:
    """Parse one JSON or YAML document from an in-memory string."""
    encoded = text.encode("utf-8")
    if len(encoded) > limits.max_file_bytes:
        raise ResourceLimitError(
            f"Scenario text is {len(encoded)} bytes; the limit is {limits.max_file_bytes}."
        )
    if fmt == "json":
        document = _load_json(text)
    elif fmt == "yaml":
        document = _load_yaml(text, limits)
    else:
        raise SerializationError(f"Unsupported scenario format {fmt!r}.")
    return scenario_from_document(document, limits=limits)


def load_scenario(path: str | Path, *, limits: ResourceLimits = DEFAULT_LIMITS) -> Scenario:
    """Read a ``.json``, ``.yml``, or ``.yaml`` scenario file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        fmt = "json"
    elif suffix in {".yml", ".yaml"}:
        fmt = "yaml"
    else:
        raise SerializationError(f"Scenario file {file_path.name} must be JSON or YAML.")
    try:
        blob = file_path.read_bytes()
    except OSError as exc:
        raise SerializationError(f"Cannot read {file_path}: {exc}") from exc
    if len(blob) > limits.max_file_bytes:
        raise ResourceLimitError(
            f"Scenario file is {len(blob)} bytes; the limit is {limits.max_file_bytes}.",
            context={"path": str(file_path)},
        )
    try:
        text = blob.decode("utf-8")
    except UnicodeError as exc:
        raise SerializationError(f"Scenario file {file_path} is not UTF-8.") from exc
    return load_scenario_text(text, fmt=fmt, limits=limits)


def dump_scenario(scenario: Scenario, path: str | Path) -> None:
    """Write a scenario as JSON or YAML, matching the file suffix."""
    file_path = Path(path)
    document = scenario.to_dict()
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        file_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return
    if suffix in {".yml", ".yaml"}:
        file_path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return
    raise SerializationError(f"Cannot write scenario format {suffix!r}.")


def _load_json(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"JSON scenario was rejected: {exc}") from exc


def _load_yaml(text: str, limits: ResourceLimits) -> object:
    loader = _BoundedLoader(text, limits)
    try:
        document = loader.get_single_data()
    except SerializationError:
        raise
    except yaml.YAMLError as exc:
        raise SerializationError(f"YAML scenario was rejected by the safe loader: {exc}") from exc
    finally:
        loader.dispose()
    return document
