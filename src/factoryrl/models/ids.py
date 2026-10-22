"""Identifier and numeric checks shared by the instance validator."""

from __future__ import annotations

import math
import re
from typing import Any

from factoryrl.exceptions import InstanceValidationError
from factoryrl.limits import ResourceLimits

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def check_id(value: object, *, what: str, limits: ResourceLimits) -> str:
    """Return ``value`` when it is a stable identifier."""
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise InstanceValidationError(
            f"{what} must be a non-empty identifier of letters, digits, '_', '.', ':', or '-'.",
            context={"what": what, "value": value},
        )
    if len(value) > limits.max_id_length:
        raise InstanceValidationError(
            f"{what} exceeds the identifier length limit ({limits.max_id_length}).",
            context={"what": what, "length": len(value)},
        )
    return value


def check_name(value: object, *, limits: ResourceLimits) -> str:
    """Return a non-empty printable instance name."""
    if not isinstance(value, str) or value.strip() == "" or value != value.strip():
        raise InstanceValidationError("Instance name must be a non-empty string without surrounding space.")
    if len(value) > limits.max_name_length or any(not ch.isprintable() for ch in value):
        raise InstanceValidationError(
            "Instance name is empty, too long, or contains non-printable characters."
        )
    return value


def as_finite_number(
    value: object,
    *,
    what: str,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> float:
    """Coerce a real JSON number to ``float`` and reject booleans, NaN, and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InstanceValidationError(f"{what} must be a finite number.", context={"value": value})
    number = float(value)
    if not math.isfinite(number):
        raise InstanceValidationError(f"{what} must be finite.", context={"value": value})
    if positive and number <= 0.0:
        raise InstanceValidationError(f"{what} must be > 0.", context={"value": number})
    if nonnegative and number < 0.0:
        raise InstanceValidationError(f"{what} must be >= 0.", context={"value": number})
    if maximum is not None and number > maximum:
        raise InstanceValidationError(
            f"{what} exceeds the configured maximum ({maximum}).",
            context={"value": number},
        )
    return number


def freeze_json(value: Any, *, what: str = "metadata", depth: int = 0, max_depth: int = 40) -> Any:
    """Deep-copy a JSON value into tuples and mapping proxies."""
    from types import MappingProxyType

    if depth > max_depth:
        raise InstanceValidationError(f"{what} is nested deeper than {max_depth}.")
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InstanceValidationError(f"{what} contains a non-finite number.")
        return value
    if isinstance(value, dict):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InstanceValidationError(f"{what} keys must be strings.")
            frozen[key] = freeze_json(item, what=what, depth=depth + 1, max_depth=max_depth)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, what=what, depth=depth + 1, max_depth=max_depth) for item in value)
    raise InstanceValidationError(f"{what} contains a non-JSON value of type {type(value).__name__}.")


def thaw_json(value: Any) -> Any:
    """Convert frozen JSON back into plain dicts and lists."""
    from types import MappingProxyType

    if isinstance(value, MappingProxyType):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
