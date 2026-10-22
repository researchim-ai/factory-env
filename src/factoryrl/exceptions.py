"""Public exceptions for FactoryRL.

Invalid schedules and invalid actions fail loudly. The simulator does not
replace a rejected action with a random legal one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class FactoryRLError(Exception):
    """Base error for every FactoryRL failure."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context or {})


class InstanceValidationError(FactoryRLError):
    """A factory instance violates the static schema or a production constraint."""


class SchemaVersionError(FactoryRLError):
    """The scenario schema version is not supported by this package."""


class ConfigurationError(FactoryRLError):
    """An environment or runtime option is inconsistent."""


class InvalidActionError(FactoryRLError):
    """The agent proposed an action outside the current legal mask."""


class InvariantViolationError(FactoryRLError):
    """An internal simulation invariant was broken. This is a simulator bug."""


class SerializationError(FactoryRLError):
    """A scenario file could not be parsed safely."""


class ResourceLimitError(FactoryRLError):
    """A scenario or episode exceeded a configured resource budget."""


class ReplayMismatchError(FactoryRLError):
    """Replaying a trace did not reproduce the recorded events or metrics."""


class BaselineError(FactoryRLError):
    """A baseline policy was used outside its contract."""
