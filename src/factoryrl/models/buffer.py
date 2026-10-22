"""Finite buffers are intentionally outside FactoryRL v0.1."""

from __future__ import annotations

from dataclasses import dataclass

from factoryrl.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Placeholder for the v0.3 finite-buffer model.

    Constructing the type is an error so a caller cannot silently attach a
    buffer that the simulator would ignore.
    """

    id: str
    capacity: int

    def __post_init__(self) -> None:
        raise ConfigurationError(
            "Finite buffers are not part of FactoryRL v0.1. "
            "They are planned for v0.3 together with blocking semantics."
        )
