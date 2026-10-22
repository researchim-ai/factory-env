"""Time scaling for observations and rewards."""

from __future__ import annotations

import math

from factoryrl.exceptions import ConfigurationError


def assert_time_scale(time_scale: float) -> float:
    """Return ``time_scale`` when it is finite and strictly positive."""
    if isinstance(time_scale, bool) or not isinstance(time_scale, (int, float)):
        raise ConfigurationError("time_scale must be a finite number > 0.")
    value = float(time_scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ConfigurationError("time_scale must be a finite number > 0.")
    return value


def scale_time(value: float, time_scale: float) -> float:
    """Convert a simulation time into observation units."""
    return float(value) / time_scale
