"""Benchmark comparisons. Schedule metrics themselves live in the core."""

from __future__ import annotations

from factoryrl.core.metrics import production_metrics
from factoryrl.exceptions import ConfigurationError

__all__ = ["production_metrics", "relative_improvement"]


def relative_improvement(baseline: float, agent: float) -> float:
    """``(baseline - agent) / baseline`` for a metric that should be minimized.

    Negative values mean the agent is worse than the baseline. ``baseline``
    must be strictly positive, which makespan is on every successful v0.1
    episode because every duration is strictly positive.
    """
    if baseline <= 0.0:
        raise ConfigurationError(
            "relative_improvement requires a baseline greater than 0.",
            context={"baseline": baseline, "agent": agent},
        )
    return (float(baseline) - float(agent)) / float(baseline)
