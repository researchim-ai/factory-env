"""Shortest processing time among operations the mask allows."""

from __future__ import annotations

import numpy as np

from factoryrl.baselines.selection import choose, dispatch_indices, only_wait_or_fail
from factoryrl.observations.builders import OperationFeature


class SPTAgent:
    """Break ties by the smallest operation index."""

    agent_id = "spt"

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        indices = dispatch_indices(action_mask)
        if len(indices) == 0:
            return only_wait_or_fail(action_mask)
        duration = observation["operations"][indices, OperationFeature.DURATION]
        return choose(indices, duration, minimize=True)
