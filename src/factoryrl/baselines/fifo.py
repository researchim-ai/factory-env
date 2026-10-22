"""First in, first out: the ready operation that has waited the longest."""

from __future__ import annotations

import numpy as np

from factoryrl.baselines.selection import choose, dispatch_indices, only_wait_or_fail
from factoryrl.observations.builders import OperationFeature


class FIFOAgent:
    """Break ties by the smallest operation index."""

    agent_id = "fifo"

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        indices = dispatch_indices(action_mask)
        if len(indices) == 0:
            return only_wait_or_fail(action_mask)
        waiting = observation["operations"][indices, OperationFeature.WAITING_TIME]
        return choose(indices, waiting, minimize=False)
