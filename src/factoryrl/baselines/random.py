"""Uniform choice among currently legal actions."""

from __future__ import annotations

import numpy as np

from factoryrl.baselines.selection import dispatch_indices, only_wait_or_fail
from factoryrl.core.rng import make_generator
from factoryrl.exceptions import BaselineError


class RandomValidAgent:
    """Sample a legal action from an independent NumPy generator."""

    agent_id = "random"

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int) -> None:
        self._rng = make_generator(seed)

    def act(self, observation: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        del observation
        if self._rng is None:
            raise BaselineError("RandomValidAgent.reset was not called.")
        indices = dispatch_indices(action_mask)
        choices = indices
        if bool(action_mask[-1]):
            choices = np.concatenate([indices, np.array([action_mask.shape[0] - 1], dtype=int)])
        if choices.size == 0:
            return only_wait_or_fail(action_mask)
        return int(self._rng.choice(choices))
