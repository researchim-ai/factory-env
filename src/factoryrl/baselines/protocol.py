"""Agent protocol. Training loops stay outside this package."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Decision policy used by the benchmark runner.

    ``reset`` receives the episode seed. ``act`` must return an index that is
    true in ``action_mask``, including the final WAIT slot when it is enabled.
    """

    def reset(self, *, seed: int) -> None:
        """Prepare the policy for one episode."""

    def act(self, observation: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        """Choose one legal action."""
