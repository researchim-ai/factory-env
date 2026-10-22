"""Discrete action space whose ``sample`` stays inside the current mask."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from gymnasium import spaces

from factoryrl.exceptions import InvalidActionError


class MaskedDiscrete(spaces.Discrete):
    """``Discrete`` space that samples a currently legal action.

    ``contains`` still accepts every index in ``0 .. n-1``. Legality is the
    mask, checked again in ``step``. Sampling does not rewrite an explicit
    illegal action into a legal one.
    """

    def __init__(self, n: int, mask_fn: Callable[[], np.ndarray]) -> None:
        super().__init__(n)
        self._mask_fn = mask_fn

    def sample(
        self,
        mask: np.ndarray | None = None,
        probability: np.ndarray | None = None,
    ) -> np.int64:
        if probability is not None:
            sampled: np.int64 = super().sample(mask=mask, probability=probability)
            return sampled
        legal = np.asarray(self._mask_fn() if mask is None else mask, dtype=bool)
        valid = np.flatnonzero(legal)
        if valid.size == 0:
            raise InvalidActionError("Cannot sample an action because the mask is empty.")
        return np.int64(self.np_random.choice(valid))
