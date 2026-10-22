"""Deterministic tie-breaks for dispatching rules."""

from __future__ import annotations

import numpy as np

from factoryrl.exceptions import BaselineError


def dispatch_indices(action_mask: np.ndarray) -> np.ndarray:
    """Return legal operation indexes, excluding the trailing WAIT slot."""
    if action_mask.ndim != 1 or action_mask.shape[0] < 1:
        raise BaselineError("action_mask must be a 1-D vector ending in the WAIT slot.")
    return np.flatnonzero(np.asarray(action_mask[:-1], dtype=bool))


def wait_index(action_mask: np.ndarray) -> int:
    return int(action_mask.shape[0] - 1)


def choose(indices: np.ndarray, values: np.ndarray, *, minimize: bool) -> int:
    """Pick the extreme value. Equal values keep the smallest operation index."""
    if len(indices) == 0:
        raise BaselineError("No dispatchable operation to choose.")
    best_index = int(indices[0])
    best_value = float(values[0])
    for raw_index, raw_value in zip(indices[1:], values[1:], strict=True):
        index = int(raw_index)
        value = float(raw_value)
        better = value < best_value if minimize else value > best_value
        tie = abs(value - best_value) <= 1e-9 and index < best_index
        if better or tie:
            best_index = index
            best_value = value
    return best_index


def only_wait_or_fail(action_mask: np.ndarray) -> int:
    if bool(action_mask[-1]):
        return wait_index(action_mask)
    raise BaselineError("The action mask has no legal dispatch or WAIT.")
