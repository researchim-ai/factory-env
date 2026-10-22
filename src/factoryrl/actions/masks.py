"""Boolean action masks. True means the action may be sent to ``step``."""

from __future__ import annotations

import numpy as np

from factoryrl.exceptions import ConfigurationError


def build_action_mask(
    dispatchable: np.ndarray,
    *,
    n_operations: int,
    max_operations: int,
    wait_allowed: bool,
) -> np.ndarray:
    """Return a mask of length ``max_operations + 1``."""
    if max_operations < n_operations:
        raise ConfigurationError("max_operations is smaller than the instance.")
    if dispatchable.shape != (n_operations,):
        raise ConfigurationError(
            "dispatchable mask has the wrong shape.",
            context={"shape": tuple(dispatchable.shape), "n_operations": n_operations},
        )
    mask = np.zeros(max_operations + 1, dtype=bool)
    mask[:n_operations] = dispatchable.astype(bool, copy=False)
    mask[max_operations] = bool(wait_allowed)
    return mask
