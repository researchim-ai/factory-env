"""Decode Gymnasium actions into engine actions."""

from __future__ import annotations

from factoryrl.core.events import WAIT_ACTION
from factoryrl.exceptions import InvalidActionError

WAIT = WAIT_ACTION


def decode_gym_action(action: int, *, n_operations: int, max_operations: int) -> int:
    """Map a discrete action to an operation index or :data:`WAIT`.

    The last legal index (``max_operations``) is WAIT. Indexes in
    ``[n_operations, max_operations)`` are padding and are illegal.
    """
    if action == max_operations:
        return WAIT
    if 0 <= action < n_operations:
        return action
    raise InvalidActionError(
        "Action index is outside the real operations and is not WAIT.",
        context={
            "action": action,
            "n_operations": n_operations,
            "max_operations": max_operations,
            "wait_index": max_operations,
        },
    )
