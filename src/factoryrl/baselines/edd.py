"""Earliest due date. Refuses to run when no selected operation has a due date."""

from __future__ import annotations

import numpy as np

from factoryrl.baselines.selection import choose, dispatch_indices, only_wait_or_fail
from factoryrl.exceptions import BaselineError
from factoryrl.observations.builders import OperationFeature


class EDDAgent:
    """Earliest due date among legal operations.

    A due-date feature of ``-1`` means the job has no due date and is ignored.
    If every legal operation lacks a due date, this agent raises instead of
    silently behaving like another rule.
    """

    agent_id = "edd"

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        indices = dispatch_indices(action_mask)
        if len(indices) == 0:
            return only_wait_or_fail(action_mask)
        due = observation["operations"][indices, OperationFeature.DUE_DATE]
        eligible = indices[due >= 0.0]
        if len(eligible) == 0:
            raise BaselineError("EDD requires at least one legal operation whose job has a due date.")
        eligible_due = observation["operations"][eligible, OperationFeature.DUE_DATE]
        return choose(eligible, eligible_due, minimize=True)
