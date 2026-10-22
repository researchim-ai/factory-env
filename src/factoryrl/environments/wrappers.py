"""Optional wrappers. The base observation schema stays unchanged."""

from __future__ import annotations

import numpy as np
from gymnasium import ObservationWrapper, spaces

from factoryrl.environments.base import FactorySchedulingEnv


class ActionMaskObservationWrapper(ObservationWrapper):
    """Copy the current action mask into the observation under ``action_mask``."""

    def __init__(self, env: FactorySchedulingEnv) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.action_space, spaces.Discrete)
        self.observation_space = spaces.Dict(
            {
                **env.observation_space.spaces,
                "action_mask": spaces.MultiBinary(int(env.action_space.n)),
            }
        )

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        base = self.env.unwrapped
        assert isinstance(base, FactorySchedulingEnv)
        updated = dict(observation)
        updated["action_mask"] = base.action_masks().astype(np.int8)
        return updated
