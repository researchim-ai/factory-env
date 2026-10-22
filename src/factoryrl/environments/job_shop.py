"""Gymnasium job-shop environment."""

from __future__ import annotations

from factoryrl.environments.base import FactorySchedulingEnv
from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.builtins import demo_instance


class JobShopEnv(FactorySchedulingEnv):
    """Static non-preemptive job shop, registered as ``FactoryRL/JobShop-v0``.

    With no instance and no generator, the environment uses the two-by-two
    acceptance scenario so ``gymnasium.make`` and ``check_env`` have a
    concrete factory. Pass ``instance`` for a real experiment.
    """

    env_id = "FactoryRL/JobShop-v0"

    def _default_instance(self) -> FactoryInstance:
        return demo_instance()
