"""Gymnasium permutation flow-shop environment."""

from __future__ import annotations

from factoryrl.environments.base import FactorySchedulingEnv
from factoryrl.models.factory import FactoryInstance
from factoryrl.models.route import assert_flow_shop
from factoryrl.scenarios.builtins import flow_demo_instance


class FlowShopEnv(FactorySchedulingEnv):
    """Flow shop on the same engine. Every job must share one machine route.

    Registered as ``FactoryRL/FlowShop-v0``. The default instance is a small
    original flow shop, not an external benchmark.
    """

    env_id = "FactoryRL/FlowShop-v0"

    def _default_instance(self) -> FactoryInstance:
        return flow_demo_instance()

    def _validate_domain(self, instance: FactoryInstance) -> None:
        assert_flow_shop(instance)
