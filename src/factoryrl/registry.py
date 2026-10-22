"""Register FactoryRL environments with Gymnasium once per process."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

ENVIRONMENTS = {
    "FactoryRL/JobShop-v0": "factoryrl.environments.job_shop:JobShopEnv",
    "FactoryRL/FlowShop-v0": "factoryrl.environments.flow_shop:FlowShopEnv",
}


def register_envs() -> None:
    """Register the v0 environments. Repeated imports do not override them."""
    for env_id, entry_point in ENVIRONMENTS.items():
        if env_id in registry:
            continue
        register(id=env_id, entry_point=entry_point)
