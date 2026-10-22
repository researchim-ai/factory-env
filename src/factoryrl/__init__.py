"""FactoryRL: a deterministic factory simulator and Gymnasium environments.

Importing this package registers ``FactoryRL/JobShop-v0`` and
``FactoryRL/FlowShop-v0``. The simulator does not control real equipment.
"""

from factoryrl.baselines import EDDAgent, FIFOAgent, RandomValidAgent, SPTAgent
from factoryrl.environments.flow_shop import FlowShopEnv
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.models import FactoryInstance, JobSpec, MachineSpec, OperationSpec
from factoryrl.registry import register_envs
from factoryrl.scenarios.builtins import (
    choice_instance,
    demo_instance,
    edd_instance,
    flow_demo_instance,
)
from factoryrl.version import __version__

register_envs()

__all__ = [
    "EDDAgent",
    "FIFOAgent",
    "FactoryInstance",
    "FlowShopEnv",
    "JobShopEnv",
    "JobSpec",
    "MachineSpec",
    "OperationSpec",
    "RandomValidAgent",
    "SPTAgent",
    "choice_instance",
    "demo_instance",
    "edd_instance",
    "flow_demo_instance",
    "__version__",
]
