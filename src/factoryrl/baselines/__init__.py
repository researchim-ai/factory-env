"""Dispatching baselines that read only the observation and the action mask."""

from factoryrl.baselines.edd import EDDAgent
from factoryrl.baselines.fifo import FIFOAgent
from factoryrl.baselines.random import RandomValidAgent
from factoryrl.baselines.spt import SPTAgent

__all__ = ["EDDAgent", "FIFOAgent", "RandomValidAgent", "SPTAgent"]
