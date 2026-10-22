"""A modest smoke timing check. This is not a public performance SLA."""

from __future__ import annotations

import time

from factoryrl.baselines import SPTAgent
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.scenarios.generator import generate_job_shop


def test_five_by_twenty_spt_finishes_quickly():
    instance = generate_job_shop(5, 20, seed=4, low=1, high=20)
    env = JobShopEnv(instance=instance, check_invariants=True, record_events=False)
    agent = SPTAgent()
    started = time.perf_counter()
    obs, info = env.reset(seed=4)
    agent.reset(seed=4)
    while True:
        action = agent.act(obs, info["action_mask"])
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - started
    assert info["success"] is True
    assert info["episode_metrics"]["operations_completed"] == 100
    assert elapsed < 10.0
