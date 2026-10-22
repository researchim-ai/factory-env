"""Random legal policies preserve the core invariants and the reward identity."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from factoryrl.baselines import FIFOAgent
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.scenarios.generator import generate_job_shop


@given(
    n_machines=st.integers(1, 3),
    n_jobs=st.integers(1, 3),
    seed=st.integers(0, 5_000),
    action_seed=st.integers(0, 5_000),
)
@settings(max_examples=30, deadline=None)
def test_random_legal_episode_is_successful_and_reward_matches_makespan(
    n_machines, n_jobs, seed, action_seed
):
    instance = generate_job_shop(n_machines, n_jobs, seed=seed, low=1, high=6)
    env = JobShopEnv(instance=instance, check_invariants=True)
    _obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(action_seed)
    total = 0.0
    previous_now = 0.0
    for _ in range(n_machines * n_jobs + 3):
        legal = np.flatnonzero(info["action_mask"])
        assert legal.size > 0
        action = int(rng.choice(legal))
        _obs, reward, terminated, truncated, info = env.step(action)
        assert info["episode_metrics"]["sim_time"] + 1e-9 >= previous_now
        previous_now = info["episode_metrics"]["sim_time"]
        total += reward
        assert not (terminated and truncated)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["success"] is True
    makespan = info["episode_metrics"]["makespan"]
    assert makespan is not None and makespan > 0
    assert abs(total + makespan) < 1e-6


def test_fifo_on_generated_instances_matches_reference_makespan():
    from tests.reference_schedule import reference_makespan

    for seed in (1, 2, 3):
        instance = generate_job_shop(2, 3, seed=seed, low=1, high=5)
        env = JobShopEnv(instance=instance)
        agent = FIFOAgent()
        obs, info = env.reset(seed=seed)
        agent.reset(seed=seed)
        actions = []
        while True:
            action = int(agent.act(obs, info["action_mask"]))
            actions.append(action)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info["episode_metrics"]["makespan"] == reference_makespan(instance, actions)
