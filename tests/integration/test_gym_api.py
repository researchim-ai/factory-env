"""Gymnasium registration, checker, masks, and replay."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import factoryrl
from factoryrl.core.replay import assert_replay
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.environments.wrappers import ActionMaskObservationWrapper
from factoryrl.scenarios.builtins import demo_instance
from factoryrl.scenarios.generator import generate_job_shop
from factoryrl.scenarios.serialization import scenario_hash


def test_both_environments_are_registered_and_pass_check_env():
    assert "FactoryRL/JobShop-v0" in gym.registry
    assert "FactoryRL/FlowShop-v0" in gym.registry
    check_env(factoryrl.JobShopEnv())
    check_env(factoryrl.FlowShopEnv())
    made = gym.make("FactoryRL/JobShop-v0")
    check_env(made.unwrapped)
    made.close()
    flow = gym.make("FactoryRL/FlowShop-v0")
    assert flow.unwrapped.env_id == "FactoryRL/FlowShop-v0"
    flow.close()


def test_reset_is_deterministic_and_samples_are_initially_legal():
    env = JobShopEnv()
    first, info_a = env.reset(seed=123)
    second, info_b = env.reset(seed=123)
    assert np.array_equal(first["operations"], second["operations"])
    assert np.array_equal(info_a["action_mask"], info_b["action_mask"])
    env.action_space.seed(123)
    for _ in range(20):
        action = int(env.action_space.sample())
        assert info_a["action_mask"][action]


def test_action_mask_wrapper_matches_info():
    wrapped = ActionMaskObservationWrapper(JobShopEnv(instance=demo_instance()))
    obs, info = wrapped.reset(seed=3)
    assert np.array_equal(obs["action_mask"].astype(bool), info["action_mask"])
    assert obs in wrapped.observation_space


def test_replay_reproduces_events_and_reset_drops_history():
    env = JobShopEnv(instance=demo_instance())
    env.reset(seed=5)
    for action in (0, 2, 1, 3):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    trace = env.export_trace(agent_id="scripted")
    assert_replay(
        demo_instance(),
        trace["actions"],
        trace["events"],
        trace["metrics"],
        seed=5,
        horizon=1000,
        allow_voluntary_idle=False,
        max_events=1_000_000,
        max_decisions=1_000_000,
        scenario_hash=scenario_hash(demo_instance()),
        expected_hash=trace["scenario_hash"],
    )
    env.reset(seed=5)
    assert len(env.export_trace()["events"]) < len(trace["events"])
    assert env._engine.clock.now == 0.0
    assert env._engine.state.operations_completed == 0


def test_record_events_false_does_not_keep_a_trace():
    env = JobShopEnv(instance=generate_job_shop(3, 5, seed=1), record_events=False)
    _obs, info = env.reset(seed=1)
    while True:
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert env.export_trace()["events"] == []
    assert info["episode_metrics"]["success"] is True
    assert info["episode_metrics"]["operations_completed"] == 15


def test_three_by_five_finishes():
    env = JobShopEnv(instance=generate_job_shop(3, 5, seed=1))
    _obs, info = env.reset(seed=1)
    while True:
        action = int(np.flatnonzero(info["action_mask"])[0])
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["success"] is True
    assert info["episode_metrics"]["makespan"] > 0
