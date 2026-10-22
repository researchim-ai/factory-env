"""Heuristics and the flow-shop adapter."""

from __future__ import annotations

import pytest
from tests.reference_schedule import reference_makespan

from factoryrl.baselines import EDDAgent, FIFOAgent, RandomValidAgent, SPTAgent
from factoryrl.environments.flow_shop import FlowShopEnv
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.exceptions import BaselineError, InstanceValidationError
from factoryrl.scenarios.builtins import (
    choice_instance,
    demo_instance,
    edd_instance,
    flow_demo_instance,
)


def _rollout(env, agent, seed: int = 0):
    obs, info = env.reset(seed=seed)
    agent.reset(seed=seed)
    actions = []
    total = 0.0
    while True:
        action = int(agent.act(obs, info["action_mask"]))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return actions, info, total


def test_choice_instance_separates_spt_and_fifo():
    env = JobShopEnv(instance=choice_instance())
    spt_actions, spt_info, spt_return = _rollout(env, SPTAgent())
    fifo_actions, fifo_info, fifo_return = _rollout(env, FIFOAgent())
    assert spt_actions[0] == 2
    assert fifo_actions[0] == 0
    assert spt_info["episode_metrics"]["makespan"] == 12.0
    assert fifo_info["episode_metrics"]["makespan"] == 21.0
    assert spt_return == -12.0
    assert fifo_return == -21.0
    assert reference_makespan(choice_instance(), spt_actions) == 12.0
    assert reference_makespan(choice_instance(), fifo_actions) == 21.0


def test_edd_uses_due_dates_and_refuses_instances_without_them():
    actions, info, _total = _rollout(JobShopEnv(instance=edd_instance()), EDDAgent())
    assert actions[0] == 0
    assert info["episode_metrics"]["success"] is True
    with pytest.raises(BaselineError):
        _rollout(JobShopEnv(instance=choice_instance()), EDDAgent())


def test_random_agent_is_reproducible_and_legal():
    env = JobShopEnv(instance=demo_instance())
    first, info_a, _ = _rollout(env, RandomValidAgent(), seed=11)
    second, info_b, _ = _rollout(env, RandomValidAgent(), seed=11)
    assert first == second
    assert info_a["episode_metrics"]["makespan"] == info_b["episode_metrics"]["makespan"]
    assert info_a["success"] is True


def test_flow_shop_rules_and_known_makespans():
    with pytest.raises(InstanceValidationError):
        FlowShopEnv(instance=demo_instance())
    env = FlowShopEnv(instance=flow_demo_instance())
    _actions, spt, _ = _rollout(env, SPTAgent())
    _actions, fifo, _ = _rollout(env, FIFOAgent())
    assert spt["episode_metrics"]["makespan"] == 7.0
    assert fifo["episode_metrics"]["makespan"] == 9.0
    assert JobShopEnv(instance=flow_demo_instance())


def test_same_timestamp_finishes_are_drained_before_the_next_decision():
    from factoryrl.scenarios.instance import instance, job, machine, operation

    built = instance(
        "sync",
        [machine("M1"), machine("M2")],
        [
            job("J1", [operation("J1-O1", "M1", 5), operation("J1-O2", "M2", 1)]),
            job("J2", [operation("J2-O1", "M2", 5), operation("J2-O2", "M1", 1)]),
        ],
    )
    env = JobShopEnv(instance=built)
    env.reset(seed=0)
    env.step(0)
    _obs, _reward, terminated, _truncated, info = env.step(2)
    assert not terminated
    types = [event["type"] for event in env.export_trace()["events"]]
    finish_at_five = [
        event
        for event in env.export_trace()["events"]
        if event["type"] == "operation_finish" and event["time"] == 5.0
    ]
    assert len(finish_at_five) == 2
    decision_indexes = [index for index, name in enumerate(types) if name == "decision_required"]
    finish_indexes = [
        index for index, event in enumerate(env.export_trace()["events"]) if event in finish_at_five
    ]
    assert max(finish_indexes) < decision_indexes[-1]
    assert info["episode_metrics"]["sim_time"] == 5.0
