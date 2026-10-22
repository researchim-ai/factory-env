"""Specification section 22: the hand schedule has makespan 23 and return -23."""

from __future__ import annotations

import json
from pathlib import Path

from factoryrl.baselines import FIFOAgent, SPTAgent
from factoryrl.core.replay import events_match
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.scenarios.builtins import demo_instance

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_demo_scripted.json"


def _run(actions: list[int]):
    env = JobShopEnv(instance=demo_instance(), render_mode="ansi")
    _obs, info = env.reset(seed=7)
    total = 0.0
    for action in actions:
        before = env.render()
        _obs, reward, terminated, truncated, info = env.step(action)
        after = env.render()
        assert before != after or reward == 0.0
        total += reward
        if terminated or truncated:
            break
    return env, info, total


def test_scripted_schedule_matches_hand_calculation():
    env, info, total = _run([0, 2, 1, 3])
    metrics = info["episode_metrics"]
    assert info["termination_reason"] == "completed"
    assert info["truncation_reason"] is None
    assert metrics["success"] is True
    assert metrics["makespan"] == 23.0
    assert metrics["mean_flow_time"] == 21.5
    assert metrics["total_tardiness"] == 0.0
    assert metrics["weighted_tardiness"] == 0.0
    assert metrics["machine_busy_time"] == {"M1": 23.0, "M2": 19.0}
    assert metrics["utilization"]["M1"] == 1.0
    assert abs(metrics["utilization"]["M2"] - 19.0 / 23.0) < 1e-9
    assert total == -23.0
    assert metrics["decision_count"] == 4
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert events_match(env.export_trace()["events"], golden["events"])


def test_spt_and_fifo_also_finish_at_23_because_both_machines_start_together():
    for agent_cls in (SPTAgent, FIFOAgent):
        env = JobShopEnv(instance=demo_instance())
        agent = agent_cls()
        obs, info = env.reset(seed=1)
        agent.reset(seed=1)
        total = 0.0
        while True:
            action = agent.act(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        assert info["episode_metrics"]["makespan"] == 23.0
        assert total == -23.0


def test_intervals_reconstructed_from_events_match_metrics():
    env, info, _total = _run([0, 2, 1, 3])
    starts = {}
    finishes = []
    for event in env.export_trace()["events"]:
        if event["type"] == "operation_start":
            starts[event["payload"]["operation_id"]] = event["time"]
        if event["type"] == "operation_finish":
            op_id = event["payload"]["operation_id"]
            finishes.append(event["time"])
            assert abs((event["time"] - starts.pop(op_id)) - event["payload"]["duration"]) < 1e-9
    assert max(finishes) == info["episode_metrics"]["makespan"]
    assert not starts


def test_render_does_not_change_a_second_run():
    first, info_a, _ = _run([0, 2, 1, 3])
    env = JobShopEnv(instance=demo_instance(), render_mode=None)
    _obs, info = env.reset(seed=7)
    for action in (0, 2, 1, 3):
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["episode_metrics"]["makespan"] == info_a["episode_metrics"]["makespan"]
    assert [event["type"] for event in env.export_trace()["events"]] == [
        event["type"] for event in first.export_trace()["events"]
    ]
