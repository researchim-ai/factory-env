"""Invalid actions, WAIT, budgets, deadlock, and the alternate reward."""

from __future__ import annotations

import pytest

from factoryrl.core.events import WAIT_ACTION
from factoryrl.core.state import OperationStatus
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.exceptions import ConfigurationError, InvalidActionError
from factoryrl.rewards.makespan import compute_reward
from factoryrl.scenarios.builtins import choice_instance, demo_instance
from factoryrl.scenarios.instance import instance, job, machine, operation


def test_illegal_dispatch_does_not_move_time():
    env = JobShopEnv(instance=demo_instance())
    env.reset(seed=1)
    before = env.export_trace()["events"]
    with pytest.raises(InvalidActionError):
        env.step(1)
    assert env.export_trace()["events"] == before
    assert env._engine.clock.now == 0.0
    assert env._engine.invalid_action_count == 1
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert not terminated and not truncated
    assert obs["operations"].shape[0] == 4


def test_wait_is_rejected_by_default_and_advances_when_enabled():
    env = JobShopEnv(instance=demo_instance())
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(env.action_space.n - 1)
    idle = JobShopEnv(instance=demo_instance(), allow_voluntary_idle=True)
    idle.reset(seed=1)
    with pytest.raises(InvalidActionError):
        idle.step(idle.action_space.n - 1)
    idle.step(0)
    _obs, reward, terminated, truncated, info = idle.step(idle.action_space.n - 1)
    assert info["delta_sim_time"] == 8.0
    assert reward == -8.0
    assert not terminated and not truncated
    assert info["episode_metrics"]["sim_time"] == 8.0


def test_boolean_action_is_rejected():
    env = JobShopEnv(instance=demo_instance())
    env.reset()
    with pytest.raises(InvalidActionError):
        env.step(True)


def test_reset_budget_too_small_is_a_configuration_error():
    with pytest.raises(ConfigurationError):
        JobShopEnv(instance=demo_instance(), max_events_per_episode=1)


def test_decision_budget_truncates():
    env = JobShopEnv(instance=demo_instance(), max_decisions_per_episode=1)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert info["truncation_reason"] == "max_decisions"
    assert info["success"] is False
    assert reward <= 0.0


def test_horizon_truncates_before_completion():
    short = instance(
        "short",
        [machine("M1")],
        [job("J1", [operation("A", "M1", 10)], due_date=3)],
        horizon=4,
    )
    env = JobShopEnv(instance=short, failure_penalty=5)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert info["truncation_reason"] == "horizon"
    assert info["episode_metrics"]["makespan"] is None
    assert info["episode_metrics"]["sim_time"] == 4.0
    assert reward == -4.0 - 5.0


def test_terminal_makespan_reward_is_paid_once():
    env = JobShopEnv(instance=demo_instance(), reward_mode="terminal_makespan")
    env.reset(seed=0)
    rewards = []
    for action in (0, 2, 1, 3):
        _obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    assert rewards[:-1] == [0.0, 0.0, 0.0]
    assert rewards[-1] == -23.0
    assert info["episode_metrics"]["success"] is True


def test_missing_due_dates_are_excluded_from_tardiness():
    env = JobShopEnv(instance=choice_instance())
    env.reset(seed=0)
    env.step(0)
    while True:
        mask = env.action_masks()
        if not mask.any():
            break
        _obs, _reward, terminated, truncated, info = env.step(int(mask.argmax()))
        if terminated or truncated:
            break
    assert info["episode_metrics"]["total_tardiness"] is None
    assert info["episode_metrics"]["weighted_tardiness"] is None


def test_deadlock_is_terminal_failure_not_success(demo):
    env = JobShopEnv(instance=demo)
    env.reset(seed=0)
    engine = env._engine
    for index, status in enumerate(engine.state.status):
        if status is OperationStatus.READY:
            engine.state.status[index] = OperationStatus.NOT_READY
            engine.state.ready_time[index] = None
    for job_index in range(engine.compiled.n_jobs):
        engine.state.job_released[job_index] = False
    engine.state.ready_by_machine = [set() for _ in engine.compiled.machine_ids]
    engine.queue = type(engine.queue)()
    engine._run(must_advance=False)
    assert engine.termination_reason == "deadlock"
    assert engine.success is False
    assert engine.diagnosis["queue_empty"] is True
    assert engine.diagnosis["unfinished_operations"]


def test_reward_helper_rejects_unknown_mode():
    with pytest.raises(ConfigurationError):
        compute_reward(
            mode="mystery",
            delta_sim_time=1,
            time_scale=1,
            terminal=False,
            success=None,
            makespan=None,
            failure_penalty=0,
        )


def test_wait_constant_is_not_an_operation_index():
    assert WAIT_ACTION < 0
