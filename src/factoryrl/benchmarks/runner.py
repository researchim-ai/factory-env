"""Run baselines on a frozen suite. The runner does not change the environment rules."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from factoryrl.baselines.edd import EDDAgent
from factoryrl.baselines.fifo import FIFOAgent
from factoryrl.baselines.protocol import Agent
from factoryrl.baselines.random import RandomValidAgent
from factoryrl.baselines.spt import SPTAgent
from factoryrl.benchmarks.metrics import relative_improvement
from factoryrl.benchmarks.splits import SUITES, SuiteCase
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.exceptions import BaselineError, ConfigurationError
from factoryrl.scenarios.serialization import scenario_hash

AGENT_BUILDERS: dict[str, Callable[[], Agent]] = {
    "random": RandomValidAgent,
    "fifo": FIFOAgent,
    "spt": SPTAgent,
    "edd": EDDAgent,
}


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One evaluated episode, including failures."""

    env_id: str
    instance_id: str
    instance_hash: str
    split: str
    seed: int
    agent_id: str
    objective: float | None
    metrics: dict[str, Any]
    status: str
    simulation_steps: int
    wall_seconds: float
    config_hash: str
    episode_return: float
    relative_improvement_vs_fifo: float | None = None


def make_agent(name: str) -> Agent:
    try:
        return AGENT_BUILDERS[name]()
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown agent {name!r}. Known agents: {', '.join(sorted(AGENT_BUILDERS))}."
        ) from exc


def run_episode(env: JobShopEnv, agent: Agent, *, seed: int) -> tuple[dict[str, Any], float, float]:
    """Execute one episode. Returns info, undiscounted return, and wall seconds."""
    started = time.perf_counter()
    observation, info = env.reset(seed=seed)
    agent.reset(seed=seed)
    total = 0.0
    guard = env.max_decisions + 2
    for _ in range(guard):
        action = int(agent.act(observation, info["action_mask"]))
        observation, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            return info, total, time.perf_counter() - started
    raise ConfigurationError("Episode exceeded the decision budget without stopping.")


def iter_cases(suite: str, split: str | None = None) -> list[SuiteCase]:
    try:
        cases = SUITES[suite]()
    except KeyError as exc:
        raise ConfigurationError(f"Unknown suite {suite!r}.") from exc
    if split is None or split == "all":
        return cases
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise ConfigurationError(f"Suite {suite!r} has no split {split!r}.")
    return selected


def run_benchmark(
    *,
    suite: str = "jobshop-small",
    agents: list[str],
    seeds: list[int],
    split: str | None = None,
    check_invariants: bool = True,
) -> list[EpisodeRecord]:
    """Evaluate every agent on every selected case and seed.

    Failures stay in the list. Relative improvement against FIFO is filled
    only when both episodes succeeded and FIFO makespan is positive.
    """
    if not agents or not seeds:
        raise ConfigurationError("Benchmark needs at least one agent and one seed.")
    cases = iter_cases(suite, split)
    records: list[EpisodeRecord] = []
    for case in cases:
        env = JobShopEnv(
            instance=case.instance,
            check_invariants=check_invariants,
            record_events=False,
        )
        for seed in seeds:
            for agent_name in agents:
                agent = make_agent(agent_name)
                try:
                    info, total, wall = run_episode(env, agent, seed=seed)
                    metrics = info["episode_metrics"]
                    success = bool(metrics["success"])
                    status = "success" if success else _failure_status(metrics)
                    objective = metrics["makespan"] if success else None
                except BaselineError as exc:
                    metrics = {"success": False, "termination_reason": "baseline_error", "error": str(exc)}
                    total = 0.0
                    wall = 0.0
                    status = "baseline_error"
                    objective = None
                    info = {"config_hash": env.config_hash}
                records.append(
                    EpisodeRecord(
                        env_id=case.env_id,
                        instance_id=case.instance_id,
                        instance_hash=scenario_hash(case.instance),
                        split=case.split,
                        seed=int(seed),
                        agent_id=agent_name,
                        objective=None if objective is None else float(objective),
                        metrics=metrics,
                        status=status,
                        simulation_steps=int(metrics.get("decision_count", 0) or 0),
                        wall_seconds=wall,
                        config_hash=str(info["config_hash"]),
                        episode_return=total,
                    )
                )
        env.close()
    return _with_fifo_reference(records)


def _failure_status(metrics: dict[str, Any]) -> str:
    if metrics.get("termination_reason"):
        return str(metrics["termination_reason"])
    if metrics.get("truncation_reason"):
        return f"truncated:{metrics['truncation_reason']}"
    return "failure"


def _with_fifo_reference(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    fifo: dict[tuple[str, int], float] = {}
    for record in records:
        if record.agent_id == "fifo" and record.objective is not None:
            fifo[(record.instance_id, record.seed)] = record.objective
    annotated: list[EpisodeRecord] = []
    for record in records:
        improvement = None
        baseline = fifo.get((record.instance_id, record.seed))
        if record.objective is not None and baseline is not None and baseline > 0.0:
            improvement = relative_improvement(baseline, record.objective)
        annotated.append(
            EpisodeRecord(
                env_id=record.env_id,
                instance_id=record.instance_id,
                instance_hash=record.instance_hash,
                split=record.split,
                seed=record.seed,
                agent_id=record.agent_id,
                objective=record.objective,
                metrics=record.metrics,
                status=record.status,
                simulation_steps=record.simulation_steps,
                wall_seconds=record.wall_seconds,
                config_hash=record.config_hash,
                episode_return=record.episode_return,
                relative_improvement_vs_fifo=improvement,
            )
        )
    return annotated
