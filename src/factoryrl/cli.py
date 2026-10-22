"""Command line: validate, simulate, benchmark, and render."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from factoryrl.baselines.edd import EDDAgent
from factoryrl.baselines.fifo import FIFOAgent
from factoryrl.baselines.protocol import Agent
from factoryrl.baselines.random import RandomValidAgent
from factoryrl.baselines.spt import SPTAgent
from factoryrl.benchmarks.reporter import prepare_run_dir, write_report
from factoryrl.benchmarks.runner import run_benchmark
from factoryrl.environments.flow_shop import FlowShopEnv
from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.exceptions import FactoryRLError
from factoryrl.rendering.gantt import render_gantt_html
from factoryrl.scenarios.serialization import load_scenario, scenario_hash
from factoryrl.version import __version__

_AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomValidAgent,
    "fifo": FIFOAgent,
    "spt": SPTAgent,
    "edd": EDDAgent,
}


def main(argv: list[str] | None = None) -> int:
    """Run a FactoryRL command. Returns a process status code."""
    parser = argparse.ArgumentParser(prog="factoryrl", description="FactoryRL scenario tools.")
    parser.add_argument("--version", action="version", version=f"factoryrl {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Check a scenario file and print its hash.")
    validate.add_argument("path")

    simulate = sub.add_parser("simulate", help="Run one baseline on a scenario file.")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--agent", default="spt", choices=sorted(_AGENTS))
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--trace", default=None)
    simulate.add_argument("--output", default=None)

    benchmark = sub.add_parser("benchmark", help="Compare agents on a named suite.")
    benchmark.add_argument("--suite", default="jobshop-small")
    benchmark.add_argument("--agents", default="random,fifo,spt")
    benchmark.add_argument("--seeds", default="11,22,33")
    benchmark.add_argument("--split", default="all")
    benchmark.add_argument("--output", default="results")

    render = sub.add_parser("render", help="Write an HTML Gantt chart from a trace.")
    render.add_argument("--trace", required=True)
    render.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.path)
        if args.command == "simulate":
            return _simulate(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "render":
            return _render(args.trace, args.output)
    except FactoryRLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _validate(path: str) -> int:
    scenario = load_scenario(path)
    print(
        json.dumps(
            {
                "ok": True,
                "name": scenario.instance.name,
                "schema_version": scenario.instance.schema_version,
                "scenario_hash": scenario_hash(scenario.instance),
                "machines": len(scenario.instance.machines),
                "jobs": len(scenario.instance.jobs),
                "environment": scenario.environment.id,
            },
            indent=2,
        )
    )
    return 0


def _simulate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.config)
    env_cls = _env_class(scenario.environment.id)
    env = env_cls(
        instance=scenario.instance,
        reward_mode=scenario.environment.reward,
        allow_voluntary_idle=scenario.environment.allow_voluntary_idle,
        time_scale=scenario.environment.time_scale,
        failure_penalty=scenario.environment.failure_penalty,
        max_events_per_episode=scenario.environment.max_events_per_episode,
        max_decisions_per_episode=scenario.environment.max_decisions_per_episode,
        record_events=True,
        check_invariants=scenario.environment.check_invariants,
    )
    agent = _AGENTS[args.agent]()
    observation, info = env.reset(seed=args.seed)
    agent.reset(seed=args.seed)
    total = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(agent.act(observation, info["action_mask"]))
        observation, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
    payload = {
        "agent": args.agent,
        "seed": args.seed,
        "scenario_hash": scenario_hash(scenario.instance),
        "return": total,
        "wall_seconds": env.wall_seconds,
        "episode_metrics": info["episode_metrics"],
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.trace:
        Path(args.trace).write_text(
            json.dumps(env.export_trace(agent_id=args.agent), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    env.close()
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    records = run_benchmark(suite=args.suite, agents=agents, seeds=seeds, split=args.split)
    run_dir = prepare_run_dir(args.output)
    write_report(
        run_dir,
        records,
        config={
            "suite": args.suite,
            "agents": agents,
            "seeds": seeds,
            "split": args.split,
            "reward": "negative_elapsed_time",
        },
    )
    print(run_dir)
    return 0


def _render(trace_path: str, output: str) -> int:
    document = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    intervals = document.get("intervals", [])
    machines = sorted({str(interval["machine_id"]) for interval in intervals})
    Path(output).write_text(render_gantt_html(intervals, machines), encoding="utf-8")
    return 0


def _env_class(env_id: str) -> type[JobShopEnv] | type[FlowShopEnv]:
    if env_id == "FactoryRL/JobShop-v0":
        return JobShopEnv
    if env_id == "FactoryRL/FlowShop-v0":
        return FlowShopEnv
    raise FactoryRLError(f"Unsupported environment id {env_id!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
