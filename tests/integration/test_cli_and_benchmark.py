"""CLI commands and the benchmark report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from factoryrl.benchmarks.runner import run_benchmark
from factoryrl.cli import main


def test_validate_and_simulate(tmp_path: Path):
    assert main(["validate", "examples/factories/demo.yaml"]) == 0
    output = tmp_path / "metrics.json"
    trace = tmp_path / "trace.json"
    gantt = tmp_path / "gantt.html"
    assert (
        main(
            [
                "simulate",
                "--config",
                "examples/factories/demo.yaml",
                "--agent",
                "spt",
                "--seed",
                "4",
                "--output",
                str(output),
                "--trace",
                str(trace),
            ]
        )
        == 0
    )
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["episode_metrics"]["makespan"] == 23.0
    assert main(["render", "--trace", str(trace), "--output", str(gantt)]) == 0
    assert "<svg" in gantt.read_text(encoding="utf-8")


def test_benchmark_keeps_every_episode(tmp_path: Path):
    records = run_benchmark(suite="jobshop-small", agents=["fifo", "spt"], seeds=[11], split="test")
    assert records
    assert all(record.status == "success" for record in records)
    assert {record.agent_id for record in records} == {"fifo", "spt"}
    choice = [record for record in records if record.instance_id == "choice"]
    makespans = {record.agent_id: record.objective for record in choice}
    assert makespans["spt"] == 12.0
    assert makespans["fifo"] == 21.0
    assert (
        main(
            [
                "benchmark",
                "--suite",
                "jobshop-small",
                "--agents",
                "fifo",
                "--seeds",
                "11",
                "--split",
                "validation",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    runs = list(tmp_path.glob("run_*"))
    assert len(runs) == 1
    summary = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["failures"] == 0
    assert (runs[0] / "episodes.csv").exists()


def test_import_does_not_pull_in_a_learning_framework():
    script = (
        "import factoryrl, sys\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'stable_baselines3' not in sys.modules\n"
        "assert 'ortools' not in sys.modules\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" if not env.get("PYTHONPATH") else "src:" + env["PYTHONPATH"]
    subprocess.check_call([sys.executable, "-c", script], env=env)
