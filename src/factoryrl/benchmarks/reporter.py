"""Write benchmark artifacts. Failures are rows, not dropped episodes."""

from __future__ import annotations

import csv
import html
import json
import platform
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from factoryrl.benchmarks.runner import EpisodeRecord
from factoryrl.version import __version__


def prepare_run_dir(output: str | Path) -> Path:
    """Create ``output/run_YYYYMMDD_HHMMSS``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_report(
    run_dir: Path,
    records: list[EpisodeRecord],
    *,
    config: dict[str, Any],
) -> None:
    """Write the machine-readable bundle and a short HTML table."""
    (run_dir / "config.json").write_text(_dumps(config), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        _dumps({"env_id": "FactoryRL/JobShop-v0", "reward": config.get("reward", "negative_elapsed_time")}),
        encoding="utf-8",
    )
    (run_dir / "versions.json").write_text(
        _dumps(
            {
                "factoryrl": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "gymnasium": gymnasium.__version__,
                "platform": platform.platform(),
                "rng_algorithm": "numpy.random.PCG64",
            }
        ),
        encoding="utf-8",
    )
    summary = summarize(records)
    (run_dir / "summary.json").write_text(_dumps(summary), encoding="utf-8")
    _write_csv(run_dir / "episodes.csv", records)
    (run_dir / "report.html").write_text(_html(summary, records), encoding="utf-8")


def summarize(records: list[EpisodeRecord]) -> dict[str, Any]:
    """Aggregate without hiding unsuccessful episodes."""
    grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_id].append(record)
    agents: dict[str, Any] = {}
    for agent_id, rows in grouped.items():
        successes = [row for row in rows if row.status == "success" and row.objective is not None]
        improvements = [
            row.relative_improvement_vs_fifo
            for row in successes
            if row.relative_improvement_vs_fifo is not None
        ]
        agents[agent_id] = {
            "episodes": len(rows),
            "successes": len(successes),
            "failures": len(rows) - len(successes),
            "mean_makespan_success": _mean([row.objective for row in successes]),
            "median_makespan_success": _median([row.objective for row in successes]),
            "mean_relative_improvement_vs_fifo_on_success": _mean(improvements),
        }
    return {
        "episodes": len(records),
        "successes": sum(1 for row in records if row.status == "success"),
        "failures": sum(1 for row in records if row.status != "success"),
        "agents": agents,
        "note": (
            "Means use successful episodes only. Failure counts are reported separately and are not dropped."
        ),
    }


def _write_csv(path: Path, records: list[EpisodeRecord]) -> None:
    fieldnames = [
        "env_id",
        "instance_id",
        "instance_hash",
        "split",
        "seed",
        "agent_id",
        "objective",
        "status",
        "makespan",
        "mean_flow_time",
        "total_tardiness",
        "decision_count",
        "event_count",
        "simulation_steps",
        "wall_seconds",
        "episode_return",
        "relative_improvement_vs_fifo",
        "config_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record.metrics
            writer.writerow(
                {
                    "env_id": record.env_id,
                    "instance_id": record.instance_id,
                    "instance_hash": record.instance_hash,
                    "split": record.split,
                    "seed": record.seed,
                    "agent_id": record.agent_id,
                    "objective": "" if record.objective is None else record.objective,
                    "status": record.status,
                    "makespan": metrics.get("makespan", ""),
                    "mean_flow_time": metrics.get("mean_flow_time", ""),
                    "total_tardiness": metrics.get("total_tardiness", ""),
                    "decision_count": metrics.get("decision_count", ""),
                    "event_count": metrics.get("event_count", ""),
                    "simulation_steps": record.simulation_steps,
                    "wall_seconds": record.wall_seconds,
                    "episode_return": record.episode_return,
                    "relative_improvement_vs_fifo": ""
                    if record.relative_improvement_vs_fifo is None
                    else record.relative_improvement_vs_fifo,
                    "config_hash": record.config_hash,
                }
            )


def _html(summary: dict[str, Any], records: list[EpisodeRecord]) -> str:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.instance_id)}</td>"
            f"<td>{html.escape(record.split)}</td>"
            f"<td>{record.seed}</td>"
            f"<td>{html.escape(record.agent_id)}</td>"
            f"<td>{html.escape(record.status)}</td>"
            f"<td>{'' if record.objective is None else f'{record.objective:.6g}'}</td>"
            "</tr>"
        )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>FactoryRL benchmark</title></head><body>'
        "<h1>FactoryRL benchmark</h1>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
        '<table border="1" cellspacing="0" cellpadding="4">'
        "<tr><th>instance</th><th>split</th><th>seed</th><th>agent</th><th>status</th><th>makespan</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: Sequence[float | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0
