"""The documented entry points run without a ML extra."""

from __future__ import annotations

import os
import subprocess
import sys


def _run(script: str, *args: str) -> None:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not previous else "src:" + previous
    subprocess.check_call([sys.executable, script, *args], env=env)


def test_quickstart_compare_and_masked_trainer():
    _run("examples/quickstart.py")
    _run("examples/compare_baselines.py")
    _run("examples/train_masked_policy.py", "--episodes", "4")
    _run("examples/train_masked_ppo.py")
