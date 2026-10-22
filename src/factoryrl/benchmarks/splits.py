"""Fixed instance ids for the small job-shop suite.

A new seed drawn from the same generator is not treated as out of distribution.
The OOD slice changes both the size and the processing-time range.
"""

from __future__ import annotations

from dataclasses import dataclass

from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.builtins import choice_instance, demo_instance, edd_instance
from factoryrl.scenarios.generator import generate_job_shop

TRAIN_SEEDS = (0, 1, 2)
VALIDATION_SEEDS = (100, 101)
TEST_SEEDS = (200, 201)
OOD_SEEDS = (300, 301)


@dataclass(frozen=True, slots=True)
class SuiteCase:
    instance_id: str
    instance: FactoryInstance
    split: str
    env_id: str = "FactoryRL/JobShop-v0"


def jobshop_small_suite() -> list[SuiteCase]:
    """Hand-built cases plus disjoint generated ids published with their seeds."""
    cases = [
        SuiteCase("demo", demo_instance(), "test"),
        SuiteCase("choice", choice_instance(), "test"),
        SuiteCase("edd", edd_instance(), "test"),
    ]
    for seed in TRAIN_SEEDS:
        cases.append(
            SuiteCase(f"train-3x5-{seed}", generate_job_shop(3, 5, seed=seed, low=1, high=10), "train")
        )
    for seed in VALIDATION_SEEDS:
        cases.append(
            SuiteCase(f"val-3x5-{seed}", generate_job_shop(3, 5, seed=seed, low=1, high=10), "validation")
        )
    for seed in TEST_SEEDS:
        cases.append(
            SuiteCase(f"test-3x5-{seed}", generate_job_shop(3, 5, seed=seed, low=1, high=10), "test")
        )
    for seed in OOD_SEEDS:
        cases.append(
            SuiteCase(
                f"ood-4x6-{seed}",
                generate_job_shop(4, 6, seed=seed, low=20, high=40),
                "ood_test",
            )
        )
    return cases


SUITES = {"jobshop-small": jobshop_small_suite}
