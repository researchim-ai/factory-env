"""Small instances owned by this repository. No third-party benchmark data."""

from __future__ import annotations

from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.instance import instance, job, machine, operation


def demo_instance() -> FactoryInstance:
    """Two jobs and two machines from the specification's acceptance scenario.

    One legal non-delay schedule has makespan 23. That checks the schedule;
    it is not a claim that 23 was proved by an external solver inside this
    constructor.
    """
    return instance(
        "demo_jobshop",
        [machine("M1"), machine("M2")],
        [
            job(
                "J1",
                [operation("J1-O1", "M1", 8), operation("J1-O2", "M2", 12)],
                due_date=50,
                priority=1,
            ),
            job(
                "J2",
                [operation("J2-O1", "M2", 7), operation("J2-O2", "M1", 15)],
                due_date=65,
                priority=2,
            ),
        ],
        horizon=1000,
    )


def choice_instance() -> FactoryInstance:
    """A two-job instance where the first dispatch changes the makespan.

    Shortest processing time finishes at 12. The other first dispatch finishes
    at 21. Neither job has a due date.
    """
    return instance(
        "choice_jobshop",
        [machine("M1"), machine("M2")],
        [
            job("J1", [operation("J1-O1", "M1", 10), operation("J1-O2", "M2", 1)]),
            job("J2", [operation("J2-O1", "M1", 1), operation("J2-O2", "M2", 10)]),
        ],
    )


def edd_instance() -> FactoryInstance:
    """Same routes as :func:`choice_instance` with due dates that favor the long job."""
    return instance(
        "edd_jobshop",
        [machine("M1"), machine("M2")],
        [
            job(
                "J1",
                [operation("J1-O1", "M1", 10), operation("J1-O2", "M2", 1)],
                due_date=5,
            ),
            job(
                "J2",
                [operation("J2-O1", "M1", 1), operation("J2-O2", "M2", 10)],
                due_date=100,
            ),
        ],
    )


def flow_demo_instance() -> FactoryInstance:
    """Permutation flow shop. SPT / Johnson order J2 then J1 has makespan 7."""
    return instance(
        "flow_demo",
        [machine("M1"), machine("M2")],
        [
            job("J1", [operation("J1-O1", "M1", 3), operation("J1-O2", "M2", 2)], due_date=20),
            job("J2", [operation("J2-O1", "M1", 1), operation("J2-O2", "M2", 4)], due_date=20),
        ],
    )
