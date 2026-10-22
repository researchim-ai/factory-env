"""Static instance validation."""

from __future__ import annotations

import math

import pytest

from factoryrl.exceptions import InstanceValidationError, SchemaVersionError
from factoryrl.models.buffer import BufferSpec
from factoryrl.scenarios.instance import instance, job, machine, operation


def _pair():
    return [machine("M1")], [job("J1", [operation("J1-O1", "M1", 3)])]


def test_valid_minimal_instance():
    built = instance("line", *_pair())
    assert built.schema_version == 1
    assert built.jobs[0].operations[0].duration == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, j: instance(" ", m, j),
        lambda m, j: instance("ok", [], j),
        lambda m, j: instance("ok", m, []),
        lambda m, j: instance("ok", [machine("M1"), machine("M1")], j),
        lambda m, j: instance("ok", [machine("M1", capacity=2)], j),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M9", 3)])]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", 0)])]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", -5)])]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", math.nan)])]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", math.inf)])]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", 3)], release_time=4)]),
        lambda m, j: instance("ok", m, [job("J1", [operation("J1-O1", "M1", 3)], priority=0)]),
        lambda m, j: instance("ok", m, j, horizon=0),
        lambda m, j: instance("ok", m, j, horizon=-1),
        lambda m, j: instance("ok", m, j, time_unit="second"),
        lambda m, j: instance("ok", m, [job("J1", [])]),
        lambda m, j: instance(
            "ok",
            m,
            [job("J1", [operation("A", "M1", 1), operation("A", "M1", 1)])],
        ),
    ],
)
def test_rejects_malformed_instances(mutate):
    machines, jobs = _pair()
    with pytest.raises((InstanceValidationError, SchemaVersionError)):
        mutate(machines, jobs)


def test_unknown_schema_version():
    with pytest.raises(SchemaVersionError):
        instance("ok", *_pair(), schema_version=99)


def test_explicit_linear_chain_is_accepted():
    built = instance(
        "chain",
        [machine("M1")],
        [
            job(
                "J1",
                [
                    operation("B", "M1", 2, predecessors=("A",)),
                    operation("A", "M1", 2),
                ],
            )
        ],
    )
    assert [op.id for op in built.jobs[0].operations] == ["B", "A"]


def test_branching_predecessors_rejected():
    with pytest.raises(InstanceValidationError):
        instance(
            "branch",
            [machine("M1")],
            [
                job(
                    "J1",
                    [
                        operation("A", "M1", 1),
                        operation("B", "M1", 1, predecessors=("A",)),
                        operation("C", "M1", 1, predecessors=("A",)),
                    ],
                )
            ],
        )


def test_cycle_rejected():
    with pytest.raises(InstanceValidationError):
        instance(
            "cycle",
            [machine("M1")],
            [
                job(
                    "J1",
                    [
                        operation("A", "M1", 1, predecessors=("B",)),
                        operation("B", "M1", 1, predecessors=("A",)),
                    ],
                )
            ],
        )


def test_operation_type_must_be_supported():
    with pytest.raises(InstanceValidationError):
        instance(
            "typed",
            [machine("M1", supported_operation_types=("drill",))],
            [job("J1", [operation("A", "M1", 1, operation_type="weld")])],
        )
    built = instance(
        "typed",
        [machine("M1", supported_operation_types=("drill",))],
        [job("J1", [operation("A", "M1", 1, operation_type="drill")])],
    )
    assert built.jobs[0].operations[0].operation_type == "drill"


def test_buffer_is_not_constructible():
    with pytest.raises(Exception, match="v0.1"):
        BufferSpec(id="B1", capacity=2)


def test_metadata_rejects_objects():
    with pytest.raises(InstanceValidationError):
        instance("ok", *_pair(), metadata={"bad": object()})
