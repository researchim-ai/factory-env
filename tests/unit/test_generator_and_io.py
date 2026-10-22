"""Generators, safe scenario IO, migrations, and external-file provenance."""

from __future__ import annotations

import json

import pytest

from factoryrl.exceptions import ResourceLimitError, SchemaVersionError, SerializationError
from factoryrl.limits import ResourceLimits
from factoryrl.scenarios.builtins import demo_instance
from factoryrl.scenarios.generator import generate_flow_shop, generate_job_shop
from factoryrl.scenarios.loaders import load_matrix_jobshop
from factoryrl.scenarios.schema import (
    EnvironmentConfig,
    Scenario,
    apply_migrations,
    migrate_v1_to_v2,
)
from factoryrl.scenarios.serialization import (
    dump_scenario,
    load_scenario,
    load_scenario_text,
    scenario_hash,
)


def test_generator_is_deterministic_and_flow_shop_is_a_flow_shop():
    first = generate_job_shop(4, 6, seed=7, low=2, high=5)
    second = generate_job_shop(4, 6, seed=7, low=2, high=5)
    assert scenario_hash(first) == scenario_hash(second)
    assert scenario_hash(first) != scenario_hash(generate_job_shop(4, 6, seed=8, low=2, high=5))
    flow = generate_flow_shop(3, 4, seed=7, low=2, high=5)
    sequence = tuple(op.machine_id for op in flow.jobs[0].operations)
    assert sequence == ("M1", "M2", "M3")
    assert all(tuple(op.machine_id for op in job.operations) == sequence for job in flow.jobs)
    fixed = generate_job_shop(
        2,
        2,
        seed=1,
        processing_time_distribution="fixed",
        fixed_value=4,
        machine_assignment="random",
        ops_per_job=3,
        due_date_factor=2,
    )
    assert all(op.duration == 4 for job in fixed.jobs for op in job.operations)
    assert all(job.due_date == 24 for job in fixed.jobs)


def test_yaml_and_json_roundtrip(tmp_path):
    original = load_scenario("examples/factories/demo.yaml")
    yaml_path = tmp_path / "demo.yaml"
    json_path = tmp_path / "demo.json"
    dump_scenario(original, yaml_path)
    dump_scenario(original, json_path)
    assert scenario_hash(load_scenario(yaml_path).instance) == scenario_hash(demo_instance())
    assert scenario_hash(load_scenario(json_path).instance) == scenario_hash(demo_instance())


def test_malicious_yaml_and_oversize_input(tmp_path):
    with pytest.raises(SerializationError):
        load_scenario_text("factory: !!python/object/apply:os.system ['echo pwned']\n", fmt="yaml")
    huge = " " * 50
    with pytest.raises(ResourceLimitError):
        load_scenario_text(huge, fmt="yaml", limits=ResourceLimits(max_file_bytes=10))
    bomb = "a: &a [" + ", ".join(["1"] * 5) + "]\n" + "b: *a\n"
    with pytest.raises(SerializationError):
        load_scenario_text(bomb, fmt="yaml", limits=ResourceLimits(max_yaml_collection=3))


def test_migration_is_explicit_and_v2_does_not_load():
    document = demo_instance().to_dict()
    migrated = migrate_v1_to_v2(document)
    assert migrated["schema_version"] == 2
    assert migrated["factory"]["metadata"]["migrated_from"] == 1
    assert apply_migrations(document, target=1)["schema_version"] == 1
    with pytest.raises(SchemaVersionError):
        load_scenario_text(json.dumps(migrated), fmt="json")


def test_matrix_loader_requires_provenance(tmp_path):
    path = tmp_path / "tiny.jsp"
    path.write_text("2 2\n0 8 1 12\n1 7 0 15\n", encoding="utf-8")
    with pytest.raises(SerializationError):
        load_matrix_jobshop(path, source="", license_id="CC0")
    loaded = load_matrix_jobshop(path, source="hand-written test", license_id="CC0-1.0", name="tiny")
    assert loaded.metadata["license_id"] == "CC0-1.0"
    assert loaded.metadata["source"] == "hand-written test"
    assert len(loaded.jobs) == 2
    assert loaded.jobs[0].operations[0].machine_id == "M0"


def test_environment_section_is_not_part_of_the_instance_hash():
    left = Scenario(demo_instance(), EnvironmentConfig(reward="negative_elapsed_time"))
    right = Scenario(demo_instance(), EnvironmentConfig(reward="terminal_makespan"))
    assert scenario_hash(left.instance) == scenario_hash(right.instance)
    assert left.to_dict()["environment"]["reward"] != right.to_dict()["environment"]["reward"]
