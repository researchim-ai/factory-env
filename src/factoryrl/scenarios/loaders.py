"""Parsers for user-supplied benchmark text.

This module does not ship third-party instances. Callers must name the source
and a license identifier before a file is turned into a factory.
"""

from __future__ import annotations

from pathlib import Path

from factoryrl.exceptions import InstanceValidationError, SerializationError
from factoryrl.hashing import sha256_bytes
from factoryrl.limits import DEFAULT_LIMITS, ResourceLimits
from factoryrl.models.factory import FactoryInstance
from factoryrl.scenarios.instance import instance, job, machine, operation


def load_matrix_jobshop(
    path: str | Path,
    *,
    source: str,
    license_id: str,
    name: str | None = None,
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> FactoryInstance:
    """Load a matrix file: first line ``n_jobs n_machines``, then one job per line.

    Each job line is ``machine duration`` pairs. Machine numbers are 0-based.
    The file is not executed. ``source`` and ``license_id`` are stored in
    metadata so the provenance stays attached to the instance.
    """
    if not source.strip() or not license_id.strip():
        raise SerializationError(
            "Refusing to import an external instance without source and license_id. "
            "Do not copy benchmark files into the repository until their license is recorded."
        )
    file_path = Path(path)
    blob = file_path.read_bytes()
    if len(blob) > limits.max_file_bytes:
        raise SerializationError("Matrix instance exceeds the file size limit.")
    lines = [
        line.strip()
        for line in blob.decode("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise SerializationError("Matrix instance is empty.")
    header = lines[0].split()
    if len(header) != 2:
        raise SerializationError("The first line must be 'n_jobs n_machines'.")
    try:
        n_jobs, n_machines = int(header[0]), int(header[1])
    except ValueError as exc:
        raise SerializationError("n_jobs and n_machines must be integers.") from exc
    if n_jobs < 1 or n_machines < 1 or len(lines) != n_jobs + 1:
        raise SerializationError("Matrix instance job lines do not match the header.")
    machines = [machine(f"M{index}") for index in range(n_machines)]
    jobs = []
    for job_number, line in enumerate(lines[1:], start=1):
        tokens = line.split()
        if len(tokens) != 2 * n_machines:
            raise InstanceValidationError(
                f"Job line {job_number} must contain {n_machines} machine/duration pairs."
            )
        operations = []
        for step in range(n_machines):
            machine_index = int(tokens[2 * step])
            duration = float(tokens[2 * step + 1])
            if machine_index < 0 or machine_index >= n_machines:
                raise InstanceValidationError(f"Machine index {machine_index} is out of range.")
            operations.append(operation(f"J{job_number}-O{step + 1}", f"M{machine_index}", duration))
        jobs.append(job(f"J{job_number}", operations))
    return instance(
        name or file_path.stem,
        machines,
        jobs,
        metadata={
            "source": source,
            "license_id": license_id,
            "sha256": sha256_bytes(blob),
            "format": "factoryrl-matrix-v1",
        },
        limits=limits,
    )
