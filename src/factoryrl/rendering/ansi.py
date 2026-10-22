"""Plain-text view of a schedule. Rendering does not change the simulation."""

from __future__ import annotations

from factoryrl.core.state import StateSnapshot


def render_ansi(snapshot: StateSnapshot) -> str:
    """Return a terminal table of machines, ready work, and finished intervals."""
    lines = [
        (
            f"t={snapshot.now:.6g}  jobs {snapshot.jobs_completed}/{snapshot.n_jobs}"
            f"  ops {snapshot.operations_completed}/{snapshot.n_operations}"
        )
    ]
    lines.append("machines:")
    for machine in snapshot.machines:
        current = machine.current_operation_id or "-"
        lines.append(
            f"  {machine.id:<8} {machine.status.value:<8} current={current:<12} "
            f"remaining={machine.remaining_time:.6g} busy={machine.busy_time:.6g} "
            f"ready={machine.ready_count}"
        )
    ready = [op for op in snapshot.operations if op.status.value == "ready"]
    lines.append("ready:")
    if not ready:
        lines.append("  (none)")
    for op in ready:
        lines.append(
            f"  {op.id:<12} machine={op.machine_id:<8} duration={op.duration:.6g} "
            f"waiting={op.waiting_time:.6g}"
        )
    lines.append("intervals:")
    if not snapshot.intervals:
        lines.append("  (none)")
    for interval in snapshot.intervals:
        lines.append(
            f"  {interval.machine_id:<8} {interval.operation_id:<12} "
            f"[{interval.start:.6g}, {interval.end:.6g}]"
        )
    return "\n".join(lines) + "\n"
