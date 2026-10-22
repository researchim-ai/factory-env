"""Independent non-delay scheduler used only by tests.

It does not share the engine's heap. It jumps to the next time at which a
ready operation has a free machine, then consumes one action.
"""

from __future__ import annotations

from factoryrl.core.events import WAIT_ACTION
from factoryrl.models.factory import FactoryInstance
from factoryrl.models.topology import compile_instance


def reference_makespan(instance: FactoryInstance, actions: list[int]) -> float:
    """Return the makespan of a non-delay schedule for ``actions``."""
    compiled = compile_instance(instance)
    n = compiled.n_operations
    pred_left = [len(preds) for preds in compiled.predecessors]
    done = [False] * n
    ready = {index for index, left in enumerate(pred_left) if left == 0}
    machine_free = [0.0] * compiled.n_machines
    running: list[tuple[float, int, int]] = []
    ends = [None] * n
    time = 0.0
    cursor = 0

    def finish_due() -> None:
        nonlocal running
        still: list[tuple[float, int, int]] = []
        for end, operation, machine in running:
            if end <= time + 1e-9 and not done[operation]:
                done[operation] = True
                ends[operation] = end
                machine_free[machine] = end
                for successor in compiled.successors[operation]:
                    pred_left[successor] -= 1
                    if pred_left[successor] == 0:
                        ready.add(successor)
            elif not done[operation]:
                still.append((end, operation, machine))
        running = still

    while not all(done):
        finish_due()
        if all(done):
            break
        dispatchable = sorted(
            op for op in ready if machine_free[compiled.operation_machine[op]] <= time + 1e-9
        )
        if dispatchable:
            if cursor >= len(actions):
                raise AssertionError("reference scheduler ran out of actions")
            choice = actions[cursor]
            cursor += 1
            if choice == WAIT_ACTION:
                raise AssertionError("reference scheduler does not implement WAIT")
            if choice not in dispatchable:
                raise AssertionError(f"{choice} is not dispatchable at t={time}: {dispatchable}")
            machine = compiled.operation_machine[choice]
            duration = compiled.operation_duration[choice]
            ready.remove(choice)
            running.append((time + duration, choice, machine))
            machine_free[machine] = time + duration
            continue
        future = [end for end, operation, _machine in running if not done[operation] and end > time + 1e-9]
        if not future:
            raise AssertionError(f"reference deadlock at t={time}")
        time = min(future)
    assert cursor == len(actions)
    return max(end for end in ends if end is not None)
