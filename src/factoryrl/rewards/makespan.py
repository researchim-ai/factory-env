"""P0 rewards whose successful return equals ``-makespan / time_scale`` at gamma 1."""

from __future__ import annotations

from factoryrl.exceptions import ConfigurationError
from factoryrl.rewards.base import REWARD_NEGATIVE_ELAPSED, REWARD_TERMINAL_MAKESPAN


def compute_reward(
    *,
    mode: str,
    delta_sim_time: float,
    time_scale: float,
    terminal: bool,
    success: bool | None,
    makespan: float | None,
    failure_penalty: float,
) -> tuple[float, dict[str, float]]:
    """Return the scalar reward and its named components.

    ``negative_elapsed_time`` pays ``-delta_sim_time / time_scale`` on every
    transition, including the last one. On a failure it also subtracts
    ``failure_penalty``. ``terminal_makespan`` pays nothing until a successful
    terminal step, which receives ``-makespan / time_scale``.
    """
    if failure_penalty < 0.0:
        raise ConfigurationError("failure_penalty must be >= 0.")
    penalty = 0.0
    if terminal and success is not True:
        penalty = -float(failure_penalty)
    if mode == REWARD_NEGATIVE_ELAPSED:
        elapsed = -float(delta_sim_time) / time_scale
        return elapsed + penalty, {"elapsed_time": elapsed, "failure_penalty": penalty}
    if mode == REWARD_TERMINAL_MAKESPAN:
        if terminal and success is True:
            if makespan is None:
                raise ConfigurationError("A successful episode is missing makespan.")
            value = -float(makespan) / time_scale
            return value + penalty, {"terminal_makespan": value, "failure_penalty": penalty}
        return penalty, {"terminal_makespan": 0.0, "failure_penalty": penalty}
    raise ConfigurationError(
        f"Unknown reward mode {mode!r}.",
        context={"reward_mode": mode},
    )
