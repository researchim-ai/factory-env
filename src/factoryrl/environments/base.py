"""Gymnasium adapter over the job-shop engine."""

from __future__ import annotations

import operator
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factoryrl.actions.dispatch import decode_gym_action
from factoryrl.actions.masks import build_action_mask
from factoryrl.core.engine import SimulationEngine
from factoryrl.core.events import Phase
from factoryrl.core.replay import export_trace
from factoryrl.environments.spaces import MaskedDiscrete
from factoryrl.exceptions import ConfigurationError, InvalidActionError
from factoryrl.hashing import sha256_json
from factoryrl.models.factory import FactoryInstance, operation_count
from factoryrl.observations.builders import (
    N_GLOBAL_FEATURES,
    N_MACHINE_FEATURES,
    N_OPERATION_FEATURES,
    OBS_SCHEMA_VERSION,
    build_observation,
)
from factoryrl.observations.normalization import assert_time_scale
from factoryrl.rendering.ansi import render_ansi
from factoryrl.rendering.gantt import render_gantt_html
from factoryrl.rewards.base import REWARD_MODES
from factoryrl.rewards.makespan import compute_reward
from factoryrl.scenarios.generator import generate_from_config, generated_shape
from factoryrl.scenarios.serialization import scenario_hash


class FactorySchedulingEnv(gym.Env[dict[str, np.ndarray], np.int64]):
    """Shared reset/step loop for job shop and flow shop.

    One action dispatches one ready operation, or waits when voluntary idle
    is enabled. The engine then runs until the next decision or the episode
    stops. ``info["action_mask"]`` matches the observation just returned.
    """

    metadata: dict[str, Any] = {"render_modes": ["ansi", "gantt_html"], "render_fps": 4}
    env_id = "FactoryRL/JobShop-v0"

    def __init__(
        self,
        instance: FactoryInstance | None = None,
        *,
        generator_config: dict[str, Any] | None = None,
        max_machines: int | None = None,
        max_operations: int | None = None,
        time_scale: float = 1.0,
        invalid_action_mode: str = "raise",
        allow_voluntary_idle: bool = False,
        reward_mode: str = "negative_elapsed_time",
        failure_penalty: float = 0.0,
        horizon: float | None = None,
        max_events_per_episode: int = 1_000_000,
        max_decisions_per_episode: int = 1_000_000,
        check_invariants: bool = True,
        record_events: bool = True,
        observation_bound: float = 1.0e6,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if invalid_action_mode != "raise":
            raise ConfigurationError(
                "v0.1 only supports invalid_action_mode='raise'. "
                "penalize and no_op are not implemented, and illegal actions are never replaced.",
                context={"invalid_action_mode": invalid_action_mode},
            )
        if reward_mode not in REWARD_MODES:
            raise ConfigurationError(f"Unknown reward mode {reward_mode!r}.")
        if isinstance(failure_penalty, bool) or float(failure_penalty) < 0.0:
            raise ConfigurationError("failure_penalty must be >= 0.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ConfigurationError(f"Unsupported render_mode {render_mode!r}.")
        if instance is not None and generator_config is not None:
            raise ConfigurationError("Pass instance or generator_config, not both.")
        self.render_mode = render_mode
        self.time_scale = assert_time_scale(time_scale)
        self.reward_mode = reward_mode
        self.failure_penalty = float(failure_penalty)
        self.allow_voluntary_idle = bool(allow_voluntary_idle)
        self._horizon_override = horizon
        self.max_events = int(max_events_per_episode)
        self.max_decisions = int(max_decisions_per_episode)
        self.check_invariants = bool(check_invariants)
        self.record_events = bool(record_events)
        self.observation_bound = float(observation_bound)
        self.generator_config = dict(generator_config) if generator_config is not None else None
        self._generator_seed = 0 if generator_config is None else int(generator_config.get("seed", 0) or 0)
        self._instance: FactoryInstance
        self._engine: SimulationEngine
        self._initial_mask: np.ndarray | None = None
        self._wall_start = time.perf_counter()
        if self.generator_config is not None:
            gen_machines, gen_ops = generated_shape(self.generator_config)
            self.max_machines = int(max_machines or gen_machines)
            self.max_operations = int(max_operations or gen_ops)
            initial = generate_from_config(self.generator_config, seed=self._generator_seed)
        else:
            initial = instance if instance is not None else self._default_instance()
            self.max_machines = int(max_machines or len(initial.machines))
            self.max_operations = int(max_operations or operation_count(initial))
        self._install_instance(initial)
        self._engine = self._make_engine()
        self._engine.reset(seed=None)
        self._build_spaces()
        self._initial_mask = self.action_masks().copy()

    def _default_instance(self) -> FactoryInstance:
        raise NotImplementedError

    def _validate_domain(self, instance: FactoryInstance) -> None:
        """Reject instances outside this environment family."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {"instance", "generator_seed"}
        if unknown:
            raise ConfigurationError(f"Unknown reset options: {sorted(unknown)}.")
        if "instance" in options and "generator_seed" in options:
            raise ConfigurationError("Pass either instance or generator_seed to reset, not both.")
        if "instance" in options:
            self.generator_config = None
            self._install_instance(options["instance"])
            self._engine = self._make_engine()
        elif self.generator_config is not None:
            if "generator_seed" in options:
                gen_seed = int(options["generator_seed"])
            elif seed is not None:
                gen_seed = int(seed)
            else:
                gen_seed = self._generator_seed
            self._generator_seed = gen_seed
            self._install_instance(generate_from_config(self.generator_config, seed=gen_seed))
            self._engine = self._make_engine()
        self._engine.reset(seed=seed)
        self._initial_mask = self.action_masks().copy()
        self._wall_start = time.perf_counter()
        observation = self._observation()
        return observation, self._info(0.0, {"elapsed_time": 0.0, "failure_penalty": 0.0})

    def step(self, action: np.int64 | int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool):
            self._engine.invalid_action_count += 1
            raise InvalidActionError("Action must be an integer.", context={"action": action})
        try:
            action_index = operator.index(action)
            internal = decode_gym_action(
                action_index,
                n_operations=self._engine.compiled.n_operations,
                max_operations=self.max_operations,
            )
        except (InvalidActionError, TypeError, ValueError) as exc:
            self._engine.invalid_action_count += 1
            if isinstance(exc, InvalidActionError):
                raise
            raise InvalidActionError("Action must be an integer.", context={"action": action}) from exc
        result = self._engine.step(internal)
        metrics = self._engine.metrics()
        terminal = bool(result.terminated or result.truncated)
        reward, components = compute_reward(
            mode=self.reward_mode,
            delta_sim_time=result.delta_sim_time,
            time_scale=self.time_scale,
            terminal=terminal,
            success=result.success if terminal else None,
            makespan=metrics["makespan"],
            failure_penalty=self.failure_penalty,
        )
        observation = self._observation()
        return (
            observation,
            float(reward),
            bool(result.terminated),
            bool(result.truncated),
            self._info(result.delta_sim_time, components),
        )

    def sample_mask(self) -> np.ndarray:
        """Mask used by ``action_space.sample``.

        Gymnasium's checker samples an action, then resets, then applies that
        same action. The sample therefore has to be legal in the initial
        decision of the episode, not in whatever state the previous step
        reached. ``step`` still rejects anything the live mask disallows.
        """
        if self._initial_mask is None:
            return self.action_masks()
        return self._initial_mask.copy()

    def action_masks(self) -> np.ndarray:
        """Boolean mask for the action that may be sent to the current state."""
        n_ops = self._engine.compiled.n_operations
        dispatchable = np.zeros(n_ops, dtype=bool)
        if self._engine.phase is Phase.DECISION:
            for index in self._engine.dispatchable_indices():
                dispatchable[index] = True
        return build_action_mask(
            dispatchable,
            n_operations=n_ops,
            max_operations=self.max_operations,
            wait_allowed=self._engine.wait_allowed(),
        )

    def render(self) -> str | None:  # type: ignore[override]
        if self.render_mode is None:
            return None
        snapshot = self._engine.snapshot()
        if self.render_mode == "ansi":
            return render_ansi(snapshot)
        if self.render_mode == "gantt_html":
            return render_gantt_html(
                self._engine.public_intervals(),
                list(self._engine.compiled.machine_ids),
            )
        raise ConfigurationError(f"Unsupported render_mode {self.render_mode!r}.")

    def close(self) -> None:
        """Nothing to release. Safe to call more than once."""
        return None

    @property
    def wall_seconds(self) -> float:
        """Process time since the last reset. Kept out of ``info`` so episodes stay reproducible."""
        return time.perf_counter() - self._wall_start

    def export_trace(self, *, agent_id: str | None = None) -> dict[str, Any]:
        """JSON-ready replay document for the current episode."""
        return export_trace(
            self._engine,
            scenario_hash=scenario_hash(self._instance),
            config_hash=self.config_hash,
            agent_id=agent_id,
        )

    @property
    def config_hash(self) -> str:
        return sha256_json(self.config_dict())

    def config_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "reward_mode": self.reward_mode,
            "invalid_action_mode": "raise",
            "allow_voluntary_idle": self.allow_voluntary_idle,
            "time_scale": self.time_scale,
            "failure_penalty": self.failure_penalty,
            "max_events_per_episode": self.max_events,
            "max_decisions_per_episode": self.max_decisions,
            "horizon": self._resolved_horizon(),
            "max_machines": self.max_machines,
            "max_operations": self.max_operations,
            "observation_schema": OBS_SCHEMA_VERSION,
            "observation_bound": self.observation_bound,
            "check_invariants": self.check_invariants,
            "record_events": self.record_events,
        }

    def _install_instance(self, candidate: FactoryInstance) -> None:
        if not isinstance(candidate, FactoryInstance):
            raise ConfigurationError("instance must be a FactoryInstance.")
        self._validate_domain(candidate)
        n_ops = operation_count(candidate)
        if len(candidate.machines) > self.max_machines or n_ops > self.max_operations:
            raise ConfigurationError(
                "Instance does not fit the environment observation capacity.",
                context={
                    "machines": len(candidate.machines),
                    "operations": n_ops,
                    "max_machines": self.max_machines,
                    "max_operations": self.max_operations,
                },
            )
        self._instance = candidate

    def _make_engine(self) -> SimulationEngine:
        return SimulationEngine(
            self._instance,
            horizon=self._resolved_horizon(),
            allow_voluntary_idle=self.allow_voluntary_idle,
            max_events=self.max_events,
            max_decisions=self.max_decisions,
            check_invariants=self.check_invariants,
            record_events=self.record_events,
        )

    def _resolved_horizon(self) -> float | None:
        if self._horizon_override is not None:
            return float(self._horizon_override)
        return self._instance.horizon

    def _build_spaces(self) -> None:
        bound = self.observation_bound
        self.observation_space = spaces.Dict(
            {
                "machines": spaces.Box(
                    low=0.0,
                    high=bound,
                    shape=(self.max_machines, N_MACHINE_FEATURES),
                    dtype=np.float32,
                ),
                "operations": spaces.Box(
                    low=-1.0,
                    high=bound,
                    shape=(self.max_operations, N_OPERATION_FEATURES),
                    dtype=np.float32,
                ),
                "machine_mask": spaces.MultiBinary(self.max_machines),
                "operation_mask": spaces.MultiBinary(self.max_operations),
                "global": spaces.Box(
                    low=-1.0,
                    high=bound,
                    shape=(N_GLOBAL_FEATURES,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = MaskedDiscrete(self.max_operations + 1, self.sample_mask)

    def _observation(self) -> dict[str, np.ndarray]:
        return build_observation(
            self._engine.snapshot(),
            max_machines=self.max_machines,
            max_operations=self.max_operations,
            time_scale=self.time_scale,
            horizon=self._resolved_horizon(),
            observation_bound=self.observation_bound,
        )

    def _info(self, delta: float, components: dict[str, float]) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks().copy(),
            "delta_sim_time": float(delta),
            "reward_components": {key: float(value) for key, value in components.items()},
            "episode_metrics": self._engine.metrics(),
            "success": self._engine.success,
            "termination_reason": self._engine.termination_reason,
            "truncation_reason": self._engine.truncation_reason,
            "scenario_hash": scenario_hash(self._instance),
            "config_hash": self.config_hash,
            "observation_schema": OBS_SCHEMA_VERSION,
            "diagnosis": self._engine.diagnosis,
        }
