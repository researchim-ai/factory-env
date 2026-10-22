# Публичный API v0.1

## Сборка экземпляра

```python
from factoryrl.scenarios.instance import instance, job, machine, operation

factory = instance(
    "demo",
    [machine("M1"), machine("M2")],
    [job("J1", [operation("J1-O1", "M1", 8), operation("J1-O2", "M2", 12)])],
)
```

Загрузка файла: `factoryrl.scenarios.serialization.load_scenario`. Неизвестная версия схемы отклоняется. `migrate_v1_to_v2` существует как заготовка и **не** делает документ версии 2 загружаемым.

Генераторы: `generate_job_shop`, `generate_flow_shop`. Один и тот же `seed` даёт один и тот же экземпляр. Алгоритм ГПСЧ: `numpy.random.PCG64` через `numpy.random.default_rng`.

## Среда

```python
JobShopEnv(
    instance=None,
    generator_config=None,
    max_machines=None,
    max_operations=None,
    time_scale=1.0,
    invalid_action_mode="raise",
    allow_voluntary_idle=False,
    reward_mode="negative_elapsed_time",  # или terminal_makespan
    failure_penalty=0.0,
    horizon=None,  # None означает горизонт экземпляра
    max_events_per_episode=1_000_000,
    max_decisions_per_episode=1_000_000,
    check_invariants=True,
    record_events=True,
    observation_bound=1e6,
    render_mode=None,  # ansi | gantt_html
)
```

`FlowShopEnv` принимает только общий маршрут станков. Без `instance` и генератора JobShop использует демонстрационный 2×2, FlowShop — свой маленький flow shop. Это нужно, чтобы `gym.make` и `check_env` имели конкретный завод. Для эксперимента передавайте свой экземпляр.

`reset(options={"instance": ...})` подменяет экземпляр, если он влезает в `max_machines` и `max_operations`. Для генератора размер пространства фиксируется при создании среды.

`info` после `reset` и `step`:

- `action_mask`
- `delta_sim_time`
- `reward_components`
- `episode_metrics`
- `success`, `termination_reason`, `truncation_reason`
- `scenario_hash`, `config_hash`, `observation_schema`
- `diagnosis` (подробности deadlock или усечения)

`env.wall_seconds` — процессорное время с последнего `reset`.

## Агенты

```python
class Agent(Protocol):
    def reset(self, *, seed: int) -> None: ...
    def act(self, observation: dict, action_mask: np.ndarray) -> int: ...
```

Готовые классы: `RandomValidAgent`, `FIFOAgent`, `SPTAgent`, `EDDAgent`. Ничья разрешается меньшим индексом операции. EDD поднимает `BaselineError`, если ни у одной допустимой операции нет due date.

## Исключения

`InstanceValidationError`, `SchemaVersionError`, `ConfigurationError`, `InvalidActionError`, `InvariantViolationError`, `SerializationError`, `ResourceLimitError`, `ReplayMismatchError`, `BaselineError`. Нарушение инварианта — дефект симулятора, не ошибка агента.
