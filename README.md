# FactoryRL

Детерминированный симулятор дискретного производства и среды Gymnasium для обучения и сравнения агентов: эвристик, солверов и RL.

**v0.1 не управляет реальными станками.** Нет записи в PLC/SCADA, нет OPC UA write и нет сертификата для safety-critical управления. Логическое время — условные минуты, а не реальное время контроллера.

## Установка

```bash
pip install -e ".[dev]"
```

Базовый пакет не тянет PyTorch, GPU и OR-Tools.

```bash
pip install -e ".[rl]"      # MaskablePPO
pip install -e ".[solver]"  # CP-SAT
```

## Быстрый старт

```python
import gymnasium as gym
import factoryrl  # регистрирует среды

env = gym.make("FactoryRL/JobShop-v0", invalid_action_mode="raise")
obs, info = env.reset(seed=42)
while True:
    mask = info["action_mask"]
    action = int(mask.nonzero()[0][0])  # или своя политика
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(info["episode_metrics"]["makespan"])
```

Канонический пример из спецификации (2 станка, 2 заказа, расписание с makespan 23):

```bash
python examples/quickstart.py
```

Сравнение FIFO и SPT на одних и тех же экземплярах:

```bash
python examples/compare_baselines.py
```

Короткое обучение маскированной линейной политики без PyTorch:

```bash
python examples/train_masked_policy.py --episodes 15
```

## Что есть в v0.1

- Свой дискретно-событийный цикл на `heapq`. События одного времени идут в порядке `(time, priority, sequence)`.
- Статический non-preemptive job shop и permutation flow shop.
- `FactoryRL/JobShop-v0` и `FactoryRL/FlowShop-v0`: маски действий, наблюдения `obs_v1`, награда `-delta_sim_time / time_scale`.
- Эвристики Random, FIFO, SPT, EDD (EDD только если у операции есть due date).
- Генераторы с фиксированным seed, JSON/YAML через безопасный парсер, replay, benchmark CLI.
- Инварианты, property-тесты и золотой след примера с makespan 23.

Незаконное действие в режиме `raise` бросает `InvalidActionError`. Симулятор не подменяет его случайным допустимым.

## Чего нет

3D, роботы, отказы станков, переналадки, конечные буферы, мультиагентность, обучение на живом производстве. Это следующие версии спецификации, не этот релиз.

Успешный эпизод классического job shop достижим любой допустимой политикой: станок освобождается по окончании операции, взаимной блокировки в этой модели нет. Качество политики видно по makespan и другим метрикам, а не по флагу `success`.

## Команды

```bash
factoryrl validate examples/factories/demo.yaml
factoryrl simulate --config examples/factories/demo.yaml --agent spt --seed 42
factoryrl benchmark --suite jobshop-small --agents fifo,spt --seeds 11 --split test --output results
factoryrl render --trace trace.json --output gantt.html
```

Проверки:

```bash
ruff check src tests examples
mypy src
pytest
```

## Документы

- [Семантика](docs/semantics.md)
- [Архитектура](docs/architecture.md)
- [API](docs/api.md)
- [Протокол бенчмарка](docs/benchmark_protocol.md)
- [Матрица трассировки](docs/traceability.md)
- [Решения ADR](docs/adr/001-simulation-backend.md)

Лицензия: Apache-2.0.
