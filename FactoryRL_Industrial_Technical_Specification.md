# FactoryRL — промышленная open-source библиотека симуляции производств и обучения AI-агентов

> **Тип документа:** техническое задание / Software Requirements Specification / Architecture RFC  
> **Статус:** v1.0, базовая спецификация для реализации  
> **Дата:** 21.10.2026  
> **Язык разработки:** Python 3.11+  
> **Предлагаемая лицензия:** Apache-2.0, при условии аудита зависимостей и наборов данных  
> **Назначение:** проектирование, разработка, верификация и публикация промышленно-качественной open-source библиотеки; **не сертификация для непосредственного управления опасным оборудованием**.

---

## 0. Как пользоваться этим документом

- **MUST / ОБЯЗАТЕЛЬНО** — критерий приемки выпуска соответствующей версии.
- **SHOULD / ЖЕЛАТЕЛЬНО** — рекомендуется, отклонение обосновывается в ADR.
- **MAY / ОПЦИОНАЛЬНО** — расширение.
- **P0** — v0.1.0; **P1** — v0.2–v0.3; **P2** — дальнейшее развитие.
- Каждый пункт вида `FR-*`, `NFR-*`, `SEC-*`, `TEST-*` должен иметь ссылку на реализацию и тест в матрице трассировки.
- Это **спецификация предполагаемой библиотеки**, а не заявление о наличии готового пакета или подтвержденных результатов производительности.

### 0.1. Граница слова «промышленного уровня»

Оно означает здесь промышленную инженерную дисциплину разработки ПО: надежные интерфейсы, детерминированность, воспроизводимость, защищенные границы исполнения, тесты инвариантов, наблюдаемость, версии схем и управляемые релизы. Оно **не** означает, что FactoryRL автоматически пригодна для safety-critical / real-time управления реальными станками. Применение для управления реальным оборудованием требует независимой инженерной оценки безопасности, требований предприятия, сертификации и отдельного контролирующего слоя.

## 1. Цели и продуктовая стратегия

### 1.1. Назначение

FactoryRL предоставляет **модульный дискретно-событийный симулятор (DES)** для задач дискретного производства и стандартные среды, пригодные для обучения и тестирования AI-агентов, эвристик и математических оптимизаторов.

### 1.2. Пользователи

1. Исследователь RL — сравнивает политики для scheduling, dispatching и maintenance.
2. Инженер промышленного ПО — строит модель линии и сценарии отказов.
3. Команда MLOps — выполняет воспроизводимые эксперименты и regression benchmarks.
4. Преподаватель — использует небольшие эталонные среды.
5. Разработчик продукта — интегрирует симулятор как Python-компонент, без обязательного GPU и UI.

### 1.3. Уникальное предложение

- Надежное событийное ядро, отделенное от Gymnasium и библиотек ML.
- Формально описанные производственные ограничения, допустимые действия, события и метрики.
- Набор сценариев с фиксированными семенами и строгой evaluation-процедурой.
- Подключаемые политики: эвристики, OR-Tools, сторонние RL.
- Воспроизводимый replay и диагностика проблем модели.
- В перспективе — расширяемые модели динамического производства и отказов.

### 1.4. Не входит в v0.1

- 3D-физика, цифровые модели роботов, физика жидкостей и тепломассообмена.
- Действия с реальным PLC/SCADA, OPC UA write, управление опасным оборудованием.
- Онлайн-обучение на подключенном действующем производстве.
- Собственный глубокий RL-фреймворк, distributed training, графические нейросети как обязательный модуль.
- Универсальный MES/ERP, полноценная складская система, календарь смен и полная multi-agent поддержка.
- Обещание оптимальности обученных политик или универсального transfer sim-to-real.

## 2. Версии и состав поставки

| Релиз | Обязательная функциональность | Условие выхода |
|---|---|---|
| **v0.1.0 P0** | Детерминированный JobShop и FlowShop; DES; Gymnasium; action masks; базовые эвристики; внешние benchmark-инстансы через загрузчики; CLI; документация; тесты и CI | Все P0 acceptance-тесты зеленые |
| **v0.2.0 P1** | FlexibleJobShop, отказ/ремонт, альтернативные машины, расширение генераторов, CP-SAT rolling horizon | Новый набор инвариантов + стабильный P0 API |
| **v0.3.0 P1** | Буферы конечной емкости, блокировки, переналадки, энерготарифы, maintenance среды, PettingZoo-адаптер | Проверены deadlock- и multi-agent-протоколы |
| **v0.4+ P2** | Графовые наблюдения, GNN, domain randomization, real-data calibration, read-only industrial adapters, ускоренный backend | Документированные схемы переноса и ограничения |
| **v1.0** | Стабильные public API и схемы, нагрузочное тестирование, полноценная совместимость, документация миграций, maintenance policy | Независимые репликации benchmark и аудит выпускных критериев |

## 3. Терминология и математическая основа

- **Job / работа**: заказ с конечным набором операций и отношениями предшествования.
- **Operation / операция**: атомарная производственная операция с обработкой на одном ресурсе.
- **Machine / станок**: ресурс с емкостью, состоянием и набором допустимых операций.
- **Dispatch decision**: назначение одной *готовой* операции на допустимый свободный станок.
- **Makespan**: `C_max = max(completion_time(job))` для конечного набора всех заказов.
- **Flow time**: `completion_time(job) - release_time(job)`.
- **Tardiness**: `max(0, completion_time(job) - due_date(job))`.
- **DES**: переход между временами значимых событий, не посекундная симуляция.
- **Simulation time**: логическое время в единых условных единицах, по умолчанию минуты; **wall time**: процессорное время.
- **Termination**: естественное завершение задачи; **truncation**: остановка по горизонту/лимиту, не эквивалентная успешному завершению.
- **Instance**: полностью определенное задание производства; **scenario**: instance + стохастические допущения/семена/режим симуляции.

В v0.1 основная задача — конечный, статический non-preemptive JobShop с фиксированными временами операций. Для каждой операции `o`: станок `m(o)`, длительность `p(o)>0`, начало `s(o)`, завершение `c(o)=s(o)+p(o)`. Ограничения: порядок операций заказа; отсутствие пересечения операций на станке емкостью 1; запуск только готовых операций. Цель начального benchmark: минимизировать `C_max`.

## 4. Архитектура и строгие границы компонентов

```text
                       API / CLI / Examples
                               |
                 +-------------+-------------+
                 |                           |
            Gymnasium Env             Benchmark Runner
                 |                           |
         Observation / Action            Agent Protocol
          Reward / Metrics           Heuristics / OR-Tools
                 |                           |
                 +-------------+-------------+
                               |
                        Simulation Core
             Clock / Events / State / RNG / Invariants
                               |
                 Factory Model / Scenario Instance
                               |
                    Storage / Replay / Export
```

### 4.1. Правила зависимостей

- `core` MUST не импортировать Gymnasium, PyTorch, Stable-Baselines3, ORM, UI.
- `models` MUST быть независимым от training framework.
- `environments` адаптирует core к Gymnasium.
- `baselines` реализует политику через протокол публичного состояния и списка допустимых действий.
- `benchmark` не должен изменять правила среды и не должен использовать тестовые решения при выборе действий.
- `rendering` MUST быть отключаемым без изменения поведения симуляции.
- `adapters` к реальным источникам данных по умолчанию READ-ONLY и НЕ входят в v0.1.

### 4.2. Предлагаемая структура репозитория

```text
factoryrl/
  pyproject.toml
  LICENSE
  README.md
  CONTRIBUTING.md
  SECURITY.md
  CODE_OF_CONDUCT.md
  CHANGELOG.md
  docs/
    architecture.md
    semantics.md
    api.md
    benchmark_protocol.md
    tutorials/
    adr/
  src/factoryrl/
    __init__.py
    registry.py
    exceptions.py
    core/
      engine.py
      clock.py
      events.py
      event_queue.py
      state.py
      rng.py
      transition.py
      replay.py
      invariants.py
    models/
      factory.py
      machine.py
      job.py
      operation.py
      route.py
      buffer.py
    scenarios/
      instance.py
      schema.py
      generator.py
      serialization.py
    environments/
      base.py
      job_shop.py
      flow_shop.py
    observations/
      builders.py
      normalization.py
    actions/
      dispatch.py
      masks.py
    rewards/
      base.py
      makespan.py
    baselines/
      protocol.py
      random.py
      fifo.py
      spt.py
      edd.py
      optional_ortools.py
    benchmarks/
      runner.py
      splits.py
      metrics.py
      reporter.py
    rendering/
      ansi.py
      gantt.py
    cli.py
  tests/
    unit/
    integration/
    property/
    regression/
    benchmarks/
    security/
  examples/
    quickstart.py
    compare_baselines.py
    train_masked_ppo.py
  .github/workflows/
    ci.yml
    release.yml
```

### 4.3. Backend abstraction

P0 может быть реализован поверх SimPy или собственной минимальной очереди `heapq`. Решение фиксируется в ADR-001 до начала разработки. Внешний API MUST скрывать выбор backend. Замена движка не должна менять семантику обработанных событий на канонических тестах.

Рекомендация: **свой минимальный event loop на heapq** для P0, поскольку он прозрачен, детерминирован и позволяет явно управлять decision points; не реализовывать обобщенный аналог SimPy. Если команда выбирает SimPy, обязательны те же контракты и тесты.

## 5. Модель данных: обязательные поля и инварианты

### 5.1. Идентификаторы, единицы, типы

- `job_id`, `operation_id`, `machine_id`: устойчивые строки; внутренние целочисленные индексы производны и не заменяют IDs.
- Время: конечное `float64`, единица — условная минута; `NaN`, `inf`, отрицательные продолжительности запрещены.
- Времена операций P0: строго `>0` для предотвращения неоднозначности событий нулевой длительности.
- Seed: целое неотрицательное, модель RNG на основе `numpy.random.Generator`, независимые потоки для разных подсистем.
- Все сериализуемые сценарии имеют `schema_version`.

### 5.2. FactoryInstance (неизменяемый вход)

| Поле | Тип | Ограничение |
|---|---|---|
| `schema_version` | int | `>=1` |
| `name` | str | непустая строка |
| `machines` | tuple[MachineSpec] | минимум 1; IDs уникальны |
| `jobs` | tuple[JobSpec] | минимум 1; IDs уникальны |
| `time_unit` | literal | P0: `minute` |
| `horizon` | float или None | если задан, `>0` |
| `metadata` | map | только JSON-совместимые значения |

### 5.3. MachineSpec

P0: `id`, `capacity=1`, `supported_operation_types` (опционально). States: `IDLE`, `BUSY` (P0); P1: `DOWN`, `REPAIR`, `MAINTENANCE`, `SETUP`. Прямые переходы между состояниями вне таблицы переходов запрещены.

### 5.4. JobSpec / OperationSpec

`JobSpec`: `id`, `release_time>=0` (P0 только 0), `due_date` (опционально), `priority>0`, `operations` непустой список. `OperationSpec`: `id`, `machine_id`, `duration>0`, `predecessor_ids` — P0 линейная цепочка, P1 DAG; каждая операция принадлежит одному job. Все machine refs существуют; граф предшествования ацикличен.

### 5.5. Runtime state

Состояние хранит `now`, статусы машин, статус каждой операции (`NOT_READY`, `READY`, `RUNNING`, `COMPLETED`), расписание событий, назначение `machine -> operation`, интервалы операций, метрики. Внешний код получает read-only snapshot, а не мутируемые внутренние структуры.

### 5.6. Инварианты (MUST)

1. На станке емкостью 1 одновременно максимум одна операция.
2. Операция одновременно максимум на одном станке.
3. Операция стартует только после всех предшественников.
4. У одной операции не более одного успешного события завершения.
5. Завершение не раньше запуска, `end - start == duration` с документированным tolerance.
6. `now` не убывает; события с прошлым временем запрещены.
7. Любой активный operation ссылается на существующую машину и job.
8. Завершенное задание имеет все операции в `COMPLETED`.
9. Не существует изменения runtime state путем обхода validated transition API.
10. После reset состояние соответствует исходному instance; эпизоды независимы.

## 6. Discrete Event Simulation engine (P0)

### 6.1. События

`JOB_RELEASE`, `OPERATION_START`, `OPERATION_FINISH`, `DECISION_REQUIRED`, `EPISODE_END`; P1: `FAILURE`, `REPAIR_START`, `REPAIR_FINISH`, `SETUP_START`, `SETUP_FINISH`.

Объект события: `time`, `priority`, `sequence`, `event_id`, `type`, `payload`. Упорядочивание: **`(time, priority, sequence)`**. Приоритеты определены в отдельной таблице и закреплены тестами. Не полагаться на случайный порядок словарей/потоков.

### 6.2. Контракт шага

- Агент вызывается **только** на decision point.
- `step(action)` валидирует action, применяет ее к текущему состоянию и выполняет DES до следующего decision point либо конца эпизода.
- Все события одного `timestamp`, которые должны предшествовать решению, обрабатываются полностью до запроса действия.
- Если несколько машин свободны одновременно, P0 допускает последовательные нулевые по времени решения; число таких решений ограничено существующими READY-операциями. Не разрешать нулевой `WAIT` бесконечно.
- Если ожидаемых будущих событий нет и остались незавершенные заказы — статус `deadlock`, возвращать диагностику и завершать как явно ошибочную/неуспешную ситуацию (не маскировать как успешный `terminated`).
- Должны быть лимиты `max_events_per_episode`, `max_decisions_per_episode`, `horizon`; с классификацией причин остановки.

### 6.3. WAIT

`WAIT` — допустимо только тогда, когда есть будущее событие, к которому можно продвинуть время. При наличии доступных операций WAIT может быть разрешен отдельным режимом `allow_voluntary_idle=True`, но тогда `step(WAIT)` MUST продвинуть время до **строго более позднего** события; если такого события нет — invalid. Default P0: `allow_voluntary_idle=False`, когда есть доступные dispatch действия.

### 6.4. Обработка некорректных действий

`invalid_action_mode`:
- `raise` — стандарт P0 (исключение `InvalidActionError` с контекстом);
- `penalize` — P1, только с определенным продвижением состояния;
- `no_op` — P1, запрещен без механизма защиты от бесконечного повторения.

Никогда не заменять silently недопустимое действие случайным допустимым.

### 6.5. Воспроизводимость и replay

- Одна и та же версия пакета + сериализованный scenario + seed + последовательность actions MUST порождать одинаковый журнал событий, состояния и метрики на поддерживаемой целевой платформе.
- Обещание бит-в-бит одинаковых floating-point результатов между разными CPU/OS/Python **не дается**; тесты допускают документированный tolerance, event ordering обязан совпадать.
- Логировать `package_version`, `schema_version`, `rng_algorithm`, `seed`, `scenario_hash`, `agent_id`, `config_hash`.
- Replay принимает instance, seed, список действий и проверяет хеш/ключевые transitions.

## 7. Публичные Python API и совместимость

### 7.1. Основной пользовательский контракт

```python
import gymnasium as gym
import factoryrl  # выполняет регистрацию сред

env = gym.make(
    "FactoryRL/JobShop-v0",
    instance=my_instance,  # экземпляр FactoryInstance
    invalid_action_mode="raise",
)
obs, info = env.reset(seed=42)
while True:
    mask = info["action_mask"]
    action = my_policy(obs, mask)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(info["episode_metrics"])
```

Этот код — **целевой public API**, который должен появиться после реализации, а не уже доступный пакет.

### 7.2. Gymnasium semantics (FR-ENV-01)

`reset(*, seed=None, options=None) -> (observation, info)`; `step(action) -> (observation, reward, terminated, truncated, info)`. Реализовать `observation_space`, `action_space`, `render()`, `close()`; проходить `check_env`.

- **terminated=True** — успешное завершение всех работ или явно смоделированное terminal-состояние с `success=False`; `info["termination_reason"]` обязательно.
- **truncated=True** — лимит симуляционного времени, событий или решений; `info["truncation_reason"]` обязательно.
- На **обычном эпизоде P0** при выполнении всех работ `terminated=True`, `truncated=False`, `success=True`.
- При физически невозможном instance конфигурация должна отклоняться **до эпизода**, если невозможность статически обнаружима.
- Deadlock динамического состояния P1: terminal `success=False`, `termination_reason="deadlock"`; нельзя записывать его в метрики успешного решения.
- `info["delta_sim_time"]` показывает модельное время между decision points.
- `info["action_mask"]` соответствует **следующему** observation.

### 7.3. Action space P0

Фиксированный `spaces.Discrete(max_operations + 1)`; допустимые операции получают стабильный индекс в пределах instance, последний индекс — `WAIT`. Маска `np.ndarray[bool]`, форма `(max_operations+1,)`, выставляет True только разрешенным действиям. Для фиксированной топологии `max_operations` равен числу операций; для генераторов используются `max_operations` + padding. Перед выполнением проверяются индекс и mask.

### 7.4. Observation space P0

Структурированное `spaces.Dict`:

```python
{
    "machines": Box(shape=(max_machines, Fm), dtype=float32),
    "operations": Box(shape=(max_operations, Fo), dtype=float32),
    "machine_mask": MultiBinary(max_machines),
    "operation_mask": MultiBinary(max_operations),
    "global": Box(shape=(Fg,), dtype=float32),
}
```

`Fm`, `Fo`, `Fg`, порядок признаков, единицы и пределы MUST быть задокументированы и версионированы. Вариант `flatten=True` — отдельная оболочка. `action_mask` размещать в `info` и предоставлять через `env.unwrapped.action_masks()` для совместимых masked-policy integrations; при необходимости добавить дополнительный wrapper, помещающий маску в observation.

**Анти-leakage**: признаки P0 включают только состояние и статические поля, доступные агенту при принятии решения. Для стохастических P1-сценариев запрещено включать будущие отказы, поступления и случайные значения, пока они не наблюдаемы.

### 7.5. Модельный Agent Protocol

```python
class Agent(Protocol):
    def reset(self, *, seed: int) -> None: ...
    def act(self, observation: dict, action_mask: np.ndarray) -> int: ...
```

Training не является частью этого протокола: библиотека не должна принуждать разработчиков наследовать конкретный класс RL-фреймворка.

## 8. Reward, objective, metrics

### 8.1. Разделение ответственности

- `objective`: оптимизируемая производственная величина (`makespan`, `total_tardiness` и др.).
- `reward`: обучение агента; разложение доступно в `info["reward_components"]`.
- `constraints`: жесткие правила, которые нельзя заменить штрафом.
- `metrics`: независимые от reward итоговые количественные характеристики.

### 8.2. P0 reward для JobShop

По умолчанию reward при каждом принятии решения равен `-delta_sim_time / time_scale` (`time_scale>0`), включая последний переход. Недисконтированная сумма `gamma=1` для завершившихся эпизодов равна `-makespan/time_scale`, если эпизод начался с `t=0` и все интервалы учтены. Для truncation/deadlock назначить отдельный явно документированный penalty либо исключать из успешного leaderboard; это нельзя смешивать с полными расписаниями без отметки статуса.

Встроить экспериментальное `terminal_makespan` reward как альтернативу, но benchmark фиксирует конфигурацию reward для воспроизводимости.

### 8.3. Метрики P0

- `makespan` (единица модельного времени);
- `mean_flow_time`;
- `total_tardiness`, `weighted_tardiness`, если есть due dates;
- `machine_busy_time` и `utilization = busy_time / makespan` для каждого станка;
- `jobs_completed`, `operations_completed`;
- `valid_action_count`, `invalid_action_count`;
- `decision_count`, `event_count`, `simulation_wall_seconds`;
- `success`, `termination_reason`, `truncation_reason`.

Метрики нельзя генерировать из потенциально неточных внешних оценок модели; они вычисляются из журналируемых переходов engine. Установка `due_date=None` должна иметь явно задокументированную семантику в tardiness-метриках (например, исключение из сумм).

## 9. Генерация сценариев и импорт benchmark

### 9.1. Генераторы P0

Параметры: `n_machines`, `n_jobs`, `processing_time_distribution` (uniform integer, fixed), стратегия формирования последовательностей станков, `seed`. Генератор должен обеспечивать корректные уникальные IDs, положительные длительности, допустимые маршруты и одинаковый результат с одинаковым seed.

### 9.2. Схемы конфигурации

- Python dataclass/typed structures — внутренний контракт.
- JSON и YAML — сериализация, валидировать Pydantic или эквивалентом на границе ввода.
- `schema_version` обязателен; неизвестные обязательные поля отклонять, расширяемые metadata явно ограничить.
- Импортируемые файлы нельзя интерпретировать как Python-код, выражения или произвольные YAML objects; использовать `yaml.safe_load` и ограничения размеров входа.
- Миграции схемы — `migrate_vN_to_vNplus1` с тестами.

### 9.3. Эталонные instances

Поддержать загрузчики стандартных задач JobShop; не копировать внешние наборы в репозиторий без проверки лицензии. Для каждого файла: происхождение, версия, checksum, known optimum/best-known из документированного источника с пометкой, если значение не доказано оптимальным.

### 9.4. Train / validation / test

Обязательны фиксированные disjoint instance IDs, публикация seeds и генераторов, separate out-of-distribution test. Генерация нового seed из того же распределения — не полноценный OOD-тест.

## 10. Baselines, обучение и оценка

### 10.1. Обязательные baselines P0

`RandomValid`, `FIFO`, `SPT`, `EDD` (только при due dates), `MWKR`/`LWKR` желательно к v0.2, `CP-SAT` optional extra `factoryrl[ortools]`. Эвристики выбирают действие **только из доступной mask**; tie-break детерминированно по stable operation ID либо заданному RNG.

### 10.2. OR-Tools

Отдельный exact/heuristic solver с `time_limit_seconds` и сохранением `solver_status`, `objective`, `best_bound`, `elapsed_time`. Для динамических сценариев в будущем — rolling horizon без знания случайных будущих событий.

### 10.3. RL-интеграции

P0: рабочий пример `MaskablePPO` либо эквивалентного masked policy решения, совместимого с выбранной версией Gymnasium. Держать обучение в `examples/` или optional extras, не в обязательном runtime. Пользовательские модели должны работать через стандартные Gymnasium интерфейсы.

### 10.4. Протокол evaluation

1. Freeze версии среды, конфигурации и хеши test instances.
2. Freeze trained policy перед тестом.
3. Установить seeds для env и baseline policy.
4. Одинаковая информация на момент решения для каждого метода.
5. Отдельно считать training compute и inference wall time.
6. Для stochastic instances — paired seeds, отчет по среднему, медиане, разбросу и 95% CI.
7. Учитывать non-success/deadlock/truncation отдельно; запрещено исключать неудачные эпизоды незаметно.
8. Экспортировать машиночитаемые результаты с метаданными и человекочитаемый отчет.
9. Не писать «RL превосходит X», пока это не подтверждено фиксированным независимым тестовым набором.

Формула сравнения для минимизируемой величины при `baseline>0`: `relative_improvement = (baseline - agent) / baseline`. Отрицательные значения означают ухудшение.

## 11. Нефункциональные требования (NFR)

| ID | Требование | Проверка / критерий |
|---|---|---|
| NFR-01 | Воспроизводимость | Повтор одного сценария + actions сохраняет order событий и метрики с заданным tolerance |
| NFR-02 | Корректность | 100% обязательных инвариантов на unit/property/regression tests |
| NFR-03 | Совместимость | Python 3.11, 3.12; Linux/macOS/Windows в CI в пределах поддерживаемой матрицы |
| NFR-04 | Производительность | Публиковать benchmark `events/sec`, `decisions/sec`, `reset latency` на фиксированной машине; *без неподтвержденного SLA* |
| NFR-05 | Масштаб тестирования | Smoke: 5x20; load: 20x200; stress: 50x1000, время и память документируются |
| NFR-06 | Память | Нет неограниченного накопления event history при `record_events=False`; memory regression tests |
| NFR-07 | API стабильность | SemVer; изменения пространств observation/action требуют нового Env ID или major version |
| NFR-08 | Логи | Структурированные события со стабильными IDs, без записи секретов |
| NFR-09 | DX | Команды установки, quickstart и одна демонстрационная тренировка из чистого окружения |
| NFR-10 | Документация | Контракты, семантика, ограничения, migration guide, FAQ и примеры |
| NFR-11 | Устойчивость | Ошибочный instance дает ясную ошибку до обучения, а не зависание |
| NFR-12 | Независимость | Установка базового пакета без PyTorch, GPU, браузера и OR-Tools |

### 11.1. Performance gates

До первого релиза измерить baseline на выделенном CI runner и зафиксировать численные пороги только после профилирования. В PR считать регрессией **>20% ухудшения медианного времени** на одном из стабильных microbenchmarks при сопоставимом окружении; требуется подтверждение повторным замером. Такие пороги — инженерные policy, не публичный SLA.

### 11.2. Наблюдаемость и диагностика

Текстовый event log и summary должны объяснять, почему операция была допущена/отклонена, кто занимал ресурс, почему завершился эпизод. Режим debug записывает переходы; по умолчанию можно отключить полный event history для памяти и скорости. Никакой зависимости результатов от вызова `render()`.

## 12. Безопасность и ограничение области применения

### SEC-01. Не выполнять непроверенный код из пользовательских конфигураций

JSON/YAML только безопасным парсером; **никаких** `eval`, `exec`, `pickle.load` из сторонних источников, небезопасных YAML-тегов, shell-команд из сценария.

### SEC-02. Ограничение ресурсов

При чтении пользовательских сценариев задать допустимые max machines/jobs/operations, размер файла, глубину вложенности, горизонт, event/decision budgets. Возможность повышать лимиты — только явной конфигурацией.

### SEC-03. Supply chain

Dependency pinning/lock для CI и examples, зависимость от PyPI только из доверенных источников, сканирование известных уязвимостей, SBOM для release, подписи/attestations релизных артефактов при возможности, least-privilege CI tokens, review обязательна перед публикацией.

### SEC-04. Реальное оборудование

FactoryRL **не предоставляет команды записи в PLC/SCADA**. Будущие адаптеры P2 работают в режиме read-only по умолчанию, требуют явного конфигурирования, сетевой изоляции и политик доступа. Управление реальным процессом — отдельный продукт и safety case, не часть релиза v1.0 без отдельной спецификации.

### SEC-05. Reporting

`SECURITY.md` с каналом private disclosure; документировать поддерживаемые версии; не публиковать чувствительные промышленные данные в example datasets.

## 13. Система тестирования

### 13.1. Unit tests

- Построение корректного/некорректного FactoryInstance.
- Машинное состояние, расписание операций, predecessors.
- Heap ordering `(time, priority, sequence)`.
- Duplicate finish events, invalid actions, WAIT semantics.
- Reset/reseed и отсутствие состояния от предыдущего эпизода.
- Корректность makespan/utilization/tardiness на ручном примере.

### 13.2. Property-based tests

Hypothesis генерирует сотни/тысячи корректных небольших JobShop instances и sequences допустимых actions; после каждого шага проверяются все core invariants, termination/timeout, отсутствие роста `now` назад и повторов finish.

### 13.3. Differential tests

Для малых детерминированных instances сравнивать минимальный найденный makespan с brute force или CP-SAT optimum там, где он доказан; сравнивать независимый расчет метрик по журналу с внутренними метриками engine.

### 13.4. Integration tests

- `gymnasium.check_env`;
- canonical Gymnasium reset/step цикл;
- masked PPO smoke train/test;
- loading и roundtrip YAML/JSON;
- benchmark runner + CSV/JSON results;
- сборка PyPI sdist/wheel и импорт из чистого окружения.

### 13.5. Regression / golden traces

Golden trace — сериализованный список events для фиксированного маленького instance. Обновляется только после review изменений семантики и release notes.

### 13.6. Негативные тесты

Пустые задачи, невозможные маршруты, отрицательные durations, неправильный action index, отсутствие разрешенных действий, horizon=0, поврежденные YAML, повторяющиеся IDs, слишком большие instances, malicious YAML tags.

### 13.7. Coverage policy

Цель **>=90% line coverage для `core`, `models`, `actions`** и 100% прохождение заранее определенных инвариантных и acceptance-тестов. Coverage не подменяет семантическую проверку.

## 14. CLI и результаты

Предлагаемые команды (целевая реализация):

```bash
factoryrl validate examples/factories/demo.yaml
factoryrl simulate --config examples/factories/demo.yaml --agent spt --seed 42
factoryrl benchmark --suite jobshop-small --agents random,fifo,spt --seeds 11,22,33 --output results/
factoryrl render --trace results/trace.json --output results/gantt.html
```

Папка результатов:

```text
results/run_YYYYMMDD_HHMMSS/
  config.json
  environment.json
  versions.json
  summary.json
  episodes.csv
  traces/          # если включено
  report.html      # optional
```

Каждая строка benchmark фиксирует: env ID, instance ID/hash, seed, agent ID/version, objective, metrics, episode status, simulation steps, wall time, config hash.

## 15. Схема конфигурации (пример обязательного P0 сценария)

```yaml
schema_version: 1
factory:
  name: demo_jobshop
  time_unit: minute
  horizon: 1000
machines:
  - id: M1
    capacity: 1
  - id: M2
    capacity: 1
jobs:
  - id: J1
    release_time: 0
    due_date: 50
    priority: 1
    operations:
      - id: J1-O1
        machine_id: M1
        duration: 8
      - id: J1-O2
        machine_id: M2
        duration: 12
  - id: J2
    release_time: 0
    due_date: 65
    priority: 2
    operations:
      - id: J2-O1
        machine_id: M2
        duration: 7
      - id: J2-O2
        machine_id: M1
        duration: 15
environment:
  id: FactoryRL/JobShop-v0
  reward: negative_elapsed_time
  invalid_action_mode: raise
  allow_voluntary_idle: false
```

Задача валидатора — до запуска проверить ссылочную целостность и совместимость полей с текущей версией схемы.

## 16. Критерии приемки v0.1.0

| ID | Критерий | Обязательный артефакт |
|---|---|---|
| AC-01 | Пакет устанавливается как `pip install factoryrl` из wheel, когда выпущен | Clean-install CI job |
| AC-02 | Не менее двух зарегистрированных сред: JobShop-v0 и FlowShop-v0 | Registry tests |
| AC-03 | 3 станка / 5 jobs: симуляция завершается по допустимой политике | Integration test + golden trace |
| AC-04 | Gymnasium contracts соблюдены | `check_env` |
| AC-05 | Все invalid actions отклоняются предсказуемо | Negative tests |
| AC-06 | 100% core invariants пройдены | Property tests |
| AC-07 | Один scenario, seed и actions воспроизводятся | Replay test |
| AC-08 | Random/FIFO/SPT сравниваются на одинаковых instances | Benchmark report |
| AC-09 | Makespan совпадает с независимым ручным/solver расчетом на канонических примерах | Differential tests |
| AC-10 | Masked RL training tutorial исполняется | Smoke training CI или nightly job |
| AC-11 | Сериализация и импорт сценария воспроизводят модель | Round-trip tests |
| AC-12 | Раздел docs содержит модели, ограничения, API, руководство | Docs build + review |
| AC-13 | CI: lint, typecheck, tests, build, security scans | GitHub Actions badges/logs |
| AC-14 | Ни одна основная зависимость не требует GPU или ML runtime | Minimal-install test |
| AC-15 | LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY опубликованы | Release checklist |

**Release gate:** нельзя выпускать как stable v0.1 при незакрытом P0 defect, нарушении инвариантов, непрохождении `check_env`, нестабильном API среды или отсутствии воспроизводимого эталонного теста.

## 17. План реализации: задачи, зависимости, трудоемкость

Оценка для одного опытного Python-инженера, при возможности периодического review от специалиста по scheduling/RL. Не является обещанием срока; RL smoke training и CI pipelines могут потребовать дополнительного времени.

| Этап | Недели | Результат | Зависит от |
|---|---:|---|---|
| A. RFC и семантика | 1–2 | ADR-001, данные, event ordering, критерии завершения | — |
| B. Сценарии и модели | 3–4 | Immutable schemas, validation, fixtures | A |
| C. DES core | 5–6 | Queue, transitions, event log, invariants | B |
| D. Gymnasium adapter | 7–8 | reset/step, masks, observations, rewards | C |
| E. Эвристики и metrics | 9 | Random/FIFO/SPT, independent metrics | D |
| F. Генераторы и benchmark | 10–11 | Instance generation, seeds, result files | B/E |
| G. Replay и тестирование | 12–13 | Property tests, differential tests, replay | C/F |
| H. RL example, CLI, docs | 14–15 | MaskablePPO demo, commands, tutorial | D/F |
| I. Hardening + release | 16 | CI matrix, security, packaging, release | Все |

### 17.1. Первая неделя — подробные задачи

- [ ] Создать репозиторий `factoryrl` и `pyproject.toml` (`src/` layout).
- [ ] Зафиксировать границы MVP и NOT IN SCOPE.
- [ ] Создать ADR-001 «Simulation backend».
- [ ] Создать ADR-002 «Event ordering and decision semantics».
- [ ] Описать Job, Operation, Machine, FactoryInstance через dataclass.
- [ ] Задать 2–3 канонических ручных instance для gold tests.
- [ ] Создать matrix требований и папку тестов.
- [ ] Включить Ruff, mypy/pyright, pytest, pre-commit, GitHub Actions.
- [ ] Определить обязательные exception types.
- [ ] Проверить лицензионную совместимость зависимостей.

### 17.2. Вторая неделя — подробные задачи

- [ ] Реализовать JSON/YAML schema validation.
- [ ] Написать запреты на malformed/malicious instances.
- [ ] Формально расписать таблицу event priorities.
- [ ] Описать условия READY, RUNNING и COMPLETED.
- [ ] Описать semantics WAIT и deadlock.
- [ ] Утвердить observation feature list и action indexing.
- [ ] Зафиксировать reward/objective/reporting semantics.
- [ ] Провести техническое review схемы с независимым разработчиком.

### 17.3. Пакеты задач для GitHub Issues

- `EPIC-CORE`: `MachineState`, `OperationState`, `EventQueue`, `TransitionEngine`, `Replay`.
- `EPIC-ENV`: Gymnasium registration, masks, observations, rewards, termination.
- `EPIC-SCENARIO`: YAML/JSON schemas, instance generators, dataset loaders.
- `EPIC-BASELINES`: Random, FIFO, SPT, EDD, optional CP-SAT.
- `EPIC-BENCH`: protocols, paired seeds, metrics, reports.
- `EPIC-QUALITY`: property testing, golden traces, CI, security checks.
- `EPIC-DOCS`: quickstart, API, extensibility examples, semantics, release notes.

У каждого issue: проблема, входы/выходы, acceptance tests, затронутые requirements IDs, зависимости и критерий готовности.

## 18. CI/CD и инженерные стандарты

### 18.1. Pull Request pipeline

1. `ruff check` и форматирование.
2. Typechecking (`mypy` или `pyright`) критических модулей.
3. Unit/integration/property tests.
4. `gymnasium.check_env` и golden replay.
5. Build `sdist/wheel`, проверка package metadata.
6. Dependency/security scan и секретов.
7. Docs build.
8. Performance smoke benchmarks (короткие; тяжелые nightly).

### 18.2. Release pipeline

- Версионирование по SemVer с учетом того, что `Env ID -vN` версионирует семантику среды отдельно.
- Release tag + changelog + compatibility notes.
- Публикация TestPyPI, installation smoke test, затем PyPI.
- SBOM, provenance/attestation — при доступной инфраструктуре.
- Release notes описывают любые изменения RNG, event ordering, observation features, rewards и datasets.
- Поддерживать как минимум предыдущую minor-версию в пределах возможностей команды; политика поддержки должна быть опубликована.

### 18.3. Технологические extras

```toml
[project.optional-dependencies]
rl = ["stable-baselines3", "sb3-contrib"]
solver = ["ortools"]
plot = ["matplotlib", "plotly"]
dev = ["pytest", "hypothesis", "ruff", "mypy"]
```

Фактические ranges/версии зафиксировать после проверки совместимости на момент реализации; не копировать примеры как гарантированно совместимые версии.

## 19. Матрица трассировки (основа)

| Requirement ID | Компонент | Тест | Релиз |
|---|---|---|---|
| FR-CORE-01: ordered DES | `core/event_queue.py` | `test_equal_timestamp_order` | P0 |
| FR-CORE-02: state invariants | `core/invariants.py` | property suite | P0 |
| FR-ENV-01: Gymnasium | `environments/base.py` | `check_env` | P0 |
| FR-ENV-02: action mask | `actions/masks.py` | mask legality suite | P0 |
| FR-ENV-03: termination | `environments/base.py` | termination/truncation suite | P0 |
| FR-RNG-01: deterministic replay | `core/rng.py`, `core/replay.py` | seeded replay | P0 |
| FR-SCN-01: config validation | `scenarios/schema.py` | invalid config suite | P0 |
| FR-BENCH-01: baseline parity | `benchmarks/runner.py` | paired instances | P0 |
| FR-OBS-01: no future leakage | `observations/builders.py` | perturb future events test | P1 |
| SEC-01: safe parsing | `scenarios/serialization.py` | malicious YAML tests | P0 |
| NFR-04: performance | `benchmarks/` | nightly profiling | P0 |

Перед началом реализации расширить матрицу на все обязательные требования. В Pull Request указывать как минимум один затронутый Requirement ID.

## 20. Реестр рисков и меры снижения

| Риск | Последствие | Контроль |
|---|---|---|
| Некорректная обработка одновременных событий | Невоспроизводимые результаты | Устойчивый ordering, golden traces |
| Агент использует будущие события | Ложное качество политики | Тесты observation leakage |
| Нет сильных baseline | Некорректная оценка AI | Heuristics + optional CP-SAT |
| WAIT вызывает вечный цикл | Зависание тренировки | Advance-to-next-event, лимиты |
| Слишком большие модели | Out-of-memory/DoS | Ограничения размера и budgets |
| Низкая скорость Python | Дорогие RL эксперименты | Profiling, необязательные backend optimizations |
| Нестабильные Env ID и rewards | Невозможность повторить статьи | SemVer, regression fixtures |
| Копирование benchmark без прав | Юридические риски | Dataset provenance/license audit |
| Симуляция не похожа на реальный завод | Непереносимость политики | Validation/calibration, scope disclaimers |
| Случайная привязка к одному RL framework | Vendor lock-in | Gymnasium + Agent Protocol |

## 21. Критерии проекта как open source

- `README.md` — установка, quickstart, 1 end-to-end пример, limitations.
- `CONTRIBUTING.md` — локальная разработка, тесты, ADR, формат PR.
- `CODE_OF_CONDUCT.md` — правила сообщества.
- `SECURITY.md` — уязвимости и поддерживаемые версии.
- `LICENSE` — полная выбранная лицензия и NOTICE при необходимости.
- `docs/` — объяснение производственной семантики и benchmark fairness.
- GitHub issue templates: bug, feature, benchmark reproducibility.
- Правило: issue с «AI лучше эвристик на X%» сопровождается конфигурацией, seeds, train/test split, hash данных и кодом репликации.

## 22. Практический минимальный сценарий приемки

Вручную создать две машины и два заказа:

- J1: M1 на 8 мин, затем M2 на 12 мин.
- J2: M2 на 7 мин, затем M1 на 15 мин.

Допустимый последовательный пример:

- `t=0`: J1-O1 на M1, J2-O1 на M2.
- `t=7`: J2-O1 завершена, J2-O2 ожидает M1.
- `t=8`: J1-O1 завершена; J1-O2 стартует на M2; J2-O2 стартует на M1.
- `t=20`: J1 готов.
- `t=23`: J2 готов.
- `makespan = 23`.

Проверить, что все переходы легальны, ни один станок не выполняет одновременно две операции, итоговое `C_max=23`, replay повторяет события, а `reward` с `time_scale=1` и `gamma=1` суммируется в `-23`. **Это тест корректности указанного расписания, а не утверждение о глобальной оптимальности instance.**

## 23. Definition of Done для каждой задачи

- [ ] Согласована формулировка requirement + API semantics.
- [ ] Реализован код с typing и документированными исключениями.
- [ ] Есть unit tests и, для core, property tests.
- [ ] Проверены сценарии edge cases.
- [ ] Обновлена документация и changelog при изменении API.
- [ ] Нет неявной зависимости от GPU, внешней сети или UI.
- [ ] CI зеленый на поддерживаемых версиях Python.
- [ ] Пройден code review.
- [ ] Обновлена матрица требований и acceptance evidence.

## 24. Решения, которые необходимо принять до первого merge

1. **ADR-001:** собственный минимальный heapq DES или SimPy adapter; P0 рекомендация — heapq.
2. **ADR-002:** точные приоритеты событий одинакового timestamp и порядок dispatch decisions.
3. **ADR-003:** формат и версии observation, masks и WAIT.
4. **ADR-004:** строгие критерии termination/truncation/deadlock.
5. **ADR-005:** нормализация времени, reward и метрик.
6. **ADR-006:** формат replay trace и степень платформенной воспроизводимости.
7. **ADR-007:** политика выпуска benchmark datasets и использования лицензированных файлов.
8. **ADR-008:** граница API между simulation core и алгоритмами обучения.

Рекомендованные стартовые решения уже указаны в документе: дискретный JobShop, event-driven steps, один агент, фиксированные машиночитаемые instance, явные constraints, masked dispatch, reward за истекшее время, независимые baselines.

## 25. Готовый backlog первого публичного релиза

**P0 / Blocker:**

- [ ] Данные `FactoryInstance`, `MachineSpec`, `JobSpec`, `OperationSpec`.
- [ ] Проверка графа операций, machine refs, длительностей и unique IDs.
- [ ] Детерминированная очередь событий и таблица приоритетов.
- [ ] Transition engine и неизменяемые snapshots.
- [ ] JobShop `reset/step` и корректная termination/truncation.
- [ ] Невалидные действия, `WAIT` и ограничения на циклы.
- [ ] Action masks + observation builders.
- [ ] `negative_elapsed_time` reward и независимые production metrics.
- [ ] FlowShop adapter (общий core, другой builder).
- [ ] Random, FIFO, SPT baseline.
- [ ] Scenario generation + seed + JSON/YAML round-trip.
- [ ] Replay и benchmark runner.
- [ ] Property tests, golden traces, env checker.
- [ ] CLI `validate`, `simulate`, `benchmark`.
- [ ] Gantt chart (можно как optional extra).
- [ ] Пример masked PPO (nightly smoke).
- [ ] PyPI packaging, CI/CD, документация, политика безопасности.

**P1 / После v0.1:**

- [ ] Альтернативные станки и routing.
- [ ] Machine failures + repairs со стабильной семантикой прерывания операции.
- [ ] Конечные буферы и блокировки.
- [ ] Setup times и тарифы энергии.
- [ ] CP-SAT rolling horizon, дополнительные метрики.
- [ ] Multi-agent PettingZoo API.

**P2 / Исследовательская экосистема:**

- [ ] Graph observations/GNN examples.
- [ ] Модели по реальным производственным данным.
- [ ] Калибровка, fidelity reports, uncertainty.
- [ ] Массовые параллельные симуляции и ускоренный backend.
- [ ] Read-only industrial-data connectors с отдельным threat model.

---

## 26. Справочные первоисточники

Проверять актуальные контракты и совместимость версий при реализации:

- Gymnasium API: https://gymnasium.farama.org/api/env/
- Gymnasium environment checker: https://gymnasium.farama.org/api/utils/#environment-checking
- PettingZoo: https://pettingzoo.farama.org/
- SimPy: https://simpy.readthedocs.io/
- OR-Tools Job Shop Scheduling: https://developers.google.com/optimization/scheduling/job_shop
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- sb3-contrib MaskablePPO: https://sb3-contrib.readthedocs.io/
- NumPy random Generator: https://numpy.org/doc/stable/reference/random/generator.html
- Python packaging: https://packaging.python.org/

---

**Итоговое правило реализации:** сначала доказать корректность простой фабрики и честность сравнений алгоритмов; затем расширять реализм. Качественный промышленный симулятор — это прежде всего строгая семантика, воспроизводимость и тестируемые ограничения, а не красивый 3D-интерфейс.
