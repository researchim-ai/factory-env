# Матрица трассировки v0.1

| ID | Где | Проверка |
|---|---|---|
| FR-CORE-01 порядок событий | `core/event_queue.py`, `core/events.py` | `tests/unit/test_event_queue.py` |
| FR-CORE-02 инварианты | `core/invariants.py`, `core/transition.py` | property-тесты и `test_edge_branches.py` |
| FR-ENV-01 Gymnasium | `environments/base.py` | `check_env` в `test_gym_api.py` |
| FR-ENV-02 маска | `actions/masks.py` | `test_gym_api.py`, `test_actions_and_rewards.py` |
| FR-ENV-03 завершение | `core/engine.py` | demo, horizon, deadlock, бюджет решений |
| FR-RNG-01 replay | `core/replay.py`, `core/rng.py` | `test_gym_api.py`, `test_edge_branches.py` |
| FR-SCN-01 схема | `scenarios/schema.py`, `scenarios/serialization.py` | `test_generator_and_io.py` |
| FR-BENCH-01 одинаковые экземпляры | `benchmarks/runner.py` | `test_cli_and_benchmark.py` |
| SEC-01 безопасный разбор | `scenarios/serialization.py` | YAML с `!!python` и лимит размера |
| AC-02 две среды | `registry.py` | `test_gym_api.py` |
| AC-03 3×5 | генератор и среда | `test_three_by_five_finishes` |
| AC-09 makespan 23 и choice 12/21 | движок и независимый планировщик | `test_demo_schedule.py`, `test_baselines_and_flow.py` |
| NFR-12 без ML-рантайма | зависимости `pyproject.toml` | `test_import_does_not_pull_in_a_learning_framework` |

Покрытие строк пакетов `core`, `models` и `actions` проверяется `pytest --cov`. Ветка покрытия включена в отчёт; цель спецификации — не меньше 90% строк. Инвариантные ветки отказа закрыты точечными тестами, но не каждой комбинацией порчи состояния.
