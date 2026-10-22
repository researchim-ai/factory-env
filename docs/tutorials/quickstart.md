# Учебный прогон

1. Установите пакет: `pip install -e ".[dev]"`.
2. Запустите `python examples/quickstart.py`. Ожидается `makespan=23.0` и `return=-23.0`.
3. Откройте `examples/factories/demo.yaml`. Это тот же завод: J1 идёт M1 (8) затем M2 (12), J2 идёт M2 (7) затем M1 (15).
4. `python examples/compare_baselines.py` показывает, что на `demo` FIFO и SPT совпадают, а на `choice` расходятся: 21 и 12.
5. `python examples/train_masked_policy.py` прогоняет короткий REINFORCE с маской. Успех эпизода здесь означает только то, что действия были допустимы.
6. MaskablePPO: `pip install -e ".[rl]"` и `python examples/train_masked_ppo.py`. Без extra скрипт объясняет, как поставить зависимость, и не тянет PyTorch в базовую установку.
