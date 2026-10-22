# ADR-003. Наблюдение, маска и WAIT

Статус: принято для v0.1.

Пространство действий — `Discrete(max_operations + 1)`. Последний индекс — WAIT. Наблюдение — словарь `obs_v1`, описанный в `docs/semantics.md`. Маска лежит в `info["action_mask"]` и в `action_masks()`.

`sample()` намеренно смотрит на маску начала эпизода, чтобы проходить `gymnasium.utils.env_checker.check_env`. Явный недопустимый `step` по-прежнему поднимает `InvalidActionError`.
