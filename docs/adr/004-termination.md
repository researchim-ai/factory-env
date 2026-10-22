# ADR-004. Завершение, усечение, deadlock

Статус: принято для v0.1.

- Все операции готовы: `terminated`, `success=true`, причина `completed`.
- Горизонт, `max_events` или `max_decisions`: `truncated`, `success=false`.
- Пустая очередь при незавершённой работе: `terminated`, `success=false`, причина `deadlock`.

Если бюджет не позволяет дойти даже до первого решения, конструктор или `reset` поднимает `ConfigurationError`, а не молчаливый пустой эпизод.
