# Архитектура

```text
CLI / примеры / Gymnasium
        |
   environments  baselines  benchmarks
        |
   observations  actions  rewards
        |
   core: очередь, переходы, инварианты, replay
        |
   models + scenarios
```

`factoryrl.core` и `factoryrl.models` не импортируют Gymnasium, PyTorch и OR-Tools. Среда — адаптер: она переводит индекс Gymnasium в индекс операции или `WAIT=-1`, собирает наблюдение и награду.

Состояние времени мутабельно только внутри движка. Наружу уходит замороженный `StateSnapshot`. Повторный `reset` собирает новое состояние из того же экземпляра; следы прошлого эпизода не переносятся.

Рендер (`ansi`, `gantt_html`) читает снимок и интервалы. От него не зависит ни одно решение и ни одна метрика.

OR-Tools живёт в `baselines/optional_ortools.py` и импортируется только при вызове солвера.
