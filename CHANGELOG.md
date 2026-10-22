# Changelog

## 0.1.0

Первый публичный набор P0.

- Детерминированное ядро job shop / flow shop на очереди `heapq`.
- Среды `FactoryRL/JobShop-v0` и `FactoryRL/FlowShop-v0`.
- Наблюдение `obs_v1`, маска действий, награда `negative_elapsed_time` и альтернатива `terminal_makespan`.
- Базовые политики Random, FIFO, SPT, EDD.
- Генераторы, безопасный JSON/YAML, replay, CLI и отчёт бенчмарка.
- Отказы, буферы, переналадки и мультиагентность не входят в этот релиз.
