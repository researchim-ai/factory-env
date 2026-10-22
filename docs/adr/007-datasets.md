# ADR-007. Наборы данных

Статус: принято для v0.1.

В репозитории только экземпляры, написанные для FactoryRL: `demo`, `choice`, `edd`, `flow_demo` и генератор. Чужой файл загружается `load_matrix_jobshop` только вместе с `source` и `license_id`. Хеш файла пишется в metadata.

Зависимости базового пакета: NumPy, Gymnasium, PyYAML, Pydantic. Лицензии BSD/MIT совместимы с Apache-2.0. OR-Tools и Stable-Baselines3 — отдельные extra.
