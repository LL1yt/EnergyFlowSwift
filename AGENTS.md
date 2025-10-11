## Контекст

- Проект исследования генераторов энергетических потоков.
- Активная ветка — SwiftPM-пакет `EnergyFlowSwift/`.
- Легаси-питон лежит в `energy_flow/`, но новые задачи выполняются в Swift.

## Принципы

- Делаем просто, но быстро: основной приоритет — производительность GPU.
- Никаких фолбэков: если GPU-функция недоступна, вызываем явную ошибку.
- CPU-вычисления только как временный хелпер, если нет GPU варианта.
- Конфигурации/логи централизованы (см. `Logger`, конфиг-структуры в Swift).
- Модульность
- Проект исследовательский, не продакшн
- Без fallback — лучше ошибка, чем костыли
- Тесты и сборку запускает пользователь вручную; агенты не дергают `swift build/test`.

## Текущая структура SwiftPM

```
EnergyFlowSwift/
  Sources/
    EFCore/         // Tensor API, GPUActor, Metal/MPS обвязка
    PyTorchSwift/   // Swift-аналоги модулей PyTorch (Embedding, Linear, TCN)
    EnergyFlow/     // TextBridge, Decoder, Trainers
    EFTrain/        // CLI тренировки
    EFTextEval/     // CLI оценки
  Tests/
    EnergyFlowSwiftTests/ // async XCTest, проверка форм и градиентов
  docs/             // планы (см. Async_GPU_Actor_Refactor_Plan.md)
```

## GPUActor — состояние

- Все GPU-хелперы (Linear, Conv1D, Elementwise, Metrics) работают через `GPUActor`.
- Каждая операция возвращает `GPUReadback<T>` — deferred readback, который разрешается после `syncBatch()`.
- Метрики KD/CE перенесены на Metal-редукции, CPU-версии удалены.
- Guard: попытка `readback.value()` до `syncBatch` → `fatalError`.

## Что уже сделано в Swift

- TextBridge использует deferred readback для проекции и метрик.
- DecoderTrainer/CombinedTrainer переходят на `GPUReadback` + `syncBatch`.
- Elementwise/Conv1D имеют `*_Deferred` API; старые методы вызывают их с `deferUntilSync: false`.
- `EFTextEval` переписан без top-level кода, с явным `@main`.

## Что дальше

1. **TCN/decoder deferred** — протянуть `GPUReadback` в Conv/LN/Mask внутри TCNBlock, упростить тренеровки.
2. **Тесты** — обновить GPU-тесты (TCN, decoder, elementwise) на новые API.
3. **Документация** — отслеживать прогресс в `docs/Async_GPU_Actor_Refactor_Plan.md` и новом плане `docs/Deferred_Readbacks_TCN.md`.

## Полезные ссылки

- Основной план GPU-акторa: `docs/Async_GPU_Actor_Refactor_Plan.md`
- План портирования Swift/MPS: `docs/Swift_MPS_Port_Plan.md`
- План по deferred readbacks (см. новый документ в docs).
