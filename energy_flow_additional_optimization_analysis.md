# Energy Flow Additional Optimization Opportunities

Краткий список наблюдаемых потенциальных проблем и предложений по их решению.

## 1. Blocking CUDA Synchronization in Metrics
- **Проблема**: Training step вызывает `torch.cuda.synchronize()` перед чтением метрик памяти, что блокирует асинхронное выполнение и замедляет шаги【F:energy_flow/training/energy_trainer.py†L296-L303】【F:energy_flow/training/energy_trainer.py†L980-L984】
- **Решение**: Ограничить вызов синхронизации профилировочными режимами или убрать его в стандартных прогонках.

## 2. Чрезмерный сброс peak memory
- **Проблема**: `torch.cuda.reset_peak_memory_stats()` выполняется в начале каждого шага аккумуляции градиентов, вызывая синхронизацию и накладные расходы【F:energy_flow/training/energy_trainer.py†L443-L446】
- **Решение**: Сбрасывать peak memory реже — только в `DEBUG_MEMORY` или на границах эпох.

## 3. Пропускной DataLoader
- **Проблема**: `DataLoader` создаётся без `num_workers` и с `lambda` как `collate_fn`, что ограничивает подготовку батчей на CPU и усложняет отладку【F:full_energy_trainer.py†L98-L105】
- **Решение**: Задать `num_workers > 0`, включить `persistent_workers=True`, перенести `collate_fn` в именованную функцию для возможной компиляции.

## 4. Детминистический cuDNN по умолчанию
- **Проблема**: Глобальная установка `torch.backends.cudnn.deterministic = True` отключает autotune и может существенно замедлять GRU【F:full_energy_trainer.py†L124-L126】
- **Решение**: Включать детерминизм только при отладке; для продуктивных запусков использовать `torch.backends.cudnn.benchmark = True`.

## 5. `.item()` в горячих циклах
- **Проблема**: В `FlowProcessor` множество вызовов `.item()` внутри циклов для логов, что приводит к синхронизации GPU→CPU на каждый элемент【F:energy_flow/core/flow_processor.py†L448-L456】
- **Решение**: Перенести подобные логи за пределы горячих путей и использовать batch-агрегацию или guard по уровню DEBUG.

## 6. Нереализованный план `torch.compile`
- **Наблюдение**: Существует подробный план внедрения `torch.compile` с ожидаемым ускорением 20–35%, но он пока не интегрирован【F:energy_flow/TORCH_COMPILE_PLAN.md†L1-L10】
- **Действие**: Реализовать `CompileManager` и поэтапно компилировать `SimpleNeuron`, `EnergyCarrier` и `FlowProcessor` согласно плану.

