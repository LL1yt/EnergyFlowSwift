Ran tool

Read file: smart_resume_training/core/config_initializer.py

Ran tool

Ran tool

Read file: core/lattice_3d/config.py

Read file: utils/centralized_config.py

Ran tool

Read file: utils/centralized_config.py

Ran tool

Read file: training/embedding_trainer/nca_adapter.py

Read file: training/embedding_trainer/nca_adapter.py

Ran tool

Read file: core/lattice_3d/enums.py

Read file: core/lattice_3d/topology.py

Read file: training/automated_training/stage_runner.py

Read file: training/automated_training/stage_runner.py

Ran tool

Read file: utils/config_manager/dynamic_config.py

Ran tool

Ran tool

Read file: core/cell_prototype/architectures/gmlp_opt_connections.py

Ниже ‒ подробный разбор того, что реально происходит во время запуска `test_phase4_full_training_cycle.py`, откуда берутся спорные сообщения в логах и какие шаги нужны, чтобы всё привести к желаемому («tiered»-соседи, `FULL_FACE` I/O-покрытие, единичная инициализация NCA-клетки, отсутствие наследия scale-factor и пр.).

---

1.  Почему в логах «🔧 PHASE 4: Hybrid mode from file: True»

---

• Источник:

```68:74:smart_resume_training/core/config_initializer.py
hybrid_mode = architecture.get("hybrid_mode", False)
logger.info(f"🔧 PHASE 4: Hybrid mode from file: {hybrid_mode}")
```

• Поле `architecture.hybrid_mode` записывается в конфиг не вами вручную, а
в методе `TrainingStageRunner._apply_hybrid_architecture()` (вызывается из
теста). Там жёстко устанавливается:

```212:228:training/automated_training/stage_runner.py
config_data["architecture"] = {
    "hybrid_mode": True,
    "neuron_architecture": "minimal_nca",
    "connection_architecture": "gated_mlp",
    ...
}
```

Таким образом «Hybrid» относится к архитектуре («нейрон = NCA, связи = gMLP»),
а не к стратегии соседей.  
Cами соседи задаются тут же:

```233:238
"neighbor_finding_strategy": "tiered",
```

Это подтверждается второй строкой вашего лога  
`Neighbor strategy: tiered`. Поэтому с точки зрения топологии всё
корректно – используется именно «tiered».

---

2.  `placement_method: "proportional"` – менять ли на `FULL_FACE`

---

• Поле читается в `core/lattice_3d/config.py`:

```641:644
combined_data["placement_strategy"] = PlacementStrategy(
    io_strategy.get("placement_method", "proportional")
)
```

• Текущая стратегия `PROPORTIONAL` распределяет точки I/O пропорционально
длинам сторон, что экономит клетки, но ограничивает размер входного /
выходного вектора.

Если «все равно потом всё прогоняем через Universal Adapter» и узким
местом становится именно ширина face, то можно сразу перейти на
`FULL_FACE` ­– это даёт:

1. максимально плотное покрытие всей грани;
2. упрощение расчёта размера embedding-вектора;
3. меньший шанс «нехватки» входных точек при последующем масштабировании.

Для быстрой проверки достаточно в StageRunner добавить всего одну строку
в `_apply_hybrid_architecture` после секции `lattice_3d`:

```python
config_data["lattice_3d"]["io_strategy"] = {"placement_method": "full_face"}
```

или прописать то же в `config/default.yaml` (если хотите системно).

---

3.  Усилить логирование Universal / Embedding Adapter

---

Добавьте в каждый публичный метод адаптера (например
`universal_embedding_adapter.convert()`):

```python
logger.info(
    f"[EMBED_ADAPTER] source='{source_name}', "
    f"target_shape={target_tensor.shape}, "
    f"for_module='{caller_module}'"
)
```

(Имя вызывающего модуля можно получить через `inspect.stack()[1].function`
или просто передавать аргументом.)

---

4.  «Initialized config … scale factor 0.01» – рудимент

---

Поле добавляется в `DynamicConfigGenerator.adjust_config_for_mode()`:

```428:433
scale_factor = self.scale_settings.get_scale(mode)
config["lattice"]["scale_factor"] = scale_factor
```

Сам `scale_factor` больше нигде не используется, вывод идёт лишь ради
отладки. Уберите строку логирования из
`ConfigInitializer._log_config_details()` или обнулите поле после
генерации – функционально ничего не изменится.

---

5.  Лог «gMLP: state=8, neighbors=26» и 23 000 параметров

---

• Линия приходит из `CentralizedConfig.log_config_summary()` и выводит
ТОЛЬКО `state_size` и `neighbor_count`.  
• Param-target (23 805) действительно прописан:

```yaml
gmlp:
  state_size: 8
  hidden_dim: 32
  target_params: 23805 # ✔ оптимизированная версия
```

Так что расхождения нет: оптимизированная gMLP-ячейка использует
`state_size = 8`, но за счёт bottleneck/gating имеет ≈23 k параметров.
Чтобы она реально создавалась:

1. В конфиге вместо `gmlp_cell` нужно указывать
   `gmlp_opt_connections` (новое имя фабрики).
2. В `TrainingStageRunner._apply_hybrid_architecture()` поменять:

```python
config_data["cell_prototype"]["prototype_name"] = "gmlp_opt_connections"
config_data["cell_prototype"]["gmlp_opt_connections"] = {
    ...  # параметры из CentralizedConfig
}
```

---

6.  Двойная инициализация `MinimalNCACell`

---

• Первая (лог `training.embedding_trainer.nca_adapter`) – это
адаптер, который создаёт отдельный экземпляр «мозга» для обработчика
входных embedding-векторов.

• Вторая (лог `core.cell_prototype.main`) – фабрика,
которая строит прототип клетки для ВСЕЙ решётки.

Чтобы не делать работу дважды:

1. Пробросьте уже созданный `EmergentNCACell` из адаптера в
   `Lattice3D` как прототип, например через аргумент
   `external_cell_instance` в `LatticeConfig` (добавьте поле).
2. В `Lattice3D.__init__` используйте переданную инстанцию,
   если она есть, вместо `create_cell_from_config`.

---

7.  `NoneType`.get в `NeighborTopology`

---

`strategy_config` приходит из `config.neighbor_strategy_config`, которое
иногда равно `None`. В `NeighborTopology.__init__` достаточно заменить:

```python
self.strategy_config = getattr(config, "neighbor_strategy_config", {}) or {}
```

или (лучше) в `_build_lattice_config_from_data` установить пустой dict,
если секция отсутствует.

---

8.  «Failed to serialize config» в Validator / Trainer

---

`json.dumps()` падает на dataclass’ах. Решение:

```python
from dataclasses import asdict
safe_cfg = asdict(config) if hasattr(config, "__dataclass_fields__") else config
logger.info(json.dumps(safe_cfg, indent=2, default=str))
```

---

9.  Нужно ли держать `nca_adapter`

---

Если решили окончательно отказаться от legacy-пути (отдельная
gMLP-ячейка + «NCA-как-нейрон»), можете:

1. Переместить логику из `nca_adapter.EmergentNCACell` в
   `minimal_nca_cell.py` как опцию `variant="emergent"`.
2. Удалить `nca_adapter`, а все вызовы заменить на
   `create_cell_from_config(..., prototype_name="minimal_nca_cell")`.

Это снимет избыточность кода и исключит вторую инициализацию.

---

10. Чек-лист действий

---

1. StageRunner:  
   • дописать `io_strategy: placement_method = full_face`,  
   • прописать `prototype_name = gmlp_opt_connections`.

2. Lattice / Topology:  
   • обработать `neighbor_strategy_config is None`,  
   • добавить возможность взять уже созданный `EmergentNCACell`.

3. CentralizedConfig / Logging:  
   • убрать вывод `scale_factor`,  
   • добавить подробные логи в Universal Adapter.

4. Validator / Trainer:  
   • сериализовать конфиги через `asdict()`.

После этих шагов вы получите:
• однократную инициализацию NCA-клетки,  
• «tiered» соседей без ошибок,  
• `FULL_FACE` схему I/O (при желании),  
• чистые логи без устаревшего scale-factor,  
• работающую оптимизированную gMLP-связь (≈23 k params).

Всё это можно сделать без изменения структуры Phase-4 теста; он
автоматически подхватит новые настройки через временный YAML, который
генерирует `TrainingStageRunner`.
