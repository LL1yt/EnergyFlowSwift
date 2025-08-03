# План перехода на архитектуру относительных координат

## Обзор изменений

### Текущая архитектура (абсолютные координаты)

- Потоки стартуют на входной плоскости Z=0
- Модель предсказывает абсолютные координаты следующей позиции
- Выходная плоскость на Z=depth
- Проблемы с нормализацией: Z=58-60 вместо Z=0
- без фолбэков, без сохранения старой легаси архитектуры. используем только новую логику или ошибка. смело удаляем старую логику.
- используем центральный конфиг и логирование.

### Новая архитектура (относительные координаты)

- **Входная плоскость**: Z = depth/2 (центр куба)
- **Выходные плоскости**: Z = 0 и Z = depth (края куба)
- **Предсказание**: относительные смещения (Δx, Δy, Δz)
- **Spawn**: на основе длины вектора смещения
- **Отражение границ**: для X/Y координат

## Детальный план реализации

### 1. Конфигурация (✅ ЗАВЕРШЕНО)

**Файл**: `energy_flow/config/energy_config.py`

**✅ РЕАЛИЗОВАННЫЕ параметры**:

```python
# Архитектура относительных координат
relative_coordinates: bool = False
center_start_enabled: bool = False
dual_output_planes: bool = False

# Система spawn на основе движения
spawn_movement_threshold_ratio: float = 0.1  # depth/10
movement_based_spawn: bool = False

# Отражение границ
boundary_reflection_enabled: bool = False

# Система важности
proximity_weight: float = 0.7
path_length_weight: float = 0.3
safe_distance_minimum: float = 0.5
```

**⏳ ТРЕБУЕТСЯ**: Обновление конфигураций debug/experiment/optimized с новыми флагами

### 2. Система нормализации (✅ ЗАВЕРШЕНО)

**Файл**: `energy_flow/utils/normalization.py`

**✅ РЕАЛИЗОВАННЫЕ изменения**:

- ✅ Удалена LEGACY поддержка (без fallback)
- ✅ Новые методы для смещений:

  ```python
  def normalize_displacement(self, displacement: torch.Tensor) -> torch.Tensor:
      # [-depth/2, depth/2] → [-1, 1]

  def denormalize_displacement(self, normalized: torch.Tensor) -> torch.Tensor:
      # [-1, 1] → [-depth/2, depth/2]
  ```

- ✅ Обновление Z-диапазона: `[0, depth]` вместо `[0, depth*2-1]`
- ✅ Фабричная функция `create_normalization_manager()` обновлена

### 3. EnergyCarrier (✅ ЗАВЕРШЕНО)

**Файл**: `energy_flow/core/energy_carrier.py`

**✅ РЕАЛИЗОВАННЫЕ изменения**:

1. ✅ **Новый выход модели**: `displacement_projection` вместо `position_projection`
2. ✅ **Удален curriculum learning**: smart initialization, forward bias
3. ✅ **Новая логика движения**:
   ```python
   def forward(self, ...):
       # Предсказываем смещения в [-1,1]
       displacement_raw = self.displacement_projection(gru_output)
       displacement_normalized = self.displacement_activation(displacement_raw)  # Tanh

       # Денормализуем в реальные смещения
       displacement_real = self.config.normalization_manager.denormalize_displacement(
           displacement_normalized
       )

       # Применяем к текущей позиции
       next_position = current_position + displacement_real
   ```
4. ✅ **Новый метод**: `_compute_next_position_relative()` для трехплоскостной архитектуры

### 4. EnergyLattice (✅ ЗАВЕРШЕНО)

**Файл**: `energy_flow/core/energy_lattice.py`

**✅ РЕАЛИЗОВАННЫЕ изменения**:

1. ✅ **Стартовые позиции в центре**:

   ```python
   def place_initial_energy(self, embeddings, mapper):
       # Размещаем все потоки на Z = depth/2
       start_z = self.depth // 2
       for (x, y), energy, batch_idx in cell_energies:
           position = torch.tensor([x, y, start_z], device=self.device)
           flow_id = self._create_flow(position, energy, batch_idx)
   ```

2. ✅ **Двойной выходной буфер**:

   ```python
   self.output_buffer_z0: Dict[Tuple[int, int], List[EnergyFlow]] = {}  # Z=0
   self.output_buffer_zdepth: Dict[Tuple[int, int], List[EnergyFlow]] = {} # Z=depth

   def _buffer_flow_to_z0_plane(self, flow_id: int): # Z=0 плоскость
   def _buffer_flow_to_zdepth_plane(self, flow_id: int): # Z=depth плоскость
   ```

3. ✅ **Система важности с безопасным делением**:

   ```python
   def calculate_flow_importance(self, flow: EnergyFlow) -> float:
       z = flow.position[2].item()
       distance_to_z0 = abs(z - 0)
       distance_to_zdepth = abs(z - self.depth)
       min_distance = min(distance_to_z0, distance_to_zdepth)

       # Безопасное деление
       safe_distance = max(min_distance, self.config.safe_distance_minimum)
       proximity = 1.0 / safe_distance

       path_importance = flow.age * self.config.path_length_weight
       return self.config.proximity_weight * proximity + path_importance
   ```

4. ✅ **Обновлены все вспомогательные методы**: `reset()`, `clear_output_buffer()`, `get_buffered_flows_count()`, `get_all_buffered_flows()`

### 5. FlowProcessor (⏳ В ПРОЦЕССЕ)

**Файл**: `energy_flow/core/flow_processor.py`

**⏳ ТРЕБУЕМЫЕ изменения**:

1. **Применение смещений**:

   ```python
   def apply_displacement(self, current_pos: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
       new_position = current_pos + displacement

       if self.config.boundary_reflection_enabled:
           new_position = self.reflect_boundaries(new_position)

       return new_position
   ```

2. **Отражение границ**:

   ```python
   def reflect_boundaries(self, position: torch.Tensor) -> torch.Tensor:
       x, y, z = position[:, 0], position[:, 1], position[:, 2]

       # Отражение X
       x = torch.where(x < 0, -x, x)
       x = torch.where(x >= self.config.lattice_width,
                      2*(self.config.lattice_width-1) - x, x)

       # Аналогично для Y
       # Z остается без изменений

       return torch.stack([x, y, z], dim=1)
   ```

3. **Spawn на основе длины смещения**:
   ```python
   def check_movement_spawn(self, displacement: torch.Tensor, flow_ids: torch.Tensor) -> List[SpawnInfo]:
       displacement_lengths = torch.norm(displacement, dim=1)  # [batch]
       threshold = self.config.lattice_depth * self.config.spawn_movement_threshold_ratio

       spawn_mask = displacement_lengths > threshold
       spawn_info = []

       for i, should_spawn in enumerate(spawn_mask):
           if should_spawn:
               delta_length = displacement_lengths[i].item()
               direction = displacement[i] / delta_length  # Нормализованное направление

               # Количество дополнительных потоков
               num_spawns = int((delta_length - threshold) // threshold)

               # Создаем spawn info
               energies = [self.get_flow_energy(flow_ids[i])] * num_spawns
               spawn_info.append(SpawnInfo(energies=energies, parent_batch_idx=i))

       return spawn_info
   ```

**⚠️ КРИТИЧЕСКИ ВАЖНО**: Обновить `_process_results_vectorized()` для работы с новыми `termination_reasons` от EnergyCarrier

### 6. Обновление debug логирования

**Файл**: `energy_flow/utils/logging.py`

**Новые debug уровни**:

```python
DEBUG_RELATIVE = 21      # Относительные координаты и смещения
DEBUG_SPAWN_MOVEMENT = 22 # Spawn на основе движения
DEBUG_REFLECTION = 23     # Отражение границ
DEBUG_IMPORTANCE = 24     # Система важности потоков
```

### 7. Последовательность внедрения

#### Этап 1: Подготовка (✅ ЗАВЕРШЕН)

- [x] ✅ Добавить новые параметры в конфигурацию
- [x] ✅ Создать систему нормализации смещений
- [ ] ⏳ Добавить новые debug уровни

#### Этап 2: Основные модели (✅ ЗАВЕРШЕН)

- [x] ✅ Переработать EnergyCarrier для предсказания смещений
- [x] ✅ Обновить EnergyLattice для трехплоскостной архитектуры
- [ ] ⏳ Модифицировать FlowProcessor для относительного движения (**ТЕКУЩИЙ**)

#### Этап 3: Дополнительные функции (⏳ СЛЕДУЮЩИЙ)

- [ ] ⏳ Реализовать систему spawn на основе длины смещения
- [ ] ⏳ Добавить отражение границ для X/Y координат
- [x] ✅ Реализовать систему важности с безопасным делением

#### Этап 4: Тестирование (⏳ ПЛАНИРУЕТСЯ)

- [ ] ⏳ Интеграционные тесты с trainer'ом

#### Этап 5: Внедрение (⏳ ПЛАНИРУЕТСЯ)

- [ ] ⏳ Обновить debug/experiment конфигурации для использования новой архитектуры

## 📍 СТАТУС НА ТЕКУЩИЙ МОМЕНТ

### ✅ Завершенные компоненты:

1. **EnergyConfig** - все параметры новой архитектуры добавлены
2. **Система нормализации** - полная поддержка смещений, удален legacy код
3. **EnergyCarrier** - предсказание смещений, удален curriculum learning
4. **EnergyLattice** - трехплоскостная архитектура, двойной буфер, система важности

### ⏳ Текущая работа:

- **FlowProcessor** - обновление для работы с относительными смещениями

### 🔄 Следующие шаги для продолжения:

1. **ПРИОРИТЕТ 1**: Завершить FlowProcessor - обновить `_process_results_vectorized()` для новых `termination_reasons`
2. **ПРИОРИТЕТ 2**: Добавить методы отражения границ и spawn на основе движения
3. **ПРИОРИТЕТ 3**: Обновить debug конфигурации для тестирования новой архитектуры

### ⚠️ Критические моменты:

- EnergyCarrier теперь возвращает новые типы `termination_reasons`: `"reached_z0_plane"`, `"reached_zdepth_plane"`, `"xy_reflection_needed"`
- FlowProcessor должен обрабатывать отражение границ для `"xy_reflection_needed"`
- Система spawn теперь основана на длине смещения, а не на эмбеддингах

### 8. Критические точки внимания

1. **Обратная совместимость**: новая архитектура управляется флагами, старая остается доступной
2. **Градиентные связи**: все операции должны сохранять градиенты для обучения
3. **Производительность**: векторизованные операции для GPU оптимизации
4. **Отладка**: подробное логирование на всех этапах для диагностики

### 9. Ожидаемые улучшения

1. **Стабильность обучения**: устранение проблем с нормализацией координат
2. **Естественное движение**: модель учится движению без принуждения
3. **Гибкость**: потоки могут двигаться в любом направлении
4. **Лучшая утилизация пространства**: использование всего объема куба
5. **Более богатые паттерны**: spawn на основе геометрии движения

### 10. Метрики успеха

- **Утилизация потоков**: % потоков достигающих выхода (цель: >90%)
- **Распределение по плоскостям**: равномерное использование обеих выходных плоскостей
- **Стабильность обучения**: отсутствие резких скачков loss
- **Производительность**: сохранение скорости обработки на RTX 5090
