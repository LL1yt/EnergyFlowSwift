"""
Тест для EnergyLattice - 3D решетка для управления энергетическими потоками
==========================================================================

Простые тесты для проверки работоспособности EnergyLattice:
- Инициализация решетки с параметрами
- Размещение входных эмбеддингов на входной стороне (z=0)
- Управление активными потоками (создание, обновление, деактивация)
- Сбор выходных эмбеддингов с выходной стороны (z=depth-1)
- Статистика и очистка потоков
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем пути для импортов
sys.path.append('/mnt/c/Users/n0n4a/projects/AA')

from energy_flow.core.energy_lattice import EnergyLattice, EnergyFlow, create_energy_lattice
from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.utils.logging import setup_logging

# Включаем debug логирование для отладки
setup_logging(debug_mode=True, level="DEBUG")

def test_energy_lattice_init():
    """Тест инициализации EnergyLattice"""
    print("\n=== Тест инициализации EnergyLattice ===")
    
    # Создаем конфиг
    config = create_debug_config()
    set_energy_config(config)
    
    # Создаем решетку
    lattice = create_energy_lattice(config)
    
    # Проверяем параметры
    print(f"✅ EnergyLattice создана: {lattice.width}x{lattice.height}x{lattice.depth}")
    print(f"✅ Максимум активных потоков: {lattice.max_active_flows}")
    print(f"✅ Входных/выходных клеток: {lattice.width * lattice.height}")
    print(f"✅ Размерность эмбеддинга на клетку: {lattice.embedding_dim}")
    print(f"✅ Устройство: {lattice.device}")
    
    # Проверяем начальное состояние
    assert len(lattice.active_flows) == 0, "Начальное состояние должно быть пустым"
    assert lattice.next_flow_id == 0, "ID следующего потока должен быть 0"
    
    stats = lattice.get_statistics()
    print(f"📊 Начальная статистика: {stats}")
    
    return lattice

def test_place_initial_energy():
    """Тест размещения входных эмбеддингов"""
    print("\n=== Тест размещения входных эмбеддингов ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    # Подготавливаем входные эмбеддинги
    batch_size = 2
    device = lattice.device
    
    # Эмбеддинги размерности input_embedding_dim (768D)
    input_embeddings = torch.randn(batch_size, config.input_embedding_dim, device=device)
    
    print(f"📝 Входные эмбеддинги: {input_embeddings.shape}")
    print(f"📝 Ожидаемое количество потоков: {lattice.width * lattice.height * batch_size}")
    
    # Размещаем энергию
    flow_ids = lattice.place_initial_energy(input_embeddings)
    
    # Проверяем результат
    print(f"✅ Создано потоков: {len(flow_ids)}")
    print(f"✅ Активных потоков: {len(lattice.get_active_flows())}")
    
    # Проверяем что потоки размещены на входной стороне (z=0)
    active_flows = lattice.get_active_flows()
    z_positions = [flow.position[2].item() for flow in active_flows]
    all_at_input = all(z == 0 for z in z_positions)
    print(f"✅ Все потоки на входной стороне (z=0): {all_at_input}")
    
    # Проверяем распределение по клеткам
    positions = [(int(flow.position[0].item()), int(flow.position[1].item())) for flow in active_flows]
    unique_positions = set(positions)
    expected_unique = lattice.width * lattice.height
    print(f"✅ Уникальных позиций: {len(unique_positions)} (ожидается: {expected_unique})")
    
    return lattice, flow_ids

def test_flow_management():
    """Тест управления потоками"""
    print("\n=== Тест управления потоками ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # Создаем тестовые потоки вручную
    position1 = torch.tensor([5, 8, 3], dtype=torch.float32, device=device)
    position2 = torch.tensor([12, 6, 7], dtype=torch.float32, device=device)
    
    energy1 = torch.randn(config.embedding_per_cell, device=device)
    energy2 = torch.randn(config.embedding_per_cell, device=device)
    
    # Создаем потоки
    flow_id1 = lattice._create_flow(position1, energy1)
    flow_id2 = lattice._create_flow(position2, energy2)
    
    print(f"✅ Созданы потоки с ID: {flow_id1}, {flow_id2}")
    print(f"✅ Активных потоков: {len(lattice.get_active_flows())}")
    
    # Обновляем поток
    new_position = torch.tensor([6, 9, 4], dtype=torch.float32, device=device)
    new_energy = torch.randn(config.embedding_per_cell, device=device)
    new_hidden = torch.randn(config.carrier_num_layers, config.carrier_hidden_size, device=device)
    
    lattice.update_flow(flow_id1, new_position, new_energy, new_hidden)
    
    # Проверяем обновление
    updated_flow = lattice.active_flows[flow_id1]
    position_updated = torch.allclose(updated_flow.position, new_position)
    print(f"✅ Поток обновлен правильно: {position_updated}")
    print(f"✅ Возраст потока: {updated_flow.age}")
    
    # Деактивируем поток
    lattice.deactivate_flow(flow_id2, "test_deactivation")
    active_after_deactivation = len(lattice.get_active_flows())
    print(f"✅ Активных потоков после деактивации: {active_after_deactivation}")
    
    return lattice

def test_spawn_flows():
    """Тест создания новых потоков от родительских"""
    print("\n=== Тест создания новых потоков ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # Создаем родительский поток
    parent_position = torch.tensor([10, 10, 5], dtype=torch.float32, device=device)
    parent_energy = torch.randn(config.embedding_per_cell, device=device)
    parent_id = lattice._create_flow(parent_position, parent_energy)
    
    print(f"✅ Родительский поток создан с ID: {parent_id}")
    
    # Энергии для новых потоков
    spawn_energies = [
        torch.randn(config.embedding_per_cell, device=device),
        torch.randn(config.embedding_per_cell, device=device),
        torch.randn(config.embedding_per_cell, device=device)
    ]
    
    # Создаем потомков
    child_ids = lattice.spawn_flows(parent_id, spawn_energies)
    
    print(f"✅ Создано потомков: {len(child_ids)}")
    print(f"✅ ID потомков: {child_ids}")
    print(f"✅ Общее количество активных потоков: {len(lattice.get_active_flows())}")
    
    # Проверяем что потомки начинают с позиции родителя
    parent_flow = lattice.active_flows[parent_id]
    for child_id in child_ids:
        child_flow = lattice.active_flows[child_id]
        same_position = torch.allclose(child_flow.position, parent_flow.position)
        correct_parent = child_flow.parent_id == parent_id
        print(f"   Потомок {child_id}: позиция={same_position}, родитель={correct_parent}")
    
    return lattice, parent_id, child_ids

def test_collect_output_energy():
    """Тест сбора выходной энергии"""
    print("\n=== Тест сбора выходной энергии ===") 
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # Создаем потоки на выходной стороне (z = depth-1)
    output_z = config.lattice_depth - 1
    
    flow_positions = [
        torch.tensor([0, 0, output_z], dtype=torch.float32, device=device),
        torch.tensor([5, 8, output_z], dtype=torch.float32, device=device),
        torch.tensor([19, 19, output_z], dtype=torch.float32, device=device),
        torch.tensor([10, 15, 5], dtype=torch.float32, device=device),  # Не на выходе
    ]
    
    energies = [torch.randn(config.embedding_per_cell, device=device) for _ in flow_positions]
    
    # Создаем потоки
    flow_ids = []
    for pos, energy in zip(flow_positions, energies):
        flow_id = lattice._create_flow(pos, energy)
        flow_ids.append(flow_id)
    
    print(f"✅ Создано потоков: {len(flow_ids)} (3 на выходе, 1 внутри)")
    
    # Собираем выходную энергию
    output_embeddings, output_flow_ids = lattice.collect_output_energy()
    
    print(f"✅ Выходных эмбеддингов: {output_embeddings.shape}")
    print(f"✅ Потоков на выходе: {len(output_flow_ids)}")
    print(f"✅ ID потоков на выходе: {output_flow_ids}")
    
    # Проверяем размерности
    expected_shape = (1, config.input_embedding_dim)
    correct_shape = output_embeddings.shape == expected_shape
    print(f"✅ Правильная размерность выхода: {correct_shape} {output_embeddings.shape} vs {expected_shape}")
    
    # Проверяем что собрали правильные потоки (только те что на выходе)
    expected_output_flows = 3  # Первые 3 потока были на выходной стороне
    correct_count = len(output_flow_ids) == expected_output_flows
    print(f"✅ Правильное количество выходных потоков: {correct_count}")
    
    return lattice, output_embeddings

def test_weighted_averaging():
    """Тест взвешенного усреднения множественных потоков"""
    print("\n=== Тест взвешенного усреднения ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    output_z = config.lattice_depth - 1
    
    # Создаем несколько потоков в одной выходной клетке (10, 10)
    test_position = [10, 10, output_z]
    
    # Поток 1: молодой с высокой энергией
    energy1 = torch.ones(config.embedding_per_cell, device=device) * 2.0  # Высокая энергия
    flow_id1 = lattice._create_flow(
        torch.tensor(test_position, dtype=torch.float32, device=device),
        energy1
    )
    
    # Поток 2: старый с низкой энергией (должен иметь больший вес из-за возраста)
    energy2 = torch.ones(config.embedding_per_cell, device=device) * 0.5  # Низкая энергия
    flow_id2 = lattice._create_flow(
        torch.tensor(test_position, dtype=torch.float32, device=device),
        energy2
    )
    
    # Искусственно увеличиваем возраст второго потока
    lattice.active_flows[flow_id2].age = 10  # Старый поток
    
    # Поток 3: средний поток
    energy3 = torch.ones(config.embedding_per_cell, device=device) * 1.0
    flow_id3 = lattice._create_flow(
        torch.tensor(test_position, dtype=torch.float32, device=device),
        energy3
    )
    lattice.active_flows[flow_id3].age = 5
    
    print(f"✅ Создано 3 потока в клетке ({test_position[0]}, {test_position[1]}):")
    print(f"   Поток {flow_id1}: энергия=2.0, возраст=0")
    print(f"   Поток {flow_id2}: энергия=0.5, возраст=10") 
    print(f"   Поток {flow_id3}: энергия=1.0, возраст=5")
    
    # Вычисляем ожидаемые веса
    weight1 = 2.0 * (1 + 0 * 0.1)   # = 2.0
    weight2 = 0.5 * (1 + 10 * 0.1)  # = 1.0  
    weight3 = 1.0 * (1 + 5 * 0.1)   # = 1.5
    total_weight = weight1 + weight2 + weight3  # = 4.5
    
    print(f"📊 Ожидаемые веса: {weight1:.1f}, {weight2:.1f}, {weight3:.1f} (сумма: {total_weight:.1f})")
    
    # Собираем выходную энергию
    output_embeddings, flow_ids = lattice.collect_output_energy()
    
    print(f"✅ Собрано потоков: {len(flow_ids)}")
    print(f"✅ ID потоков: {flow_ids}")
    print(f"✅ Размер выходного эмбеддинга: {output_embeddings.shape}")
    
    # Проверяем что взвешенное усреднение работает
    weighted_avg_applied = len(flow_ids) == 3
    correct_output_shape = output_embeddings.shape == (1, config.lattice_width * config.lattice_height)
    
    print(f"✅ Взвешенное усреднение применено: {weighted_avg_applied}")
    print(f"✅ Правильная размерность выхода: {correct_output_shape}")
    
    return lattice, output_embeddings

def test_statistics_and_cleanup():
    """Тест статистики и очистки"""
    print("\n=== Тест статистики и очистки ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # Создаем несколько потоков
    for i in range(5):
        position = torch.tensor([i, i, i], dtype=torch.float32, device=device)
        energy = torch.randn(config.embedding_per_cell, device=device)
        flow_id = lattice._create_flow(position, energy)
        
        # Деактивируем некоторые
        if i % 2 == 0:
            lattice.deactivate_flow(flow_id, "test")
    
    print(f"✅ Создано 5 потоков, деактивировано 3")
    
    # Статистика до очистки
    stats_before = lattice.get_statistics()
    print(f"📊 Статистика до очистки: {stats_before}")
    
    # Очистка неактивных потоков
    lattice._cleanup_inactive_flows()
    
    # Статистика после очистки
    stats_after = lattice.get_statistics()
    print(f"📊 Статистика после очистки: {stats_after}")
    
    # Полный сброс
    lattice.reset()
    stats_reset = lattice.get_statistics()
    print(f"📊 Статистика после сброса: {stats_reset}")
    
    # Проверяем что сброс работает
    reset_successful = (
        len(lattice.active_flows) == 0 and
        lattice.next_flow_id == 0 and
        stats_reset['current_active'] == 0
    )
    print(f"✅ Сброс выполнен правильно: {reset_successful}")
    
    return stats_before, stats_after, stats_reset

def test_buffered_flow_collection():
    """Тест новой буферизованной системы сбора потоков"""
    print("\n=== Тест буферизованной системы сбора ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # 1. Создаем потоки вручную
    test_flows = []
    for i in range(5):
        position = torch.tensor([i*2, i*3, 0], dtype=torch.float32, device=device)
        energy = torch.randn(config.embedding_per_cell, device=device) * (i + 1)  # Разная энергия
        flow_id = lattice._create_flow(position, energy)
        test_flows.append(flow_id)
    
    print(f"🌱 Создано {len(test_flows)} тестовых потоков")
    
    # 2. Перемещаем потоки к выходу
    for i, flow_id in enumerate(test_flows):
        flow = lattice.active_flows[flow_id]
        new_position = flow.position.clone()
        
        if i < 2:
            # Потоки 0,1: точно на выходной стороне (z = depth-1)
            new_position[2] = config.lattice_depth - 1
        elif i < 4:
            # Потоки 2,3: вышли за пределы (z > depth-1)
            new_position[2] = config.lattice_depth + 2
        else:
            # Поток 4: не дошел до выхода
            new_position[2] = config.lattice_depth - 3
        
        # Устанавливаем возраст для тестирования взвешенного усреднения
        lattice.active_flows[flow_id].age = i * 2
        
        lattice.update_flow(flow_id, new_position, flow.energy, flow.hidden_state)
    
    # 3. Проверяем буфер
    buffered_count = lattice.get_buffered_flows_count()
    print(f"📦 Потоков в буфере: {buffered_count}")
    print(f"📦 Клеток с потоками в буфере: {len(lattice.output_buffer)}")
    
    # 4. Собираем энергию из буфера
    output_embeddings, collected_ids = lattice.collect_buffered_energy()
    
    print(f"🏁 Собрано из буфера: {len(collected_ids)} потоков")
    print(f"🏁 ID собранных потоков: {collected_ids}")
    print(f"📦 Размер выходного эмбеддинга: {output_embeddings.shape}")
    
    # 5. Проверяем что потоки 0,1,2,3 собраны (4 не должен быть)
    expected_collected = [0, 1, 2, 3]  # Потоки которые достигли выхода
    collected_correctly = set(collected_ids) == set(expected_collected)
    print(f"✅ Собраны правильные потоки: {collected_correctly}")
    
    # 6. Проверяем что буфер автоматически НЕ очищается (для FlowProcessor)
    buffered_after = lattice.get_buffered_flows_count()
    buffer_persists = buffered_after == buffered_count  # Буфер должен остаться
    print(f"📦 Буфер сохранился для повторного использования: {buffer_persists}")
    
    # 7. Очищаем буфер вручную
    lattice.clear_output_buffer()
    buffered_after_clear = lattice.get_buffered_flows_count()
    print(f"🧹 Буфер очищен: {buffered_after_clear == 0}")
    
    return output_embeddings, collected_ids

def test_energy_flow_lifecycle():
    """Тест полного жизненного цикла потока"""
    print("\n=== Тест жизненного цикла потока ===")
    
    config = create_debug_config()
    set_energy_config(config)
    lattice = create_energy_lattice(config)
    
    device = lattice.device
    
    # 1. Размещаем входную энергию
    input_embedding = torch.randn(1, config.input_embedding_dim, device=device)
    initial_flow_ids = lattice.place_initial_energy(input_embedding)
    
    print(f"🌱 Размещено {len(initial_flow_ids)} начальных потоков")
    
    # 2. Симулируем движение потоков к выходу
    active_flows = lattice.get_active_flows()
    moved_flow_ids = []
    
    # Тестируем разные сценарии выхода
    for i, flow in enumerate(active_flows[:5]):  # Берем первые 5 для разных тестов
        new_position = flow.position.clone()
        
        if i < 2:
            # Потоки 0,1: точно на выходной стороне (z = depth-1)
            new_position[2] = config.lattice_depth - 1
        elif i < 4:
            # Потоки 2,3: вышли за пределы (z > depth-1) - должны быть скорректированы
            new_position[2] = config.lattice_depth + 2  # Выходим за пределы
        else:
            # Поток 4: почти на выходе, но не дошел
            new_position[2] = config.lattice_depth - 2
        
        lattice.update_flow(
            flow.id,
            new_position,
            flow.energy,
            flow.hidden_state
        )
        moved_flow_ids.append(flow.id)
    
    print(f"🚀 Переместили {len(moved_flow_ids)} потоков: 2 на выход, 2 за пределы, 1 почти на выходе")
    
    # Проверяем позиции перемещенных потоков
    print(f"🔍 Проверяем позиции перемещенных потоков:")
    for flow_id in moved_flow_ids:
        if flow_id in lattice.active_flows:
            flow = lattice.active_flows[flow_id]
            z_pos = flow.position[2].item()
            print(f"   Поток {flow_id}: z={z_pos:.1f}, активен={flow.is_active}")
    
    # 3. Собираем выходную энергию
    output_embeddings, completed_flow_ids = lattice.collect_output_energy()
    
    print(f"🏁 Собрали энергию от {len(completed_flow_ids)} потоков")
    print(f"📦 Выходной эмбеддинг: {output_embeddings.shape}")
    
    # Проверяем детали сбора
    expected_collected = 4  # 2 на выходе + 2 за пределами = 4 потока
    print(f"🔍 Ожидали собрать: {expected_collected} потоков (2 на выходе + 2 за пределами)")
    print(f"🔍 ID собранных потоков: {completed_flow_ids}")
    
    # 4. Финальная статистика
    final_stats = lattice.get_statistics()
    print(f"📊 Финальная статистика: {final_stats}")
    
    # Проверяем что жизненный цикл завершился корректно
    lifecycle_success = (
        len(completed_flow_ids) >= expected_collected and  # Собрали нужные потоки
        output_embeddings.shape[0] == 1 and               # Правильный batch size
        output_embeddings.shape[1] == lattice.width * lattice.height  # Размер выходной решетки
    )
    print(f"✅ Жизненный цикл завершен успешно: {lifecycle_success}")
    
    # Дополнительные проверки логики
    print(f"🔍 Проверки логики:")
    print(f"   - Размер выходного эмбеддинга: {output_embeddings.shape} (ожидается: [1, {lattice.width * lattice.height}])")
    print(f"   - Потоки собраны: {len(completed_flow_ids)}/{expected_collected}")
    print(f"   - Активных потоков осталось: {final_stats['current_active']}")
    
    return output_embeddings, final_stats

def main():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов для EnergyLattice")
    print("=" * 50)
    
    try:
        # Базовые тесты
        lattice = test_energy_lattice_init()
        lattice, flow_ids = test_place_initial_energy()
        
        # Функциональные тесты
        lattice = test_flow_management()
        lattice, parent_id, child_ids = test_spawn_flows()
        lattice, output_emb = test_collect_output_energy()
        lattice, weighted_output = test_weighted_averaging()
        stats = test_statistics_and_cleanup()
        
        # Тесты буферизованной системы
        buffer_output, buffer_ids = test_buffered_flow_collection()
        
        # Интеграционный тест
        final_output, final_stats = test_energy_flow_lifecycle()
        
        print("\n" + "=" * 50)
        print("✅ Все тесты EnergyLattice прошли успешно!")
        print(f"📊 Устройство: {lattice.device}")
        print(f"📊 Размер решетки: {lattice.width}x{lattice.height}x{lattice.depth}")
        print(f"📊 Максимум потоков: {lattice.max_active_flows}")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)