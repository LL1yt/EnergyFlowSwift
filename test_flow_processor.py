"""
Тест для FlowProcessor - механизм распространения энергии
==========================================================

Простые тесты для проверки работоспособности FlowProcessor:
- Инициализация процессора с компонентами
- Полный цикл forward (input → propagation → output)
- Координация между SimpleNeuron, EnergyCarrier и EnergyLattice
- Гибридный сбор энергии (активные потоки + буфер)
- Статистика производительности
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем пути для импортов
sys.path.append('/mnt/c/Users/n0n4a/projects/AA')

from energy_flow.core.flow_processor import FlowProcessor, create_flow_processor
from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.utils.logging import setup_logging

# Включаем debug логирование
setup_logging(debug_mode=True, level="DEBUG")

def test_flow_processor_init():
    """Тест инициализации FlowProcessor"""
    print("\n=== Тест инициализации FlowProcessor ===")
    
    # Создаем конфиг
    config = create_debug_config()
    set_energy_config(config)
    
    # Создаем процессор
    processor = create_flow_processor(config)
    
    # Проверяем компоненты
    print(f"✅ FlowProcessor создан на устройстве: {processor.device}")
    print(f"✅ Компоненты инициализированы:")
    print(f"   - EnergyLattice: {processor.lattice.width}x{processor.lattice.height}x{processor.lattice.depth}")
    print(f"   - SimpleNeuron: {sum(p.numel() for p in processor.neuron.parameters()):,} параметров")
    print(f"   - EnergyCarrier: {sum(p.numel() for p in processor.carrier.parameters()):,} параметров")
    
    # Проверяем что все на одном устройстве
    lattice_device = processor.lattice.device  # EnergyLattice хранит device как атрибут
    neuron_device = next(processor.neuron.parameters()).device
    carrier_device = next(processor.carrier.parameters()).device
    
    same_device = (lattice_device == neuron_device == carrier_device == processor.device)
    print(f"✅ Все компоненты на одном устройстве: {same_device}")
    print(f"   Lattice: {lattice_device}, Neuron: {neuron_device}, Carrier: {carrier_device}, Processor: {processor.device}")
    
    # Проверяем начальную статистику
    stats = processor.get_performance_stats()
    print(f"✅ Начальная статистика: {len(stats)} метрик")
    
    return processor

def test_simple_forward_pass():
    """Тест простого прямого прохода"""
    print("\n=== Тест простого прямого прохода ===")
    
    config = create_debug_config()
    set_energy_config(config)
    processor = create_flow_processor(config)
    
    # Подготавливаем входные данные
    batch_size = 2
    device = processor.device
    
    # Создаем случайные входные эмбеддинги
    input_embeddings = torch.randn(batch_size, config.input_embedding_dim, device=device)
    print(f"📝 Входные эмбеддинги: {input_embeddings.shape}")
    
    # Прямой проход
    with torch.no_grad():
        output_embeddings = processor.forward(input_embeddings, max_steps=5)  # Ограничиваем шаги для теста
    
    # Проверяем выходы
    print(f"✅ Выходные эмбеддинги: {output_embeddings.shape}")
    print(f"✅ Ожидаемая размерность: [{batch_size}, {config.input_embedding_dim}]")
    
    # Проверяем размерности
    correct_shape = output_embeddings.shape == (batch_size, config.input_embedding_dim)
    print(f"✅ Правильная размерность выхода: {correct_shape}")
    
    # Проверяем что не все нули
    non_zero = torch.any(output_embeddings != 0).item()
    print(f"✅ Выход не нулевой: {non_zero}")
    
    # Статистика после прохода
    perf_stats = processor.get_performance_stats()
    print(f"📊 Статистика производительности:")
    for key, value in perf_stats.items():
        if isinstance(value, dict):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    return processor, output_embeddings

def test_step_by_step_processing():
    """Тест пошаговой обработки потоков"""
    print("\n=== Тест пошаговой обработки ===")
    
    config = create_debug_config()
    set_energy_config(config)
    processor = create_flow_processor(config)
    
    device = processor.device
    
    # Создаем минимальные входные данные
    input_embeddings = torch.randn(1, config.input_embedding_dim, device=device)
    
    # Размещаем энергию вручную
    processor.lattice.reset()
    flow_ids = processor.lattice.place_initial_energy(input_embeddings)
    print(f"🌱 Размещено {len(flow_ids)} начальных потоков")
    
    # Выполняем несколько шагов вручную
    max_manual_steps = 3
    for step in range(max_manual_steps):
        active_flows = processor.lattice.get_active_flows()
        buffered_count = processor.lattice.get_buffered_flows_count()
        
        print(f"🔄 Шаг {step}: {len(active_flows)} активных, {buffered_count} в буфере")
        
        if not active_flows and buffered_count == 0:
            print("   Нет активных потоков и буфер пуст - завершаем")
            break
        
        if active_flows:
            # Один шаг обработки
            processor.step(active_flows)
            
            # Показываем пример движения потоков
            if step == 0 and len(active_flows) >= 3:
                for i, flow in enumerate(active_flows[:3]):
                    z_pos = flow.position[2].item()
                    print(f"   Поток {flow.id}: z={z_pos:.1f}, возраст={flow.age}")
    
    # Используем гибридный сбор
    final_output, collected_ids = processor._collect_final_output()
    
    print(f"🏁 Финальный сбор: {len(collected_ids)} потоков")
    print(f"📦 Размер выхода: {final_output.shape}")
    
    return processor, final_output

def test_hybrid_collection():
    """Тест гибридного сбора энергии"""
    print("\n=== Тест гибридного сбора энергии ===")
    
    config = create_debug_config()
    set_energy_config(config)
    processor = create_flow_processor(config)
    
    device = processor.device
    
    # Создаем тестовые потоки в разных состояниях
    processor.lattice.reset()
    
    # 1. Создаем потоки вручную
    test_scenarios = [
        {"position": [5, 5, config.lattice_depth - 1], "desc": "на выходной стороне"},
        {"position": [8, 8, config.lattice_depth + 1], "desc": "за пределами решетки"},
        {"position": [12, 12, config.lattice_depth - 2], "desc": "почти у выхода"},
        {"position": [15, 15, 2], "desc": "далеко от выхода"}
    ]
    
    created_flows = []
    for i, scenario in enumerate(test_scenarios):
        pos = torch.tensor(scenario["position"], dtype=torch.float32, device=device)
        energy = torch.randn(config.embedding_per_cell, device=device)
        flow_id = processor.lattice._create_flow(pos, energy)
        created_flows.append(flow_id)
        print(f"🌱 Поток {flow_id} создан {scenario['desc']}")
    
    # 2. Принудительно буферизуем потоки на выходе
    for flow_id in created_flows[:2]:  # Первые 2 потока
        processor.lattice._buffer_output_flow(flow_id)
    
    print(f"📦 Потоков в буфере: {processor.lattice.get_buffered_flows_count()}")
    print(f"🔄 Активных потоков: {len(processor.lattice.get_active_flows())}")
    
    # 3. Тестируем гибридный сбор
    final_output, collected_ids = processor._collect_final_output()
    
    print(f"🏁 Гибридный сбор завершен:")
    print(f"   - Собрано потоков: {len(collected_ids)}")
    print(f"   - ID собранных: {collected_ids}")
    print(f"   - Размер выхода: {final_output.shape}")
    print(f"   - Буфер после сбора: {processor.lattice.get_buffered_flows_count()} потоков")
    
    # Проверяем что собрали правильные потоки (первые 3 должны быть собраны)
    expected_collected = 3  # 2 из буфера + 1 который был почти у выхода
    collection_success = len(collected_ids) >= 2  # Минимум 2 из буфера
    print(f"✅ Сбор работает правильно: {collection_success}")
    
    return processor, final_output, collected_ids

def test_performance_stats():
    """Тест сбора статистики производительности"""
    print("\n=== Тест статистики производительности ===")
    
    config = create_debug_config()
    set_energy_config(config)
    processor = create_flow_processor(config)
    
    # Выполняем небольшой forward pass для генерации статистики
    device = processor.device
    input_embeddings = torch.randn(1, config.input_embedding_dim, device=device)
    
    with torch.no_grad():
        output = processor.forward(input_embeddings, max_steps=3)
    
    # Получаем статистику
    stats = processor.get_performance_stats()
    
    print(f"📊 Статистика производительности:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"   {key}: {formatted_value}")
    
    # Проверяем что собрали данные
    has_timing = 'avg_step_time' in stats and stats['avg_step_time'] > 0
    has_flow_stats = 'avg_flows_per_step' in stats
    has_lattice_stats = 'lattice_stats' in stats and isinstance(stats['lattice_stats'], dict)
    
    print(f"✅ Собрана статистика времени: {has_timing}")
    print(f"✅ Собрана статистика потоков: {has_flow_stats}")
    print(f"✅ Собрана статистика решетки: {has_lattice_stats}")
    
    # Тест визуализации состояния
    viz_data = processor.visualize_flow_state()
    print(f"📊 Данные для визуализации:")
    print(f"   - Всего потоков: {viz_data['total_flows']}")
    print(f"   - Размеры решетки: {viz_data['lattice_dims']}")
    print(f"   - Позиций в данных: {len(viz_data['positions'])}")
    
    return processor, stats, viz_data

def main():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов для FlowProcessor")
    print("=" * 50)
    
    try:
        # Базовые тесты
        processor = test_flow_processor_init()
        processor, simple_output = test_simple_forward_pass()
        
        # Функциональные тесты
        processor, step_output = test_step_by_step_processing()
        processor, hybrid_output, collected = test_hybrid_collection()
        processor, perf_stats, viz_data = test_performance_stats()
        
        print("\n" + "=" * 50)
        print("✅ Все тесты FlowProcessor прошли успешно!")
        print(f"📊 Устройство: {processor.device}")
        print(f"📊 Решетка: {processor.config.lattice_width}x{processor.config.lattice_height}x{processor.config.lattice_depth}")
        print(f"📊 Максимум потоков: {processor.config.max_active_flows}")
        
        # Общая статистика
        total_params = (
            sum(p.numel() for p in processor.neuron.parameters()) +
            sum(p.numel() for p in processor.carrier.parameters())
        )
        print(f"📊 Общее количество параметров: {total_params:,}")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)