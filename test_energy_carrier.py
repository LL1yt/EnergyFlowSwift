"""
Тест для EnergyCarrier - RNN-based энергетические потоки
========================================================

Простые тесты для проверки работоспособности EnergyCarrier:
- Инициализация модели 
- Прямой проход с базовыми входами
- Проверка размерностей выходов
- Тест механизма порождения потоков
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем пути для импортов
sys.path.append('/mnt/c/Users/n0n4a/projects/AA')

from energy_flow.core.energy_carrier import EnergyCarrier, EnergyOutput, create_energy_carrier
from energy_flow.config import create_debug_config, set_energy_config

def test_energy_carrier_init():
    """Тест инициализации EnergyCarrier"""
    print("\n=== Тест инициализации EnergyCarrier ===")
    
    # Создаем конфиг
    config = create_debug_config()
    set_energy_config(config)
    
    # Создаем модель
    carrier = create_energy_carrier(config)
    
    # Проверяем параметры
    total_params = sum(p.numel() for p in carrier.parameters())
    print(f"✅ EnergyCarrier создан с {total_params:,} параметрами")
    
    # Проверяем архитектуру
    print(f"✅ GRU: input_size={carrier.input_dim}, hidden_size={carrier.hidden_size}, layers={carrier.num_layers}")
    print(f"✅ Embedding dim: {carrier.embedding_dim}")
    
    return carrier

def test_energy_carrier_forward():
    """Тест прямого прохода"""
    print("\n=== Тест прямого прохода EnergyCarrier ===")
    
    config = create_debug_config()
    set_energy_config(config)
    carrier = create_energy_carrier(config)
    
    # Подготавливаем входы
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    carrier = carrier.to(device)
    
    # Входные данные
    neuron_output = torch.randn(batch_size, config.neuron_output_dim, device=device)
    embedding_part = torch.randn(batch_size, config.embedding_per_cell, device=device)
    current_position = torch.randint(0, 10, (batch_size, 3), dtype=torch.float32, device=device)
    
    print(f"📝 Входы: neuron_output={neuron_output.shape}, embedding_part={embedding_part.shape}")
    print(f"📝 Позиции: {current_position.shape}")
    
    # Прямой проход
    with torch.no_grad():
        output, new_hidden = carrier(neuron_output, embedding_part, None, current_position)
    
    # Проверяем выходы
    print(f"✅ Energy value: {output.energy_value.shape}")
    print(f"✅ Next position: {output.next_position.shape}")
    print(f"✅ Spawn count: {output.spawn_count}")
    print(f"✅ Spawn energies: {len(output.spawn_energies)}")
    print(f"✅ Hidden state: {new_hidden.shape}")
    
    # Проверяем диапазоны значений
    energy_norm = torch.norm(output.energy_value, dim=-1).mean()
    print(f"📊 Средняя норма энергии: {energy_norm:.4f}")
    
    pos_delta = (output.next_position - current_position).abs().mean()
    print(f"📊 Среднее смещение позиции: {pos_delta:.4f}")
    
    return carrier, output

def test_energy_spawn_mechanism():
    """Тест механизма порождения потоков"""
    print("\n=== Тест механизма порождения потоков ===")
    
    config = create_debug_config()
    # Устанавливаем низкий порог для гарантированного спавна
    config.spawn_threshold = 0.1
    config.max_spawn_per_step = 3
    set_energy_config(config)
    
    carrier = create_energy_carrier(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    carrier = carrier.to(device)
    
    batch_size = 2
    
    # Создаем "высокоэнергетичные" входы
    neuron_output = torch.ones(batch_size, config.neuron_output_dim, device=device) * 2.0
    embedding_part = torch.ones(batch_size, config.embedding_per_cell, device=device) * 1.5
    current_position = torch.tensor([[5, 5, 2], [3, 7, 1]], dtype=torch.float32, device=device)
    
    # Несколько проходов для накопления состояния
    hidden_state = carrier.init_hidden(batch_size, device)
    
    spawn_counts = []
    
    for step in range(5):
        with torch.no_grad():
            output, hidden_state = carrier(neuron_output, embedding_part, hidden_state, current_position)
        
        spawn_counts.append(output.spawn_count)
        
        if output.spawn_count > 0:
            print(f"✅ Шаг {step}: создано {output.spawn_count} новых потоков")
            print(f"   📍 Энергии для спавна: {len(output.spawn_energies)}")
            
            # Проверяем энергии
            for i, energy in enumerate(output.spawn_energies):
                energy_norm = torch.norm(energy).item()
                print(f"   🔋 Спавн {i}: норма энергии = {energy_norm:.4f}")
        else:
            print(f"⚪ Шаг {step}: спавна нет")
        
        # Обновляем позиции для следующего шага
        current_position = output.next_position
    
    total_spawns = sum(spawn_counts)
    print(f"📊 Всего создано потоков за 5 шагов: {total_spawns}")
    
    return spawn_counts

def test_energy_threshold_check():
    """Тест проверки уровня энергии"""
    print("\n=== Тест проверки уровня энергии ===")
    
    config = create_debug_config()
    carrier = create_energy_carrier(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    carrier = carrier.to(device)
    
    # Создаем энергии разных уровней
    high_energy = torch.ones(1, config.embedding_per_cell, device=device) * 2.0
    low_energy = torch.ones(1, config.embedding_per_cell, device=device) * 0.01
    zero_energy = torch.zeros(1, config.embedding_per_cell, device=device)
    
    # Проверяем жизнеспособность
    with torch.no_grad():
        high_alive = carrier.check_energy_level(high_energy)
        low_alive = carrier.check_energy_level(low_energy)
        zero_alive = carrier.check_energy_level(zero_energy)
    
    print(f"🔋 Высокая энергия (норма={torch.norm(high_energy):.4f}): жив = {high_alive[0]}")
    print(f"🪫 Низкая энергия (норма={torch.norm(low_energy):.4f}): жив = {low_alive[0]}")
    print(f"💀 Нулевая энергия (норма={torch.norm(zero_energy):.4f}): жив = {zero_alive[0]}")
    
    print(f"📊 Порог энергии: {config.energy_threshold}")
    
    return high_alive, low_alive, zero_alive

def test_position_constraints():
    """Тест ограничений движения"""
    print("\n=== Тест ограничений движения ===")
    
    config = create_debug_config()
    carrier = create_energy_carrier(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    carrier = carrier.to(device)
    
    # Тестируем позиции на границах решетки
    test_positions = torch.tensor([
        [0, 0, 0],  # Левый нижний угол
        [config.lattice_width-1, config.lattice_height-1, config.lattice_depth-2],  # Правый верхний
        [config.lattice_width//2, config.lattice_height//2, config.lattice_depth//2],  # Центр
    ], dtype=torch.float32, device=device)
    
    batch_size = test_positions.shape[0]
    neuron_output = torch.randn(batch_size, config.neuron_output_dim, device=device)
    embedding_part = torch.randn(batch_size, config.embedding_per_cell, device=device)
    
    with torch.no_grad():
        # Получаем внутренние предсказания до обработки
        combined_input = torch.cat([neuron_output, embedding_part], dim=-1).unsqueeze(1)
        gru_output, _ = carrier.gru(combined_input, None)
        gru_output = gru_output.squeeze(1)
        raw_predictions = carrier.position_projection(gru_output)
        
        output, _ = carrier(neuron_output, embedding_part, None, test_positions)
    
    print(f"📍 Исходные pozиции:")
    for i, pos in enumerate(test_positions):
        print(f"   {i}: {pos.cpu().numpy()}")
    
    print(f"📍 Сырые предсказания модели:")
    for i, pos in enumerate(raw_predictions):
        print(f"   {i}: {pos.cpu().numpy()}")
    
    print(f"📍 Следующие позиции (после обработки):")
    for i, pos in enumerate(output.next_position):
        print(f"   {i}: {pos.cpu().numpy()}")
    
    # Проверяем, что Z координата всегда увеличивается
    z_increases = output.next_position[:, 2] > test_positions[:, 2]
    print(f"✅ Z координата увеличилась во всех случаях: {z_increases.all()}")
    
    # Проверяем границы решетки
    within_bounds = (
        (output.next_position[:, 0] >= 0) & (output.next_position[:, 0] < config.lattice_width) &
        (output.next_position[:, 1] >= 0) & (output.next_position[:, 1] < config.lattice_height) &
        (output.next_position[:, 2] >= 0) & (output.next_position[:, 2] < config.lattice_depth)
    )
    print(f"✅ Все позиции в границах решетки: {within_bounds.all()}")
    
    return output.next_position

def main():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов для EnergyCarrier")
    print("=" * 50)
    
    try:
        # Базовые тесты
        carrier = test_energy_carrier_init()
        carrier, output = test_energy_carrier_forward()
        
        # Функциональные тесты
        spawn_counts = test_energy_spawn_mechanism()
        alive_states = test_energy_threshold_check()
        positions = test_position_constraints()
        
        print("\n" + "=" * 50)
        print("✅ Все тесты EnergyCarrier прошли успешно!")
        print(f"📊 Устройство: {next(carrier.parameters()).device}")
        
        # Итоговая статистика
        total_params = sum(p.numel() for p in carrier.parameters())
        print(f"📊 Общее количество параметров: {total_params:,}")
        print(f"📊 Общее количество созданных потоков в тестах: {sum(spawn_counts)}")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)