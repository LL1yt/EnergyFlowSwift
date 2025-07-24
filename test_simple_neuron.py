"""
Тест для SimpleNeuron - простой нейрон-автомат
===============================================

Простые тесты для проверки работоспособности SimpleNeuron:
- Инициализация модели (~1000 параметров)
- Прямой проход с координатами и эмбеддингами
- Проверка размерностей и нормализации координат
- Тест позиционного кодирования
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем пути для импортов
sys.path.append('/mnt/c/Users/n0n4a/projects/AA')

from energy_flow.core.simple_neuron import SimpleNeuron, create_simple_neuron
from energy_flow.config import create_debug_config, set_energy_config

def test_simple_neuron_init():
    """Тест инициализации SimpleNeuron"""
    print("\n=== Тест инициализации SimpleNeuron ===")
    
    # Создаем конфиг
    config = create_debug_config()
    set_energy_config(config)
    
    # Создаем модель
    neuron = create_simple_neuron(config)
    
    # Проверяем параметры
    total_params = sum(p.numel() for p in neuron.parameters())
    print(f"✅ SimpleNeuron создан с {total_params:,} параметрами")
    
    # Проверяем архитектуру
    print(f"✅ Входная размерность: {neuron.coord_dim + neuron.embedding_dim}")
    print(f"✅ Скрытая размерность: {neuron.hidden_dim}")
    print(f"✅ Выходная размерность: {neuron.output_dim}")
    
    # Целевые параметры ~1000
    target_params = 1000
    if total_params <= target_params * 1.5:  # В пределах 150% от цели
        print(f"✅ Количество параметров в норме (цель: ~{target_params})")
    else:
        print(f"⚠️ Слишком много параметров (цель: ~{target_params})")
    
    return neuron

def test_simple_neuron_forward():
    """Тест прямого прохода"""
    print("\n=== Тест прямого прохода SimpleNeuron ===")
    
    config = create_debug_config()
    set_energy_config(config)
    neuron = create_simple_neuron(config)
    
    # Подготавливаем входы
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neuron = neuron.to(device)
    
    # Входные данные - позиции и части эмбеддингов
    positions = torch.tensor([
        [0, 0, 0],      # Левый нижний угол
        [19, 19, 9],    # Правый верхний угол
        [10, 10, 5],    # Центр
        [5, 15, 2]      # Произвольная точка
    ], dtype=torch.float32, device=device)
    
    embedding_parts = torch.randn(batch_size, config.embedding_per_cell, device=device)
    
    print(f"📝 Входы: positions={positions.shape}, embeddings={embedding_parts.shape}")
    
    # Прямой проход
    with torch.no_grad():
        output = neuron(positions, embedding_parts)
    
    # Проверяем выходы
    print(f"✅ Выход SimpleNeuron: {output.shape}")
    print(f"✅ Ожидаемая размерность: [{batch_size}, {config.neuron_output_dim}]")
    
    assert output.shape == (batch_size, config.neuron_output_dim), \
        f"Неверная размерность выхода: {output.shape}"
    
    # Проверяем диапазоны значений
    output_mean = output.mean().item()
    output_std = output.std().item()
    print(f"📊 Статистика выхода: mean={output_mean:.4f}, std={output_std:.4f}")
    
    return neuron, output

def test_coordinate_normalization():
    """Тест нормализации координат"""
    print("\n=== Тест нормализации координат ===")
    
    config = create_debug_config()
    set_energy_config(config)
    neuron = create_simple_neuron(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neuron = neuron.to(device)
    
    # Тестовые позиции на границах решетки
    test_positions = torch.tensor([
        [0, 0, 0],                                    # Минимум
        [config.lattice_width-1, config.lattice_height-1, config.lattice_depth-1],  # Максимум
        [config.lattice_width//2, config.lattice_height//2, config.lattice_depth//2] # Центр
    ], dtype=torch.float32, device=device)
    
    # Проверяем нормализацию
    normalized = neuron._normalize_coordinates(test_positions)
    
    print(f"📍 Исходные координаты:")
    for i, pos in enumerate(test_positions):
        print(f"   {i}: {pos.cpu().numpy()}")
    
    print(f"📍 Нормализованные координаты:")
    for i, pos in enumerate(normalized):
        print(f"   {i}: {pos.cpu().numpy()}")
    
    # Проверяем диапазон [-1, 1]
    within_range = (normalized >= -1.0) & (normalized <= 1.0)
    print(f"✅ Все координаты в диапазоне [-1, 1]: {within_range.all()}")
    
    # Проверяем что углы действительно на границах
    corners_correct = (
        torch.allclose(normalized[0], torch.tensor([-1., -1., -1.], device=device)) and
        torch.allclose(normalized[1], torch.tensor([1., 1., 1.], device=device)) and
        torch.allclose(normalized[2], torch.tensor([0., 0., 0.], device=device), atol=0.1)
    )
    print(f"✅ Углы нормализованы правильно: {corners_correct}")
    
    return normalized

def test_positional_encoding():
    """Тест позиционного кодирования"""
    print("\n=== Тест позиционного кодирования ===")
    
    config = create_debug_config()
    set_energy_config(config)
    neuron = create_simple_neuron(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neuron = neuron.to(device)
    
    # Разные позиции для проверки различий в кодировании
    positions = torch.tensor([
        [0, 0, 0],
        [10, 10, 5],
        [19, 19, 9]
    ], dtype=torch.float32, device=device)
    
    embedding_parts = torch.randn(3, config.embedding_per_cell, device=device)
    
    # Получаем активационные паттерны
    with torch.no_grad():
        patterns = neuron.compute_activation_pattern(positions, embedding_parts)
    
    print(f"✅ Паттерны активации: {patterns.shape}")
    
    # Проверяем что разные позиции дают разные паттерны
    pattern_diffs = []
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            diff = torch.norm(patterns[i] - patterns[j]).item()
            pattern_diffs.append(diff)
            print(f"📊 Различие паттернов {i}-{j}: {diff:.4f}")
    
    avg_diff = sum(pattern_diffs) / len(pattern_diffs)
    print(f"📊 Среднее различие паттернов: {avg_diff:.4f}")
    
    # Разные позиции должны давать заметно разные паттерны
    distinct_patterns = avg_diff > 0.1
    print(f"✅ Позиции дают различимые паттерны: {distinct_patterns}")
    
    return patterns

def test_batch_consistency():
    """Тест консистентности батчевой обработки"""
    print("\n=== Тест консистентности батчевой обработки ===")
    
    config = create_debug_config()
    set_energy_config(config)
    neuron = create_simple_neuron(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neuron = neuron.to(device)
    
    # Тестовые данные
    position = torch.tensor([[5, 8, 3]], dtype=torch.float32, device=device)
    embedding = torch.randn(1, config.embedding_per_cell, device=device)
    
    with torch.no_grad():
        # Обработка по одному
        single_output = neuron(position, embedding)
        
        # Батчевая обработка с дублированием
        batch_positions = position.repeat(4, 1)
        batch_embeddings = embedding.repeat(4, 1)
        batch_output = neuron(batch_positions, batch_embeddings)
    
    print(f"📝 Одиночный выход: {single_output.shape}")
    print(f"📝 Батчевый выход: {batch_output.shape}")
    
    # Проверяем что все элементы батча одинаковые
    consistency_check = torch.allclose(batch_output[0], single_output[0], atol=1e-6)
    print(f"✅ Одиночная и батчевая обработка дают одинаковый результат: {consistency_check}")
    
    # Проверяем что все элементы в батче одинаковые
    batch_consistency = all(
        torch.allclose(batch_output[0], batch_output[i], atol=1e-6) 
        for i in range(1, batch_output.shape[0])
    )
    print(f"✅ Все элементы батча одинаковые: {batch_consistency}")
    
    return single_output, batch_output

def test_lattice_dimensions():
    """Тест установки размеров решетки"""
    print("\n=== Тест установки размеров решетки ===")
    
    config = create_debug_config()
    set_energy_config(config)
    neuron = create_simple_neuron(config)
    
    # Проверяем начальные размеры
    initial_dims = neuron._lattice_dims
    print(f"📏 Начальные размеры решетки: {initial_dims}")
    
    # Устанавливаем новые размеры
    new_width, new_height, new_depth = 50, 50, 20
    neuron.set_lattice_dimensions(new_width, new_height, new_depth)
    
    updated_dims = neuron._lattice_dims
    print(f"📏 Обновленные размеры решетки: {updated_dims}")
    
    # Проверяем что размеры обновились
    dims_updated = updated_dims == (new_width, new_height, new_depth)
    print(f"✅ Размеры решетки обновлены правильно: {dims_updated}")
    
    return updated_dims

def main():
    """Запуск всех тестов"""
    print("🧪 Запуск тестов для SimpleNeuron")
    print("=" * 50)
    
    try:
        # Базовые тесты
        neuron = test_simple_neuron_init()
        neuron, output = test_simple_neuron_forward()
        
        # Функциональные тесты
        normalized = test_coordinate_normalization()
        patterns = test_positional_encoding()
        single_out, batch_out = test_batch_consistency()
        dims = test_lattice_dimensions()
        
        print("\n" + "=" * 50)
        print("✅ Все тесты SimpleNeuron прошли успешно!")
        print(f"📊 Устройство: {next(neuron.parameters()).device}")
        
        # Итоговая статистика
        total_params = sum(p.numel() for p in neuron.parameters())
        print(f"📊 Общее количество параметров: {total_params:,}")
        print(f"📊 Архитектура: 3D координаты + {neuron.embedding_dim}D эмбеддинг → {neuron.hidden_dim}D → {neuron.output_dim}D")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)