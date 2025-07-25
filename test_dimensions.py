#!/usr/bin/env python3
"""
Тест размерностей для энергетической архитектуры
===============================================

Проверяет правильность преобразований:
768D (DistilBERT) -> 400 скаляров -> обработка -> 768D (для сравнения с teacher)
"""

import torch
import sys
import os

# Default device будет установлен при импорте energy_config

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.core.embedding_mapper import EnergyFlowMapper
from energy_flow.core import FlowProcessor
from energy_flow.utils.logging import setup_logging

def test_full_dimensions_pipeline():
    """Тест полной цепочки размерностей"""
    # Включаем debug логирование для диагностики устройств
    setup_logging(debug_mode=True, debug_categories=['energy'])
    
    print("🧪 Тест размерностей energy_flow архитектуры")
    print("=" * 50)
    
    # Конфигурация
    config = create_debug_config()
    set_energy_config(config)
    
    print(f"📊 Конфигурация:")
    print(f"   Решетка: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    print(f"   Поверхность: {config.lattice_width * config.lattice_height} клеток")
    print(f"   Эмбеддинг учителя: {config.input_embedding_dim_from_teacher}D")
    
    # Создаем компоненты
    mapper = EnergyFlowMapper(config)
    processor = FlowProcessor(config)
    
    # Эмуляция входных данных от DistilBERT
    batch_size = 2
    teacher_embeddings = torch.randn(batch_size, config.input_embedding_dim_from_teacher)  # Автоматически на GPU
    
    print(f"\n🔄 Тестирование преобразований:")
    print(f"   Вход: {teacher_embeddings.shape}")
    
    # 1. Тест входного маппинга
    print("\n1️⃣ Входной маппинг (768D -> скаляры):")
    cell_energies = mapper.map_to_surface(teacher_embeddings)
    print(f"   Создано {len(cell_energies)} потоков энергии")
    
    # Проверяем, что каждый поток получил скалярную энергию
    sample_energy = cell_energies[0][1]  # (position, energy, batch_idx)
    print(f"   Размер энергии на клетку: {sample_energy.shape}")
    assert sample_energy.shape == torch.Size([1]), f"Энергия должна быть скалярной, получили {sample_energy.shape}"
    
    # 2. Тест полного прохода через FlowProcessor
    print("\n2️⃣ Полный проход через FlowProcessor:")
    try:
        output_embeddings = processor.forward(teacher_embeddings, max_steps=5)
        print(f"   Выход: {output_embeddings.shape}")
        assert output_embeddings.shape == teacher_embeddings.shape, \
            f"Размерности не совпадают: {output_embeddings.shape} != {teacher_embeddings.shape}"
        
        print("   ✅ Размерности согласованы!")
        
        # 3. Проверка диапазона значений
        print("\n3️⃣ Проверка диапазонов значений:")
        print(f"   Входной эмбеддинг: [{teacher_embeddings.min():.3f}, {teacher_embeddings.max():.3f}]")
        print(f"   Выходной эмбеддинг: [{output_embeddings.min():.3f}, {output_embeddings.max():.3f}]")
        
        # 4. Проверка готовности к сравнению с teacher'ом
        print("\n4️⃣ Готовность к сравнению с teacher'ом:")
        target_embeddings = torch.randn_like(teacher_embeddings)  # Эмуляция эмбеддинга-ответа
        
        # MSE Loss
        mse_loss = torch.nn.functional.mse_loss(output_embeddings, target_embeddings)
        print(f"   MSE Loss: {mse_loss.item():.6f}")
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            output_embeddings.view(-1), target_embeddings.view(-1), dim=0
        )
        print(f"   Cosine Similarity: {cos_sim.item():.6f}")
        
        print("\n🎉 Все тесты пройдены! Архитектура готова для обучения.")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_energy_thresholds():
    """Тест оптимизированных порогов энергии"""
    print("\n🔧 Тест энергетических порогов:")
    
    config = create_debug_config()
    print(f"   energy_threshold: {config.energy_threshold}")
    print(f"   spawn_threshold: {config.spawn_threshold}")
    print(f"   max_spawn_per_step: {config.max_spawn_per_step}")
    print(f"   carrier_dropout: {config.carrier_dropout}")
    
    # Создаем тестовые скалярные энергии в диапазоне [-1, 1]
    test_energies = torch.tensor([-0.9, -0.1, 0.0, 0.05, 0.1, 0.6, 0.8, 1.0])
    
    print("\n   Тест survival energy:")
    for energy in test_energies:
        survives = abs(energy.item()) > config.energy_threshold
        print(f"     Энергия {energy.item():+.2f}: {'✅ выживает' if survives else '❌ умирает'}")
    
    print("\n   Тест spawn energy:")
    spawn_probs = torch.sigmoid(test_energies)  # Эмуляция spawn probability
    for i, prob in enumerate(spawn_probs):
        spawns = prob.item() > config.spawn_threshold
        print(f"     Prob {prob.item():.3f}: {'🐣 spawn' if spawns else '⭕ no spawn'}")

if __name__ == "__main__":
    success = test_full_dimensions_pipeline()
    test_energy_thresholds()
    
    if success:
        print("\n🚀 Архитектура готова к созданию тренера!")
    else:
        print("\n🛠️ Требуется доработка архитектуры.")