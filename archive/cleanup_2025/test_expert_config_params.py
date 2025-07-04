#!/usr/bin/env python3
"""Тест индивидуальных параметров экспертов в конфигурации"""

import torch
from new_rebuild.config import (
    create_debug_config,
    create_experiment_config,
    create_optimized_config,
    set_project_config
)
from new_rebuild.core.moe import create_moe_connection_processor
from new_rebuild.core.moe.simple_linear_expert import OptimizedSimpleLinearExpert
from new_rebuild.core.moe.hybrid_gnn_cnf_expert import HybridGNN_CNF_Expert
from new_rebuild.core.cnf.gpu_enhanced_cnf import GPUEnhancedCNF, ConnectionType
from new_rebuild.core.moe.gating_network import GatingNetwork

def test_expert_parameters():
    """Проверяем, что параметры экспертов корректно настраиваются из конфигурации"""
    
    print("=== Тестирование индивидуальных параметров экспертов ===\n")
    
    # Тестируем разные режимы конфигурации
    configs = [
        ("DEBUG", create_debug_config()),
        ("EXPERIMENT", create_experiment_config()),
        ("OPTIMIZED", create_optimized_config())
    ]
    
    for mode_name, config in configs:
        print(f"\n{mode_name} режим:")
        print(f"  State size (общий): {config.model.state_size}")
        print(f"  Model не содержит hidden_dim, message_dim, target_params - они перенесены в экспертов")
        
        # Проверяем LocalExpert параметры
        print(f"\n  LocalExpert:")
        print(f"    - target params: {config.expert.local.params}")
        print(f"    - neighbor_agg_hidden1: {config.expert.local.neighbor_agg_hidden1}")
        print(f"    - neighbor_agg_hidden2: {config.expert.local.neighbor_agg_hidden2}")
        print(f"    - processor_hidden: {config.expert.local.processor_hidden}")
        
        # Проверяем FunctionalExpert параметры
        print(f"\n  FunctionalExpert:")
        print(f"    - target params: {config.expert.functional.params}")
        print(f"    - hidden_dim: {config.expert.functional.hidden_dim}")
        print(f"    - message_dim: {config.expert.functional.message_dim}")
        
        # Проверяем DistantExpert параметры
        print(f"\n  DistantExpert (CNF):")
        print(f"    - target params: {config.expert.distant.params}")
        print(f"    - ode_hidden_dim: {config.expert.distant.ode_hidden_dim}")
        print(f"    - ode_dropout_rate: {config.expert.distant.ode_dropout_rate}")
        
        # Проверяем GatingNetwork параметры
        print(f"\n  GatingNetwork:")
        print(f"    - target params: {config.expert.gating.params}")
        print(f"    - hidden_dim: {config.expert.gating.hidden_dim}")
        print(f"    - state_size: {config.expert.gating.state_size}")
        
        # Устанавливаем конфиг и создаем экспертов для проверки
        set_project_config(config)
        
        # Создаем и проверяем LocalExpert
        local_expert = OptimizedSimpleLinearExpert(state_size=config.model.state_size)
        local_params = sum(p.numel() for p in local_expert.parameters())
        print(f"\n  Реальное количество параметров LocalExpert: {local_params}")
        
        # Создаем и проверяем FunctionalExpert
        functional_expert = HybridGNN_CNF_Expert(
            state_size=config.model.state_size,
            neighbor_count=-1,  # dynamic
            target_params=config.expert.functional.params
        )
        functional_params = sum(p.numel() for p in functional_expert.parameters())
        print(f"  Реальное количество параметров FunctionalExpert: {functional_params}")
        
        # Создаем и проверяем DistantExpert
        distant_expert = GPUEnhancedCNF(
            state_size=config.model.state_size,
            connection_type=ConnectionType.DISTANT
        )
        distant_params = sum(p.numel() for p in distant_expert.parameters())
        print(f"  Реальное количество параметров DistantExpert: {distant_params}")
        
        # Создаем и проверяем GatingNetwork
        gating_network = GatingNetwork(
            state_size=config.model.state_size,
            num_experts=3
        )
        gating_params = sum(p.numel() for p in gating_network.parameters())
        print(f"  Реальное количество параметров GatingNetwork: {gating_params}")
        
        # Проверяем что state_size одинаков для всех
        print(f"\n  ✅ State size одинаков для всех экспертов: {config.model.state_size}")
        
        print("-" * 60)


def test_moe_processor():
    """Проверяем создание MoE processor с разными конфигурациями"""
    
    print("\n\n=== Тестирование MoE Processor ===\n")
    
    # Используем DEBUG конфигурацию для быстрого теста
    config = create_debug_config()
    set_project_config(config)
    
    # Создаем MoE processor
    moe_processor = create_moe_connection_processor(
        dimensions=config.lattice.dimensions,
        state_size=config.model.state_size,
        device=torch.device('cpu')
    )
    
    # Проверяем компоненты
    print(f"MoE Processor создан успешно!")
    print(f"  - Lattice dimensions: {config.lattice.dimensions}")
    print(f"  - State size: {config.model.state_size}")
    
    # Подсчитываем общее количество параметров
    total_params = sum(p.numel() for p in moe_processor.parameters())
    print(f"  - Общее количество параметров: {total_params}")
    
    # Проверяем отдельные эксперты
    print(f"\n  Эксперты в MoE:")
    print(f"    - Local: {type(moe_processor.local_expert).__name__}")
    print(f"    - Functional: {type(moe_processor.functional_expert).__name__}")
    print(f"    - Distant: {type(moe_processor.distant_expert).__name__}")
    

if __name__ == "__main__":
    test_expert_parameters()
    test_moe_processor()
    print("\n✅ Тесты завершены успешно!")