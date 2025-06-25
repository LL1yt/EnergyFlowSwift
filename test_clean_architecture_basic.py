#!/usr/bin/env python3
"""
Базовый тест Clean 3D Cellular Neural Network
============================================

Проверяем основные компоненты:
1. Новая модульная конфигурация работает
2. MoE архитектура (эксперты, gating) создается и выполняется
3. Параметры компонентов соответствуют целевым значениям в конфиге
"""

import torch
import logging
import sys
import os
import unittest

# Добавляем путь к new_rebuild
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from new_rebuild.config import (
    get_project_config,
    set_project_config,
    reset_global_config,
    ProjectConfig,
)
from new_rebuild.core.moe import (
    GatingNetwork,
    SimpleLinearExpert,
    HybridGNN_CNF_Expert,
    MoEConnectionProcessor,
)
from new_rebuild.core.cells import GNNCell  # GNNCell является базовой

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestCleanArchitecture(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом"""
        reset_global_config()
        config = ProjectConfig()
        config.logging.debug_mode = False
        set_project_config(config)
        logger.info(f"--- Запуск теста: {self._testMethodName} ---")

    def tearDown(self):
        """Очистка после каждого теста"""
        reset_global_config()
        logger.info(f"--- Завершение теста: {self._testMethodName} ---\n")

    def test_project_config_new_structure(self):
        """Тест новой модульной конфигурации"""
        logger.info("Тестируем новую модульную структуру ProjectConfig")
        config = get_project_config()

        self.assertTrue(hasattr(config, "lattice"))
        self.assertTrue(hasattr(config, "gnn"))
        self.assertTrue(hasattr(config, "expert"))
        self.assertTrue(hasattr(config.expert, "gating"))
        self.assertTrue(hasattr(config.expert, "local"))
        self.assertTrue(hasattr(config.expert, "functional"))
        self.assertTrue(hasattr(config.expert, "distant"))

        self.assertEqual(config.lattice.dimensions, (5, 5, 5))
        self.assertEqual(config.expert.gating.params, 808)
        self.assertEqual(config.expert.local.params, 2059)
        self.assertTrue(config.expert.enabled)

        logger.info("✅ Новая структура конфигурации работает корректно")

    def test_gnn_base_cell(self):
        """Тест базовой GNN клетки"""
        logger.info("Тестируем создание и работу базовой GNN клетки")
        config = get_project_config()
        cell = GNNCell(
            state_size=config.gnn.state_size,
            neighbor_count=config.neighbors.base_neighbor_count,
            message_dim=config.gnn.message_dim,
            hidden_dim=config.gnn.hidden_dim,
            external_input_size=config.gnn.external_input_size,
        )
        total_params = sum(p.numel() for p in cell.parameters())
        logger.info(f"✅ GNN клетка создана, параметры: {total_params}")

        # Forward pass
        batch_size = 2
        neighbor_states = torch.randn(
            batch_size, config.neighbors.base_neighbor_count, config.gnn.state_size
        )
        own_state = torch.randn(batch_size, config.gnn.state_size)
        external_input = torch.randn(batch_size, config.gnn.external_input_size)

        output = cell(neighbor_states, own_state, external_input)
        self.assertEqual(output.shape, (batch_size, config.gnn.state_size))
        logger.info("✅ Forward pass GNN клетки выполнен успешно")

    def test_moe_architecture_components(self):
        """Тест создания компонентов MoE архитектуры"""
        logger.info("Тестируем создание и параметры компонентов MoE")
        config = get_project_config()

        # Gating Network
        gating = GatingNetwork(state_size=config.gnn.state_size, num_experts=3)
        gating_params = sum(p.numel() for p in gating.parameters())
        self.assertAlmostEqual(gating_params, config.expert.gating.params, delta=100)
        logger.info(
            f"✅ GatingNetwork создан: {gating_params} параметров (цель: {config.expert.gating.params})"
        )

        # Local Expert
        local_expert = SimpleLinearExpert(state_size=config.gnn.state_size)
        local_params = sum(p.numel() for p in local_expert.parameters())
        self.assertAlmostEqual(local_params, config.expert.local.params, delta=150)
        logger.info(
            f"✅ LocalExpert создан: {local_params} параметров (цель: {config.expert.local.params})"
        )

        # Functional Expert
        functional_expert = HybridGNN_CNF_Expert(state_size=config.gnn.state_size)
        functional_params = sum(p.numel() for p in functional_expert.parameters())
        self.assertAlmostEqual(
            functional_params, config.expert.functional.params, delta=500
        )
        logger.info(
            f"✅ FunctionalExpert создан: {functional_params} параметров (цель: {config.expert.functional.params})"
        )

    def test_moe_processor_forward_pass(self):
        """Тест forward pass через MoE Connection Processor"""
        logger.info("Тестируем полный forward pass через MoE процессор")
        config = get_project_config()
        moe_processor = MoEConnectionProcessor(
            state_size=config.gnn.state_size,
            lattice_dimensions=config.lattice.dimensions,
        )
        moe_processor.to(config.current_device)

        batch_size = 1
        num_neighbors = 10
        state_size = config.gnn.state_size

        current_state = torch.randn(state_size).to(config.current_device)
        neighbor_states = torch.randn(num_neighbors, state_size).to(
            config.current_device
        )

        result = moe_processor(
            current_state=current_state,
            neighbor_states=neighbor_states,
            cell_idx=0,
            neighbor_indices=list(range(1, num_neighbors + 1)),
        )

        self.assertIn("new_state", result)
        self.assertEqual(result["new_state"].shape, (state_size,))
        self.assertIn("expert_weights", result)
        self.assertEqual(result["expert_weights"].shape, (3,))
        logger.info("✅ Forward pass MoE процессора выполнен успешно")


if __name__ == "__main__":
    logger.info("🚀 ЗАПУСК ТЕСТОВ CLEAN АРХИТЕКТУРЫ С НОВОЙ КОНФИГУРАЦИЕЙ")
    unittest.main()
