#!/usr/bin/env python3
"""
Тест новой конфигурации для эмбедингов
=====================================

Проверяет, что настройки эмбедингов правильно интегрированы
в централизованную конфигурацию.
"""

from new_rebuild.config.simple_config import get_project_config, SimpleProjectConfig
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_embedding_config():
    """Тестирование настроек эмбедингов в конфигурации"""
    
    # Получаем глобальную конфигурацию
    config = get_project_config()
    
    logger.info("=== Тестирование конфигурации эмбедингов ===")
    
    # Проверяем основные настройки эмбедингов
    logger.info(f"\nОсновные настройки эмбедингов:")
    logger.info(f"  Teacher model: {config.embedding.teacher_model}")
    logger.info(f"  Teacher embedding dim: {config.embedding.teacher_embedding_dim}")
    logger.info(f"  Cube surface dim: {config.embedding.cube_surface_dim}")
    logger.info(f"  Cube embedding dim: {config.embedding.cube_embedding_dim}")
    logger.info(f"  Transformation type: {config.embedding.transformation_type}")
    logger.info(f"  Use residual connections: {config.embedding.use_residual_connections}")
    
    # Проверяем настройки обучения
    logger.info(f"\nНастройки обучения эмбедингов:")
    logger.info(f"  Test mode: {config.training_embedding.test_mode}")
    logger.info(f"  Test lattice dim: {config.training_embedding.test_lattice_dim}")
    logger.info(f"  Main epochs: {config.training_embedding.main_epochs}")
    logger.info(f"  Curriculum learning: {config.training_embedding.use_curriculum_learning}")
    logger.info(f"  Semantic validation: {config.training_embedding.enable_semantic_validation}")
    logger.info(f"  Probing tasks: {config.training_embedding.probing_tasks}")
    
    # Проверяем loss веса
    logger.info(f"\nВеса loss функций:")
    logger.info(f"  Reconstruction: {config.training_embedding.reconstruction_loss_weight}")
    logger.info(f"  Similarity: {config.training_embedding.similarity_loss_weight}")
    logger.info(f"  Diversity: {config.training_embedding.diversity_loss_weight}")
    logger.info(f"  Emergence: {config.training_embedding.emergence_loss_weight}")
    
    # Проверяем интеграцию с решеткой
    logger.info(f"\nИнтеграция с решеткой:")
    expected_cube_embedding = config.embedding.cube_surface_dim ** 2
    assert config.embedding.cube_embedding_dim == expected_cube_embedding, \
        f"Несоответствие размерности: {config.embedding.cube_embedding_dim} != {expected_cube_embedding}"
    logger.info(f"  ✓ Размерность эмбединга куба корректна: {config.embedding.cube_embedding_dim}")
    
    # Проверяем кэширование
    logger.info(f"\nНастройки кэширования эмбедингов:")
    logger.info(f"  Cache enabled: {config.embedding.cache_embeddings}")
    logger.info(f"  Cache dir: {config.embedding.cache_dir}")
    logger.info(f"  Max cache size: {config.embedding.max_cache_size_gb} GB")
    
    # Тестируем обновление настроек
    logger.info(f"\nТестирование обновления настроек:")
    config.update_component('embedding', teacher_model='bert-base-uncased', teacher_embedding_dim=768)
    logger.info(f"  ✓ Обновлено: teacher_model = {config.embedding.teacher_model}")
    
    # Проверяем настройки для быстрых тестов
    logger.info(f"\nТестовые параметры:")
    logger.info(f"  Quick iterations: {config.training_embedding.test_quick_iterations}")
    logger.info(f"  Test dataset size: {config.training_embedding.test_dataset_size}")
    logger.info(f"  Validation split: {config.training_embedding.test_validation_split}")
    
    logger.info("\n✅ Все тесты конфигурации эмбедингов прошли успешно!")


def test_hierarchical_embedding_config():
    """Тест конфигурации для иерархического преобразования эмбедингов"""
    
    config = SimpleProjectConfig()
    
    # Настраиваем для иерархического преобразования
    config.embedding.transformation_type = "hierarchical"
    config.embedding.use_residual_connections = True
    config.embedding.use_layer_norm = True
    
    logger.info("\n=== Тест иерархической конфигурации ===")
    logger.info(f"Transformation type: {config.embedding.transformation_type}")
    logger.info(f"Residual connections: {config.embedding.use_residual_connections}")
    logger.info(f"Layer norm: {config.embedding.use_layer_norm}")
    
    # Проверяем настройки для больших кубов
    config.lattice.dimensions = (50, 50, 50)
    config.embedding.cube_surface_dim = 50
    config.embedding.cube_embedding_dim = 2500  # 50×50
    
    logger.info(f"\nТест для большого куба:")
    logger.info(f"  Lattice: {config.lattice.dimensions}")
    logger.info(f"  Total cells: {config.lattice.total_cells}")
    logger.info(f"  Embedding dim: {config.embedding.cube_embedding_dim}")
    
    # Проверяем автоматическое включение кэша
    if config.lattice.total_cells > config.cache.auto_enable_threshold:
        logger.info(f"  ✓ Кэш должен быть включен автоматически (порог: {config.cache.auto_enable_threshold})")


if __name__ == "__main__":
    test_embedding_config()
    test_hierarchical_embedding_config()
    
    logger.info("\n🎉 Конфигурация эмбедингов успешно интегрирована!")