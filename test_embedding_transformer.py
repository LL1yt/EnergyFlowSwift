#!/usr/bin/env python3
"""
Тест EmbeddingTransformer
========================

Проверяет работу преобразователя эмбедингов между Teacher моделью и кубом.
"""

import logging
import torch

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from new_rebuild.config.simple_config import get_project_config
from new_rebuild.core.common.embedding_transformer import (
    create_embedding_transformer, 
    test_embedding_transformer
)
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_basic_transformer():
    """Тест базового линейного преобразователя"""
    logger.info("=== Тест базового преобразователя ===")
    
    config = get_project_config()
    config.embedding.transformation_type = "linear"
    config.embedding.use_residual_connections = True
    
    transformer, metrics = test_embedding_transformer(config, batch_size=8)
    
    logger.info(f"Результаты базового преобразователя:")
    logger.info(f"  MSE Loss: {metrics['mse_loss']:.6f}")
    logger.info(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    logger.info(f"  Parameters: {metrics['parameter_count']:,}")
    
    return metrics


def test_hierarchical_transformer():
    """Тест иерархического преобразователя"""
    logger.info("\n=== Тест иерархического преобразователя ===")
    
    config = get_project_config()
    config.embedding.transformation_type = "hierarchical"
    config.embedding.use_residual_connections = True
    config.embedding.use_layer_norm = True
    
    transformer, metrics = test_embedding_transformer(config, batch_size=8)
    
    logger.info(f"Результаты иерархического преобразователя:")
    logger.info(f"  MSE Loss: {metrics['mse_loss']:.6f}")
    logger.info(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    logger.info(f"  Parameters: {metrics['parameter_count']:,}")
    
    return metrics


def test_different_cube_sizes():
    """Тест различных размеров кубов"""
    logger.info("\n=== Тест различных размеров кубов ===")
    
    cube_sizes = [27, 37, 50]  # 27×27=729, 37×37=1369, 50×50=2500
    results = {}
    
    for size in cube_sizes:
        logger.info(f"\nТестируем куб {size}×{size}×{size}:")
        
        config = get_project_config()
        config.embedding.cube_surface_dim = size
        config.embedding.cube_embedding_dim = size * size
        config.embedding.transformation_type = "hierarchical"
        
        transformer, metrics = test_embedding_transformer(config, batch_size=4)
        results[size] = metrics
        
        logger.info(f"  Compression ratio: {transformer.get_compression_ratio():.2f}")
    
    return results


def test_batch_sizes():
    """Тест различных размеров батчей"""
    logger.info("\n=== Тест различных размеров батчей ===")
    
    config = get_project_config()
    config.embedding.transformation_type = "hierarchical"
    
    transformer = create_embedding_transformer(config)
    
    batch_sizes = [1, 4, 16, 32]
    
    for batch_size in batch_sizes:
        logger.info(f"\nБатч размер: {batch_size}")
        
        # Создаем тестовые данные
        teacher_embeddings = torch.randn(batch_size, config.embedding.teacher_embedding_dim)
        
        # Тестируем преобразования
        cube_embeddings = transformer.transform_to_cube(teacher_embeddings)
        reconstructed = transformer.transform_from_cube(cube_embeddings)
        
        # Проверяем размерности
        assert teacher_embeddings.shape == reconstructed.shape
        assert cube_embeddings.shape == (batch_size, config.embedding.cube_surface_dim, 
                                       config.embedding.cube_surface_dim)
        
        mse = torch.nn.functional.mse_loss(reconstructed, teacher_embeddings)
        logger.info(f"  ✓ MSE: {mse:.6f}")


def compare_transformation_types():
    """Сравнение различных типов преобразований"""
    logger.info("\n=== Сравнение типов преобразований ===")
    
    types = ["linear", "hierarchical"]
    results = {}
    
    for transform_type in types:
        logger.info(f"\nТип: {transform_type}")
        
        config = get_project_config()
        config.embedding.transformation_type = transform_type
        config.embedding.use_residual_connections = True
        
        transformer, metrics = test_embedding_transformer(config, batch_size=16)
        results[transform_type] = metrics
    
    # Сравниваем результаты
    logger.info("\n📊 Сравнение результатов:")
    for transform_type, metrics in results.items():
        logger.info(f"  {transform_type}:")
        logger.info(f"    MSE: {metrics['mse_loss']:.6f}")
        logger.info(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        logger.info(f"    Parameters: {metrics['parameter_count']:,}")
    
    return results


if __name__ == "__main__":
    logger.info("🚀 Запуск тестов EmbeddingTransformer")
    
    # Запускаем все тесты
    basic_results = test_basic_transformer()
    hierarchical_results = test_hierarchical_transformer()
    cube_size_results = test_different_cube_sizes()
    test_batch_sizes()
    comparison_results = compare_transformation_types()
    
    logger.info("\n🎉 Все тесты EmbeddingTransformer завершены успешно!")
    
    # Выводим итоговую статистику
    logger.info("\n📈 Итоговая статистика:")
    logger.info(f"  Лучшая cosine similarity: {max(basic_results['cosine_similarity'], hierarchical_results['cosine_similarity']):.6f}")
    logger.info(f"  Наименьший MSE: {min(basic_results['mse_loss'], hierarchical_results['mse_loss']):.6f}")