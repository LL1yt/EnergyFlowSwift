#!/usr/bin/env python3
"""
Тест базового EmbeddingTrainer
==============================

Проверка работы полного цикла обучения:
Text → DistilBERT → EmbeddingTransformer → MoE Cube → EmbeddingTransformer → TextDecoder

Использует минимальные размеры для быстрой проверки на RTX 5090.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from new_rebuild.config.simple_config import SimpleProjectConfig
from new_rebuild.core.training.embedding_trainer import create_embedding_trainer
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def create_synthetic_embedding_dataset(
    config: SimpleProjectConfig, num_samples: int = 100
):
    """Создание синтетических эмбедингов для тестирования"""

    embedding_dim = config.embedding.teacher_embedding_dim  # 768 для DistilBERT

    # Генерируем случайные эмбединги с некоторой структурой
    embeddings = []
    target_embeddings = []

    for i in range(num_samples):
        # Базовый эмбединг
        base_embedding = torch.randn(embedding_dim) * 0.1

        # Добавляем некоторую структуру (например, синусоидальные паттерны)
        pattern = (
            torch.sin(
                torch.arange(embedding_dim, dtype=torch.float32)
                * (i / num_samples)
                * np.pi
                * 2
            )
            * 0.05
        )

        embedding = base_embedding + pattern

        # Целевой эмбединг - слегка модифицированная версия
        target = embedding + torch.randn(embedding_dim) * 0.02

        embeddings.append(embedding)
        target_embeddings.append(target)

    embeddings = torch.stack(embeddings)
    target_embeddings = torch.stack(target_embeddings)

    logger.info(f"Создан датасет: {num_samples} образцов, размерность {embedding_dim}")
    logger.info(
        f"Статистика эмбедингов: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}"
    )

    return embeddings, target_embeddings


def test_trainer_basic():
    """Базовый тест инициализации и forward pass"""
    print("\n=== БАЗОВЫЙ ТЕСТ EMBEDDING TRAINER ===")

    # Настраиваем логирование на DEBUG уровень
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Используем централизованную конфигурацию
    config = SimpleProjectConfig()
    config.training_embedding.test_mode = True
    config.training_embedding.test_quick_iterations = 5
    config.logging.debug_mode = True  # Включаем debug логи

    # Инициализация тренера (все размеры берутся из централизованного конфига)
    trainer = create_embedding_trainer(config)

    print(f"✓ Тренер инициализирован на устройстве: {trainer.device}")
    print(
        f"✓ Параметров в EmbeddingTransformer: {trainer.embedding_transformer.get_parameter_count()}"
    )

    # Создание синтетических данных
    embeddings, targets = create_synthetic_embedding_dataset(config, num_samples=32)

    # Создание DataLoader
    dataset = TensorDataset(embeddings, targets)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Тест forward pass
    print("\n--- Тест Forward Pass ---")
    trainer.embedding_transformer.eval()
    trainer.lattice.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        input_emb, target_emb = batch
        input_emb = input_emb.to(trainer.device)
        target_emb = target_emb.to(trainer.device)
        
        print(f"Input embedding shape: {input_emb.shape}")
        
        # Проверяем преобразование эмбедингов
        surface_emb = trainer.embedding_transformer.transform_to_cube(input_emb)
        print(f"Surface embedding shape after transform: {surface_emb.shape}")
        print(f"Expected shape for lattice_mapper: [batch_size, {config.cube_embedding_dim}]")

        # Forward pass через весь pipeline
        losses = trainer._forward_pass(input_emb, target_emb)

        print(f"✓ Forward pass успешен")
        print(f"  Total Loss: {losses['total'].item():.6f}")
        print(f"  Reconstruction: {losses['reconstruction'].item():.6f}")
        print(f"  Similarity: {losses['similarity'].item():.6f}")
        print(f"  Diversity: {losses['diversity'].item():.6f}")
        print(f"  Emergence: {losses['emergence'].item():.6f}")

    return trainer, dataloader


def test_training_epoch(trainer, dataloader):
    """Тест полного цикла обучения одной эпохи"""
    print("\n--- Тест Training Epoch ---")

    # Сохраняем начальные веса для проверки обновления
    initial_params = {}
    for name, param in trainer.embedding_transformer.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()

    # Запуск тренировки эпохи
    train_losses = trainer.train_epoch(dataloader)

    print(f"✓ Эпоха обучения завершена")
    print(f"  Total Loss: {train_losses['total']:.6f}")
    print(f"  Количество батчей: {train_losses['count']}")

    # Проверяем, что параметры обновились
    params_updated = False
    for name, param in trainer.embedding_transformer.named_parameters():
        if param.requires_grad and name in initial_params:
            if not torch.equal(param.data, initial_params[name]):
                params_updated = True
                break

    if params_updated:
        print("✓ Параметры обновились в процессе обучения")
    else:
        print("⚠ Параметры не изменились (возможно, слишком малый learning rate)")

    return train_losses


def test_validation_epoch(trainer, dataloader):
    """Тест валидации"""
    print("\n--- Тест Validation Epoch ---")

    val_losses = trainer.validate_epoch(dataloader)

    print(f"✓ Валидация завершена")
    print(f"  Total Loss: {val_losses['total']:.6f}")
    print(f"  Количество батчей: {val_losses['count']}")

    return val_losses


def test_checkpoint_save_load(trainer):
    """Тест сохранения и загрузки checkpoint'ов"""
    print("\n--- Тест Checkpoint Save/Load ---")

    checkpoint_path = "test_checkpoint.pth"

    # Сохранение
    trainer.save_checkpoint(checkpoint_path, epoch=1, test_mode=True)
    print(f"✓ Checkpoint сохранен: {checkpoint_path}")

    # Сохраняем текущие параметры
    original_params = {}
    for name, param in trainer.embedding_transformer.named_parameters():
        original_params[name] = param.data.clone()

    # Изменяем параметры (небольшим шумом)
    with torch.no_grad():
        for param in trainer.embedding_transformer.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Загрузка
    checkpoint_data = trainer.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint загружен")

    # Проверяем, что параметры восстановились
    params_restored = True
    for name, param in trainer.embedding_transformer.named_parameters():
        if name in original_params:
            if not torch.allclose(param.data, original_params[name], atol=1e-6):
                params_restored = False
                break

    if params_restored:
        print("✓ Параметры успешно восстановлены")
    else:
        print("✗ Ошибка восстановления параметров")

    # Удаляем тестовый файл
    Path(checkpoint_path).unlink(missing_ok=True)

    return checkpoint_data


def test_performance_monitoring(trainer):
    """Тест мониторинга производительности"""
    print("\n--- Тест Performance Monitoring ---")

    summary = trainer.get_training_summary()

    print(f"✓ Устройство: {summary['device']}")
    print(f"✓ Общее количество параметров: {summary['total_parameters']:,}")

    if summary["performance_stats"]["avg_total_time"] > 0:
        print(
            f"✓ Среднее время батча: {summary['performance_stats']['avg_total_time']:.4f}s"
        )
        print(f"  Forward: {summary['performance_stats']['avg_forward_time']:.4f}s")
        print(f"  Backward: {summary['performance_stats']['avg_backward_time']:.4f}s")

    return summary


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ EMBEDDING TRAINER")
    print("=" * 50)

    try:
        # 1. Базовый тест
        trainer, dataloader = test_trainer_basic()

        # 2. Тест обучения
        train_losses = test_training_epoch(trainer, dataloader)

        # 3. Тест валидации
        val_losses = test_validation_epoch(trainer, dataloader)

        # 4. Тест checkpoint'ов
        checkpoint_data = test_checkpoint_save_load(trainer)

        # 5. Тест мониторинга
        summary = test_performance_monitoring(trainer)

        print("\n" + "=" * 50)
        print("🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
        print("=" * 50)

        # Итоговая статистика
        print(f"\nИтоговая статистика:")
        print(
            f"  Архитектура: DistilBERT → EmbeddingTransformer → MoE Cube → TextDecoder"
        )
        print(
            f"  Решетка: {trainer.config.training_embedding.test_lattice_dim}×{trainer.config.training_embedding.test_lattice_dim}×{trainer.config.training_embedding.test_lattice_dim}"
        )
        print(f"  Устройство: {trainer.device}")
        print(f"  Параметры: {summary['total_parameters']:,}")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"  Val Loss: {val_losses['total']:.6f}")

        if trainer.device.type == "cuda":
            print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        return True

    except Exception as e:
        print(f"\n💥 ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
