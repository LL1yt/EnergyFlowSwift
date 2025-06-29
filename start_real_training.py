#!/usr/bin/env python3
"""
Скрипт для запуска реального обучения 3D Cellular Neural Network
Использует unified dataset loader и оптимизированную конфигурацию для 8x8x8 куба
"""

import torch
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from unified_dataset_loader import create_training_dataloader, DatasetConfig
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.config.config_components import (
    LatticeSettings, ModelSettings, TrainingSettings, TrainingEmbeddingSettings,
    EmbeddingSettings, DeviceSettings, LoggingSettings, CacheSettings,
    SpatialSettings, MemorySettings
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_real_training_config() -> SimpleProjectConfig:
    """Создает оптимальную конфигурацию для реального обучения на 8x8x8 кубе"""
    
    config = SimpleProjectConfig(
        # 8x8x8 куб для начала (512 клеток)
        lattice=LatticeSettings(
            dimensions=(8, 8, 8),
            face_placement_strategy="random"
        ),
        
        # Модель оптимизирована для RTX 5090
        model=ModelSettings(
            state_size=64,           # Достаточно для emergent behavior
            hidden_dim=128,          # Увеличиваем для лучшего обучения
            neighbor_count=6         # 3D соседи (6-connectivity)
        ),
        
        # Агрессивные параметры обучения для быстрой конвергенции
        training=TrainingSettings(
            learning_rate=0.001,     # Conservative start
            batch_size=16,           # Увеличено для лучшей статистики
            optimizer_type="adamw",  # AdamW для стабильности
            weight_decay=0.01,
            grad_clip_norm=1.0,
            scheduler_type="cosine_annealing",
            warmup_steps=100
        ),
        
        # Специализированные настройки для embedding training
        training_embedding=TrainingEmbeddingSettings(
            test_mode=False,                 # Реальное обучение
            num_epochs=50,                   # Достаточно для первых экспериментов
            validation_interval=1,           # Валидация каждую эпоху
            save_checkpoint_every=5,         # Checkpoint каждые 5 эпох
            log_interval=10,                 # Лог каждые 10 батчей
            early_stopping_patience=10,      # Early stopping через 10 эпох без улучшения
            
            # Loss weights (начальные значения из плана)
            reconstruction_weight=1.0,       # Основная задача
            similarity_weight=0.5,           # Семантическая похожесть
            diversity_weight=0.2,            # Разнообразие представлений
            emergence_weight=0.1,            # Emergent behavior
            
            # Специфичные параметры
            target_embedding_dim=64,         # Сжимаем 768 → 64 для куба 8x8x8
            teacher_model="distilbert-base-uncased",
            use_teacher_forcing=True,
            lattice_steps=5                  # Количество шагов распространения
        ),
        
        # Настройки эмбеддингов
        embedding=EmbeddingSettings(
            input_dim=768,                   # DistilBERT dimension
            output_dim=64,                   # Для куба 8x8x8
            use_projection=True,
            projection_layers=[768, 256, 64],
            dropout_rate=0.1
        ),
        
        # Максимально используем RTX 5090
        device=DeviceSettings(
            prefer_cuda=True,
            force_device=None,               # Автоопределение
            debug_mode=False,
            mixed_precision=True             # Включаем для ускорения
        ),
        
        # Оптимизация кэширования
        cache=CacheSettings(
            use_gpu_cache=True,
            gpu_cache_size_mb=4096,          # 4GB cache для RTX 5090
            use_connection_cache=True,
            cache_directory="cache/real_training",
            auto_cleanup=True
        ),
        
        # Пространственная оптимизация
        spatial=SpatialSettings(
            use_spatial_optimization=True,
            chunk_size=64,                   # Оптимальный размер для 8x8x8
            overlap_size=8,
            use_gpu_spatial_hashing=True,
            morton_encoding=True
        ),
        
        # Управление памятью для больших датасетов
        memory=MemorySettings(
            max_memory_usage_gb=24,          # Оставляем запас на RTX 5090
            memory_cleanup_interval=100,
            use_memory_mapping=True,
            cache_embedding_results=True
        ),
        
        # Продвинутое логирование
        logging=LoggingSettings(
            level=logging.INFO,
            enable_caller_tracking=True,
            enable_anti_duplication=True,
            log_performance_metrics=True,
            save_logs_to_file=True,
            log_directory="logs/real_training"
        )
    )
    
    logger.info("✅ Real training configuration created for 8x8x8 cube")
    return config


def setup_experiment_tracking(experiment_name: str) -> Path:
    """Настраивает отслеживание эксперимента"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем папки для различных артефактов
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "metrics").mkdir(exist_ok=True)
    (experiment_dir / "config").mkdir(exist_ok=True)
    
    logger.info(f"📊 Experiment tracking setup: {experiment_dir}")
    return experiment_dir


def save_experiment_config(config: SimpleProjectConfig, experiment_dir: Path):
    """Сохраняет конфигурацию эксперимента"""
    config_file = experiment_dir / "config" / "training_config.json"
    
    # Конвертируем config в сериализуемый формат
    config_dict = {
        "lattice": {
            "dimensions": config.lattice.dimensions,
            "face_placement_strategy": config.lattice.face_placement_strategy
        },
        "model": {
            "state_size": config.model.state_size,
            "hidden_dim": config.model.hidden_dim,
            "neighbor_count": config.model.neighbor_count
        },
        "training": {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "optimizer_type": config.training.optimizer_type
        },
        "training_embedding": {
            "num_epochs": config.training_embedding.num_epochs,
            "reconstruction_weight": config.training_embedding.reconstruction_weight,
            "similarity_weight": config.training_embedding.similarity_weight,
            "diversity_weight": config.training_embedding.diversity_weight,
            "emergence_weight": config.training_embedding.emergence_weight,
            "target_embedding_dim": config.training_embedding.target_embedding_dim,
            "lattice_steps": config.training_embedding.lattice_steps
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"💾 Configuration saved: {config_file}")


def create_dataset_for_training() -> tuple:
    """Создает датасет для обучения с оптимальными параметрами"""
    
    dataset_config = DatasetConfig(
        use_dialogue_cache=True,
        use_prepared_embeddings=True,
        use_cache_embeddings=True,
        use_snli_generator=False,           # Отключаем для первого запуска
        max_samples_per_source=1000,        # Ограничиваем для быстрого старта
        shuffle_sources=True,
        embedding_dim=768,
        min_embedding_norm=0.1,
        max_embedding_norm=50.0
    )
    
    dataloader, stats = create_training_dataloader(
        config=dataset_config,
        batch_size=16,
        shuffle=True,
        num_workers=2  # Параллельная загрузка данных
    )
    
    logger.info(f"📊 Dataset created: {stats['total_samples']} samples")
    return dataloader, stats


def run_training_epoch(trainer: EmbeddingTrainer, dataloader, epoch: int, experiment_dir: Path) -> Dict[str, float]:
    """Запускает одну эпоху обучения с детальным логированием"""
    
    logger.info(f"\n🚀 Starting Epoch {epoch + 1}")
    epoch_start_time = time.time()
    
    # Training phase
    trainer.model.train()
    train_losses = {"total": 0.0, "reconstruction": 0.0, "similarity": 0.0, "diversity": 0.0, "emergence": 0.0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        # Извлекаем эмбеддинги из батча
        embeddings = batch['embedding']  # [batch_size, 768]
        
        # Прогоняем через trainer
        try:
            losses = trainer.train_step(embeddings)
            
            # Обновляем статистику
            for key in train_losses.keys():
                if key in losses:
                    train_losses[key] += losses[key]
            
            num_batches += 1
            
            # Логируем каждые 10 батчей
            if (batch_idx + 1) % 10 == 0:
                batch_time = time.time() - batch_start_time
                current_loss = losses.get('total', 0.0)
                logger.info(f"  Batch {batch_idx + 1:3d}: loss={current_loss:.6f}, time={batch_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Усредняем loss'ы
    if num_batches > 0:
        for key in train_losses.keys():
            train_losses[key] /= num_batches
    
    epoch_time = time.time() - epoch_start_time
    
    logger.info(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
    logger.info(f"   Average losses: {', '.join([f'{k}={v:.6f}' for k, v in train_losses.items()])}")
    
    # Сохраняем метрики
    metrics_file = experiment_dir / "metrics" / f"epoch_{epoch + 1}_metrics.json"
    metrics = {
        "epoch": epoch + 1,
        "train_losses": train_losses,
        "epoch_time": epoch_time,
        "num_batches": num_batches,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return train_losses


def main():
    """Главная функция для запуска реального обучения"""
    
    print("🚀 STARTING REAL 3D CELLULAR NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Настройка эксперимента
    experiment_name = "first_8x8x8_real_training"
    experiment_dir = setup_experiment_tracking(experiment_name)
    
    # Создание конфигурации
    logger.info("⚙️ Creating training configuration...")
    config = create_real_training_config()
    save_experiment_config(config, experiment_dir)
    
    # Создание датасета
    logger.info("📂 Loading unified dataset...")
    dataloader, dataset_stats = create_dataset_for_training()
    
    # Сохраняем статистику датасета
    with open(experiment_dir / "config" / "dataset_stats.json", 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Создание trainer'а
    logger.info("🧠 Initializing EmbeddingTrainer...")
    trainer = EmbeddingTrainer(config)
    
    # Основной цикл обучения
    logger.info(f"🎯 Starting training for {config.training_embedding.num_epochs} epochs...")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training_embedding.num_epochs):
        
        # Обучение
        train_losses = run_training_epoch(trainer, dataloader, epoch, experiment_dir)
        current_loss = train_losses['total']
        
        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            
            # Сохраняем лучшую модель
            best_model_path = experiment_dir / "checkpoints" / "best_model.pth"
            trainer.save_checkpoint(str(best_model_path), epoch=epoch + 1, loss=current_loss)
            logger.info(f"💾 New best model saved: loss={current_loss:.6f}")
            
        else:
            patience_counter += 1
            logger.info(f"⚠️ No improvement for {patience_counter} epochs")
        
        # Регулярные checkpoint'ы
        if (epoch + 1) % config.training_embedding.save_checkpoint_every == 0:
            checkpoint_path = experiment_dir / "checkpoints" / f"epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch=epoch + 1, loss=current_loss)
            logger.info(f"💾 Regular checkpoint saved: epoch_{epoch + 1}.pth")
        
        # Early stopping
        if patience_counter >= config.training_embedding.early_stopping_patience:
            logger.info(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Финальное сохранение
    final_model_path = experiment_dir / "checkpoints" / "final_model.pth"
    trainer.save_checkpoint(str(final_model_path), epoch=epoch + 1, loss=current_loss)
    
    # Сводка эксперимента
    summary = {
        "experiment_name": experiment_name,
        "total_epochs": epoch + 1,
        "best_loss": best_loss,
        "final_loss": current_loss,
        "dataset_samples": dataset_stats['total_samples'],
        "config_summary": {
            "lattice_size": config.lattice.dimensions,
            "state_size": config.model.state_size,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate
        },
        "completion_time": datetime.now().isoformat()
    }
    
    with open(experiment_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 Experiment results saved to: {experiment_dir}")
    print(f"🏆 Best loss achieved: {best_loss:.6f}")
    print(f"📈 Total samples processed: {dataset_stats['total_samples']}")
    print(f"\n🚀 Ready for analysis and next steps!")


if __name__ == "__main__":
    main()