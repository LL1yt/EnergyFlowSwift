#!/usr/bin/env python3
"""
Упрощенный скрипт для реального обучения 3D Cellular Neural Network
==================================================================

Использует ТОЛЬКО центральную конфигурацию из new_rebuild.config
Следует принципам CLAUDE.md - никаких локальных конфигураций
"""

import torch
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def convert_tensors_to_json(obj):
    """Convert PyTorch tensors to JSON-serializable values."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_json(item) for item in obj]
    else:
        return obj


from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.core.training.utils import create_training_dataloader
from new_rebuild.utils.logging import setup_logging, get_logger

# === ИСПОЛЬЗУЕМ ТОЛЬКО ЦЕНТРАЛЬНУЮ КОНФИГУРАЦИЮ ===
config = SimpleProjectConfig()

# Initialize centralized logging system
setup_logging(
    debug_mode=config.logging.debug_mode,
    level=config.logging.level,  # ИСПРАВЛЕНО: Передаем level из конфигурации
    log_file=(
        config.logging.log_file
        if getattr(config.logging, "log_to_file", False)
        else None
    ),
    enable_context=True,
)

logger = get_logger(__name__)
logger.info("⚙️ Loading central configuration...")


def setup_experiment_tracking(experiment_name: str) -> Path:
    """Настраивает отслеживание эксперимента"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Создаем папки для различных артефактов
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "metrics").mkdir(exist_ok=True)

    logger.info(f"📊 Experiment tracking setup: {experiment_dir}")
    return experiment_dir


def run_training_epoch(
    trainer: EmbeddingTrainer, dataloader, epoch: int, experiment_dir: Path
) -> Dict[str, float]:
    """Запускает одну эпоху обучения"""

    logger.info(f"\n🚀 Starting Epoch {epoch + 1}")
    epoch_start_time = time.time()

    # Use trainer's train_epoch method directly
    train_losses = trainer.train_epoch(dataloader)

    epoch_time = time.time() - epoch_start_time

    logger.info(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
    logger.info(
        f"   Average losses: {', '.join([f'{k}={v:.6f}' for k, v in train_losses.items()])}"
    )

    # Сохраняем метрики
    metrics_file = experiment_dir / "metrics" / f"epoch_{epoch + 1}_metrics.json"
    metrics_file.parent.mkdir(exist_ok=True)

    metrics = {
        "epoch": epoch + 1,
        "train_losses": convert_tensors_to_json(train_losses),
        "epoch_time": epoch_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return train_losses


def main():
    """Главная функция для запуска реального обучения"""

    logger.info("🚀 STARTING REAL 3D CELLULAR NEURAL NETWORK TRAINING")
    logger.info("Using CENTRAL CONFIG ONLY (new_rebuild.config)")
    logger.info("=" * 60)

    # Проверяем что включен режим реального обучения
    if config.training_embedding.test_mode:
        logger.warning(
            "⚠️ test_mode=True in config! Switch to real training mode in config_components.py"
        )
        logger.error("\n❌ CONFIGURATION ERROR:")
        logger.error("test_mode=True in central config!")
        logger.error("Edit new_rebuild/config/config_components.py:")
        logger.error("  Change: test_mode: bool = False")
        return

    logger.info("✅ Real training mode enabled")
    logger.info(f"📏 Lattice size: {config.lattice.dimensions}")
    logger.info(
        f"🎯 Target embedding dim: {config.training_embedding.target_embedding_dim}"
    )
    logger.info(f"📊 Epochs: {config.training_embedding.num_epochs}")
    logger.info(f"🔥 Batch size: {config.training_embedding.embedding_batch_size}")

    # Настройка эксперимента
    experiment_name = f"real_training_{config.lattice.dimensions[0]}x{config.lattice.dimensions[1]}x{config.lattice.dimensions[2]}"
    experiment_dir = setup_experiment_tracking(experiment_name)

    # Создание датасета (используем настройки из конфига)
    logger.info("📂 Loading unified dataset...")
    # Для прогоночного обучения используем только 658 сэмплов (из dialogue cache)
    max_samples = config.training_embedding.test_dataset_size

    dataloader, dataset_stats = create_training_dataloader(config=config, shuffle=True)

    logger.info(f"📊 Dataset loaded: {dataset_stats.total_samples} total samples")

    # Сохраняем статистику датасета
    with open(experiment_dir / "dataset_stats.json", "w") as f:
        stats_dict = {
            "total_samples": dataset_stats.total_samples,
            "embedding_dim": dataset_stats.embedding_dim,
            "source_distribution": dataset_stats.source_distribution,
            "type_distribution": dataset_stats.type_distribution,
        }
        json.dump(stats_dict, f, indent=2)

    # Создание trainer'а
    logger.info("🧠 Initializing EmbeddingTrainer...")
    trainer = EmbeddingTrainer(config)

    # Основной цикл обучения
    num_epochs = config.training_embedding.num_epochs
    logger.info(f"🎯 Starting training for {num_epochs} epochs...")

    best_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = config.training_embedding.early_stopping_patience

    for epoch in range(num_epochs):

        # Обучение
        train_losses = run_training_epoch(trainer, dataloader, epoch, experiment_dir)
        current_loss = train_losses["total"]

        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0

            # Сохраняем лучшую модель
            best_model_path = experiment_dir / "checkpoints" / "best_model.pth"
            trainer.save_checkpoint(
                str(best_model_path), epoch=epoch + 1, loss=current_loss
            )
            logger.info(f"💾 New best model saved: loss={current_loss:.6f}")

        else:
            patience_counter += 1
            logger.info(f"⚠️ No improvement for {patience_counter} epochs")

        # Регулярные checkpoint'ы (используем настройку из конфига)
        save_interval = config.training_embedding.save_checkpoint_every
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = experiment_dir / "checkpoints" / f"epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(
                str(checkpoint_path), epoch=epoch + 1, loss=current_loss
            )
            logger.info(f"💾 Regular checkpoint saved: epoch_{epoch + 1}.pth")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(
                f"🛑 Early stopping triggered after {patience_counter} epochs without improvement"
            )
            break

    # Финальное сохранение
    final_model_path = experiment_dir / "checkpoints" / "final_model.pth"
    trainer.save_checkpoint(str(final_model_path), epoch=epoch + 1, loss=current_loss)

    # Сводка эксперимента
    summary = {
        "experiment_name": experiment_name,
        "total_epochs": epoch + 1,
        "best_loss": convert_tensors_to_json(best_loss),
        "final_loss": convert_tensors_to_json(current_loss),
        "dataset_samples": dataset_stats.total_samples,
        "config_summary": {
            "lattice_size": config.lattice.dimensions,
            "state_size": config.model.state_size,
            "batch_size": config.training_embedding.embedding_batch_size,
            "learning_rate": config.training.learning_rate,
            "test_mode": config.training_embedding.test_mode,
        },
        "completion_time": datetime.now().isoformat(),
    }

    with open(experiment_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n🎉 TRAINING COMPLETED!")
    logger.info(f"📊 Experiment results saved to: {experiment_dir}")
    logger.info(f"🏆 Best loss achieved: {best_loss:.6f}")
    logger.info(f"📈 Total samples processed: {dataset_stats.total_samples}")
    logger.info(f"\n🚀 Ready for analysis and next steps!")


if __name__ == "__main__":
    main()
