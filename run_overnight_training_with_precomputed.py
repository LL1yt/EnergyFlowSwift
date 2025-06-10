#!/usr/bin/env python3
"""
OVERNIGHT TRAINING с предварительно вычисленными эмбеддингами
Использует большой датасет готовых эмбеддингов для эффективного обучения
"""

import torch
import torch.nn as nn
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os
import signal

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
)
from utils.config_manager.config_manager import ConfigManager
from precomputed_embedding_loader import (
    PrecomputedEmbeddingLoader,
    create_precomputed_dataset,
)
from model_weights_manager import ModelWeightsManager
from config_converter import convert_config_dict_to_object

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "logs/overnight_training_precomputed.log", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PrecomputedOvernightTrainer:
    """Overnight Trainer с предварительно вычисленными эмбеддингами"""

    def __init__(self, embeddings_file: str = None):
        self.trainer = None
        self.dataset = None
        self.config = None
        self.weights_manager = ModelWeightsManager()
        self.embedding_loader = PrecomputedEmbeddingLoader()
        self.embeddings_file = embeddings_file
        self.should_stop = False
        self.best_similarity = 0.0
        self.training_log = []

        # Обработчик сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("[START] PrecomputedOvernightTrainer initialized")

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful остановки"""
        logger.info(
            f"[SIGNAL] Received signal {signum}, stopping training gracefully..."
        )
        self.should_stop = True

    def auto_select_dataset(self) -> str:
        """Автоматический выбор самого нового датасета"""
        logger.info("[SEARCH] Auto-selecting dataset...")

        datasets = self.embedding_loader.list_available_datasets()

        if not datasets:
            raise FileNotFoundError(
                "No precomputed datasets found! "
                "Please run generate_large_embedding_dataset.py first."
            )

        # Выбираем самый новый и большой датасет
        latest_dataset = datasets[0]  # Уже отсортированы по времени

        logger.info(f"[LOAD] Selected dataset: {latest_dataset['filename']}")
        logger.info(f"   Size: {latest_dataset['size']:,} pairs")
        logger.info(f"   Teacher model: {latest_dataset['teacher_model']}")
        logger.info(f"   File size: {latest_dataset['file_size_mb']:.1f} MB")

        return latest_dataset["file_path"]

    def setup_training(self):
        """Настройка обучения"""
        logger.info("[SETUP] Setting up training components...")

        # 1. Автоматически выбираем датасет если не указан
        if self.embeddings_file is None:
            self.embeddings_file = self.auto_select_dataset()

        # 2. Загружаем конфигурацию
        config_manager = ConfigManager()
        config_dict = config_manager.get_config()
        self.config = convert_config_dict_to_object(config_dict)

        # 3. Создаем trainer
        self.trainer = EmergentCubeTrainer(self.config)
        self.trainer.to("cuda" if torch.cuda.is_available() else "cpu")

        # 4. Загружаем готовые эмбеддинги
        logger.info("[LOAD] Loading precomputed embeddings...")
        self.dataset = self.embedding_loader.load_dataset(self.embeddings_file)

        # 5. Проверяем данные
        sample = self.dataset[0]
        q_emb, a_emb = sample
        logger.info(f"[OK] Dataset loaded successfully:")
        logger.info(f"   Question embedding norm: {q_emb.norm().item():.6f}")
        logger.info(f"   Answer embedding norm: {a_emb.norm().item():.6f}")
        logger.info(f"   Dataset size: {len(self.dataset):,}")

        if q_emb.norm().item() < 0.1 or a_emb.norm().item() < 0.1:
            raise ValueError("Dataset contains zero embeddings!")

        logger.info("[OK] Training setup completed successfully")

    def run_training(self, max_epochs: int = 999999, batch_size: int = 2048):
        """Запуск обучения с готовыми эмбеддингами"""
        logger.info(
            f"[TARGET] Starting overnight training with precomputed embeddings:"
        )
        logger.info(f"   Max epochs: {max_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Dataset size: {len(self.dataset):,}")
        logger.info(f"   Device: {next(self.trainer.parameters()).device}")

        # Создаем DataLoader - больший batch_size возможен с готовыми эмбеддингами
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 для Windows совместимости
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Оптимизатор
        optimizer = torch.optim.AdamW(self.trainer.parameters(), lr=0.0001)

        epoch = 0
        start_time = time.time()

        try:
            while epoch < max_epochs and not self.should_stop:
                epoch_start_time = time.time()

                # Training epoch
                total_loss = 0.0
                total_similarity = 0.0
                num_batches = 0

                for batch_idx, (question_emb, answer_emb) in enumerate(dataloader):
                    if self.should_stop:
                        break

                    # Перемещаем на device
                    device = next(self.trainer.parameters()).device
                    question_emb = question_emb.to(device)
                    answer_emb = answer_emb.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.trainer.forward(question_emb)

                    # КРИТИЧНО: Адаптируем target embedding к 225D через тот же адаптер
                    with torch.no_grad():
                        adapted_target = self.trainer.base_trainer.adapter(answer_emb)

                    # Targets
                    targets = {
                        "target_embedding": adapted_target,
                        "target_surface": outputs["input_surface"],
                    }

                    # Loss computation
                    losses = self.trainer.compute_loss(outputs, targets)

                    # Правильное суммирование loss'ов в скаляр
                    total_loss_tensor = torch.tensor(
                        0.0, device=device, requires_grad=True
                    )
                    for loss_name, loss_value in losses.items():
                        if torch.is_tensor(loss_value) and loss_value.requires_grad:
                            if loss_value.dim() > 0:
                                loss_value = loss_value.mean()
                            total_loss_tensor = total_loss_tensor + loss_value

                    # Backward pass
                    total_loss_tensor.backward()

                    # Gradient clipping для стабильности
                    torch.nn.utils.clip_grad_norm_(
                        self.trainer.parameters(), max_norm=1.0
                    )

                    optimizer.step()

                    # Metrics - используем adapted target для правильной размерности
                    with torch.no_grad():
                        similarity = (
                            torch.cosine_similarity(
                                outputs["final_output"], adapted_target, dim=-1
                            )
                            .mean()
                            .item()
                        )

                    total_loss += total_loss_tensor.item()
                    total_similarity += similarity
                    num_batches += 1

                # Epoch metrics
                avg_loss = total_loss / max(num_batches, 1)
                avg_similarity = total_similarity / max(num_batches, 1)
                epoch_time = time.time() - epoch_start_time

                # Logging
                epoch += 1

                # Детальный лог каждые 5 эпох или если отличные результаты
                if epoch % 5 == 0 or epoch <= 10 or avg_similarity > 0.4:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"Similarity: {avg_similarity:.4f} | "
                        f"Time: {epoch_time:.1f}s | "
                        f"Batches: {num_batches}"
                    )

                # Сохранение прогресса
                log_entry = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "similarity": avg_similarity,
                    "time": epoch_time,
                    "batches": num_batches,
                    "timestamp": datetime.now().isoformat(),
                }
                self.training_log.append(log_entry)

                # Best model tracking
                if avg_similarity > self.best_similarity:
                    self.best_similarity = avg_similarity
                    logger.info(f"[BEST] New best similarity: {avg_similarity:.4f}")

                    # Сохраняем лучшую модель
                    self.weights_manager.save_latest_weights(
                        self.trainer,
                        self.config.to_dict(),
                        metadata={
                            "epoch": epoch,
                            "loss": avg_loss,
                            "similarity": avg_similarity,
                            "training_type": "overnight_precomputed",
                            "dataset_size": len(self.dataset),
                            "embeddings_file": Path(self.embeddings_file).name,
                        },
                    )

                # Checkpoint каждые 25 эпох
                if epoch % 25 == 0:
                    self.weights_manager.create_training_checkpoint(
                        self.trainer,
                        self.config.to_dict(),
                        epoch,
                        avg_loss,
                        avg_similarity,
                        metadata={
                            "training_type": "overnight_precomputed",
                            "dataset_size": len(self.dataset),
                            "embeddings_file": Path(self.embeddings_file).name,
                        },
                    )

                # Особые отметки прогресса
                if avg_similarity > 0.7:
                    logger.info(f"[EXCELLENT] OUTSTANDING RESULTS! Similarity > 70%")
                elif avg_similarity > 0.5:
                    logger.info(f"[TARGET] EXCELLENT PROGRESS! Similarity > 50%")
                elif avg_similarity > 0.3:
                    logger.info(f"[PROGRESS] GOOD PROGRESS! Similarity > 30%")

                # Сохранение лога каждые 50 эпох
                if epoch % 50 == 0:
                    self._save_training_log()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            self._finalize_training(epoch, time.time() - start_time)

    def _save_training_log(self):
        """Сохранение лога обучения"""
        log_path = f"logs/overnight_training_precomputed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def _finalize_training(self, final_epoch: int, total_time: float):
        """Финализация обучения"""
        logger.info(f"[DONE] Training completed:")
        logger.info(f"   Final epoch: {final_epoch}")
        logger.info(f"   Total time: {total_time/3600:.1f} hours")
        logger.info(f"   Best similarity: {self.best_similarity:.4f}")
        logger.info(f"   Dataset used: {Path(self.embeddings_file).name}")
        logger.info(f"   Dataset size: {len(self.dataset):,} pairs")

        # Финальное сохранение
        self._save_training_log()

        # Создаем milestone checkpoint
        self.weights_manager.create_milestone_checkpoint(
            self.trainer,
            self.config.to_dict(),
            f"overnight_precomputed_final_{final_epoch}",
            {
                "final_epoch": final_epoch,
                "total_time_hours": total_time / 3600,
                "best_similarity": self.best_similarity,
                "total_batches": len(self.training_log),
                "dataset_size": len(self.dataset),
                "embeddings_file": Path(self.embeddings_file).name,
            },
        )

        logger.info("[OK] Training finalization completed")


def main():
    """Главная функция"""
    print("[START] OVERNIGHT TRAINING С ПРЕДВАРИТЕЛЬНО ВЫЧИСЛЕННЫМИ ЭМБЕДДИНГАМИ")
    print("=" * 70)
    print("Auto-selects latest dataset or specify with --dataset argument")
    print("Larger batch sizes possible with precomputed embeddings")
    print("Optimal batch_size 2048+ for RTX 5090")
    print("=" * 70)

    # Простой argument parsing
    embeddings_file = None
    if len(sys.argv) > 1 and sys.argv[1] == "--dataset" and len(sys.argv) > 2:
        embeddings_file = sys.argv[2]
        print(f"Using specified dataset: {embeddings_file}")

    trainer = PrecomputedOvernightTrainer(embeddings_file)

    try:
        trainer.setup_training()
        trainer.run_training(
            max_epochs=999999,  # Unlimited
            batch_size=2048,  # Larger batch possible with precomputed embeddings
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
