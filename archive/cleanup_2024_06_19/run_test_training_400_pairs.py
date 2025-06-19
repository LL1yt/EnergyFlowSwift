#!/usr/bin/env python3
"""
ТЕСТОВОЕ ОБУЧЕНИЕ на датасете из 400 пар
Проверяет работоспособность системы перед переходом на большие датасеты
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
from precomputed_embedding_loader import PrecomputedEmbeddingLoader
from model_weights_manager import ModelWeightsManager
from config_converter import convert_config_dict_to_object

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/test_training_400_pairs.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TestTrainer400Pairs:
    """Тестовый тренер для датасета из 400 пар"""

    def __init__(self):
        self.trainer = None
        self.dataset = None
        self.config = None
        self.weights_manager = ModelWeightsManager()
        self.embedding_loader = PrecomputedEmbeddingLoader()
        self.should_stop = False
        self.best_similarity = 0.0
        self.training_log = []
        self.start_time = None

        # Обработчик сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("[START] TestTrainer400Pairs initialized")

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful остановки"""
        logger.info(
            f"[SIGNAL] Received signal {signum}, stopping training gracefully..."
        )
        self.should_stop = True

    def setup_training(self):
        """Настройка обучения"""
        logger.info("[SETUP] Setting up test training...")

        # 1. Находим датасет
        datasets = self.embedding_loader.list_available_datasets()
        if not datasets:
            raise FileNotFoundError(
                "No datasets found! Run generate_large_embedding_dataset.py first."
            )

        # Выбираем самый новый датасет
        latest_dataset = datasets[0]
        embeddings_file = latest_dataset["file_path"]

        logger.info(f"[LOAD] Using dataset: {latest_dataset['filename']}")
        logger.info(f"   Size: {latest_dataset['size']} pairs")
        logger.info(f"   Teacher model: {latest_dataset['teacher_model']}")

        # 2. Загружаем конфигурацию
        config_manager = ConfigManager()
        config_dict = config_manager.get_config()
        self.config = convert_config_dict_to_object(config_dict)

        # 3. Создаем trainer
        self.trainer = EmergentCubeTrainer(self.config)
        self.trainer.to("cuda" if torch.cuda.is_available() else "cpu")

        # 4. Загружаем готовые эмбеддинги
        self.dataset = self.embedding_loader.load_dataset(embeddings_file)

        # 5. Проверяем данные
        sample = self.dataset[0]
        q_emb, a_emb = sample
        logger.info(f"[OK] Dataset loaded for testing:")
        logger.info(f"   Question embedding norm: {q_emb.norm().item():.6f}")
        logger.info(f"   Answer embedding norm: {a_emb.norm().item():.6f}")
        logger.info(f"   Dataset size: {len(self.dataset)} pairs")

        logger.info("[OK] Test training setup completed")

    def run_test_training(self, max_epochs: int = 100, batch_size: int = 64):
        """Запуск тестового обучения"""
        self.start_time = time.time()

        logger.info(f"[TARGET] Starting TEST training (limited epochs):")
        logger.info(f"   Max epochs: {max_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Dataset size: {len(self.dataset)} pairs")
        logger.info(f"   Device: {next(self.trainer.parameters()).device}")

        # Создаем DataLoader
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

                    # Адаптируем target embedding к 225D
                    with torch.no_grad():
                        adapted_target = self.trainer.base_trainer.adapter(answer_emb)

                    # Targets
                    targets = {
                        "target_embedding": adapted_target,
                        "target_surface": outputs["input_surface"],
                    }

                    # Loss computation
                    losses = self.trainer.compute_loss(outputs, targets)

                    # Суммирование loss'ов
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
                    torch.nn.utils.clip_grad_norm_(
                        self.trainer.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    # Metrics
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

                epoch += 1

                # Лог каждые 5 эпох или важные события
                if (
                    epoch % 5 == 0
                    or epoch <= 10
                    or avg_similarity > self.best_similarity
                ):
                    logger.info(
                        f"Epoch {epoch:3d}/{max_epochs} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"Similarity: {avg_similarity:.4f} | "
                        f"Time: {epoch_time:.1f}s"
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
                    logger.info(
                        f"[BEST] New best similarity: {avg_similarity:.4f} (epoch {epoch})"
                    )

                # Отметки прогресса
                if avg_similarity > 0.5:
                    logger.info(f"[EXCELLENT] Great progress! Similarity > 50%")
                elif avg_similarity > 0.3:
                    logger.info(f"[GOOD] Good progress! Similarity > 30%")
                elif avg_similarity > 0.15:
                    logger.info(f"[PROGRESS] Decent progress! Similarity > 15%")

        except KeyboardInterrupt:
            logger.info("[STOP] Training interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Training error: {e}")
            raise
        finally:
            self._finalize_test_training(epoch)

    def _finalize_test_training(self, final_epoch: int):
        """Финализация тестового обучения"""
        total_time = time.time() - self.start_time

        logger.info(f"[DONE] Test training completed:")
        logger.info(f"   Final epoch: {final_epoch}")
        logger.info(f"   Total time: {total_time/60:.1f} minutes")
        logger.info(f"   Best similarity: {self.best_similarity:.4f}")
        logger.info(f"   Dataset size: {len(self.dataset)} pairs")

        # Создаем понятное имя для результата
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = f"test_400pairs_{final_epoch}epochs_{self.best_similarity:.3f}sim_{timestamp}"

        # Сохраняем результат с понятным названием
        if self.best_similarity > 0.1:  # Сохраняем только если есть прогресс
            logger.info(f"[SAVE] Saving test result: {result_name}")

            self.weights_manager.create_milestone_checkpoint(
                self.trainer,
                self.config.to_dict(),
                result_name,
                {
                    "training_type": "test_400_pairs",
                    "final_epoch": final_epoch,
                    "total_time_minutes": total_time / 60,
                    "best_similarity": self.best_similarity,
                    "dataset_size": len(self.dataset),
                    "timestamp": timestamp,
                    "description": f"Test training on 400 pairs dataset, {final_epoch} epochs, best similarity {self.best_similarity:.3f}",
                },
            )

        # Сохраняем лог обучения
        log_path = f"logs/test_training_400pairs_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "training_info": {
                        "type": "test_400_pairs",
                        "final_epoch": final_epoch,
                        "total_time_minutes": total_time / 60,
                        "best_similarity": self.best_similarity,
                        "dataset_size": len(self.dataset),
                    },
                    "training_log": self.training_log,
                },
                f,
                indent=2,
            )

        # Оценка результатов
        self._evaluate_results()

        logger.info("[OK] Test training finalization completed")

    def _evaluate_results(self):
        """Оценка результатов тестового обучения"""
        logger.info(f"\n[STATS] === ОЦЕНКА РЕЗУЛЬТАТОВ ===")

        if self.best_similarity > 0.4:
            logger.info(
                f"[EXCELLENT] Отличный результат! Similarity {self.best_similarity:.3f}"
            )
            logger.info(f"[RECOMMEND] Рекомендую переходить к большим датасетам")
        elif self.best_similarity > 0.2:
            logger.info(
                f"[GOOD] Хороший результат! Similarity {self.best_similarity:.3f}"
            )
            logger.info(f"[RECOMMEND] Можно пробовать большие датасеты или больше эпох")
        elif self.best_similarity > 0.1:
            logger.info(
                f"[OK] Приемлемый результат. Similarity {self.best_similarity:.3f}"
            )
            logger.info(
                f"[RECOMMEND] Стоит попробовать больше эпох или настроить параметры"
            )
        else:
            logger.info(
                f"[WARNING] Низкий результат. Similarity {self.best_similarity:.3f}"
            )
            logger.info(f"[RECOMMEND] Нужно проверить настройки обучения")

        # Анализ динамики
        if len(self.training_log) > 10:
            early_sim = sum(log["similarity"] for log in self.training_log[:5]) / 5
            late_sim = sum(log["similarity"] for log in self.training_log[-5:]) / 5
            improvement = late_sim - early_sim

            logger.info(f"[ANALYSIS] Прогресс обучения:")
            logger.info(f"   Начальная similarity: {early_sim:.4f}")
            logger.info(f"   Финальная similarity: {late_sim:.4f}")
            logger.info(f"   Улучшение: {improvement:.4f}")

            if improvement > 0.05:
                logger.info(f"[GOOD] Хорошая динамика обучения!")
            elif improvement > 0.01:
                logger.info(f"[OK] Умеренная динамика обучения")
            else:
                logger.info(f"[WARNING] Слабая динамика, возможно нужно больше эпох")


def main():
    """Главная функция"""
    print("[START] ТЕСТОВОЕ ОБУЧЕНИЕ НА 400 ПАРАХ")
    print("=" * 50)
    print("Проверяет работоспособность перед большими датасетами")
    print("Ограниченное количество эпох для быстрой проверки")
    print("Сохраняет результат с понятным названием")
    print("=" * 50)

    trainer = TestTrainer400Pairs()

    try:
        trainer.setup_training()
        trainer.run_test_training(
            max_epochs=100,  # Ограниченное количество для теста
            batch_size=2048,  # Меньший batch для стабильности
        )
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
