"""
🚀 Dynamic Training Script с автоматической конфигурацией
Использует Dynamic Configuration System для оптимального использования железа
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import gc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Настройка окружения для обучения"""
    # Оптимизация PyTorch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Настройка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # Используем 90% GPU памяти

    logger.info("🔧 Environment setup completed")


class DynamicTrainingManager:
    """Менеджер обучения с динамической конфигурацией"""

    def __init__(
        self, forced_mode: Optional[str] = None, custom_scale: Optional[float] = None
    ):
        """
        Args:
            forced_mode: Принудительный режим (development, research, validation, production)
                        Если None, будет автоопределение
            custom_scale: Пользовательский scale factor (переопределяет режим по умолчанию)
        """
        self.forced_mode = forced_mode
        self.custom_scale = custom_scale
        self.config_manager = None
        self.dynamic_config = None
        self.trainer = None

        # Загружаем конфигурацию
        self._load_dynamic_config()

    def _load_dynamic_config(self):
        """Загрузка динамической конфигурации"""
        try:
            from utils.config_manager.config_manager import (
                ConfigManager,
                ConfigManagerSettings,
            )

            # Настройки с включенной динамической конфигурацией
            settings = ConfigManagerSettings(
                enable_dynamic_config=True,
                dynamic_config_mode=self.forced_mode or "auto",
                auto_hardware_detection=True,
                enable_hot_reload=False,  # Отключаем во время обучения
            )

            # Создание ConfigManager
            self.config_manager = ConfigManager(settings)

            # Применяем custom scale если указан
            if self.custom_scale:
                self._apply_custom_scale()

            # Получение динамической конфигурации
            self.dynamic_config = {
                "lattice": self.config_manager.get_config("lattice"),
                "embeddings": self.config_manager.get_config("embeddings"),
                "training": self.config_manager.get_config("training"),
                "gmlp": self.config_manager.get_config("gmlp"),
            }

            # Информация о режиме
            dynamic_info = self.config_manager.get_dynamic_config_info()
            if dynamic_info:
                logger.info(f"🎯 Loaded dynamic config: {dynamic_info['mode']} mode")
                logger.info(f"   Scale factor: {dynamic_info['scale_factor']}")

            # Выводим основную информацию
            lattice = self.dynamic_config["lattice"]
            embeddings = self.dynamic_config["embeddings"]
            training = self.dynamic_config["training"]

            logger.info(f"📊 Configuration loaded:")
            logger.info(f"   Lattice: {lattice['xs']}x{lattice['ys']}x{lattice['zs']}")
            logger.info(f"   Total neurons: {lattice['total_neurons']:,}")
            logger.info(f"   Embedding dim: {embeddings['embedding_dim']:,}")
            logger.info(f"   Batch size: {training['batch_size']}")
            logger.info(f"   Learning rate: {training['learning_rate']}")

        except Exception as e:
            logger.error(f"❌ Failed to load dynamic config: {e}")
            raise

    def _apply_custom_scale(self):
        """Применить custom scale factor"""
        try:
            from utils.config_manager.dynamic_config import DynamicConfigManager

            # Создаем dynamic config manager
            dynamic_manager = DynamicConfigManager()

            # Определяем режим для применения custom scale
            mode = self.forced_mode or "development"

            # Временно изменяем scale в dynamic manager
            original_scale = getattr(dynamic_manager.generator.scale_settings, mode)
            setattr(dynamic_manager.generator.scale_settings, mode, self.custom_scale)

            # Генерируем новую конфигурацию
            new_config = dynamic_manager.create_config_for_mode(mode)

            # Объединяем с текущей конфигурацией ConfigManager
            self.config_manager.merge_dynamic_config(new_config)

            # Восстанавливаем оригинальный scale
            setattr(dynamic_manager.generator.scale_settings, mode, original_scale)

            logger.info(f"🎯 Applied custom scale factor: {self.custom_scale}")

        except Exception as e:
            logger.error(f"❌ Failed to apply custom scale: {e}")
            raise

    def create_trainer(self):
        """Создание trainer с динамической конфигурацией"""
        try:
            from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
                EmergentCubeTrainer,
                EmergentTrainingConfig,
            )
            from training.embedding_trainer.neural_cellular_automata import (
                create_nca_config,
            )

            # Создаем конфигурацию trainer'а на основе динамической конфигурации
            lattice_config = self.dynamic_config["lattice"]
            gmlp_config = self.dynamic_config["gmlp"]
            training_config = self.dynamic_config["training"]

            # Конфигурация EmergentTrainingConfig
            trainer_config = EmergentTrainingConfig(
                teacher_model="distilbert-base-uncased",
                cube_dimensions=(
                    lattice_config["xs"],
                    lattice_config["ys"],
                    lattice_config["zs"],
                ),
                # gMLP конфигурация из динамической системы
                gmlp_config={
                    "state_size": gmlp_config["state_size"],
                    "neighbor_count": gmlp_config["neighbor_count"],
                    "hidden_dim": gmlp_config["hidden_dim"],
                    "external_input_size": gmlp_config["external_input_size"],
                    "use_memory": True,
                    "activation": "gelu",
                    "dropout": 0.1,
                    "spatial_connections": True,
                },
                # Training параметры
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                epochs=training_config["epochs"],
                # Оптимизации для больших размеров
                mixed_precision=True,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                # NCA конфигурация
                enable_nca=True,
                nca_config=create_nca_config(
                    update_probability=0.7,
                    residual_learning_rate=0.1,
                    enable_pattern_detection=True,
                ),
            )

            # Создаем trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.trainer = EmergentCubeTrainer(trainer_config, device=device)

            logger.info(f"✅ Trainer created successfully")
            logger.info(f"   Device: {device}")
            logger.info(f"   Mixed precision: {trainer_config.mixed_precision}")
            logger.info(
                f"   Gradient checkpointing: {trainer_config.gradient_checkpointing}"
            )

            return self.trainer

        except Exception as e:
            logger.error(f"❌ Failed to create trainer: {e}")
            raise

    def prepare_dataset(self, limit: Optional[int] = None):
        """Подготовка датасета для обучения с использованием precomputed embeddings"""
        try:
            # Используем существующую систему PrecomputedEmbeddingLoader
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).parent))

            from precomputed_embedding_loader import PrecomputedEmbeddingLoader

            # Создаем загрузчик
            loader = PrecomputedEmbeddingLoader()

            # Находим доступные датасеты
            datasets = loader.list_available_datasets()
            if not datasets:
                raise FileNotFoundError(
                    "No precomputed datasets found! Run generate_large_embedding_dataset.py first."
                )

            # Выбираем самый новый и большой датасет
            latest_dataset = datasets[0]
            embeddings_file = latest_dataset["file_path"]

            logger.info(f"📁 Using precomputed dataset: {latest_dataset['filename']}")
            logger.info(f"   Available size: {latest_dataset['size']} pairs")
            logger.info(f"   Teacher model: {latest_dataset['teacher_model']}")

            # Загружаем датасет
            dataset = loader.load_dataset(embeddings_file)

            # Ограничиваем размер если нужно
            if limit and limit < len(dataset):
                from torch.utils.data import Subset
                import torch

                # Случайная выборка для разнообразия
                indices = torch.randperm(len(dataset))[:limit]
                dataset = Subset(dataset, indices)
                logger.info(f"   Limited to: {limit} pairs")

            logger.info(f"📁 Dataset prepared:")
            logger.info(f"   Final size: {len(dataset)} pairs")

            # Проверяем образец
            sample = dataset[0]
            q_emb, a_emb = sample
            logger.info(f"   Question embedding shape: {q_emb.shape}")
            logger.info(f"   Answer embedding shape: {a_emb.shape}")
            logger.info(f"   Question norm: {q_emb.norm().item():.6f}")
            logger.info(f"   Answer norm: {a_emb.norm().item():.6f}")

            return dataset

        except Exception as e:
            logger.error(f"❌ Failed to prepare dataset: {e}")
            raise

    def run_training(
        self,
        dataset_limit: Optional[int] = None,
        epochs: int = None,
        batch_size: int = None,
    ):
        """Запуск обучения"""
        try:
            # Подготовка
            setup_environment()

            # Создание trainer
            trainer = self.create_trainer()

            # Подготовка данных
            dataset = self.prepare_dataset(limit=dataset_limit)

            # Используем параметры из конфигурации если не указано
            if epochs is None:
                epochs = self.dynamic_config["training"]["epochs"]
            if batch_size is None:
                batch_size = self.dynamic_config["training"]["batch_size"]

            # Создаем DataLoader
            from torch.utils.data import DataLoader

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # 0 для Windows совместимости
                pin_memory=True if torch.cuda.is_available() else False,
            )

            # Оптимизатор
            optimizer = torch.optim.AdamW(
                trainer.parameters(),
                lr=self.dynamic_config["training"]["learning_rate"],
            )

            # Запуск обучения
            logger.info(f"🚀 Starting dynamic training:")
            logger.info(f"   Dataset size: {len(dataset)}")
            logger.info(f"   Epochs: {epochs}")
            logger.info(f"   Batch size: {batch_size}")
            logger.info(
                f"   Estimated time: {self._estimate_training_time(len(dataset), epochs)}"
            )

            start_time = time.time()
            best_similarity = 0.0
            training_log = []

            for epoch in range(epochs):
                epoch_start = time.time()

                # Training epoch
                total_loss = 0.0
                total_similarity = 0.0
                num_batches = 0

                for batch_idx, (question_emb, answer_emb) in enumerate(dataloader):
                    # Перемещаем на device
                    device = next(trainer.parameters()).device
                    question_emb = question_emb.to(device)
                    answer_emb = answer_emb.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = trainer.forward(question_emb)

                    # Адаптируем target embedding
                    with torch.no_grad():
                        adapted_target = trainer.base_trainer.adapter(answer_emb)

                    # Targets
                    targets = {
                        "target_embedding": adapted_target,
                        "target_surface": outputs["input_surface"],
                    }

                    # Loss computation
                    losses = trainer.compute_loss(outputs, targets)

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
                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
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
                epoch_time = time.time() - epoch_start

                # Лог каждые 5 эпох или важные события
                if epoch % 5 == 0 or epoch <= 10 or avg_similarity > best_similarity:
                    logger.info(
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"Similarity: {avg_similarity:.4f} | "
                        f"Time: {epoch_time:.1f}s"
                    )

                # Сохранение прогресса
                training_log.append(
                    {
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "similarity": avg_similarity,
                        "time": epoch_time,
                    }
                )

                # Best model tracking
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    logger.info(
                        f"[BEST] New best similarity: {avg_similarity:.4f} (epoch {epoch+1})"
                    )

                # Очистка памяти каждые 10 эпох
                if (epoch + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            total_time = time.time() - start_time

            # Сохранение результата с указанием scale
            self._save_training_results(
                trainer, len(dataset), epochs, best_similarity, total_time, training_log
            )

            logger.info(f"🎉 Training completed:")
            logger.info(f"   Total time: {total_time/60:.1f} minutes")
            logger.info(f"   Final similarity: {avg_similarity:.4f}")
            logger.info(f"   Best similarity: {best_similarity:.4f}")

            return {
                "final_similarity": avg_similarity,
                "best_similarity": best_similarity,
                "total_time": total_time,
                "epochs": epochs,
                "dataset_size": len(dataset),
                "training_log": training_log,
            }

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def _save_training_results(
        self,
        trainer,
        dataset_size: int,
        epochs: int,
        best_similarity: float,
        total_time: float,
        training_log: list,
    ):
        """Сохранение результатов обучения с указанием scale в названии"""
        try:
            from datetime import datetime
            from model_weights_manager import ModelWeightsManager

            # Получаем информацию о динамической конфигурации
            dynamic_info = self.config_manager.get_dynamic_config_info()
            mode = dynamic_info.get("mode", "unknown") if dynamic_info else "unknown"
            scale_factor = (
                dynamic_info.get("scale_factor", "unknown")
                if dynamic_info
                else "unknown"
            )

            # Создаем понятное имя с указанием scale
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_name = f"dynamic_{mode}_scale{scale_factor}_{dataset_size}pairs_{epochs}epochs_{best_similarity:.3f}sim_{timestamp}"

            # Сохраняем checkpoint если есть прогресс
            if best_similarity > 0.1:
                weights_manager = ModelWeightsManager()

                logger.info(f"[SAVE] Saving dynamic training result: {result_name}")

                weights_manager.create_milestone_checkpoint(
                    trainer,
                    self.config_manager.get_config(),
                    result_name,
                    {
                        "training_type": "dynamic_training",
                        "mode": mode,
                        "scale_factor": scale_factor,
                        "cube_dimensions": f"{self.dynamic_config['lattice']['xs']}x{self.dynamic_config['lattice']['ys']}x{self.dynamic_config['lattice']['zs']}",
                        "total_neurons": self.dynamic_config["lattice"][
                            "total_neurons"
                        ],
                        "embedding_dim": self.dynamic_config["embeddings"][
                            "embedding_dim"
                        ],
                        "dataset_size": dataset_size,
                        "epochs": epochs,
                        "best_similarity": best_similarity,
                        "total_time_minutes": total_time / 60,
                        "timestamp": timestamp,
                        "description": f"Dynamic training {mode} mode (scale={scale_factor}), {dataset_size} pairs, {epochs} epochs, best similarity {best_similarity:.3f}",
                    },
                )

                # Сохраняем лог обучения
                import json

                log_path = (
                    f"logs/dynamic_training_{mode}_scale{scale_factor}_{timestamp}.json"
                )
                with open(log_path, "w") as f:
                    json.dump(
                        {
                            "training_info": {
                                "type": "dynamic_training",
                                "mode": mode,
                                "scale_factor": scale_factor,
                                "dataset_size": dataset_size,
                                "epochs": epochs,
                                "best_similarity": best_similarity,
                                "total_time_minutes": total_time / 60,
                            },
                            "dynamic_config": self.dynamic_config,
                            "training_log": training_log,
                        },
                        f,
                        indent=2,
                    )

                logger.info(f"✅ Results saved with scale indication: {result_name}")
            else:
                logger.info(
                    f"⚠️ Similarity too low ({best_similarity:.3f}), skipping checkpoint save"
                )

        except Exception as e:
            logger.error(f"❌ Failed to save training results: {e}")

    def _estimate_training_time(self, dataset_size: int, epochs: int) -> str:
        """Оценка времени обучения"""
        lattice = self.dynamic_config["lattice"]
        total_neurons = lattice["total_neurons"]

        # Примерная оценка на основе размера решетки
        if total_neurons < 1000:
            time_per_epoch = 5  # секунд
        elif total_neurons < 50000:
            time_per_epoch = 15
        elif total_neurons < 500000:
            time_per_epoch = 60
        elif total_neurons < 2000000:
            time_per_epoch = 180  # 3 минуты
        else:
            time_per_epoch = 300  # 5 минут

        total_minutes = (time_per_epoch * epochs) / 60

        if total_minutes < 60:
            return f"~{total_minutes:.0f} minutes"
        else:
            return f"~{total_minutes/60:.1f} hours"


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamic Training Script with Precomputed Embeddings"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "development", "research", "validation", "production"],
        default="development",
        help="Configuration mode (default: development for scale=0.01)",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit dataset size (uses full dataset if not specified)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (uses config default if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (uses config default if not specified)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Custom scale factor (overrides mode default)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Создание менеджера обучения
        training_manager = DynamicTrainingManager(
            forced_mode=args.mode if args.mode != "auto" else None,
            custom_scale=args.scale,
        )

        # Запуск обучения
        results = training_manager.run_training(
            dataset_limit=args.dataset_limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Результаты
        logger.info("📈 Training Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")

        logger.info("✅ Dynamic training completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
