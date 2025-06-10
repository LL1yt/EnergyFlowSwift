#!/usr/bin/env python3
"""
🧠 Smart Resume Training - автоматический поиск и продолжение обучения
Ищет совместимые чекпоинты на основе динамической конфигурации
"""

import os
import sys
import torch
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import glob
from warmup_scheduler import create_warmup_scheduler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SmartResumeManager:
    """Менеджер умного возобновления обучения"""

    def __init__(
        self, forced_mode: Optional[str] = None, custom_scale: Optional[float] = None
    ):
        """
        Args:
            forced_mode: Принудительный режим конфигурации
            custom_scale: Пользовательский scale factor
        """
        self.forced_mode = forced_mode
        self.custom_scale = custom_scale
        self.config_manager = None
        self.current_config = None

        # Инициализируем конфигурацию
        self._initialize_config()

    def _initialize_config(self):
        """Инициализация текущей конфигурации"""
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
                enable_hot_reload=False,
                # Отключаем автоматическое сканирование старых конфигов
                config_search_paths=[],  # Пустой список = только main_config.yaml + динамическая система
                # Передаем custom scale сразу при инициализации
                custom_scale_factor=self.custom_scale_factor,
            )

            # Создание ConfigManager
            self.config_manager = ConfigManager(settings)

            # Получение текущей динамической конфигурации
            self.current_config = {
                "lattice": self.config_manager.get_config("lattice"),
                "embeddings": self.config_manager.get_config("embeddings"),
                "training": self.config_manager.get_config("training"),
                "gmlp": self.config_manager.get_config("gmlp"),
            }

            # Информация о режиме
            dynamic_info = self.config_manager.get_dynamic_config_info()
            if dynamic_info:
                logger.info(f"🎯 Current config: {dynamic_info['mode']} mode")
                logger.info(f"   Scale factor: {dynamic_info['scale_factor']}")

            lattice = self.current_config["lattice"]
            logger.info(f"📊 Target configuration:")
            logger.info(f"   Lattice: {lattice['xs']}x{lattice['ys']}x{lattice['zs']}")
            logger.info(f"   Total neurons: {lattice['total_neurons']:,}")
            logger.info(
                f"   Embedding dim: {self.current_config['embeddings']['embedding_dim']:,}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize config: {e}")
            raise

    def find_compatible_checkpoints(
        self, checkpoints_dir: str = "checkpoints"
    ) -> List[Dict[str, Any]]:
        """
        Ищет совместимые чекпоинты на основе текущей конфигурации

        Returns:
            Список совместимых чекпоинтов, отсортированных по приоритету
        """
        logger.info(f"🔍 Searching for compatible checkpoints in {checkpoints_dir}")

        checkpoints_path = Path(checkpoints_dir)
        if not checkpoints_path.exists():
            logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
            return []

        compatible_checkpoints = []
        current_signature = self._get_config_signature(self.current_config)

        # Ищем во всех подпапках
        search_patterns = [
            checkpoints_path / "latest" / "*.pt",
            checkpoints_path / "versioned" / "*" / "*.pt",
            checkpoints_path / "*.pt",  # Прямо в корне
        ]

        for pattern in search_patterns:
            for checkpoint_file in glob.glob(str(pattern)):
                checkpoint_info = self._analyze_checkpoint(
                    checkpoint_file, current_signature
                )
                if checkpoint_info:
                    compatible_checkpoints.append(checkpoint_info)

        # Сортируем по приоритету (совместимость, потом время)
        compatible_checkpoints.sort(
            key=lambda x: (
                -x["compatibility_score"],  # Высокая совместимость первая
                -x["timestamp_score"],  # Новые чекпоинты первые
            )
        )

        logger.info(f"🎯 Found {len(compatible_checkpoints)} compatible checkpoints")

        # Показываем топ 5
        for i, cp in enumerate(compatible_checkpoints[:5]):
            logger.info(
                f"   {i+1}. {cp['name']} (score: {cp['compatibility_score']:.2f})"
            )
            logger.info(f"      {cp['description']}")

        return compatible_checkpoints

    def _analyze_checkpoint(
        self, checkpoint_path: str, current_signature: Dict
    ) -> Optional[Dict[str, Any]]:
        """Анализирует совместимость чекпоинта"""
        try:
            # Загружаем метаданные чекпоинта
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

            # Извлекаем конфигурацию
            checkpoint_config = checkpoint_data.get("config", {})
            checkpoint_metadata = checkpoint_data.get("metadata", {})

            if not checkpoint_config:
                return None

            # Вычисляем сигнатуру чекпоинта
            checkpoint_signature = self._get_config_signature(checkpoint_config)

            # Оценка совместимости
            compatibility_score = self._calculate_compatibility(
                current_signature, checkpoint_signature
            )

            # Минимальный порог совместимости
            if compatibility_score < 0.5:
                return None

            # Извлекаем временную метку
            timestamp_str = checkpoint_metadata.get("timestamp", "1970-01-01T00:00:00")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_score = timestamp.timestamp()
            except:
                timestamp_score = 0

            # Формируем информацию о чекпоинте
            checkpoint_info = {
                "path": checkpoint_path,
                "name": Path(checkpoint_path).name,
                "config": checkpoint_config,
                "metadata": checkpoint_metadata,
                "signature": checkpoint_signature,
                "compatibility_score": compatibility_score,
                "timestamp_score": timestamp_score,
                "timestamp": timestamp_str,
                "description": self._generate_checkpoint_description(
                    checkpoint_metadata
                ),
            }

            return checkpoint_info

        except Exception as e:
            logger.debug(f"Failed to analyze checkpoint {checkpoint_path}: {e}")
            return None

    def _get_config_signature(self, config: Dict) -> Dict:
        """Создает сигнатуру конфигурации для сравнения"""
        signature = {}

        # Lattice параметры
        if "lattice" in config:
            lattice = config["lattice"]
            signature["lattice_dims"] = (
                lattice.get("xs", 0),
                lattice.get("ys", 0),
                lattice.get("zs", 0),
            )
            signature["total_neurons"] = lattice.get("total_neurons", 0)

        # Embedding параметры
        if "embeddings" in config:
            embeddings = config["embeddings"]
            signature["embedding_dim"] = embeddings.get("embedding_dim", 0)

        # gMLP параметры
        if "gmlp" in config:
            gmlp = config["gmlp"]
            signature["state_size"] = gmlp.get("state_size", 0)
            signature["hidden_dim"] = gmlp.get("hidden_dim", 0)
            signature["neighbor_count"] = gmlp.get("neighbor_count", 0)

        return signature

    def _calculate_compatibility(self, current: Dict, checkpoint: Dict) -> float:
        """Вычисляет оценку совместимости (0.0 - 1.0)"""
        score = 0.0
        total_weight = 0.0

        # Веса важности разных параметров
        weights = {
            "lattice_dims": 0.4,  # Очень важно - архитектура
            "total_neurons": 0.2,  # Важно для размера модели
            "embedding_dim": 0.3,  # Критично для входных данных
            "state_size": 0.05,  # Менее критично
            "hidden_dim": 0.03,  # Менее критично
            "neighbor_count": 0.02,  # Менее критично
        }

        for param, weight in weights.items():
            if param in current and param in checkpoint:
                current_val = current[param]
                checkpoint_val = checkpoint[param]

                if current_val == checkpoint_val:
                    score += weight  # Полное совпадение
                elif param == "lattice_dims":
                    # Для размеров решетки проверяем близость
                    if isinstance(current_val, tuple) and isinstance(
                        checkpoint_val, tuple
                    ):
                        similarity = self._tuple_similarity(current_val, checkpoint_val)
                        score += weight * similarity
                else:
                    # Для численных параметров вычисляем относительную близость
                    if isinstance(current_val, (int, float)) and isinstance(
                        checkpoint_val, (int, float)
                    ):
                        if current_val > 0 and checkpoint_val > 0:
                            ratio = min(current_val, checkpoint_val) / max(
                                current_val, checkpoint_val
                            )
                            score += weight * ratio

                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _tuple_similarity(self, tuple1: tuple, tuple2: tuple) -> float:
        """Вычисляет сходство между кортежами (например, размеры решетки)"""
        if len(tuple1) != len(tuple2):
            return 0.0

        similarities = []
        for a, b in zip(tuple1, tuple2):
            if a == b:
                similarities.append(1.0)
            elif a > 0 and b > 0:
                ratio = min(a, b) / max(a, b)
                similarities.append(ratio)
            else:
                similarities.append(0.0)

        return sum(similarities) / len(similarities)

    def _generate_checkpoint_description(self, metadata: Dict) -> str:
        """Генерирует описание чекпоинта"""
        parts = []

        # Тип обучения
        training_type = metadata.get("training_type", "unknown")
        parts.append(training_type)

        # Параметры модели
        params = metadata.get("trainable_params", metadata.get("model_params"))
        if params:
            if params < 1000:
                parts.append(f"{params} params")
            elif params < 1000000:
                parts.append(f"{params//1000}K params")
            else:
                parts.append(f"{params//1000000}M params")

        # Дополнительная информация
        if "best_similarity" in metadata:
            parts.append(f"sim={metadata['best_similarity']:.3f}")

        if "epochs" in metadata:
            parts.append(f"{metadata['epochs']} epochs")

        return " | ".join(parts) if parts else "checkpoint"

    def auto_resume_training(
        self,
        dataset_limit: Optional[int] = None,
        additional_epochs: int = 10,
        **training_kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Автоматически находит лучший чекпоинт и продолжает обучение

        Args:
            dataset_limit: Ограничение размера датасета
            additional_epochs: Сколько дополнительных эпох обучать
            **training_kwargs: Дополнительные параметры обучения

        Returns:
            Результаты обучения или None если не удалось
        """
        logger.info(f"🚀 Starting smart resume training")

        # Ищем совместимые чекпоинты
        compatible_checkpoints = self.find_compatible_checkpoints()

        if not compatible_checkpoints:
            logger.warning(
                f"⚠️ No compatible checkpoints found, starting fresh training"
            )
            return self._start_fresh_training(
                dataset_limit, additional_epochs, **training_kwargs
            )

        # Выбираем лучший чекпоинт
        best_checkpoint = compatible_checkpoints[0]
        logger.info(f"🎯 Selected checkpoint: {best_checkpoint['name']}")
        logger.info(f"   Compatibility: {best_checkpoint['compatibility_score']:.2f}")
        logger.info(f"   Description: {best_checkpoint['description']}")

        # Загружаем и продолжаем обучение
        return self._resume_from_checkpoint(
            best_checkpoint, dataset_limit, additional_epochs, **training_kwargs
        )

    def _start_fresh_training(
        self, dataset_limit: Optional[int], epochs: int, **kwargs
    ) -> Dict[str, Any]:
        """Запускает новое обучение"""
        logger.info(f"🆕 Starting fresh training")

        from run_dynamic_training import DynamicTrainingManager

        training_manager = DynamicTrainingManager(
            forced_mode=self.forced_mode, custom_scale=self.custom_scale
        )

        return training_manager.run_training(
            dataset_limit=dataset_limit, epochs=epochs, **kwargs
        )

    def _resume_from_checkpoint(
        self,
        checkpoint_info: Dict[str, Any],
        dataset_limit: Optional[int],
        additional_epochs: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Продолжает обучение с чекпоинта"""
        logger.info(f"▶️ Resuming training from checkpoint: {checkpoint_info['name']}")
        logger.info(
            f"   Checkpoint similarity: {checkpoint_info['metadata'].get('best_similarity', 'unknown')}"
        )
        logger.info(f"   Checkpoint timestamp: {checkpoint_info['timestamp']}")

        try:
            # Создаем trainer с текущей конфигурацией
            from run_dynamic_training import DynamicTrainingManager

            training_manager = DynamicTrainingManager(
                forced_mode=self.forced_mode, custom_scale=self.custom_scale
            )

            # Создаем trainer и загружаем веса
            trainer = training_manager.create_trainer()
            checkpoint_data = torch.load(checkpoint_info["path"], map_location="cpu")
            trainer.load_state_dict(checkpoint_data["model_state_dict"], strict=False)

            logger.info(f"✅ Checkpoint weights loaded successfully")

            # Извлекаем количество пройденных эпох из метаданных
            checkpoint_metadata = checkpoint_info.get("metadata", {})
            completed_epochs = checkpoint_metadata.get("epochs", 0)

            logger.info(
                f"🎓 Resuming from epoch {completed_epochs + 1}, will train {additional_epochs} more epochs"
            )

            # Запускаем обучение с resume
            return training_manager.run_training(
                dataset_limit=dataset_limit,
                epochs=additional_epochs,
                resume_trainer=trainer,
                start_epoch=completed_epochs,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.info(f"🆕 Falling back to fresh training")
            return self._start_fresh_training(
                dataset_limit, additional_epochs, **kwargs
            )


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Resume Training with Auto Checkpoint Detection"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "development", "research", "validation", "production"],
        default="development",
        help="Configuration mode",
    )
    parser.add_argument(
        "--dataset-limit", type=int, default=None, help="Limit dataset size"
    )
    parser.add_argument(
        "--additional-epochs", type=int, default=10, help="Additional epochs to train"
    )
    parser.add_argument("--scale", type=float, default=None, help="Custom scale factor")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (uses config default if not specified)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list compatible checkpoints, don't train",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Создание умного менеджера возобновления
        resume_manager = SmartResumeManager(
            forced_mode=args.mode if args.mode != "auto" else None,
            custom_scale=args.scale,
        )

        if args.list_only:
            # Только показываем совместимые чекпоинты
            compatible_checkpoints = resume_manager.find_compatible_checkpoints()

            if compatible_checkpoints:
                logger.info(f"\n📋 Compatible checkpoints found:")
                for i, cp in enumerate(compatible_checkpoints):
                    logger.info(f"   {i+1}. {cp['name']}")
                    logger.info(f"      Score: {cp['compatibility_score']:.2f}")
                    logger.info(f"      {cp['description']}")
                    logger.info(f"      Path: {cp['path']}")
                    logger.info("")
            else:
                logger.info(f"❌ No compatible checkpoints found")

            return

        # Запуск умного возобновления обучения
        results = resume_manager.auto_resume_training(
            dataset_limit=args.dataset_limit,
            additional_epochs=args.additional_epochs,
            batch_size=args.batch_size,
        )

        if results:
            logger.info("📈 Training Results:")
            for key, value in results.items():
                if key != "training_log":  # Пропускаем детальный лог
                    logger.info(f"   {key}: {value}")

        logger.info("✅ Smart resume training completed!")

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Smart resume failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
