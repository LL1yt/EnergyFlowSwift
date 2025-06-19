"""
Progressive Configuration Manager - Менеджер прогрессивных конфигураций

Этот модуль управляет конфигурациями для разных стадий автоматизированного обучения,
реализуя стратегию постепенного увеличения сложности.

Стратегия обучения:
- Stage 1: Маленький датасет, много эпох (изучение основ)
- Stage 2: Средний датасет, средние эпохи (консолидация)
- Stage 3: Большой датасет, мало эпох (финальная подстройка)
- Stage 4: Очень большой датасет, мало эпох (мастерство)
- Stage 5: Массивный датасет, минимум эпох (совершенство)
"""

import logging
from typing import Dict, Any, Optional

from .types import StageConfig

logger = logging.getLogger(__name__)


class ProgressiveConfigManager:
    """Менеджер прогрессивных конфигураций для автоматизированного обучения"""

    def __init__(
        self,
        dataset_limit_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
    ):
        """
        Args:
            dataset_limit_override: Переопределить dataset_limit для всех стадий (для тестирования)
            batch_size_override: Переопределить batch_size для всех стадий (для ускорения)
        """
        self.dataset_limit_override = dataset_limit_override
        self.batch_size_override = batch_size_override

        # Базовые конфигурации стадий
        self._base_configs = {
            1: {
                "dataset_limit": 2000,
                "epochs": 20,
                "batch_size": 32,
                "description": "Foundation Learning (small data, many epochs)",
            },
            2: {
                "dataset_limit": 5000,
                "epochs": 15,
                "batch_size": 64,
                "description": "Consolidation (medium data, medium epochs)",
            },
            3: {
                "dataset_limit": 10000,
                "epochs": 12,
                "batch_size": 64,
                "description": "Refinement (large data, fewer epochs)",
            },
            4: {
                "dataset_limit": 20000,
                "epochs": 8,
                "batch_size": 128,
                "description": "Mastery (very large data, few epochs)",
            },
            5: {
                "dataset_limit": 50000,
                "epochs": 5,
                "batch_size": 128,
                "description": "Perfection (massive data, minimal epochs)",
            },
        }

        # Минимальное логирование инициализации
        if dataset_limit_override or batch_size_override:
            logger.warning(
                f"[CONFIG] Overrides: dataset={dataset_limit_override}, batch={batch_size_override}"
            )

    def get_stage_config(self, stage: int) -> StageConfig:
        """
        Получает конфигурацию для определенной стадии обучения

        Args:
            stage: Номер стадии (1-5, для больших используется стадия 5)

        Returns:
            StageConfig: Конфигурация стадии
        """
        # Возвращаем конфигурацию или последнюю если stage слишком большой
        base_config = self._base_configs.get(stage, self._base_configs[5])
        config = base_config.copy()

        # Переопределяем параметры если заданы override
        if self.dataset_limit_override or self.batch_size_override:
            if self.dataset_limit_override:
                config["dataset_limit"] = self.dataset_limit_override
                config["description"] += f" (dataset: {self.dataset_limit_override})"

            if self.batch_size_override:
                config["batch_size"] = self.batch_size_override
                config["description"] += f" (batch: {self.batch_size_override})"

        return StageConfig(
            dataset_limit=config["dataset_limit"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            description=config["description"],
            stage=stage,
        )

    def get_all_stages_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Получает информацию обо всех стадиях для предварительного анализа

        Returns:
            Dict: Информация о всех стадиях с оценкой времени
        """
        stages_info = {}

        for stage in range(1, 6):
            config = self.get_stage_config(stage)
            estimated_time = self.estimate_stage_time(config, mode="development")

            stages_info[stage] = {
                "config": config,
                "estimated_time_minutes": estimated_time,
                "summary": f"Stage {stage}: {config.description}",
            }

        return stages_info

    def estimate_stage_time(
        self, config: StageConfig, mode: str = "development"
    ) -> float:
        """
        Оценивает время выполнения стадии в минутах

        Args:
            config: Конфигурация стадии
            mode: Режим обучения (development, research, production)

        Returns:
            float: Оценочное время в минутах
        """
        dataset_size = config.dataset_limit
        epochs = config.epochs
        batch_size = config.batch_size

        # Примерная оценка на основе размера датасета и режима
        if mode == "development":
            time_per_1k_examples = 2  # минут
        elif mode == "research":
            time_per_1k_examples = 5
        else:
            time_per_1k_examples = 10

        estimated_minutes = (dataset_size / 1000) * time_per_1k_examples * epochs / 10

        # Минимальное время для инициализации и маленьких датасетов
        min_time_minutes = 8.0  # минимум 8 минут (увеличено)
        if dataset_size <= 100:  # Для очень маленьких датасетов
            min_time_minutes = 12.0  # минимум 12 минут
        elif dataset_size <= 1000:  # Для маленьких датасетов
            min_time_minutes = 10.0  # минимум 10 минут

        # Корректировка для больших batch size (ускоряют обучение)
        if batch_size >= 128:
            estimated_minutes *= 0.5  # 50% ускорение
        elif batch_size >= 64:
            estimated_minutes *= 0.7  # 30% ускорение

        return max(estimated_minutes, min_time_minutes)

    def validate_stage_config(self, config: StageConfig) -> bool:
        """Валидирует конфигурацию стадии (с минимальным логированием)"""
        if config.dataset_limit <= 0:
            logger.error(f"❌ Invalid dataset_limit: {config.dataset_limit}")
            return False

        if config.epochs <= 0:
            logger.error(f"❌ Invalid epochs: {config.epochs}")
            return False

        if config.batch_size <= 0:
            logger.error(f"❌ Invalid batch_size: {config.batch_size}")
            return False

        return True
