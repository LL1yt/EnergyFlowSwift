#!/usr/bin/env python3
"""
Базовые интерфейсы для работы с кубом
====================================

Определяет общие интерфейсы, используемые как в обучении, так и в инференсе.
Обеспечивает единообразный API для работы с эмбедингами и кубом.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union
import torch
import torch.nn as nn

from ...utils.logging import get_logger
from ...config.simple_config import SimpleProjectConfig

logger = get_logger(__name__)


class EmbeddingProcessor(ABC):
    """Базовый интерфейс для обработки эмбедингов"""
    
    @abstractmethod
    def transform_to_cube(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Преобразование эмбедингов в формат куба (Teacher → Cube)"""
        pass
    
    @abstractmethod
    def transform_from_cube(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Преобразование эмбедингов из формата куба (Cube → Teacher)"""
        pass


class CubeInterface(ABC):
    """Базовый интерфейс для работы с 3D кубом"""
    
    def __init__(self, config: SimpleProjectConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    @abstractmethod
    def forward(self, input_embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        """Прямой проход через куб"""
        pass
    
    @abstractmethod
    def get_surface_cells(self, surface: str = "front") -> torch.Tensor:
        """Получение клеток поверхности куба"""
        pass
    
    @abstractmethod
    def set_surface_cells(self, surface: str, embeddings: torch.Tensor):
        """Установка эмбедингов на поверхность куба"""
        pass


class TrainingInterface(ABC):
    """Интерфейс для обучения куба"""
    
    @abstractmethod
    def train_epoch(self, dataloader, optimizer, **kwargs) -> Dict[str, float]:
        """Обучение одной эпохи"""
        pass
    
    @abstractmethod
    def validate_epoch(self, dataloader, **kwargs) -> Dict[str, float]:
        """Валидация одной эпохи"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **metadata):
        """Сохранение checkpoint'а"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Загрузка checkpoint'а"""
        pass


class InferenceInterface(ABC):
    """Интерфейс для продуктивной работы куба"""
    
    @abstractmethod
    def process_text(self, text: str) -> str:
        """Обработка текста: Text → Embedding → Cube → Embedding → Text"""
        pass
    
    @abstractmethod
    def process_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Обработка эмбединга: Embedding → Cube → Embedding"""
        pass
    
    @abstractmethod
    def batch_process(self, texts: list) -> list:
        """Батчевая обработка текстов"""
        pass


class ContinualLearningInterface(ABC):
    """Интерфейс для постоянного обучения"""
    
    @abstractmethod
    def update_from_feedback(self, input_text: str, expected_output: str, 
                           actual_output: str, feedback_score: float):
        """Обновление модели на основе пользовательского фидбэка"""
        pass
    
    @abstractmethod
    def enable_learning_mode(self):
        """Включение режима постоянного обучения"""
        pass
    
    @abstractmethod
    def disable_learning_mode(self):
        """Выключение режима постоянного обучения"""
        pass


class EmbeddingCache(ABC):
    """Интерфейс для кэширования эмбедингов"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Получение эмбединга из кэша"""
        pass
    
    @abstractmethod
    def put(self, key: str, embedding: torch.Tensor):
        """Сохранение эмбединга в кэш"""
        pass
    
    @abstractmethod
    def clear(self):
        """Очистка кэша"""
        pass


class ValidationInterface(ABC):
    """Интерфейс для валидации качества обучения"""
    
    @abstractmethod
    def compute_semantic_similarity(self, output_embeddings: torch.Tensor, 
                                  target_embeddings: torch.Tensor) -> float:
        """Вычисление семантического сходства"""
        pass
    
    @abstractmethod
    def run_probing_tasks(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Запуск probing задач для проверки понимания"""
        pass
    
    @abstractmethod
    def generate_samples(self, embeddings: torch.Tensor, num_samples: int = 5) -> list:
        """Генерация примеров для визуального контроля качества"""
        pass


# === ФАБРИЧНЫЕ ФУНКЦИИ ===

def create_cube_interface(mode: str, config: SimpleProjectConfig) -> CubeInterface:
    """Фабричная функция для создания интерфейса куба"""
    if mode == "training":
        from ..training.cube_trainer import TrainingCubeInterface
        return TrainingCubeInterface(config)
    elif mode == "inference":
        from ..inference.cube_interface import CubeInferenceInterface
        return CubeInferenceInterface(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'training' or 'inference'")


def create_embedding_processor(config: SimpleProjectConfig) -> EmbeddingProcessor:
    """Фабричная функция для создания процессора эмбедингов"""
    from .embedding_transformer import EmbeddingTransformer
    return EmbeddingTransformer(config)