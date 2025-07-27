"""
Базовый класс для провайдеров данных
===================================

Определяет единый интерфейс для всех источников данных в dataset модуле
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Iterator
import torch
from pathlib import Path

from ...utils.logging import get_logger

logger = get_logger(__name__)


class BaseDataProvider(ABC):
    """
    Базовый класс для всех провайдеров данных
    
    Определяет единый интерфейс для получения текстовых пар и эмбеддингов
    """
    
    def __init__(self, name: str, config):
        """
        Args:
            name: Название провайдера
            config: DatasetConfig с настройками
        """
        self.name = name
        self.config = config
        self.device = torch.device(config.device or 'cuda')
        self._is_initialized = False
        self._cached_data = None
        
        logger.info(f"🔧 Initializing {self.name} provider on {self.device}")
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Проверить доступность источника данных
        
        Returns:
            True если источник готов к использованию
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Инициализировать провайдер (загрузка моделей, данных и т.д.)
        
        Returns:
            True если инициализация успешна
        """
        pass
    
    @abstractmethod
    def get_text_pairs(self, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Получить пары текстов (input, target)
        
        Args:
            max_samples: Максимальное количество пар (None = все доступные)
            
        Returns:
            Список кортежей (input_text, target_text)
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить эмбеддинги для пар текстов
        
        Args:
            max_samples: Максимальное количество пар
            
        Returns:
            Tuple (input_embeddings, target_embeddings) каждый размером [N, embed_dim]
        """
        pass
    
    def get_mixed_data(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Получить и тексты и эмбеддинги одновременно
        
        Args:
            max_samples: Максимальное количество образцов
            
        Returns:
            Словарь с ключами: 'text_pairs', 'input_embeddings', 'target_embeddings'
        """
        text_pairs = self.get_text_pairs(max_samples)
        input_embeddings, target_embeddings = self.get_embeddings(max_samples)
        
        return {
            'text_pairs': text_pairs,
            'input_embeddings': input_embeddings,
            'target_embeddings': target_embeddings,
            'source': self.name,
            'count': len(text_pairs)
        }
    
    def validate_embeddings(self, embeddings: torch.Tensor, name: str = "embeddings") -> bool:
        """
        Валидация эмбеддингов на корректность
        
        Args:
            embeddings: Тензор эмбеддингов для проверки
            name: Название для логирования
            
        Returns:
            True если эмбеддинги валидны
        """
        if not self.config.validate_embeddings:
            return True
        
        try:
            # Проверка формы
            if embeddings.dim() != 2:
                logger.warning(f"❌ {name}: неправильная размерность {embeddings.shape}")
                return False
            
            # Проверка на NaN/Inf
            if self.config.check_nan_inf:
                if torch.isnan(embeddings).any():
                    logger.warning(f"❌ {name}: содержит NaN значения")
                    return False
                if torch.isinf(embeddings).any():
                    logger.warning(f"❌ {name}: содержит Inf значения")
                    return False
            
            # Проверка норм
            norms = torch.norm(embeddings, dim=1)
            min_norm, max_norm = norms.min().item(), norms.max().item()
            
            if min_norm < self.config.min_embedding_norm:
                logger.warning(f"❌ {name}: слишком малая норма {min_norm:.6f}")
                return False
            if max_norm > self.config.max_embedding_norm:
                logger.warning(f"❌ {name}: слишком большая норма {max_norm:.6f}")
                return False
            
            logger.debug(f"✅ {name}: валидация пройдена, norm range [{min_norm:.4f}, {max_norm:.4f}]")
            return True
            
        except Exception as e:
            logger.error(f"❌ {name}: ошибка валидации - {e}")
            return False
    
    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Нормализация эмбеддингов если включена в конфигурации
        
        Args:
            embeddings: Эмбеддинги для нормализации
            
        Returns:
            Нормализованные эмбеддинги
        """
        if not self.config.normalize_embeddings:
            return embeddings
        
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        logger.debug(f"📐 Normalized embeddings: {embeddings.shape}")
        return normalized
    
    def ensure_initialized(self) -> bool:
        """
        Убеждается что провайдер инициализирован
        
        Returns:
            True если провайдер готов к работе
        """
        if not self._is_initialized:
            if not self.is_available():
                logger.error(f"❌ {self.name}: источник данных недоступен")
                return False
            
            if not self.initialize():
                logger.error(f"❌ {self.name}: не удалось инициализировать провайдер")
                return False
                
            self._is_initialized = True
            logger.info(f"✅ {self.name}: провайдер успешно инициализирован")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику по провайдеру
        
        Returns:
            Словарь со статистикой
        """
        if not self.ensure_initialized():
            return {'error': 'Provider not initialized'}
        
        try:
            # Базовая статистика
            text_pairs = self.get_text_pairs(max_samples=100)  # Выборка для статистики
            
            stats = {
                'name': self.name,
                'is_available': self.is_available(),
                'is_initialized': self._is_initialized,
                'device': str(self.device),
                'sample_count': len(text_pairs),
            }
            
            if text_pairs:
                # Статистика по длинам текстов
                input_lengths = [len(pair[0].split()) for pair in text_pairs]
                target_lengths = [len(pair[1].split()) for pair in text_pairs]
                
                stats.update({
                    'avg_input_length': sum(input_lengths) / len(input_lengths),
                    'avg_target_length': sum(target_lengths) / len(target_lengths),
                    'max_input_length': max(input_lengths),
                    'max_target_length': max(target_lengths)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ {self.name}: ошибка получения статистики - {e}")
            return {'error': str(e)}
    
    def __str__(self) -> str:
        return f"{self.name}Provider(initialized={self._is_initialized}, device={self.device})"
    
    def __repr__(self) -> str:
        return self.__str__()