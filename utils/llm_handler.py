"""
LLM Handler wrapper для совместимости с существующим кодом.

Этот модуль предоставляет wrapper для LLMHandler из data.embedding_loader,
чтобы обеспечить совместимость с кодом, который ищет utils.llm_handler.
"""

# Импортируем класс из правильного места
from data.embedding_loader.format_handlers import LLMHandler as BaseLLMHandler
from data.embedding_loader.format_handlers import SUPPORTED_LLM_MODELS as BASE_SUPPORTED_LLM_MODELS
import torch
from typing import List

class LLMHandler(BaseLLMHandler):
    """
    Wrapper для LLMHandler с дополнительными методами совместимости
    """
    
    def generate_embedding(self, text: str, pooling_strategy: str = "mean") -> torch.Tensor:
        """
        Генерация одного эмбединга (wrapper для совместимости)
        
        Args:
            text: Входной текст
            pooling_strategy: Стратегия агрегации
            
        Returns:
            torch.Tensor: Эмбединг текста [hidden_size]
        """
        # Используем generate_embeddings с одним текстом
        embeddings = self.generate_embeddings([text], pooling_strategy)
        return embeddings.squeeze(0)  # Убираем batch dimension
    
    def generate_embeddings_batch(self, texts: List[str], pooling_strategy: str = "mean") -> torch.Tensor:
        """
        Альтернативное название для generate_embeddings
        """
        return self.generate_embeddings(texts, pooling_strategy)

def create_llm_handler(model_key: str) -> LLMHandler:
    """
    Фабричная функция для создания LLM handler с нашим wrapper
    
    Args:
        model_key: Ключ модели из SUPPORTED_LLM_MODELS
        
    Returns:
        LLMHandler: Настроенный wrapper handler
    """
    if model_key not in BASE_SUPPORTED_LLM_MODELS:
        raise ValueError(f"Unsupported model key: {model_key}. "
                        f"Supported: {list(BASE_SUPPORTED_LLM_MODELS.keys())}")
    
    model_name = BASE_SUPPORTED_LLM_MODELS[model_key]
    return LLMHandler(model_name)  # Используем наш wrapper класс

# Re-export основные компоненты
SUPPORTED_LLM_MODELS = BASE_SUPPORTED_LLM_MODELS

__all__ = [
    'LLMHandler',
    'SUPPORTED_LLM_MODELS', 
    'create_llm_handler'
] 