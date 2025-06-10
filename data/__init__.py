"""
Data Package for 3D Cellular Neural Network

Этот пакет содержит модули для работы с данными:
- embedding_loader: Загрузка и обработка эмбеддингов (Teacher LLM Encoder)
- embedding_adapter: Адаптация эмбеддингов для 3D куба
- tokenizer: Токенизация текстов
- data_visualization: 3D визуализация решетки

Phase 2.3: EmbeddingReshaper - Мост между модулями
"""

# Экспорты основных модулей
from . import embedding_loader
from . import embedding_adapter
from . import tokenizer
from . import data_visualization

# Версия пакета
__version__ = "2.3.0"  # Обновлено для Phase 2.3
__phase__ = "Phase 2.3: Embedding Processing"

# Статус пакета
__status__ = "Active Development - Embedding Processing"

# Список доступных модулей
__all__ = ["embedding_loader", "embedding_adapter", "tokenizer", "data_visualization"]


def get_data_package_info():
    """Возвращает информацию о пакете data"""
    return {
        "version": __version__,
        "phase": __phase__,
        "status": __status__,
        "modules": __all__,
        "description": "Data processing modules for 3D Cellular Neural Network",
    }


def list_available_modules():
    """Выводит список доступных модулей"""
    print("[DATA] Data Package Modules:")
    for module in __all__:
        print(f"  - {module}")
    return __all__
