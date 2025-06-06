"""
Data Package for 3D Cellular Neural Network

Этот пакет содержит модули для работы с данными:
- embedding_loader: Загрузка и обработка эмбеддингов
- tokenizer: Токенизация текстов
- data_visualization: 3D визуализация решетки

Phase 2: Core Functionality
"""

# Экспорты основных модулей
from . import embedding_loader
from . import tokenizer  
from . import data_visualization

# Версия пакета
__version__ = "2.0.0"
__phase__ = "Phase 2: Core Functionality"

# Статус пакета
__status__ = "Active Development"

# Список доступных модулей
__all__ = [
    'embedding_loader',
    'tokenizer', 
    'data_visualization'
]

def get_data_package_info():
    """Возвращает информацию о пакете data"""
    return {
        'version': __version__,
        'phase': __phase__,
        'status': __status__,
        'modules': __all__,
        'description': 'Data processing modules for 3D Cellular Neural Network'
    }

def list_available_modules():
    """Выводит список доступных модулей"""
    print("📦 Data Package Modules:")
    for module in __all__:
        print(f"  - {module}")
    return __all__ 