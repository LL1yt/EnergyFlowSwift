"""
Модуль Data Visualization - 3D Визуализация Клеточной Нейронной Сети

Этот модуль предоставляет инструменты для визуализации 3D клеточной нейронной сети,
включая отображение состояний клеток, I/O точек, анимацию распространения сигналов
и интерактивные дашборды.

Основные классы:
- VisualizationConfig: Конфигурация визуализации
- Lattice3DVisualizer: Основной визуализатор решетки
- IOPointVisualizer: Специализированная визуализация I/O точек
- load_visualization_config: Загрузка конфигурации из YAML

Использование:
    from data.data_visualization import create_visualizer, create_io_visualizer
    
    # Создание визуализатора
    visualizer = create_visualizer()
    
    # Визуализация решетки
    fig = visualizer.visualize_lattice(lattice)
    fig.show()
"""

from .main import (
    VisualizationConfig,
    load_visualization_config,
    RenderEngine,
    VisualizationMode,
    ExportFormat
)

# Импортируем визуализаторы только если они созданы
try:
    from .visualizers import Lattice3DVisualizer, IOPointVisualizer
    _VISUALIZERS_AVAILABLE = True
except ImportError:
    _VISUALIZERS_AVAILABLE = False
    Lattice3DVisualizer = None
    IOPointVisualizer = None


# =============================================================================
# УДОБНЫЕ ФУНКЦИИ СОЗДАНИЯ
# =============================================================================

def create_visualizer(config_path: str = None) -> 'Lattice3DVisualizer':
    """
    Создает визуализатор с загруженной конфигурацией.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Lattice3DVisualizer: Готовый к использованию визуализатор
        
    Raises:
        ImportError: Если визуализаторы недоступны
    """
    if not _VISUALIZERS_AVAILABLE:
        raise ImportError("Visualizers are not available. Please check the installation.")
        
    config = load_visualization_config(config_path)
    return Lattice3DVisualizer(config)


def create_io_visualizer(config_path: str = None) -> 'IOPointVisualizer':
    """
    Создает визуализатор I/O точек.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        IOPointVisualizer: Визуализатор I/O точек
        
    Raises:
        ImportError: Если визуализаторы недоступны
    """
    if not _VISUALIZERS_AVAILABLE:
        raise ImportError("Visualizers are not available. Please check the installation.")
        
    config = load_visualization_config(config_path)
    return IOPointVisualizer(config)


def quick_visualize_lattice(lattice, title: str = "3D Neural Network", 
                           show_io_points: bool = True) -> 'go.Figure':
    """
    Быстрая визуализация решетки с настройками по умолчанию.
    
    Args:
        lattice: Объект решетки для визуализации
        title: Заголовок визуализации
        show_io_points: Показывать ли I/O точки
        
    Returns:
        go.Figure: Plotly фигура для отображения
    """
    if not _VISUALIZERS_AVAILABLE:
        raise ImportError("Visualizers are not available. Please check the installation.")
        
    config = VisualizationConfig()
    config.title = title
    
    visualizer = Lattice3DVisualizer(config)
    return visualizer.visualize_lattice(lattice)


def quick_visualize_io_strategy(io_placer, face=None) -> 'go.Figure':
    """
    Быстрая визуализация стратегии размещения I/O точек.
    
    Args:
        io_placer: Объект размещения I/O точек
        face: Грань для визуализации (по умолчанию FRONT)
        
    Returns:
        go.Figure: Plotly фигура для отображения
    """
    if not _VISUALIZERS_AVAILABLE:
        raise ImportError("Visualizers are not available. Please check the installation.")
        
    from core.lattice_3d import Face
    if face is None:
        face = Face.FRONT
        
    config = VisualizationConfig()
    visualizer = IOPointVisualizer(config)
    return visualizer.visualize_io_strategy(io_placer, face)


# =============================================================================
# ЭКСПОРТ МОДУЛЯ
# =============================================================================

__all__ = [
    # Основные классы
    'VisualizationConfig',
    
    # Функции загрузки
    'load_visualization_config',
    
    # Функции создания
    'create_visualizer',
    'create_io_visualizer',
    
    # Быстрые функции
    'quick_visualize_lattice',
    'quick_visualize_io_strategy',
    
    # Enums
    'RenderEngine',
    'VisualizationMode',
    'ExportFormat'
]

# Добавляем визуализаторы в экспорт только если они доступны
if _VISUALIZERS_AVAILABLE:
    __all__.extend(['Lattice3DVisualizer', 'IOPointVisualizer'])


# =============================================================================
# ИНФОРМАЦИЯ О МОДУЛЕ
# =============================================================================

__version__ = "1.0.0"
__author__ = "3D Cellular Neural Network Project"
__description__ = "3D visualization tools for cellular neural networks"

# Проверяем доступность зависимостей
_DEPENDENCIES_STATUS = {
    'plotly': True,
    'numpy': True,
    'torch': True,
    'visualizers': _VISUALIZERS_AVAILABLE
}

def get_module_info() -> dict:
    """
    Возвращает информацию о модуле и статусе зависимостей.
    
    Returns:
        dict: Информация о модуле
    """
    return {
        'version': __version__,
        'description': __description__,
        'dependencies_status': _DEPENDENCIES_STATUS,
        'visualizers_available': _VISUALIZERS_AVAILABLE,
        'classes_available': [name for name in __all__ if name.endswith('Visualizer')]
    } 