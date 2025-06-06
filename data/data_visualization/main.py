"""
Модуль Data Visualization - 3D Визуализация Клеточной Нейронной Сети

Этот модуль предоставляет инструменты для визуализации 3D клеточной нейронной сети,
включая отображение состояний клеток, I/O точек, анимацию распространения сигналов
и интерактивные дашборды.

Основные компоненты:
- VisualizationConfig: Конфигурация визуализации
- Lattice3DVisualizer: Основной визуализатор решетки
- IOPointVisualizer: Специализированная визуализация I/O точек
- AnimationController: Управление анимацией
- InteractiveDashboard: Интерактивный дашборд с метриками

Биологическая аналогия: как МРТ или fMRI сканирование мозга, показывающее
активность нейронных сетей в реальном времени.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime
import json

# Импорты из других модулей проекта
from core.lattice_3d import Lattice3D, IOPointPlacer, Face, PlacementStrategy
from core.signal_propagation import SignalPropagator


# =============================================================================
# БАЗОВЫЕ ТИПЫ И КОНСТАНТЫ
# =============================================================================

class RenderEngine(Enum):
    """Движки рендеринга для визуализации"""
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    MAYAVI = "mayavi"


class VisualizationMode(Enum):
    """Режимы визуализации"""
    STATIC = "static"         # Статичное отображение
    ANIMATED = "animated"     # Анимация
    INTERACTIVE = "interactive"  # Интерактивный режим
    DASHBOARD = "dashboard"   # Дашборд с метриками


class ExportFormat(Enum):
    """Поддерживаемые форматы экспорта"""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    MP4 = "mp4"
    GIF = "gif"
    PDF = "pdf"


# Типы для координат и цветов
Coordinates3D = Tuple[int, int, int]
RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]


# =============================================================================
# КОНФИГУРАЦИЯ ВИЗУАЛИЗАЦИИ
# =============================================================================

@dataclass
class VisualizationConfig:
    """
    Конфигурация для модуля визуализации.
    
    Содержит все настройки для отображения 3D решетки, анимации,
    интерактивности и экспорта результатов.
    """
    
    # Основные настройки дисплея
    title: str = "3D Cellular Neural Network Visualization"
    width: int = 1200
    height: int = 800
    background_color: str = "#f8f9fa"
    
    # Рендеринг
    engine: RenderEngine = RenderEngine.PLOTLY
    quality: str = "high"
    anti_aliasing: bool = True
    max_fps: int = 60
    
    # Настройки решетки
    cell_size: float = 0.8
    cell_opacity: float = 0.7
    show_connections: bool = False
    connection_opacity: float = 0.2
    
    # Цвета активации клеток
    cell_colors: Dict[str, str] = field(default_factory=lambda: {
        "inactive": "#e9ecef",
        "active_low": "#28a745",
        "active_medium": "#ffc107",
        "active_high": "#dc3545"
    })
    
    # Пороги активации
    activation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.1,
        "medium": 0.5,
        "high": 0.8
    })
    
    # I/O точки
    input_point_size: float = 1.5
    input_point_color: str = "#007bff"
    output_point_size: float = 1.5
    output_point_color: str = "#28a745"
    show_flow_arrows: bool = True
    
    # Анимация
    enable_animation: bool = True
    animation_speed: float = 1.0
    trail_length: int = 5
    wave_color: str = "#17a2b8"
    pulse_active_cells: bool = True
    
    # Интерактивность
    enable_interaction: bool = True
    hover_info: List[str] = field(default_factory=lambda: ["position", "activation", "neighbors"])
    clickable_cells: bool = True
    
    # Производительность
    max_lattice_size: Tuple[int, int, int] = (50, 50, 50)
    level_of_detail: bool = True
    adaptive_quality: bool = True
    
    # Экспорт
    export_formats: List[ExportFormat] = field(default_factory=lambda: [
        ExportFormat.PNG, ExportFormat.HTML, ExportFormat.MP4
    ])
    export_resolution: Tuple[int, int] = (1920, 1080)
    video_fps: int = 24
    
    # Дашборд
    enable_dashboard: bool = True
    update_interval: int = 100  # миллисекунд
    show_metrics: List[str] = field(default_factory=lambda: [
        "total_activation", "convergence_rate", "io_utilization"
    ])
    
    def __post_init__(self):
        """Валидация конфигурации после создания"""
        self._validate_settings()
        self._normalize_colors()
        
    def _validate_settings(self):
        """Проверка корректности настроек"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
            
        if not (0 <= self.cell_opacity <= 1):
            raise ValueError("Cell opacity must be between 0 and 1")
            
        for threshold in self.activation_thresholds.values():
            if not (0 <= threshold <= 1):
                raise ValueError("Activation thresholds must be between 0 and 1")
                
    def _normalize_colors(self):
        """Нормализация цветовых значений"""
        # Проверяем, что все цвета в правильном формате
        for color_name, color_value in self.cell_colors.items():
            if not color_value.startswith('#'):
                logging.warning(f"Color {color_name} should start with #")


# =============================================================================
# ФУНКЦИИ СОЗДАНИЯ И ЗАГРУЗКИ
# =============================================================================

def load_visualization_config(config_path: Optional[str] = None) -> VisualizationConfig:
    """
    Загружает конфигурацию визуализации из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        VisualizationConfig: Объект конфигурации
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "default.yaml"
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        # Извлекаем настройки визуализации
        viz_config = config_data.get('data_visualization', {})
        
        # Создаем объект конфигурации
        config = VisualizationConfig()
        
        # Обновляем поля из YAML
        if 'display' in viz_config:
            display = viz_config['display']
            config.title = display.get('title', config.title)
            config.width = display.get('width', config.width)
            config.height = display.get('height', config.height)
            config.background_color = display.get('background_color', config.background_color)
            
        return config
        
    except Exception as e:
        logging.error(f"Error loading visualization config: {e}")
        return VisualizationConfig()  # Возвращаем дефолтную конфигурацию


# =============================================================================
# ЭКСПОРТ МОДУЛЯ
# =============================================================================

__all__ = [
    # Основные классы
    'VisualizationConfig',
    
    # Функции
    'load_visualization_config',
    
    # Enums
    'RenderEngine',
    'VisualizationMode',
    'ExportFormat'
] 