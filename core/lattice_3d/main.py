"""
Модуль Lattice 3D - Трехмерная Решетка Клеток

Этот модуль реализует трехмерную решетку "умных клеток" для нейронной сети
клеточного типа. Основные компоненты:

- LatticeConfig: Конфигурация решетки
- Position3D: Работа с 3D координатами  
- NeighborTopology: Система соседства и граничные условия
- Lattice3D: Главный класс решетки

Биологическая аналогия: кора головного мозга как 3D ткань из взаимосвязанных 
нейронов, где каждый нейрон связан с ближайшими соседями.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Импорты из других модулей проекта
from core.cell_prototype import CellPrototype, create_cell_from_config


# =============================================================================
# БАЗОВЫЕ ТИПЫ И КОНСТАНТЫ
# =============================================================================

class BoundaryCondition(Enum):
    """Типы граничных условий для решетки"""
    WALLS = "walls"           # Границы блокируют сигналы
    PERIODIC = "periodic"     # Решетка замыкается в тор
    ABSORBING = "absorbing"   # Сигналы затухают на границах
    REFLECTING = "reflecting" # Сигналы отражаются от границ


class Face(Enum):
    """Грани решетки для ввода/вывода"""
    FRONT = "front"   # Z = 0
    BACK = "back"     # Z = max
    LEFT = "left"     # X = 0
    RIGHT = "right"   # X = max
    TOP = "top"       # Y = max
    BOTTOM = "bottom" # Y = 0


# Типы для координат и размеров
Coordinates3D = Tuple[int, int, int]
Dimensions3D = Tuple[int, int, int]


# =============================================================================
# КОНФИГУРАЦИЯ РЕШЕТКИ
# =============================================================================

@dataclass
class LatticeConfig:
    """
    Конфигурация трехмерной решетки клеток.
    
    Содержит все параметры, необходимые для создания и настройки решетки:
    размеры, граничные условия, параметры производительности и интеграции.
    """
    
    # Основные параметры решетки
    dimensions: Dimensions3D = (8, 8, 8)
    boundary_conditions: BoundaryCondition = BoundaryCondition.WALLS
    
    # Производительность
    parallel_processing: bool = True
    gpu_enabled: bool = True
    batch_size: int = 1
    
    # Инициализация состояний
    initialization_method: str = "normal"
    initialization_std: float = 0.1
    initialization_mean: float = 0.0
    
    # Топология
    neighbors: int = 6
    validate_connections: bool = True
    cache_neighbors: bool = True
    
    # Интерфейсы ввода/вывода
    input_face: Face = Face.FRONT
    output_face: Face = Face.BACK
    embedding_mapping: str = "linear"
    
    # Диагностика
    enable_logging: bool = True
    log_level: str = "INFO"
    track_performance: bool = True
    validate_states: bool = True
    
    # Оптимизация
    memory_efficient: bool = True
    use_checkpointing: bool = False
    mixed_precision: bool = False
    
    # Интеграция с cell_prototype
    cell_config: Optional[Dict[str, Any]] = None
    auto_sync_cell_config: bool = True
    
    def __post_init__(self):
        """Валидация и нормализация конфигурации после создания"""
        self._validate_dimensions()
        self._normalize_boundary_conditions()
        self._setup_logging()
        
    def _validate_dimensions(self):
        """Проверка корректности размеров решетки"""
        if len(self.dimensions) != 3:
            raise ValueError("Dimensions must be a 3-tuple (X, Y, Z)")
            
        if any(dim < 1 for dim in self.dimensions):
            raise ValueError("All dimensions must be positive")
            
        if any(dim > 1000 for dim in self.dimensions):
            logging.warning(f"Large dimensions {self.dimensions} may cause performance issues")
            
    def _normalize_boundary_conditions(self):
        """Нормализация граничных условий"""
        if isinstance(self.boundary_conditions, str):
            try:
                self.boundary_conditions = BoundaryCondition(self.boundary_conditions)
            except ValueError:
                raise ValueError(f"Unknown boundary condition: {self.boundary_conditions}")
                
    def _setup_logging(self):
        """Настройка логирования для модуля"""
        if self.enable_logging:
            logging.basicConfig(level=getattr(logging, self.log_level.upper()))
            
    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        
    @property
    def device(self) -> torch.device:
        """Устройство для вычислений (CPU или GPU)"""
        if self.gpu_enabled and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def load_lattice_config(config_path: Optional[str] = None) -> LatticeConfig:
    """
    Загрузка конфигурации решетки из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации. Если None, используется default.yaml
        
    Returns:
        LatticeConfig: Загруженная конфигурация
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "default.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        # Извлекаем секцию lattice_3d
        lattice_data = yaml_data.get('lattice_3d', {})
        
        # Преобразуем в параметры LatticeConfig
        config_params = {
            'dimensions': tuple(lattice_data.get('dimensions', [8, 8, 8])),
            'boundary_conditions': lattice_data.get('boundary_conditions', 'walls'),
            'parallel_processing': lattice_data.get('parallel_processing', True),
            'gpu_enabled': lattice_data.get('gpu_enabled', True),
            'batch_size': lattice_data.get('batch_size', 1),
        }
        
        # Добавляем вложенные параметры
        if 'initialization' in lattice_data:
            init_data = lattice_data['initialization']
            config_params.update({
                'initialization_method': init_data.get('method', 'normal'),
                'initialization_std': init_data.get('std', 0.1),
                'initialization_mean': init_data.get('mean', 0.0),
            })
            
        if 'topology' in lattice_data:
            topo_data = lattice_data['topology']
            config_params.update({
                'neighbors': topo_data.get('neighbors', 6),
                'validate_connections': topo_data.get('validate_connections', True),
                'cache_neighbors': topo_data.get('cache_neighbors', True),
            })
            
        if 'io_interfaces' in lattice_data:
            io_data = lattice_data['io_interfaces']
            config_params.update({
                'input_face': Face(io_data.get('input_face', 'front')),
                'output_face': Face(io_data.get('output_face', 'back')),
                'embedding_mapping': io_data.get('embedding_mapping', 'linear'),
            })
            
        if 'diagnostics' in lattice_data:
            diag_data = lattice_data['diagnostics']
            config_params.update({
                'enable_logging': diag_data.get('enable_logging', True),
                'log_level': diag_data.get('log_level', 'INFO'),
                'track_performance': diag_data.get('track_performance', True),
                'validate_states': diag_data.get('validate_states', True),
            })
            
        if 'optimization' in lattice_data:
            opt_data = lattice_data['optimization']
            config_params.update({
                'memory_efficient': opt_data.get('memory_efficient', True),
                'use_checkpointing': opt_data.get('use_checkpointing', False),
                'mixed_precision': opt_data.get('mixed_precision', False),
            })
        
        # Загружаем конфигурацию cell_prototype если требуется автосинхронизация
        integration_data = yaml_data.get('integration', {})
        if integration_data.get('cell_prototype', {}).get('auto_sync', True):
            try:
                config_params['cell_config'] = load_cell_config()
                config_params['auto_sync_cell_config'] = True
                logging.info("Auto-synced with cell_prototype configuration")
            except Exception as e:
                logging.warning(f"Could not auto-sync cell_prototype config: {e}")
                config_params['auto_sync_cell_config'] = False
        
        return LatticeConfig(**config_params)
        
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}. Using default configuration.")
        return LatticeConfig()
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise


# =============================================================================
# СИСТЕМА КООРДИНАТ 3D
# =============================================================================

class Position3D:
    """
    Класс для работы с трехмерными координатами в решетке.
    
    Предоставляет функции для:
    - Преобразования между 3D координатами и линейными индексами
    - Валидации координат
    - Вычисления расстояний и соседства
    """
    
    def __init__(self, dimensions: Dimensions3D):
        """
        Инициализация системы координат.
        
        Args:
            dimensions: Размеры решетки (X, Y, Z)
        """
        self.dimensions = dimensions
        self.x_size, self.y_size, self.z_size = dimensions
        self.total_positions = self.x_size * self.y_size * self.z_size
        
    def to_linear_index(self, coords: Coordinates3D) -> int:
        """
        Преобразование 3D координат в линейный индекс.
        
        Использует порядок: Z * (X * Y) + Y * X + X
        
        Args:
            coords: 3D координаты (x, y, z)
            
        Returns:
            int: Линейный индекс
        """
        x, y, z = coords
        self._validate_coordinates(coords)
        return z * (self.x_size * self.y_size) + y * self.x_size + x
        
    def to_3d_coordinates(self, linear_index: int) -> Coordinates3D:
        """
        Преобразование линейного индекса в 3D координаты.
        
        Args:
            linear_index: Линейный индекс
            
        Returns:
            Coordinates3D: 3D координаты (x, y, z)
        """
        if not 0 <= linear_index < self.total_positions:
            raise ValueError(f"Linear index {linear_index} out of range [0, {self.total_positions})")
            
        z = linear_index // (self.x_size * self.y_size)
        remainder = linear_index % (self.x_size * self.y_size)
        y = remainder // self.x_size
        x = remainder % self.x_size
        
        return (x, y, z)
        
    def _validate_coordinates(self, coords: Coordinates3D) -> None:
        """Валидация 3D координат"""
        x, y, z = coords
        if not (0 <= x < self.x_size):
            raise ValueError(f"X coordinate {x} out of range [0, {self.x_size})")
        if not (0 <= y < self.y_size):
            raise ValueError(f"Y coordinate {y} out of range [0, {self.y_size})")
        if not (0 <= z < self.z_size):
            raise ValueError(f"Z coordinate {z} out of range [0, {self.z_size})")
            
    def is_valid_coordinates(self, coords: Coordinates3D) -> bool:
        """Проверка валидности координат без исключения"""
        try:
            self._validate_coordinates(coords)
            return True
        except ValueError:
            return False
            
    def get_all_coordinates(self) -> List[Coordinates3D]:
        """Получение списка всех валидных координат в решетке"""
        coordinates = []
        for z in range(self.z_size):
            for y in range(self.y_size):
                for x in range(self.x_size):
                    coordinates.append((x, y, z))
        return coordinates
        
    def manhattan_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> int:
        """Манхэттенское расстояние между двумя точками"""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) + abs(coord1[2] - coord2[2])
        
    def euclidean_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> float:
        """Евклидово расстояние между двумя точками"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))


# =============================================================================
# ТОПОЛОГИЯ СОСЕДСТВА
# =============================================================================

class NeighborTopology:
    """
    Система управления соседством клеток в 3D решетке.
    
    Реализует различные типы граничных условий и предоставляет
    эффективные методы для получения соседей каждой клетки.
    """
    
    # Направления к 6 соседям в 3D пространстве
    NEIGHBOR_DIRECTIONS = [
        (-1,  0,  0),  # Влево
        ( 1,  0,  0),  # Вправо
        ( 0, -1,  0),  # Вниз
        ( 0,  1,  0),  # Вверх
        ( 0,  0, -1),  # Назад
        ( 0,  0,  1),  # Вперед
    ]
    
    def __init__(self, config: LatticeConfig):
        """
        Инициализация системы соседства.
        
        Args:
            config: Конфигурация решетки
        """
        self.config = config
        self.dimensions = config.dimensions
        self.boundary_conditions = config.boundary_conditions
        self.position_system = Position3D(self.dimensions)
        
        # Кэш для карты соседства
        self._neighbor_cache: Dict[int, List[int]] = {}
        self._cache_initialized = False
        
        if config.cache_neighbors:
            self._build_neighbor_cache()
            
    def get_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """
        Получение соседей для заданных координат.
        
        Args:
            coords: 3D координаты клетки
            
        Returns:
            List[Coordinates3D]: Список координат соседних клеток
        """
        x, y, z = coords
        neighbors = []
        
        for dx, dy, dz in self.NEIGHBOR_DIRECTIONS:
            neighbor_coords = (x + dx, y + dy, z + dz)
            
            # Применяем граничные условия
            processed_coords = self._apply_boundary_conditions(neighbor_coords)
            
            if processed_coords is not None:
                neighbors.append(processed_coords)
                
        return neighbors
        
    def get_neighbor_indices(self, linear_index: int) -> List[int]:
        """
        Получение индексов соседей для линейного индекса.
        
        Args:
            linear_index: Линейный индекс клетки
            
        Returns:
            List[int]: Список линейных индексов соседних клеток
        """
        if self.config.cache_neighbors and self._cache_initialized:
            return self._neighbor_cache.get(linear_index, [])
            
        coords = self.position_system.to_3d_coordinates(linear_index)
        neighbor_coords = self.get_neighbors(coords)
        
        return [self.position_system.to_linear_index(nc) for nc in neighbor_coords]
        
    def _apply_boundary_conditions(self, coords: Coordinates3D) -> Optional[Coordinates3D]:
        """
        Применение граничных условий к координатам.
        
        Args:
            coords: Координаты (могут быть вне границ)
            
        Returns:
            Optional[Coordinates3D]: Обработанные координаты или None если недоступны
        """
        x, y, z = coords
        x_size, y_size, z_size = self.dimensions
        
        if self.boundary_conditions == BoundaryCondition.WALLS:
            # Стенки: координаты вне границ недоступны
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return coords
            
        elif self.boundary_conditions == BoundaryCondition.PERIODIC:
            # Периодические: решетка замыкается в тор
            return (x % x_size, y % y_size, z % z_size)
            
        elif self.boundary_conditions == BoundaryCondition.ABSORBING:
            # Поглощающие: границы поглощают сигналы (как стенки для топологии)
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return coords
            
        elif self.boundary_conditions == BoundaryCondition.REFLECTING:
            # Отражающие: координаты отражаются от границ
            if x < 0:
                x = -x
            elif x >= x_size:
                x = 2 * x_size - x - 1
                
            if y < 0:
                y = -y
            elif y >= y_size:
                y = 2 * y_size - y - 1
                
            if z < 0:
                z = -z
            elif z >= z_size:
                z = 2 * z_size - z - 1
                
            # Проверяем, что отраженные координаты валидны
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return (x, y, z)
            
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary_conditions}")
            
    def _build_neighbor_cache(self):
        """
        Предварительное построение кэша соседства для всех клеток.
        
        Повышает производительность при частых запросах соседей.
        """
        logging.info("Building neighbor cache for lattice...")
        
        for linear_index in range(self.position_system.total_positions):
            coords = self.position_system.to_3d_coordinates(linear_index)
            neighbor_coords = self.get_neighbors(coords)
            neighbor_indices = [self.position_system.to_linear_index(nc) for nc in neighbor_coords]
            self._neighbor_cache[linear_index] = neighbor_indices
            
        self._cache_initialized = True
        logging.info(f"Neighbor cache built: {len(self._neighbor_cache)} entries")
        
    def validate_topology(self) -> Dict[str, Any]:
        """
        Валидация топологии решетки и возврат статистики.
        
        Returns:
            Dict[str, Any]: Статистика топологии
        """
        stats = {
            'total_cells': self.position_system.total_positions,
            'boundary_conditions': self.boundary_conditions.value,
            'neighbor_counts': {},
            'symmetry_check': True,
            'connectivity_check': True,
        }
        
        # Подсчет количества соседей для каждой клетки
        neighbor_counts = []
        for i in range(self.position_system.total_positions):
            neighbors = self.get_neighbor_indices(i)
            neighbor_counts.append(len(neighbors))
            
        # Статистика количества соседей
        unique_counts = list(set(neighbor_counts))
        for count in unique_counts:
            stats['neighbor_counts'][str(count)] = neighbor_counts.count(count)
            
        # Проверка симметрии соседства
        for i in range(self.position_system.total_positions):
            neighbors = self.get_neighbor_indices(i)
            for neighbor_idx in neighbors:
                neighbor_neighbors = self.get_neighbor_indices(neighbor_idx)
                if i not in neighbor_neighbors:
                    stats['symmetry_check'] = False
                    logging.warning(f"Asymmetric neighbor relationship: {i} -> {neighbor_idx}")
                    
        logging.info(f"Topology validation completed: {stats}")
        return stats


# =============================================================================
# ОСНОВНОЙ КЛАСС РЕШЕТКИ (ЗАГЛУШКА ДЛЯ СЛЕДУЮЩЕГО ЭТАПА)
# =============================================================================

class Lattice3D(nn.Module):
    """
    Основной класс трехмерной решетки клеток.
    
    [ЗАГЛУШКА - будет реализован в следующем этапе]
    
    Управляет сеткой клеток cell_prototype, их состояниями и взаимодействием.
    """
    
    def __init__(self, config: LatticeConfig):
        """Инициализация решетки (базовая версия)"""
        super().__init__()
        self.config = config
        
        # Будет реализовано на следующем этапе
        logging.info(f"Lattice3D initialized with config: {config.dimensions}")
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Заглушка для forward pass"""
        logging.info("Lattice3D.forward() called - implementation pending")
        return states


# =============================================================================
# УТИЛИТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def create_lattice_from_config(config_path: Optional[str] = None) -> Lattice3D:
    """
    Создание решетки на основе конфигурационного файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Lattice3D: Созданная и настроенная решетка
    """
    config = load_lattice_config(config_path)
    return Lattice3D(config)


def validate_lattice_config(config: LatticeConfig) -> Dict[str, Any]:
    """
    Валидация конфигурации решетки.
    
    Args:
        config: Конфигурация для проверки
        
    Returns:
        Dict[str, Any]: Результаты валидации
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Проверка размеров
    if config.total_cells > 100000:
        validation_results['warnings'].append(
            f"Large lattice size ({config.total_cells} cells) may impact performance"
        )
        
    # Проверка GPU
    if config.gpu_enabled and not torch.cuda.is_available():
        validation_results['warnings'].append(
            "GPU enabled but CUDA not available, falling back to CPU"
        )
        
    # Проверка конфигурации клеток
    if config.cell_config is None and config.auto_sync_cell_config:
        validation_results['recommendations'].append(
            "Consider loading cell_prototype configuration for better integration"
        )
        
    return validation_results


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ
# =============================================================================

def load_cell_config() -> Dict[str, Any]:
    """
    Загрузка конфигурации cell_prototype для совместимости.
    
    Returns:
        Dict[str, Any]: Базовая конфигурация cell_prototype
    """
    return {
        'cell_prototype': {
            'input_size': 12,
            'state_size': 8,
            'architecture': {
                'hidden_size': 16,
                'activation': 'tanh',
                'use_bias': True
            }
        }
    }


# =============================================================================
# ЭКСПОРТЫ МОДУЛЯ
# =============================================================================

__all__ = [
    # Основные классы
    'Lattice3D',
    'LatticeConfig', 
    'Position3D',
    'NeighborTopology',
    
    # Енумы
    'BoundaryCondition',
    'Face',
    
    # Функции
    'load_lattice_config',
    'create_lattice_from_config',
    'validate_lattice_config',
    
    # Типы
    'Coordinates3D',
    'Dimensions3D',
] 