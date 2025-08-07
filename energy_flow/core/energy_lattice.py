"""
Energy Lattice - 3D решетка для управления энергетическими потоками
===================================================================

Управляет активными потоками, их позициями и взаимодействием с решеткой.
Работает только с входной и выходной сторонами куба.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
import numpy as np

from ..utils.logging import get_logger, log_memory_state
from ..config import get_energy_config, create_debug_config, set_energy_config
from ..utils.device_manager import get_device_manager

logger = get_logger(__name__)


class Position3D(NamedTuple):
    """3D позиция в решетке"""
    x: int
    y: int
    z: int


@dataclass
class EnergyFlow:
    """Представление одного энергетического потока"""
    id: int
    position: torch.Tensor  # [3] - текущая позиция (НОРМАЛИЗОВАННАЯ в [-1,1])
    energy: torch.Tensor    # [embedding_dim] - энергия/эмбеддинг
    hidden_state: torch.Tensor  # [num_layers, hidden_size] - состояние GRU
    batch_idx: int = 0  # Индекс в батче
    parent_id: Optional[int] = None
    age: int = 0  # Количество шагов жизни потока (использется для проекционной системы)
    is_active: bool = True
    steps_taken: int = 0  # Дополнительный счетчик для двухуровневой проекции
    distance_to_surface: float = 0.0  # Расстояние до ближайшей выходной поверхности
    projected_surface: str = "unknown"  # "z0" или "zdepth" - куда проецируется


class EnergyLattice(nn.Module):
    """
    3D решетка для управления энергетическими потоками
    
    Основные функции:
    - Размещение входных эмбеддингов на входной стороне
    - Управление активными потоками
    - Сбор выходных эмбеддингов с выходной стороны
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: EnergyConfig с настройками
        """
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        
        # Размеры решетки
        self.width = config.lattice_width
        self.height = config.lattice_height
        self.depth = config.lattice_depth
        
        # Параметры потоков
        self.max_active_flows = config.max_active_flows
        self.energy_dim = 1  # Скалярная энергия от mapper'а
        
        # Активные потоки
        self.active_flows: Dict[int, EnergyFlow] = {}
        self.next_flow_id = 0
        
        # УДАЛЕНО: Буферная система больше не используется в новой архитектуре прямой работы с потоками
        # Все данные хранятся напрямую в active_flows до момента сбора и удаления
        
        # Статистика
        self.stats = {
            'total_created': 0,
            'total_completed': 0,
            'total_died': 0,
            'max_concurrent': 0
        }
        
        
        # Предвычисляем нормализованную сетку координат для эффективного округления
        self._precompute_normalized_lattice_grid()
        
        # Предвычисляем mapping: нормализованные координаты -> surface_idx
        self._precompute_normalized_to_surface_mapping()
        
        logger.info(f"EnergyLattice initialized: {self.width}x{self.height}x{self.depth}")
        logger.info(f"Input/output cells: {self.width * self.height}, max flows: {self.max_active_flows}")
    
    def _precompute_normalized_lattice_grid(self):
        """Предвычисляет нормализованные координаты всех позиций решетки."""
        # Создаем сетку всех возможных дискретных координат
        x_coords = torch.arange(self.width, dtype=torch.float32)
        y_coords = torch.arange(self.height, dtype=torch.float32)  
        z_coords = torch.arange(self.depth + 1, dtype=torch.float32)  # +1 для включения depth
        
        # Создаем meshgrid для всех координат
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Формируем тензор координат [W, H, D+1, 3]
        lattice_coords = torch.stack([X, Y, Z], dim=-1)
        
        # Нормализуем все координаты в [-1, 1]
        normalized_coords = self.config.normalization_manager.normalize_coordinates(
            lattice_coords.view(-1, 3)
        ).view(self.width, self.height, self.depth + 1, 3)
        
        # Сохраняем предвычисленную сетку
        self.normalized_lattice_grid = normalized_coords
        
        logger.debug(f"Precomputed normalized lattice grid: {normalized_coords.shape}")
        logger.debug(f"Normalized coordinate ranges: X[{normalized_coords[..., 0].min():.3f}, {normalized_coords[..., 0].max():.3f}], "
                    f"Y[{normalized_coords[..., 1].min():.3f}, {normalized_coords[..., 1].max():.3f}], "
                    f"Z[{normalized_coords[..., 2].min():.3f}, {normalized_coords[..., 2].max():.3f}]")
    
    def _precompute_normalized_to_surface_mapping(self):
        """Предвычисляет mapping: (norm_x, norm_y) -> surface_idx для избежания денормализации."""
        self.normalized_to_surface_idx = {}
        
        # Для каждой дискретной позиции на поверхности
        for y in range(self.height):
            for x in range(self.width):
                # Получаем нормализованные координаты этой позиции
                raw_coords = torch.tensor([x, y, 0], dtype=torch.float32)  # Z не важно для surface
                norm_coords = self.config.normalization_manager.normalize_coordinates(
                    raw_coords.unsqueeze(0)
                )[0]
                
                # Создаем ключ (с округлением для консистентности)
                norm_key = (round(norm_coords[0].item(), 6), round(norm_coords[1].item(), 6))
                
                # Вычисляем surface_idx (линеаризация)
                surface_idx = y * self.width + x
                
                # Сохраняем mapping
                self.normalized_to_surface_idx[norm_key] = surface_idx
        
        logger.debug(f"Precomputed normalized->surface mapping for {len(self.normalized_to_surface_idx)} positions")
    
    def round_to_nearest_lattice_position(self, normalized_positions: torch.Tensor) -> torch.Tensor:
        """
        Округляет нормализованные позиции до ближайших координат решетки.
        ОПТИМИЗИРОВАННАЯ ВЕКТОРИЗОВАННАЯ ВЕРСИЯ для GPU параллелизма.
        
        Args:
            normalized_positions: [batch, 3] нормализованные позиции в [-1, 1]
            
        Returns:
            [batch, 3] нормализованные координаты ближайших позиций решетки
        """
        batch_size = normalized_positions.shape[0]
        device = normalized_positions.device
        
        # Переносим предвычисленную сетку на нужное устройство если необходимо
        if self.normalized_lattice_grid.device != device:
            self.normalized_lattice_grid = self.normalized_lattice_grid.to(device)
        
        # ВЕКТОРИЗОВАННАЯ ВЕРСИЯ: обрабатываем весь батч одновременно
        grid_flat = self.normalized_lattice_grid.view(-1, 3)  # [N_grid, 3]
        
        # Эффективное вычисление расстояний для всего батча
        # Используем broadcasting: [batch, 1, 3] - [1, N_grid, 3] = [batch, N_grid, 3]
        positions_expanded = normalized_positions.unsqueeze(1)  # [batch, 1, 3]
        grid_expanded = grid_flat.unsqueeze(0)  # [1, N_grid, 3]
        
        # Вычисляем разности
        diff = positions_expanded - grid_expanded  # [batch, N_grid, 3]
        
        # Вычисляем евклидовы расстояния
        distances = torch.norm(diff, dim=2)  # [batch, N_grid]
        
        # Находим индексы ближайших точек для всего батча
        nearest_indices = torch.argmin(distances, dim=1)  # [batch]
        
        # Извлекаем ближайшие позиции
        rounded_positions = grid_flat[nearest_indices]  # [batch, 3]
        
        return rounded_positions
    
    def get_normalized_buffer_key(self, normalized_position: torch.Tensor) -> Tuple[float, float]:
        """
        Получает ключ для буферизации из нормализованной позиции.
        
        Округляет позицию до ближайшей точки предвычисленной решетки,
        затем использует нормализованные X, Y координаты как ключ.
        
        Args:
            normalized_position: [3] нормализованная позиция в [-1, 1]
            
        Returns:
            (norm_x, norm_y): ключ для буферизации
        """
        # Округляем до ближайшей точки решетки
        rounded_pos = self.round_to_nearest_lattice_position(normalized_position.unsqueeze(0))[0]
        
        # Используем нормализованные X, Y как ключ (округляем для консистентности float хеширования)
        norm_x = round(rounded_pos[0].item(), 6)  # 6 знаков после запятой для точности
        norm_y = round(rounded_pos[1].item(), 6)
        
        return (norm_x, norm_y)
    
    def calculate_distance_to_nearest_surface(self, normalized_position: torch.Tensor) -> Tuple[float, str]:
        """
        Вычисляет расстояние до ближайшей выходной поверхности.
        
        Args:
            normalized_position: [3] нормализованная позиция в [-1, 1]
            
        Returns:
            (distance, surface_name): расстояние и название поверхности ("z0" или "zdepth")
        """
        norm_z = normalized_position[2].item()
        
        # Получаем нормализованные значения выходных поверхностей
        norm_z0 = self.config.normalization_manager._normalize_to_range(
            torch.tensor([0.0]), self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0].item()
        
        norm_zdepth = self.config.normalization_manager._normalize_to_range(
            torch.tensor([float(self.depth)]), self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0].item()
        
        # Вычисляем расстояния до обеих поверхностей
        distance_to_z0 = abs(norm_z - norm_z0)
        distance_to_zdepth = abs(norm_z - norm_zdepth)
        
        # Возвращаем ближайшую поверхность
        if distance_to_z0 <= distance_to_zdepth:
            return distance_to_z0, "z0"
        else:
            return distance_to_zdepth, "zdepth"
    
    # УДАЛЕНО: @property output_buffer - больше не нужен в архитектуре прямой работы с потоками
    
    def place_initial_energy(self, embeddings: torch.Tensor, mapper=None) -> List[int]:
        """
        Размещает входные эмбеддинги на входной плоскости в центре куба (Z = depth/2)
        ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: батчевое создание потоков
        
        DUAL OUTPUT PLANES АРХИТЕКТУРА:
        - Входная плоскость: Z = depth/2 (центр куба, normalized Z = 0.0)
        - Выходные плоскости: Z = 0 (normalized Z = -1.0) И Z = depth (normalized Z = +1.0)
        
        Потоки могут двигаться к любой из двух выходных плоскостей - модель сама выбирает
        оптимальное направление для каждого эмбеддинга.
        
        Args:
            embeddings: [batch, embedding_dim] - входные эмбеддинги (768D)
            mapper: EnergyFlowMapper для проекции (обязательно)
            
        Returns:
            flow_ids: Список ID созданных потоков
        """
        import time
        start_time = time.time()
        
        batch_size = embeddings.shape[0]
        
        # Проверяем размерность
        expected_dim = self.config.input_embedding_dim_from_teacher
        if embeddings.shape[1] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {embeddings.shape[1]}")
        
        # Очищаем неактивные потоки
        self._cleanup_inactive_flows()
        
        if mapper is None:
            raise ValueError("EnergyFlowMapper is required! No fallback logic allowed.")
        
        # Используем маппер для проекции 768D -> surface_dim
        cell_energies = mapper.map_to_surface(embeddings)
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: стартовая Z-координата в центре куба
        start_z = self.depth // 2  # Z = depth/2 (центр куба)
        
        # ОПТИМИЗАЦИЯ: логируем нормализацию только в debug режиме
        if logger.isEnabledFor(20):  # DEBUG_CONVERGENCE level
            test_raw = torch.tensor([0, 0, start_z], dtype=torch.float32, device=self.device)
            test_norm = self.config.normalization_manager.normalize_coordinates(test_raw.unsqueeze(0))[0]
            test_z0_raw = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)  
            test_z0_norm = self.config.normalization_manager.normalize_coordinates(test_z0_raw.unsqueeze(0))[0]
            test_zdepth_raw = torch.tensor([0, 0, self.depth], dtype=torch.float32, device=self.device)
            test_zdepth_norm = self.config.normalization_manager.normalize_coordinates(test_zdepth_raw.unsqueeze(0))[0]
            
            logger.debug_convergence(f"🔍 NORMALIZATION DEBUG:")
            logger.debug_convergence(f"  Raw start center Z={start_z} → normalized Z={test_norm[2]:.6f}")
            logger.debug_convergence(f"  Raw output Z=0 → normalized Z={test_z0_norm[2]:.6f}")  
            logger.debug_convergence(f"  Raw output Z={self.depth} → normalized Z={test_zdepth_norm[2]:.6f}")
            logger.debug_convergence(f"  Z normalization range: {self.config.normalization_manager.ranges.z_range}")
        
        # ОПТИМИЗАЦИЯ: Батчевое создание потоков
        flow_ids = self._batch_create_flows(cell_energies, start_z)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🏗️ Created {len(flow_ids)} initial flows on center input plane (raw Z={start_z}) in {elapsed_time:.2f}s")
        logger.info(f"🎯 Triplaner architecture: input Z={start_z}, outputs Z=0 and Z={self.depth}")
        return flow_ids
    
    def _batch_create_flows(self, cell_energies, start_z: int) -> List[int]:
        """
        Батчевое создание потоков - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
        Используем векторизованные операции вместо циклов
        
        Args:
            cell_energies: Список кортежей ((x, y), energy, batch_idx)
            start_z: Z-координата для всех потоков
            
        Returns:
            flow_ids: Список созданных ID потоков
        """
        # Ограничиваем количество потоков
        num_flows = min(len(cell_energies), self.max_active_flows)
        if num_flows < len(cell_energies):
            logger.warning(f"Limiting flows to {self.max_active_flows} (requested: {len(cell_energies)})")
            cell_energies = cell_energies[:num_flows]
        
        if num_flows == 0:
            return []
        
        # ОПТИМИЗАЦИЯ: Создаем все тензоры разом
        # Извлекаем данные через быстрые list comprehensions
        positions_xy = [ce[0] for ce in cell_energies]  # [(x, y), ...]
        energies_tensors = [ce[1] for ce in cell_energies]  # [tensor, ...]
        batch_indices = [ce[2] for ce in cell_energies]  # [idx, ...]
        
        # Создаем матрицу позиций векторизованно
        positions_tensor = torch.tensor(
            [[x, y, start_z] for x, y in positions_xy],
            dtype=torch.float32, 
            device=self.device
        )
        
        # Векторизованная нормализация всех позиций одним вызовом
        normalized_positions = self.config.normalization_manager.normalize_coordinates(positions_tensor)
        
        # ОПТИМИЗАЦИЯ: Векторизованное вычисление расстояний
        # Вычисляем расстояния до обеих поверхностей векторизованно
        norm_z_values = normalized_positions[:, 2]  # [num_flows]
        
        # Нормализованные Z для выходных поверхностей
        norm_z0 = self.config.normalization_manager._normalize_to_range(
            torch.tensor([0.0], device=self.device), 
            self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0]
        
        norm_zdepth = self.config.normalization_manager._normalize_to_range(
            torch.tensor([float(self.depth)], device=self.device), 
            self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0]
        
        # Векторизованные расстояния
        distances_to_z0 = torch.abs(norm_z_values - norm_z0)
        distances_to_zdepth = torch.abs(norm_z_values - norm_zdepth)
        
        # Определяем ближайшие поверхности
        is_closer_to_z0 = distances_to_z0 <= distances_to_zdepth
        distances = torch.where(is_closer_to_z0, distances_to_z0, distances_to_zdepth)
        surfaces = ["z0" if is_z0 else "zdepth" for is_z0 in is_closer_to_z0]
        
        # ОПТИМИЗАЦИЯ: Создаем все hidden states одним тензором
        num_layers = self.config.carrier_num_layers
        hidden_size = self.config.carrier_hidden_size
        all_hidden_states = torch.zeros(
            num_flows, num_layers, hidden_size, 
            device=self.device
        )
        
        # Генерируем ID потоков
        flow_ids = list(range(self.next_flow_id, self.next_flow_id + num_flows))
        self.next_flow_id += num_flows
        
        # Создаем все потоки батчем
        for i in range(num_flows):
            flow = EnergyFlow(
                id=flow_ids[i],
                position=normalized_positions[i],
                energy=energies_tensors[i],
                hidden_state=all_hidden_states[i],
                batch_idx=batch_indices[i],
                parent_id=None,
                age=0,
                is_active=True,
                steps_taken=0,
                distance_to_surface=distances[i].item(),
                projected_surface=surfaces[i]
            )
            
            self.active_flows[flow_ids[i]] = flow
            
            # Логируем только первые несколько для диагностики
            if i < 5 and logger.isEnabledFor(17):  # DEBUG_INIT level
                x, y = positions_xy[i]
                logger.debug_init(f"🅫 Created flow {flow_ids[i]}: raw=({x}, {y}, {start_z}) → "
                                f"norm=({normalized_positions[i][0]:.3f}, {normalized_positions[i][1]:.3f}, "
                                f"{normalized_positions[i][2]:.3f}), embedding_magnitude={torch.norm(energies_tensors[i]):.3f}")
        
        # Обновляем статистику
        self.stats['total_created'] += num_flows
        self.stats['max_concurrent'] = max(self.stats['max_concurrent'], len(self.active_flows))
        
        return flow_ids
    
    def _create_flow(self, position: torch.Tensor, energy: torch.Tensor, 
                    batch_idx: int = 0,
                    parent_id: Optional[int] = None,
                    hidden_state: Optional[torch.Tensor] = None) -> int:
        """Создает новый энергетический поток"""
        flow_id = self.next_flow_id
        self.next_flow_id += 1
        
        # Создаем пустое скрытое состояние если не передано
        if hidden_state is None:
            # Размеры берем из конфига
            num_layers = self.config.carrier_num_layers
            hidden_size = self.config.carrier_hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, device=self.device)
        
        # Вычисляем расстояние до ближайшей поверхности
        distance_to_surface, projected_surface = self.calculate_distance_to_nearest_surface(position)
        
        flow = EnergyFlow(
            id=flow_id,
            position=position,  # Позиция уже нормализованная
            energy=energy,
            hidden_state=hidden_state,
            batch_idx=batch_idx,
            parent_id=parent_id,
            age=0,
            is_active=True,
            steps_taken=0,  # Инициализируем счетчик шагов
            distance_to_surface=distance_to_surface,
            projected_surface=projected_surface
        )
        
        self.active_flows[flow_id] = flow
        self.stats['total_created'] += 1
        self.stats['max_concurrent'] = max(self.stats['max_concurrent'], len(self.active_flows))
        
        return flow_id
    
    def get_active_flows(self) -> List[EnergyFlow]:
        """Возвращает список активных потоков"""
        return [flow for flow in self.active_flows.values() if flow.is_active]
    
    def update_flow(self, flow_id: int, 
                   new_position: torch.Tensor,
                   new_energy: torch.Tensor,
                   new_hidden: torch.Tensor):
        """Обновляет состояние потока для трехплоскостной архитектуры"""
        if flow_id not in self.active_flows:
            return
        
        flow = self.active_flows[flow_id]
        flow.position = new_position
        flow.energy = new_energy
        flow.hidden_state = new_hidden
        flow.age += 1
        
        # Буферизуем потоки при достижении любой из выходных плоскостей
        z_pos = new_position[2].item()
        if z_pos <= 0:
            # Достигнута левая выходная плоскость (Z=0)
            self._buffer_flow_to_z0_plane(flow_id)
            logger.debug(f"Flow {flow_id} reached Z=0 output plane at age {flow.age} (buffered)")
        elif z_pos >= self.depth:
            # Достигнута правая выходная плоскость (Z=depth)
            self._buffer_flow_to_zdepth_plane(flow_id)
            logger.debug(f"Flow {flow_id} reached Z={self.depth} output plane at age {flow.age} (buffered)")
    
    def spawn_flows(self, parent_id: int, spawn_energies: List[torch.Tensor]) -> List[int]:
        """
        Создает новые потоки от родительского
        
        Args:
            parent_id: ID родительского потока
            spawn_energies: Список энергий для новых потоков
            
        Returns:
            new_flow_ids: ID созданных потоков
        """
        # ДИАГНОСТИКА: логируем причины отказов в spawn
        if parent_id not in self.active_flows:
            logger.debug_spawn(f"🚫 Spawn failed: parent {parent_id} not in active_flows")
            return []
        
        parent = self.active_flows[parent_id]
        new_flow_ids = []
        requested_count = len(spawn_energies)
        max_flows_reached = False
        
        for i, energy in enumerate(spawn_energies):
            if len(self.active_flows) >= self.max_active_flows:
                # Логируем только первые 3 случая max_active_flows ограничения
                if not hasattr(self, '_spawn_limit_log_counter'):
                    self._spawn_limit_log_counter = 0
                
                if self._spawn_limit_log_counter < 3:
                    logger.debug_spawn(f"🚫 Spawn limited: max_active_flows={self.max_active_flows} reached at spawn {i}/{requested_count}")
                    self._spawn_limit_log_counter += 1
                elif self._spawn_limit_log_counter == 3:
                    logger.debug_spawn(f"... (дальнейшие логи max_active_flows ограничений скрыты)")
                    self._spawn_limit_log_counter += 1
                
                max_flows_reached = True
                break
            
            # Новые потоки начинают с позиции родителя
            flow_id = self._create_flow(
                parent.position.clone(),
                energy,
                batch_idx=parent.batch_idx,  # Наследуем batch_idx от родителя
                parent_id=parent_id,
                hidden_state=parent.hidden_state.clone()
            )
            new_flow_ids.append(flow_id)
        
        # ДИАГНОСТИКА: детальная статистика spawn'а (только первые примеры)
        created_count = len(new_flow_ids)
        if requested_count > 0:
            # Логируем только первые 3 примера spawn'а за обработку
            if not hasattr(self, '_spawn_log_counter'):
                self._spawn_log_counter = 0
            
            if self._spawn_log_counter < 3:
                logger.debug_spawn(f"✅ Spawn result: parent_{parent_id} requested={requested_count} → created={created_count}")
                if created_count < requested_count:
                    logger.debug_spawn(f"⚠️ Spawn limited: {requested_count - created_count} flows not created " +
                                     (f"(max_flows_reached)" if max_flows_reached else "(unknown reason)"))
                self._spawn_log_counter += 1
            elif self._spawn_log_counter == 3:
                logger.debug_spawn(f"... (дальнейшие индивидуальные логи spawn скрыты для чистоты вывода)")
                self._spawn_log_counter += 1
        
        return new_flow_ids
    
    def deactivate_flow(self, flow_id: int, reason: str = "energy_depleted"):
        """Деактивирует поток"""
        if flow_id in self.active_flows:
            self.active_flows[flow_id].is_active = False
            if reason != "reached_output":
                self.stats['total_died'] += 1
            logger.debug(f"Flow {flow_id} deactivated: {reason}")
    
    def update_flow(self, flow_id: int, new_position: torch.Tensor, new_energy: torch.Tensor, new_hidden: torch.Tensor):
        """Обновляет состояние потока"""
        if flow_id in self.active_flows:
            flow = self.active_flows[flow_id]
            flow.position = new_position.clone()
            flow.energy = new_energy.clone()
            flow.hidden_state = new_hidden.clone()
            flow.age += 1
    
    def batch_deactivate_flows(self, dead_flow_ids: torch.Tensor, 
                              energy_dead_mask: torch.Tensor,
                              backward_dead_mask: torch.Tensor, 
                              bounds_dead_mask: torch.Tensor):
        """ВЕКТОРИЗОВАННАЯ деактивация множества потоков одновременно"""
        # Преобразуем тензоры в списки Python для избежания дальнейших GPU-CPU синхронизаций
        dead_ids = dead_flow_ids.detach().cpu().tolist()
        energy_dead = energy_dead_mask.detach().cpu().tolist()
        backward_dead = backward_dead_mask.detach().cpu().tolist()
        bounds_dead = bounds_dead_mask.detach().cpu().tolist()
        
        deactivated_count = 0
        for i, flow_id in enumerate(dead_ids):
            if flow_id in self.active_flows:
                self.active_flows[flow_id].is_active = False
                deactivated_count += 1
                
                # Определяем причину векторизованно
                if energy_dead[i]:
                    reason = "energy_depleted"
                elif backward_dead[i]:
                    reason = "backward_z_movement"
                else:  # bounds_dead[i]
                    reason = "out_of_bounds"
                
                # Обновляем статистику без детального логирования
                self.stats['total_died'] += 1
        
        if deactivated_count > 0:
            logger.debug(f"Batch deactivated {deactivated_count} flows")
    
    def batch_update_flows(self, alive_flow_ids: torch.Tensor,
                          alive_positions: torch.Tensor,
                          alive_energies: torch.Tensor,
                          alive_hidden: torch.Tensor):
        """ВЕКТОРИЗОВАННОЕ обновление множества потоков одновременно"""
        # Сбрасываем счетчики логирования для нового батча
        self._spawn_log_counter = 0
        self._spawn_limit_log_counter = 0
        # Преобразуем только ID в CPU, остальное оставляем на GPU
        alive_ids = alive_flow_ids.detach().cpu().tolist()
        
        updated_count = 0
        position_changes = []
        
        for i, flow_id in enumerate(alive_ids):
            if flow_id in self.active_flows:
                flow = self.active_flows[flow_id]
                
                # ДИАГНОСТИКА: логируем изменения позиций для первых нескольких потоков
                if updated_count < 5:
                    old_pos = flow.position.clone()
                    new_pos = alive_positions[i]
                    pos_diff = torch.norm(new_pos - old_pos).item()
                    position_changes.append(f"flow_{flow_id}[{old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}]→[{new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f}](diff={pos_diff:.3f})")
                
                # Обновляем БЕЗ .clone() для ускорения (данные уже отделены от графа)
                flow.position = alive_positions[i]
                flow.energy = alive_energies[i] 
                flow.hidden_state = alive_hidden[i]
                flow.age += 1
                updated_count += 1
        
        if updated_count > 0:
            # ДИАГНОСТИКА: общая статистика позиций
            z_positions = alive_positions[:, 2]
            z_min, z_max, z_mean = z_positions.min().item(), z_positions.max().item(), z_positions.mean().item()
            
            logger.debug(f"🔄 Batch updated {updated_count} flows: Z range [{z_min:.3f}, {z_max:.3f}], mean={z_mean:.3f}")
            
            # Детальные примеры изменений позиций
            if position_changes:
                logger.debug(f"🔄 Position changes: {'; '.join(position_changes)}")
    
    def _mark_flow_completed_z0_plane(self, flow_id: int):
        """НОВАЯ АРХИТЕКТУРА: Помечает поток как завершенный на Z=0 плоскости без буферизации"""
        if flow_id not in self.active_flows:
            return
            
        flow = self.active_flows[flow_id]
        
        # Проецируем позицию на Z=0 плоскость в нормализованных координатах
        normalized_z0_value = self.config.normalization_manager._normalize_to_range(
            torch.tensor([0.0]), self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0]
        
        # Обновляем позицию потока
        projected_position = torch.tensor([
            flow.position[0],  # X, Y уже нормализованные
            flow.position[1], 
            normalized_z0_value  # Проецируем на нормализованную Z=0 плоскость
        ], device=self.device)
        
        flow.position = projected_position
        flow.projected_surface = "z0"
        flow.is_active = False  # Помечаем как завершенный
        self.stats['total_completed'] += 1
        
        logger.debug(f"Flow {flow_id} marked completed on Z=0 plane")
    
    def _mark_flow_completed_zdepth_plane(self, flow_id: int):
        """НОВАЯ АРХИТЕКТУРА: Помечает поток как завершенный на Z=depth плоскости без буферизации"""
        if flow_id not in self.active_flows:
            return
            
        flow = self.active_flows[flow_id]
        
        # Проецируем позицию на Z=depth плоскость в нормализованных координатах
        normalized_zdepth_value = self.config.normalization_manager._normalize_to_range(
            torch.tensor([float(self.depth)]), self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0]
        
        # Обновляем позицию потока
        projected_position = torch.tensor([
            flow.position[0],  # X, Y уже нормализованные
            flow.position[1], 
            normalized_zdepth_value  # Проецируем на нормализованную Z=depth плоскость
        ], device=self.device)
        
        flow.position = projected_position
        flow.projected_surface = "zdepth"
        flow.is_active = False  # Помечаем как завершенный
        self.stats['total_completed'] += 1
        
        logger.debug(f"Flow {flow_id} marked completed on Z={self.depth} plane")

    # УДАЛЕНО: _buffer_flow_to_z0_plane() и _buffer_flow_to_zdepth_plane()
    # Заменены на _mark_flow_completed_*_plane() в новой архитектуре прямой работы с потоками
    
    # УДАЛЕНО: get_buffered_flows_count(), clear_output_buffer(), get_all_buffered_flows()
    # В новой архитектуре прямой работы с потоками буферы не используются
    
    def calculate_flow_importance(self, flow: EnergyFlow) -> float:
        """
        Вычисляет важность потока для трехплоскостной архитектуры
        
        Система важности:
        1. Близость к выходной плоскости (главный фактор)
        2. Длина пути (количество шагов)
        
        Args:
            flow: EnergyFlow для оценки важности
            
        Returns:
            importance: float - важность потока для взвешенного усреднения
        """
        z = flow.position[2].item()
        
        # Расстояние до ближайшей выходной плоскости
        distance_to_z0 = abs(z - 0)
        distance_to_zdepth = abs(z - self.depth)
        min_distance = min(distance_to_z0, distance_to_zdepth)
        
        # Безопасное деление - избегаем деления на ноль
        safe_distance = max(min_distance, self.config.safe_distance_minimum)
        proximity_importance = 1.0 / safe_distance
        
        # Важность длины пути
        path_importance = flow.age * self.config.path_length_weight
        
        # Комбинированная важность
        total_importance = (self.config.proximity_weight * proximity_importance + 
                          self.config.path_length_weight * path_importance)
        
        return total_importance
    
    def collect_completed_flows_direct(self) -> Tuple[torch.Tensor, List[int]]:
        """
        НОВАЯ АРХИТЕКТУРА: Собирает энергию напрямую из завершенных потоков без буферизации
        
        Returns:
            output_embeddings: [batch, embedding_dim] - собранные эмбеддинги  
            flow_ids: ID завершенных потоков
        """
        # Находим все завершенные потоки (неактивные с известной проекционной поверхностью)
        completed_flows = [
            flow for flow in self.active_flows.values() 
            if not flow.is_active and flow.projected_surface != "unknown"
        ]
        
        if not completed_flows:
            logger.debug("No completed flows found")
            return torch.zeros(0, self.config.input_embedding_dim_from_teacher, device=self.device), []
            
        logger.debug(f"Found {len(completed_flows)} completed flows for direct collection")
        
        # Группируем потоки по нормализованным координатам
        grouped_flows = {}
        for flow in completed_flows:
            key = self.get_normalized_buffer_key(flow.position)
            if key not in grouped_flows:
                grouped_flows[key] = []
            grouped_flows[key].append(flow)
        
        flow_ids = []
        output_embeddings = []
        aggregation_stats = []
        
        # Итерируем по группам потоков
        for key, flows_in_cell in grouped_flows.items():
            # Собираем ID всех потоков в этой клетке
            for flow in flows_in_cell:
                flow_ids.append(flow.id)
            
            # Агрегация энергии в клетке
            if len(flows_in_cell) == 1:
                # Один поток - просто берем его энергию
                aggregated = flows_in_cell[0].energy
                stats = f"single_flow(id={flows_in_cell[0].id})"
            else:
                # Несколько потоков - взвешенное усреднение с ИСПРАВЛЕННОЙ весовой системой
                energies = torch.stack([flow.energy for flow in flows_in_cell])
                
                weights = []
                for flow in flows_in_cell:
                    energy_magnitude = torch.norm(flow.energy).item()
                    
                    # ИСПРАВЛЕННЫЙ фактор близости для нормализованных координат
                    proximity_factor = 1.0 / (1.0 + flow.distance_to_surface)
                    
                    # Фактор количества шагов
                    steps_factor = 1.0 + flow.steps_taken * 0.1
                    
                    # Комбинированный вес
                    weight = energy_magnitude * proximity_factor * steps_factor
                    weights.append(weight)
                
                weights = torch.tensor(weights, device=self.device)
                weights = weights / weights.sum()  # Нормализуем
                
                # Взвешенное усреднение
                aggregated = (energies * weights.unsqueeze(-1)).sum(dim=0)
                
                # Статистика
                flow_ids_cell = [flow.id for flow in flows_in_cell]
                avg_age = sum(flow.age for flow in flows_in_cell) / len(flows_in_cell)
                stats = f"weighted_avg({len(flows_in_cell)}_flows, ids={flow_ids_cell}, avg_age={avg_age:.1f})"
                
                logger.debug(f"Cell {key}: {stats}")
            
            aggregation_stats.append(stats)
            output_embeddings.append(aggregated)
        
        # Собираем в тензор
        output_embeddings = torch.stack(output_embeddings)  # [num_cells, embedding_dim]
        
        # Восстанавливаем исходную размерность эмбеддинга
        output_embeddings = output_embeddings.view(-1)[:self.config.input_embedding_dim_from_teacher]
        output_embeddings = output_embeddings.unsqueeze(0)  # [1, 768]
        
        # Статистика
        cells_with_flows = len(aggregation_stats)
        multi_flow_cells = len([stats for stats in aggregation_stats if 'weighted_avg' in stats])
        
        logger.info(f"Direct collection: {len(flow_ids)} completed flows from {cells_with_flows} cells, "
                   f"{multi_flow_cells} cells with multiple flows")
        
        return output_embeddings, flow_ids
    
    def collect_completed_flows_surface_direct(self) -> Tuple[torch.Tensor, List[int]]:
        """
        НОВАЯ АРХИТЕКТУРА: Собирает surface embeddings напрямую из завершенных потоков
        
        Returns:
            output_surface_embeddings: [batch, surface_dim] - surface embeddings
            completed_flows: ID завершенных потоков
        """
        # Находим все завершенные потоки
        completed_flows = [
            flow for flow in self.active_flows.values() 
            if not flow.is_active and flow.projected_surface != "unknown"
        ]
        
        if not completed_flows:
            logger.debug("No completed flows found for surface collection")
            # Возвращаем нулевые surface embeddings с сохранением градиентов
            surface_dim = self.width * self.height
            reference_tensor = None
            for flow in self.active_flows.values():
                if flow.energy.requires_grad:
                    reference_tensor = flow.energy
                    break
            if reference_tensor is not None:
                zero_tensor = torch.zeros(1, surface_dim, device=self.device, dtype=reference_tensor.dtype, requires_grad=True)
                return zero_tensor * 0.0, []
            else:
                return torch.zeros(1, surface_dim, device=self.device), []
        
        # Определяем размер батча
        batch_indices = {flow.batch_idx for flow in completed_flows}
        batch_size = max(batch_indices) + 1 if batch_indices else 1
        
        # Создаем surface tensor с градиентной связью
        surface_dim = self.width * self.height
        reference_energy = completed_flows[0].energy
        if reference_energy.requires_grad:
            output_surface = torch.zeros(batch_size, surface_dim, device=self.device, dtype=reference_energy.dtype)
            # Создаем градиентную связь
            all_energies = torch.stack([flow.energy for flow in completed_flows])
            energy_sum = all_energies.sum()
            output_surface = output_surface + energy_sum * 0.0
        else:
            output_surface = torch.zeros(batch_size, surface_dim, device=self.device)
        
        flow_ids = []
        
        logger.debug(f"Direct surface collection: {len(completed_flows)} completed flows, batch_size={batch_size}")
        
        # Группируем потоки по нормализованным координатам
        grouped_flows = {}
        for flow in completed_flows:
            key = self.get_normalized_buffer_key(flow.position)
            if key not in grouped_flows:
                grouped_flows[key] = []
            grouped_flows[key].append(flow)
        
        # Итерируем по группам потоков
        for (norm_x, norm_y), flows in grouped_flows.items():
            # Получаем surface_idx из предвычисленного mapping
            surface_idx = self.normalized_to_surface_idx.get((norm_x, norm_y))
            if surface_idx is None:
                logger.warning(f"No surface mapping for normalized coords ({norm_x:.6f}, {norm_y:.6f})")
                continue
            
            # Собираем ID всех потоков в этой клетке
            cell_flow_ids = [flow.id for flow in flows]
            flow_ids.extend(cell_flow_ids)
            
            if len(flows) == 1:
                # Один поток - просто берем его энергию
                flow = flows[0]
                aggregated_energy = flow.energy
                batch_idx = flow.batch_idx
                logger.debug(f"Cell ({norm_x:.3f},{norm_y:.3f})->[idx={surface_idx}]: single_flow(id={flow.id}, batch={batch_idx})")
            else:
                # Несколько потоков - взвешенное усреднение с ИСПРАВЛЕННОЙ весовой системой
                energies = []
                weights = []
                batch_indices_cell = []
                
                for flow in flows:
                    energy_magnitude = torch.norm(flow.energy).item()
                    
                    # ИСПРАВЛЕННЫЙ фактор близости для нормализованных координат
                    proximity_factor = 1.0 / (1.0 + flow.distance_to_surface)
                    
                    # Фактор количества шагов
                    steps_factor = 1.0 + flow.steps_taken * 0.1
                    
                    # Комбинированный вес
                    weight = energy_magnitude * proximity_factor * steps_factor
                    
                    energies.append(flow.energy)
                    weights.append(weight)
                    batch_indices_cell.append(flow.batch_idx)
                
                # Нормализуем веса
                weights = torch.tensor(weights, device=self.device)
                weights = weights / weights.sum()
                
                # Взвешенное усреднение энергий
                energies_tensor = torch.stack(energies)
                aggregated_energy = (energies_tensor * weights.unsqueeze(-1)).sum(dim=0)
                
                # Для множественных потоков берем batch_idx первого потока
                batch_idx = batch_indices_cell[0]
                
                avg_age = sum(flow.age for flow in flows) / len(flows)
                logger.debug(f"Cell ({norm_x:.3f},{norm_y:.3f})->[idx={surface_idx}]: weighted_avg({len(flows)}_flows, ids={cell_flow_ids}, avg_age={avg_age:.1f}, batch={batch_idx})")
            
            # Размещаем агрегированную энергию в surface tensor
            if 0 <= batch_idx < batch_size:
                if aggregated_energy.numel() == 1:
                    output_surface[batch_idx, surface_idx] = aggregated_energy.squeeze()
                else:
                    output_surface[batch_idx, surface_idx] = aggregated_energy[0]
            else:
                logger.warning(f"Invalid batch_idx: {batch_idx} (expected 0 <= batch_idx < {batch_size})")
        
        logger.info(f"Direct surface collection: {len(flow_ids)} completed flows across {len(grouped_flows)} cells")
        return output_surface, flow_ids

    # УДАЛЕНО: collect_buffered_energy() - заменен на collect_completed_flows_direct()
    # в новой архитектуре прямой работы с потоками
    
    # УДАЛЕНО: collect_buffered_surface_energy() - заменен на collect_completed_flows_surface_direct()
    
    # УДАЛЕНО: collect_buffered_surface_energy() и collect_output_energy()
    # Заменены на collect_completed_flows_surface_direct() и collect_completed_flows_direct()
    # в новой архитектуре прямой работы с потоками
    
    def _cleanup_inactive_flows(self):
        """Удаляет неактивные потоки из памяти"""
        inactive_ids = [fid for fid, flow in self.active_flows.items() if not flow.is_active]
        
        for flow_id in inactive_ids:
            del self.active_flows[flow_id]
        
        if inactive_ids:
            logger.debug(f"Cleaned up {len(inactive_ids)} inactive flows")
    
    def get_statistics(self) -> Dict[str, any]:
        """Возвращает статистику решетки"""
        active_count = len(self.get_active_flows())
        
        return {
            **self.stats,
            'current_active': active_count,
            'utilization': active_count / self.max_active_flows if self.max_active_flows > 0 else 0
        }
    
    def _cleanup_inactive_flows(self):
        """Удаляет неактивные потоки из памяти"""
        inactive_ids = [fid for fid, flow in self.active_flows.items() if not flow.is_active]
        
        for flow_id in inactive_ids:
            del self.active_flows[flow_id]
        
        if inactive_ids:
            logger.debug(f"Cleaned up {len(inactive_ids)} inactive flows")
    
    def reset(self):
        """Сбрасывает состояние решетки (НОВАЯ АРХИТЕКТУРА: без буферизации)"""
        self.active_flows.clear()
        # УДАЛЕНО: очистка буферов в новой архитектуре без буферизации
        # Буферы остаются для обратной совместимости, но не используются
        self.next_flow_id = 0
        self.stats = {
            'total_created': 0,
            'total_completed': 0,
            'total_died': 0,
            'max_concurrent': 0
        }
        logger.info("EnergyLattice reset (direct flows architecture - no buffering)")


def create_energy_lattice(config=None) -> EnergyLattice:
    """Фабричная функция для создания EnergyLattice"""
    return EnergyLattice(config)