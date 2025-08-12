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

from ..utils.logging import get_logger, log_memory_state, gated_log, summarize_step, format_first_n
from ..config import get_energy_config, create_debug_config, set_energy_config
from ..utils.device_manager import get_device_manager
from .tensorized_storage import TensorizedFlowStorage

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
        
        # Необязательное тензорное хранилище активных потоков
        self.tensor_storage: Optional[TensorizedFlowStorage] = None
        if getattr(self.config, 'tensorized_storage_enabled', False):
            self.tensor_storage = TensorizedFlowStorage(
                max_flows=self.max_active_flows,
                embedding_dim=1,
                hidden_layers=self.config.carrier_num_layers,
                hidden_size=self.config.carrier_hidden_size,
                device=self.device,
                lattice_width=self.width,
                lattice_height=self.height
            )
        
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
        
        # ОПТИМИЗАЦИЯ: Предкешируем нормализованные Z плоскости для избежания повторных вычислений
        self._precompute_normalized_z_planes()
        
        # Scratch buffers for allocation reuse in hot paths
        self._scratch: Dict[str, torch.Tensor] = {}
        
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
    
    def _precompute_normalized_z_planes(self):
        """Предвычисляет нормализованные значения Z для выходных плоскостей."""
        # Вычисляем один раз при инициализации
        self.norm_z0 = self.config.normalization_manager._normalize_to_range(
            torch.tensor([0.0], device=self.device), 
            self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0].item()
        
        self.norm_zdepth = self.config.normalization_manager._normalize_to_range(
            torch.tensor([float(self.depth)], device=self.device), 
            self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0].item()
        
        # Также вычисляем нормализованное значение для центра (входная плоскость)
        self.norm_zcenter = self.config.normalization_manager._normalize_to_range(
            torch.tensor([float(self.depth // 2)], device=self.device), 
            self.config.normalization_manager.ranges.z_range[0], 
            self.config.normalization_manager.ranges.z_range[1]
        )[0].item()
        
        logger.debug(f"Precomputed normalized Z planes: z0={self.norm_z0:.6f}, "
                    f"center={self.norm_zcenter:.6f}, zdepth={self.norm_zdepth:.6f}")
    
    def round_to_nearest_lattice_position(self, normalized_positions: torch.Tensor) -> torch.Tensor:
        """
        Округляет нормализованные позиции до ближайших координат решетки.
        АРИФМЕТИЧЕСКАЯ ВЕРСИЯ O(1) - прямое вычисление без поиска.
        
        Args:
            normalized_positions: [batch, 3] нормализованные позиции в [-1, 1]
            
        Returns:
            [batch, 3] нормализованные координаты ближайших позиций решетки
        """
        device = normalized_positions.device
        
        # АРИФМЕТИЧЕСКАЯ КВАНТИЗАЦИЯ: O(1) операция вместо O(N*M) поиска
        # Преобразуем из [-1, 1] в индексы решетки [0, size-1]
        
        # X координата: [-1, 1] -> [0, width-1]
        idx_x = ((normalized_positions[:, 0] + 1) * 0.5 * (self.width - 1))
        idx_x = idx_x.round().clamp(0, self.width - 1)
        
        # Y координата: [-1, 1] -> [0, height-1]
        idx_y = ((normalized_positions[:, 1] + 1) * 0.5 * (self.height - 1))
        idx_y = idx_y.round().clamp(0, self.height - 1)
        
        # Z координата: [-1, 1] -> [0, depth]
        idx_z = ((normalized_positions[:, 2] + 1) * 0.5 * self.depth)
        idx_z = idx_z.round().clamp(0, self.depth)
        
        # Преобразуем обратно в нормализованные координаты [-1, 1]
        norm_x = (idx_x / (self.width - 1)) * 2 - 1
        norm_y = (idx_y / (self.height - 1)) * 2 - 1
        norm_z = (idx_z / self.depth) * 2 - 1
        
        # Собираем результат
        rounded_positions = torch.stack([norm_x, norm_y, norm_z], dim=-1)
        
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
        
        # Используем предвычисленные нормализованные значения выходных поверхностей
        # Вычисляем расстояния до обеих поверхностей
        distance_to_z0 = abs(norm_z - self.norm_z0)
        distance_to_zdepth = abs(norm_z - self.norm_zdepth)
        
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
        summary = summarize_step({'flows': len(flow_ids), 'start_z': start_z, 'depth': self.depth, 'time_s': round(elapsed_time, 2)}, prefix='INIT')
        logger.info(f"🏗️ Created initial flows. {summary}")
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
        
        # Используем предвычисленные нормализованные Z для выходных поверхностей
        # Векторизованные расстояния
        distances_to_z0 = torch.abs(norm_z_values - self.norm_z0)
        distances_to_zdepth = torch.abs(norm_z_values - self.norm_zdepth)
        
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
        
        # Ленивое диагностическое логирование первых примеров (одна агрегированная строка)
        def _created_examples_msg():
            k = min(5, num_flows)
            raw_examples = [(positions_xy[i][0], positions_xy[i][1], start_z) for i in range(k)]
            norm_examples = [
                (
                    float(normalized_positions[i][0]),
                    float(normalized_positions[i][1]),
                    float(normalized_positions[i][2])
                ) for i in range(k)
            ]
            mags = [float(torch.norm(energies_tensors[i]).item()) for i in range(k)]
            return (
                f"🅫 Created {num_flows} flows (showing {k}): raw={raw_examples}, "
                f"norm={norm_examples}, |emb|={mags}"
            )
        gated_log(
            logger,
            'DEBUG_INIT',
            step=0,
            key='flows_created_examples',
            msg_or_factory=_created_examples_msg,
            first_n_steps=1,
            every=0,
        )
        
        # Синхронизация с тензорным хранилищем (если включено)
        if self.tensor_storage is not None and num_flows > 0:
            # Энергии как [N,1]
            energies_matrix = torch.stack(energies_tensors, dim=0).view(num_flows, 1).to(self.device)
            batch_indices_tensor = torch.tensor(batch_indices[:num_flows], device=self.device, dtype=torch.long)
            # allocate в tensor_storage с теми же flow_ids
            _ = self.tensor_storage.allocate_flows(
                num_flows=num_flows,
                initial_positions=normalized_positions[:num_flows],
                initial_energies=energies_matrix,
                batch_indices_list=batch_indices_tensor.tolist(),
                parent_ids=None,
                flow_ids=flow_ids,
                initial_hidden=all_hidden_states[:num_flows]
            )
        
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
    
    
    def spawn_flows(self, parent_id: int, spawn_energies: List[torch.Tensor], start_position: Optional[torch.Tensor] = None) -> List[int]:
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
            
            # Новые потоки начинают с позиции родителя ДО перемещения (или явно заданной позиции)
            start_pos = start_position if start_position is not None else parent.position
            flow_id = self._create_flow(
                start_pos,
                energy,
                batch_idx=parent.batch_idx,  # Наследуем batch_idx от родителя
                parent_id=parent_id,
                hidden_state=parent.hidden_state.clone()
            )
            new_flow_ids.append(flow_id)
        
        # Синхронизация с тензорным хранилищем (если включено)
        if self.tensor_storage is not None and new_flow_ids:
            num_new = len(new_flow_ids)
            # Позиции: один start_pos для всех
            positions = torch.stack([start_pos for _ in range(num_new)]).to(self.device)
            # Энергии: [N,1]
            energies_matrix = torch.stack(spawn_energies[:num_new], dim=0).view(num_new, 1).to(self.device)
            batch_indices_list = [parent.batch_idx for _ in range(num_new)]
            # Подготовим hidden states от родителя
            parent_hidden_matrix = torch.stack([parent.hidden_state for _ in range(num_new)], dim=0).to(self.device)
            self.tensor_storage.allocate_flows(
                num_flows=num_new,
                initial_positions=positions,
                initial_energies=energies_matrix,
                batch_indices_list=batch_indices_list,
                parent_ids=[parent_id for _ in range(num_new)],
                flow_ids=new_flow_ids,
                initial_hidden=parent_hidden_matrix
            )
        
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
                    # Копия не требуется: мы переassign'им flow.position ниже, старый тензор останется доступным
                    old_pos = flow.position
                    new_pos = alive_positions[i]
                    pos_diff = torch.norm(new_pos - old_pos).item()
                    position_changes.append(f"flow_{flow_id}[{old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}]→[{new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f}](diff={pos_diff:.3f})")
                
                # Обновляем БЕЗ .clone() для ускорения (данные уже отделены от графа)
                flow.position = alive_positions[i]
                flow.energy = alive_energies[i] 
                flow.hidden_state = alive_hidden[i]
                flow.age += 1
                # Обновляем кеш distance_to_surface (в нормализованном пространстве)
                z_norm = flow.position[2].item()
                dist_z0 = abs(z_norm - self.norm_z0)
                dist_zd = abs(z_norm - self.norm_zdepth)
                flow.distance_to_surface = min(dist_z0, dist_zd)
                updated_count += 1
        
        # Обновляем тензорное хранилище (если включено)
        if self.tensor_storage is not None and alive_flow_ids.numel() > 0:
            # Приводим энергии к форме [N,1] при необходимости
            energies_tensor = alive_energies
            if energies_tensor.dim() == 1:
                energies_tensor = energies_tensor.view(-1, 1)
            self.tensor_storage.batch_update(
                flow_ids=alive_flow_ids,
                new_positions=alive_positions,
                new_energies=energies_tensor,
                new_hidden=alive_hidden,
                increment_age=True
            )
        
        if updated_count > 0:
            # Компактная сводка и ленивые примеры
            z_positions = alive_positions[:, 2]
            z_min, z_max, z_mean = z_positions.min().item(), z_positions.max().item(), z_positions.mean().item()
            gated_log(
                logger,
                'DEBUG',
                step=None,
                key='lattice_batch_update',
                msg_or_factory=lambda: f"🔄 Batch updated {updated_count} flows: Z range [{z_min:.3f}, {z_max:.3f}], mean={z_mean:.3f}",
                first_n_steps=1,
                every=0,
            )
            if position_changes:
                gated_log(
                    logger,
                    'DEBUG',
                    step=None,
                    key='lattice_position_changes',
                    msg_or_factory=lambda: f"🔄 Position changes: {'; '.join(position_changes)}",
                    first_n_steps=1,
                    every=0,
                )
    
    def _mark_flow_completed_z0_plane(self, flow_id: int):
        """НОВАЯ АРХИТЕКТУРА: Помечает поток как завершенный на Z=0 плоскости без буферизации"""
        if flow_id not in self.active_flows:
            return
            
        flow = self.active_flows[flow_id]
        
        # Используем предвычисленное нормализованное значение Z=0 плоскости
        # Обновляем позицию потока
        projected_position = torch.tensor([
            flow.position[0],  # X, Y уже нормализованные
            flow.position[1], 
            self.norm_z0  # Проецируем на нормализованную Z=0 плоскость
        ], device=self.device)
        
        flow.position = projected_position
        flow.projected_surface = "z0"
        flow.is_active = False  # Помечаем как завершенный
        self.stats['total_completed'] += 1
        
        # Обновляем тензорное хранилище (если включено)
        if self.tensor_storage is not None:
            self.tensor_storage.mark_completed([flow_id], surface_type='z0')
        
        logger.debug(f"Flow {flow_id} marked completed on Z=0 plane")
    
    def _mark_flow_completed_zdepth_plane(self, flow_id: int):
        """НОВАЯ АРХИТЕКТУРА: Помечает поток как завершенный на Z=depth плоскости без буферизации"""
        if flow_id not in self.active_flows:
            return
            
        flow = self.active_flows[flow_id]
        
        # Используем предвычисленное нормализованное значение Z=depth плоскости
        # Обновляем позицию потока
        projected_position = torch.tensor([
            flow.position[0],  # X, Y уже нормализованные
            flow.position[1], 
            self.norm_zdepth  # Проецируем на нормализованную Z=depth плоскость
        ], device=self.device)
        
        flow.position = projected_position
        flow.projected_surface = "zdepth"
        flow.is_active = False  # Помечаем как завершенный
        self.stats['total_completed'] += 1
        
        # Обновляем тензорное хранилище (если включено)
        if self.tensor_storage is not None:
            self.tensor_storage.mark_completed([flow_id], surface_type='zdepth')
        
        logger.debug(f"Flow {flow_id} marked completed on Z={self.depth} plane")

    def mark_flows_completed_z0_batch(self, flow_ids: torch.Tensor):
        """Батчевое помечение потоков завершенными на плоскости Z=0 (нормализованной)."""
        if flow_ids is None or flow_ids.numel() == 0:
            return
        ids = flow_ids.detach().cpu().tolist()
        for fid in ids:
            self._mark_flow_completed_z0_plane(fid)

    def mark_flows_completed_zdepth_batch(self, flow_ids: torch.Tensor):
        """Батчевое помечение потоков завершенными на плоскости Z=depth (нормализованной)."""
        if flow_ids is None or flow_ids.numel() == 0:
            return
        ids = flow_ids.detach().cpu().tolist()
        for fid in ids:
            self._mark_flow_completed_zdepth_plane(fid)

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
    
    def collect_completed_flows_direct(self, mapper, expected_batch_size: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает энергию завершенных потоков, агрегирует по surface и восстанавливает 768D.
        Требует tensorized_storage_enabled. Без фолбэков.
        """
        if self.tensor_storage is None:
            raise RuntimeError("Tensorized storage is required for 768D collection (tensorized_storage_enabled=True)")
        output_surface, flow_ids = self.collect_completed_flows_surface_direct_tensorized()
        if output_surface.numel() == 0 or output_surface.shape[0] == 0:
            # Явная ошибка вместо скрытого нуля — пусть тренер обработает это явно
            raise RuntimeError("No completed flows available for reconstruction at collection time")
        embeddings_768 = mapper.output_collector.reconstruction(output_surface)
        summary = summarize_step({'flows': len(flow_ids), 'cells': output_surface.shape[1]}, prefix='COLLECT-768D')
        logger.info(summary)
        return embeddings_768, flow_ids

    def collect_completed_flows_surface_direct(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает surface embeddings напрямую из TensorizedFlowStorage.
        Требует включенного tensorized_storage_enabled. Без фолбэков.
        """
        if self.tensor_storage is None:
            raise RuntimeError("Tensorized storage is required for surface collection (tensorized_storage_enabled=True)")
        return self.collect_completed_flows_surface_direct_tensorized()
    
    def _get_scratch(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, fill: Optional[float] = None) -> torch.Tensor:
        """Возвращает предварительно выделенный тензор нужной формы, при необходимости перевыделяет."""
        t = self._scratch.get(name)
        need_alloc = (t is None) or (t.dtype != dtype) or (t.device != device) or (tuple(t.shape) != tuple(shape))
        if need_alloc:
            t = torch.empty(*shape, device=device, dtype=dtype)
            self._scratch[name] = t
        if fill is not None:
            t.fill_(fill)
        return t

    def collect_completed_flows_surface_direct_tensorized(self) -> Tuple[torch.Tensor, List[int]]:
        """Полностью векторизованный сбор surface embeddings напрямую из TensorizedFlowStorage (без Python-циклов) с переиспользованием буферов."""
        if self.tensor_storage is None:
            raise RuntimeError("Tensorized storage is required for tensorized surface collection")
        storage = self.tensor_storage
        device = self.device
        surface_dim = self.width * self.height
        
        import time
        t0 = time.perf_counter()
        # Маска завершенных
        completed_mask = storage.is_completed & (storage.projected_surface != 0)
        completed_indices = torch.where(completed_mask)[0]
        if completed_indices.numel() == 0:
            out = self._get_scratch('output_surface_empty', (1, surface_dim), torch.float32, device, fill=0.0)
            return out, []
        
        # Данные завершенных
        pos = storage.positions[completed_indices]            # [N,3] нормализованные
        energy = storage.energies[completed_indices].view(-1) # [N]
        batch_idx = storage.batch_indices[completed_indices]  # [N]
        steps_taken = storage.steps_taken[completed_indices].to(device).float()
        
        # Расстояние до ближайшей поверхности (нормализованное пространство)
        z = pos[:, 2]
        dist = torch.minimum((z - self.norm_z0).abs(), (z - self.norm_zdepth).abs())  # [N]
        
        # Квантование X/Y в индексы поверхности (используем кэш если включен)
        if getattr(self.config, 'cache_surface_indices_enabled', False):
            surface_idx = storage.cached_surface_idx[completed_indices]
        else:
            idx_x = (((pos[:, 0] + 1) * 0.5) * (self.width - 1)).round().clamp(0, self.width - 1).long()
            idx_y = (((pos[:, 1] + 1) * 0.5) * (self.height - 1)).round().clamp(0, self.height - 1).long()
            surface_idx = idx_y * self.width + idx_x  # [N]
        
        # Ключ группировки: (batch, surface)
        key = batch_idx.long() * surface_dim + surface_idx  # [N]
        uniq, inv = torch.unique(key, return_inverse=True)
        num_groups = uniq.shape[0]
        
        # Batch size (sync point, acceptable)
        batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
        
        # Веса для softmax внутри группы (mixed precision опционально)
        use_mp = getattr(self.config, 'collection_use_mixed_precision', False)
        mp_dtype = getattr(self.config, 'collection_dtype', torch.bfloat16)
        if use_mp:
            wr_energy = energy.abs().to(mp_dtype)
            wr_dist = dist.to(mp_dtype)
            wr_steps = steps_taken.to(mp_dtype)
            weights_raw = wr_energy * (1.0 / (1.0 + wr_dist)) * (1.0 + wr_steps * 0.1)
        else:
            weights_raw = energy.abs() * (1.0 / (1.0 + dist)) * (1.0 + steps_taken * 0.1)  # [N]
        
        # Стабильный per-group softmax с помощью scatter_reduce
        if not hasattr(torch, 'scatter_reduce'):
            raise RuntimeError("Required torch.scatter_reduce is not available in this Torch version")
        
        # group_max for numerical stability (allocate/reuse and fill with -inf)
        group_max = self._get_scratch('group_max', (num_groups,), weights_raw.dtype, device, fill=float('-inf'))
        group_max = torch.scatter_reduce(group_max, 0, inv, weights_raw, reduce='amax', include_self=True)  # returns tensor
        exps = torch.exp(weights_raw - group_max[inv])
        # group sums (allocate/reuse and zero-fill)
        denom = self._get_scratch('denom', (num_groups,), exps.dtype, device, fill=0.0)
        denom = torch.scatter_reduce(denom, 0, inv, exps, reduce='sum', include_self=True)
        weights = exps / (denom[inv] + 1e-12)
        
        # Взвешенная агрегация энергий по ключу (batch,surface) с переиспользованием буфера
        if use_mp:
            weighted_energy = weights.to(torch.float32) * energy.to(torch.float32)
        else:
            weighted_energy = weights * energy  # [N]
        output_flat = self._get_scratch('output_flat', (batch_size * surface_dim,), weighted_energy.dtype, device)
        output_flat.zero_()
        output_flat.index_add_(0, key, weighted_energy)
        output_surface = output_flat.view(batch_size, surface_dim)
        
        # Собираем flow_ids
        flow_ids = [storage.index_to_id[idx.item()] for idx in completed_indices if idx.item() in storage.index_to_id]
        t_total = (time.perf_counter() - t0) * 1000.0
        try:
            logger.debug(f"SURF-COLLECT tensorized: N={completed_indices.numel()}, groups={num_groups}, batch={batch_size}, time={t_total:.2f}ms")
        except Exception:
            pass
        return output_surface, flow_ids

    # УДАЛЕНО: collect_buffered_energy() - заменен на collect_completed_flows_direct()
    # в новой архитектуре прямой работы с потоками
    
    # УДАЛЕНО: collect_buffered_surface_energy() - заменен на collect_completed_flows_surface_direct()
    
    # УДАЛЕНО: collect_buffered_surface_energy() и collect_output_energy()
    # Заменены на collect_completed_flows_surface_direct() и collect_completed_flows_direct()
    # в новой архитектуре прямой работы с потоками
    
    
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
        # Деаллокация в тензорном хранилище (если включено)
        if self.tensor_storage is not None:
            # Освобождаем все активные ID
            all_ids = list(self.active_flows.keys())
            if all_ids:
                self.tensor_storage.deallocate_flows(all_ids)
        
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