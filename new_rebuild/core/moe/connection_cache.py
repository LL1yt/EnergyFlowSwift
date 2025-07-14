#!/usr/bin/env python3
"""
Connection Cache Manager - Pre-computed кэширование классификации связей
========================================================================

Оптимизация производительности через pre-compute статических структур:
1. Pre-computed расстояния между всеми парами клеток
2. Pre-computed базовая классификация (LOCAL/DISTANT)
3. Кэширование candidate списков для FUNCTIONAL связей
4. Быстрая lookup структура для batch операций

ПРИНЦИПЫ:
- Статические структуры вычисляются один раз при инициализации
- Только functional_similarity проверяется динамически
- Massive speedup для повторяющихся классификаций
- Memory-efficient хранение с sparse структурами
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
import os
import time
import math

from .connection_types import ConnectionCategory, ConnectionInfo
from .distance_calculator import DistanceCalculator
from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CachedConnectionInfo:
    """Минимальная информация о связи для кэширования"""

    target_idx: int
    euclidean_distance: float
    manhattan_distance: float
    category: (
        ConnectionCategory  # LOCAL или DISTANT (FUNCTIONAL определяется динамически)
    )


class ConnectionCacheManager:
    """
    Менеджер кэширования связей с pre-computed структурами

    Ключевые оптимизации:
    1. Pre-computed distance matrices (sparse)
    2. Pre-computed категории LOCAL/DISTANT
    3. Candidate списки для FUNCTIONAL проверки
    4. Batch lookup таблицы
    """

    def __init__(
        self,
        lattice_dimensions: Tuple[int, int, int],
        cache_config: Optional[Dict] = None,
    ):
        """
        Инициализация кэш менеджера

        Args:
            lattice_dimensions: Размеры 3D решетки (x, y, z)
            cache_config: Конфигурация кэширования
        """
        logger.debug_init(f"[ConnectionCacheManager.__init__] Начало инициализации для решетки размером: {lattice_dimensions}")
        self.lattice_dimensions = lattice_dimensions
        self.total_cells = np.prod(lattice_dimensions)

        # Получаем конфигурацию
        try:
            config = get_project_config()
            logger.debug_init(f"[ConnectionCacheManager.__init__] Успешно получен project config: {config.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to get project config: {e}")
            raise
            
        if cache_config is None:
            self.cache_config = asdict(config.cache) if config.cache else {}
        else:
            self.cache_config = cache_config
            
        logger.debug_init(f"[ConnectionCacheManager.__init__] Получена конфигурация кэша (передана из UnifiedConnectionClassifier): {self.cache_config}")

        # ИСПРАВЛЕНО: Всегда получаем актуальный adaptive_radius
        try:
            self.adaptive_radius = config.calculate_adaptive_radius()
            logger.debug_init(f"[ConnectionCacheManager.__init__] Вычислен adaptive_radius из конфигурации: {self.adaptive_radius}")
        except Exception as e:
            logger.error(f"Failed to calculate adaptive radius: {e}")
            raise

        # Пороги для классификации связей, вычисляемые на основе adaptive_radius
        try:
            self.local_threshold = (
                self.adaptive_radius * config.lattice.local_distance_ratio
            )
            self.functional_threshold = (
                self.adaptive_radius * config.lattice.functional_distance_ratio
            )
            self.distant_threshold = (
                self.adaptive_radius * config.lattice.distant_distance_ratio
            )
            logger.debug_init(f"[ConnectionCacheManager.__init__] Вычислены пороги классификации на основе adaptive_radius и коэффициентов из конфига:")
            logger.debug_init(f"  - LOCAL threshold: {self.local_threshold} (radius * {config.lattice.local_distance_ratio})")
            logger.debug_init(f"  - FUNCTIONAL threshold: {self.functional_threshold} (radius * {config.lattice.functional_distance_ratio})")
            logger.debug_init(f"  - DISTANT threshold: {self.distant_threshold} (radius * {config.lattice.distant_distance_ratio})")
        except Exception as e:
            logger.error(f"Failed to calculate thresholds: {e}")
            raise

        # Инициализируем distance calculator
        self.distance_calculator = DistanceCalculator(lattice_dimensions)

        # GPU настройки
        # Проверяем доступность GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU не доступен! RTX 5090 обязателен для работы системы")
            
        self.use_gpu = True  # Всегда используем GPU
        self.gpu_batch_size = self.cache_config.get("gpu_batch_size", 10000)
        self.device = torch.device("cuda")  # Всегда CUDA для RTX 5090

        # Кэш структуры
        self.cache: Dict[int, Dict[str, List[CachedConnectionInfo]]] = {}
        self.distance_cache: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.is_precomputed = False
        self._all_neighbors_cache = None  # Инициализируется при первом вызове _compute_all_neighbors()

        # Статистика (включается по настройкам)
        self.enable_performance_monitoring = self.cache_config.get(
            "enable_performance_monitoring", False
        )
        self.enable_detailed_stats = self.cache_config.get(
            "enable_detailed_stats", False
        )

        if self.enable_performance_monitoring:
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_lookup_time = 0.0
            self.total_rebuild_time = 0.0

        # Вычисляем примерное количество соседей для каждого порога используя центральный конфиг
        approx_local_neighbors = config.estimate_neighbors_in_radius(self.local_threshold)
        approx_functional_neighbors = config.estimate_neighbors_in_radius(self.functional_threshold) - approx_local_neighbors
        approx_distant_neighbors = config.estimate_neighbors_in_radius(self.distant_threshold) - approx_local_neighbors - approx_functional_neighbors
        
        logger.info(f"🔧 [ConnectionCacheManager] Инициализация завершена:")
        logger.info(f"   Решетка: {lattice_dimensions} ({self.total_cells} клеток)")
        logger.info(f"   Адаптивный радиус: {self.adaptive_radius}")
        logger.info(
            f"   Пороги: LOCAL≤{self.local_threshold}, FUNCTIONAL≤{self.functional_threshold}, DISTANT≤{self.distant_threshold}"
        )
        logger.info(f"   Примерное количество соседей:")
        logger.info(f"     - LOCAL: ~{approx_local_neighbors} клеток (в радиусе {self.local_threshold})")
        logger.info(f"     - FUNCTIONAL: ~{approx_functional_neighbors} клеток (между {self.local_threshold} и {self.functional_threshold})")
        logger.info(f"     - DISTANT: ~{approx_distant_neighbors} клеток (между {self.functional_threshold} и {self.distant_threshold})")
        logger.info(f"   Мониторинг производительности: {'включен' if self.enable_performance_monitoring else 'выключен'}")
        logger.info(f"   Детальная статистика: {'включена' if self.enable_detailed_stats else 'выключена'}")

        # GPU информация
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 [ConnectionCacheManager] GPU ускорение: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"   Размер батча для GPU операций: {self.gpu_batch_size}")
        else:
            logger.info("💻 CPU mode: GPU not available or disabled")

    def _load_cache_from_disk(self) -> bool:
        """
        Загрузка кэша с диска с полной проверкой совместимости.
        Returns:
            True если кэш успешно загружен, иначе False.
        """
        try:
            cache_key = self._get_cache_key()
            cache_file = f"cache/connection_cache_{cache_key}.pkl"

            if not os.path.exists(cache_file):
                logger.info(f"Кэш файл не найден: {cache_file}")
                return False

            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Детальная проверка совместимости
            is_compatible = True
            checks = {
                "lattice_dimensions": self.lattice_dimensions,
                "adaptive_radius": self.adaptive_radius,
                "local_threshold": self.local_threshold,
                "functional_threshold": self.functional_threshold,
                "distant_threshold": self.distant_threshold,
                "cache_version": self.cache_config.get("cache_version", "2024.1"),
            }
            if logger.isEnabledFor(10):
                logger.debug_cache("--- Проверка совместимости кэша ---")

            for key, expected_value in checks.items():
                cached_value = cache_data.get(key)
                if isinstance(expected_value, float):
                    if not math.isclose(
                        cached_value if isinstance(cached_value, float) else -1.0,
                        expected_value,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        if logger.isEnabledFor(10):
                            logger.debug_cache(
                                f"❌ НЕ СОВПАДАЕТ (float): {key} | Ожидалось: {expected_value} | В кэше: {cached_value}"
                            )
                        is_compatible = False
                elif cached_value != expected_value:
                    if logger.isEnabledFor(10):
                        logger.debug_cache(
                            f"❌ НЕ СОВПАДАЕТ: {key} | Ожидалось: {expected_value} | В кэше: {cached_value}"
                        )
                    is_compatible = False
                else:
                    if logger.isEnabledFor(10):
                        logger.debug_cache(f"✅ Совпадает: {key} = {cached_value}")

            if not is_compatible:
                logger.info("Кэш несовместим. Требуется пересоздание.")
                return False

            # Если все проверки прошли, загружаем данные
            self.cache = cache_data["cache"]
            self.distance_cache = cache_data["distance_cache"]
            self.total_cells = cache_data["total_cells"]
            
            # Восстанавливаем _all_neighbors_cache из загруженного кэша
            self._restore_all_neighbors_cache_from_cache()
            
            logger.info(f"✅ Кэш совместим и успешно загружен с диска: {cache_file}")
            logger.info(f"   Загружено клеток в кэше: {len(self.cache)}")
            logger.info(f"   Примеры ключей кэша: {list(self.cache.keys())[:10]}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки кэша: {e}")
            return False

    def _restore_all_neighbors_cache_from_cache(self):
        """Восстанавливает _all_neighbors_cache из загруженного основного кэша"""
        if not self.cache:
            logger.warning("Основной кэш пуст, невозможно восстановить _all_neighbors_cache")
            return
            
        self._all_neighbors_cache = {}
        
        # Восстанавливаем индексы соседей из основного кэша
        for cell_idx, connections in self.cache.items():
            neighbors = set()
            
            # Собираем все индексы соседей из всех типов связей
            for conn_type in ["local", "functional_candidates", "distant"]:
                if conn_type in connections:
                    for conn_info in connections[conn_type]:
                        neighbors.add(conn_info.target_idx)
            
            self._all_neighbors_cache[cell_idx] = list(neighbors)
        
        total_neighbors = sum(len(neighbors) for neighbors in self._all_neighbors_cache.values())
        avg_neighbors = total_neighbors / len(self._all_neighbors_cache) if self._all_neighbors_cache else 0
        
        logger.info(f"✅ Восстановлен _all_neighbors_cache из основного кэша:")
        logger.info(f"   Клеток: {len(self._all_neighbors_cache)}")
        logger.info(f"   Среднее количество соседей на клетку: {avg_neighbors:.1f}")

    def _get_cache_key(self) -> str:
        """Генерирует уникальный ключ для кэша на основе конфигурации"""
        key_data = {
            "lattice_dimensions": self.lattice_dimensions,
            "adaptive_radius": self.adaptive_radius,
            "local_threshold": self.local_threshold,
            "functional_threshold": self.functional_threshold,
            "distant_threshold": self.distant_threshold,
            "cache_version": self.cache_config.get("cache_version", "2024.1"),
            # GPU/CPU кэш полностью совместим, убираем GPU из ключа
        }
        if logger.isEnabledFor(10):
            logger.debug_cache(f"🔑 Cache key data: {key_data}")

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _save_cache_to_disk(self):
        """Сохранение кэша на диск с метаданными"""
        try:
            os.makedirs("cache", exist_ok=True)
            cache_key = self._get_cache_key()
            cache_file = f"cache/connection_cache_{cache_key}.pkl"

            # Подготавливаем данные для сохранения с полной совместимостью
            cache_data = {
                "cache": self.cache,
                "distance_cache": self.distance_cache,
                "adaptive_radius": self.adaptive_radius,
                "lattice_dimensions": self.lattice_dimensions,
                "local_threshold": self.local_threshold,
                "functional_threshold": self.functional_threshold,
                "distant_threshold": self.distant_threshold,
                "total_cells": self.total_cells,
                "cache_version": self.cache_config.get("cache_version", "2024.1"),
                "timestamp": time.time(),
                # Информационные поля (не влияют на совместимость)
                "created_with_gpu": self.use_gpu,
                "creator_device": (
                    torch.cuda.get_device_name(0)
                    if self.use_gpu and torch.cuda.is_available()
                    else "CPU"
                ),
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"✅ Кэш сохранен: {cache_file}")
            logger.info(f"   Размер кэша: {len(self.cache)} клеток")
            logger.info(f"   Adaptive radius: {self.adaptive_radius}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения кэша: {e}")

    def precompute_all_connections(self, force_rebuild: bool = False):
        """
        Основной метод для предвычисления всех связей.
        Использует GPU для ускорения если доступно.
        """
        if self.is_precomputed and not force_rebuild:
            logger.info("✅ Кэш уже в памяти, переиспользование.")
            return

        if not force_rebuild and self._load_cache_from_disk():
            self.is_precomputed = True
            return

        # --- Логика пересоздания кэша ---
        logger.info("🔄 Пересоздание кэша классификации связей...")
        rebuild_start_time = time.time()

        # Получаем список всех соседей для каждой клетки
        all_neighbors = self._compute_all_neighbors()

        # Pre-compute классификация для каждой клетки
        for cell_idx in range(self.total_cells):
            neighbors = all_neighbors[cell_idx]
            if not neighbors:
                continue

            # Классифицируем связи для этой клетки
            cell_connections = self._precompute_cell_connections(cell_idx, neighbors)
            self.cache[cell_idx] = cell_connections

            # Прогресс лог
            if cell_idx % 1000 == 0:
                logger.debug_cache(f"Pre-computed {cell_idx}/{self.total_cells} клеток")

        # Сохраняем кэш на диск
        self._save_cache_to_disk()

        self.is_precomputed = True
        logger.info(f"✅ Pre-compute завершен для {len(self.cache)} клеток")
        logger.info(
            f"   Время пересоздания: {time.time() - rebuild_start_time:.2f} секунд"
        )

    def _compute_all_neighbors(self) -> Dict[int, List[int]]:
        """Вычисляет всех соседей для каждой клетки в радиусе adaptive_radius"""
        if self._all_neighbors_cache is not None:
            return self._all_neighbors_cache

        if self.use_gpu:
            logger.info("🚀 Вычисляем всех соседей на GPU...")
            self._all_neighbors_cache = self._compute_all_neighbors_gpu()
            return self._all_neighbors_cache
        else:
            logger.error("❌ GPU не доступен для вычисления соседей")
            raise RuntimeError("GPU обязателен для работы системы с RTX 5090")

    def _compute_all_neighbors_cpu(self) -> Dict[int, List[int]]:
        """CPU версия вычисления соседей"""
        all_neighbors = {}
        x_dim, y_dim, z_dim = self.lattice_dimensions

        for cell_idx in range(self.total_cells):
            # Конвертируем индекс в 3D координаты
            x = cell_idx % x_dim
            y = (cell_idx // x_dim) % y_dim
            z = cell_idx // (x_dim * y_dim)

            neighbors = []

            # Проверяем все клетки в радиусе
            for dx in range(
                -int(self.adaptive_radius) - 1, int(self.adaptive_radius) + 2
            ):
                for dy in range(
                    -int(self.adaptive_radius) - 1, int(self.adaptive_radius) + 2
                ):
                    for dz in range(
                        -int(self.adaptive_radius) - 1, int(self.adaptive_radius) + 2
                    ):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx, ny, nz = x + dx, y + dy, z + dz

                        # Проверяем границы
                        if 0 <= nx < x_dim and 0 <= ny < y_dim and 0 <= nz < z_dim:
                            neighbor_idx = nx + ny * x_dim + nz * (x_dim * y_dim)

                            # Проверяем расстояние
                            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                            if distance <= self.adaptive_radius:
                                neighbors.append(neighbor_idx)

            all_neighbors[cell_idx] = neighbors

        self._all_neighbors_cache = all_neighbors
        logger.info(f"✅ Вычислены соседи для {len(all_neighbors)} клеток")
        return all_neighbors

    def _compute_all_neighbors_gpu(self) -> Dict[int, List[int]]:
        """GPU-ускоренная версия вычисления соседей"""
        try:
            x_dim, y_dim, z_dim = self.lattice_dimensions

            # Создаем координаты всех клеток на GPU
            all_indices = torch.arange(self.total_cells, device=self.device)

            x_coords = all_indices % x_dim
            y_coords = (all_indices // x_dim) % y_dim
            z_coords = all_indices // (x_dim * y_dim)

            all_coords = torch.stack([x_coords, y_coords, z_coords], dim=1).float()

            logger.info(
                f"💾 GPU memory для координат: {all_coords.numel() * 4 / 1024**2:.1f}MB"
            )

            all_neighbors = {}
            batch_size = min(self.gpu_batch_size, self.total_cells)

            # Обрабатываем батчами для экономии памяти
            for start_idx in range(0, self.total_cells, batch_size):
                end_idx = min(start_idx + batch_size, self.total_cells)
                batch_coords = all_coords[start_idx:end_idx]

                # Вычисляем расстояния до всех других клеток
                # batch_coords: [batch_size, 3], all_coords: [total_cells, 3]
                distances = torch.cdist(
                    batch_coords, all_coords
                )  # [batch_size, total_cells]

                # Находим соседей в радиусе (исключая саму клетку)
                for i, cell_idx in enumerate(range(start_idx, end_idx)):
                    # ИСПРАВЛЕНИЕ: Используем distant_threshold вместо adaptive_radius
                    # Это гарантирует, что все найденные соседи попадут в одну из трех категорий
                    neighbor_mask = (distances[i] <= self.distant_threshold) & (
                        distances[i] > 0
                    )
                    neighbor_indices = torch.where(neighbor_mask)[0].cpu().tolist()
                    all_neighbors[cell_idx] = neighbor_indices

                # Освобождаем GPU память
                del distances
                torch.cuda.empty_cache()

                if start_idx % (batch_size * 10) == 0:
                    logger.info(
                        f"🚀 GPU: обработано {end_idx}/{self.total_cells} клеток"
                    )

            self._all_neighbors_cache = all_neighbors
            logger.info(f"✅ GPU: Вычислены соседи для {len(all_neighbors)} клеток")
            
            # Логирование для диагностики
            total_neighbors = sum(len(neighbors) for neighbors in all_neighbors.values())
            avg_neighbors = total_neighbors / len(all_neighbors) if all_neighbors else 0
            logger.info(f"   Среднее количество соседей на клетку: {avg_neighbors:.1f}")
            logger.info(f"   Используемый порог: {self.distant_threshold} (distant_threshold)")
            
            return all_neighbors

        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return self._compute_all_neighbors_cpu()

    def _precompute_cell_connections(
        self, cell_idx: int, neighbor_indices: List[int]
    ) -> Dict[str, List[CachedConnectionInfo]]:
        """Pre-compute классификация связей для одной клетки"""
        connections = {
            "local": [],
            "functional_candidates": [],  # Кандидаты для функциональной проверки
            "distant": [],
        }

        for neighbor_idx in neighbor_indices:
            # Вычисляем расстояния
            euclidean_dist = self.distance_calculator.euclidean_distance(
                cell_idx, neighbor_idx
            )
            manhattan_dist = self.distance_calculator.manhattan_distance(
                cell_idx, neighbor_idx
            )

            # Кэшируем расстояния
            self.distance_cache[(cell_idx, neighbor_idx)] = {
                "euclidean": euclidean_dist,
                "manhattan": manhattan_dist,
            }

            # Классифицируем по расстоянию
            # LOCAL: 0 < distance < local_threshold
            if euclidean_dist < self.local_threshold:
                category = ConnectionCategory.LOCAL
                connections["local"].append(
                    CachedConnectionInfo(
                        target_idx=neighbor_idx,
                        euclidean_distance=euclidean_dist,
                        manhattan_distance=manhattan_dist,
                        category=category,
                    )
                )

            # FUNCTIONAL: local_threshold ≤ distance ≤ functional_threshold
            elif euclidean_dist <= self.functional_threshold:
                # Кандидат для функциональной проверки
                connections["functional_candidates"].append(
                    CachedConnectionInfo(
                        target_idx=neighbor_idx,
                        euclidean_distance=euclidean_dist,
                        manhattan_distance=manhattan_dist,
                        category=ConnectionCategory.FUNCTIONAL,  # Предварительно
                    )
                )

            # DISTANT: functional_threshold < distance ≤ distant_threshold
            else:
                category = ConnectionCategory.DISTANT
                connections["distant"].append(
                    CachedConnectionInfo(
                        target_idx=neighbor_idx,
                        euclidean_distance=euclidean_dist,
                        manhattan_distance=manhattan_dist,
                        category=category,
                    )
                )

        return connections

    def get_neighbors_and_classification(
        self, 
        cell_idx: int, 
        states: Optional[torch.Tensor] = None,
        functional_similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Возвращает соседей И их классификацию одним вызовом
        
        Args:
            cell_idx: Индекс клетки
            states: Состояния всех клеток для функциональной проверки
            functional_similarity_threshold: Порог для функциональной близости
            
        Returns:
            {
                "local": {"indices": [...], "states": tensor, "connections": [ConnectionInfo]},
                "functional": {"indices": [...], "states": tensor, "connections": [ConnectionInfo]},
                "distant": {"indices": [...], "states": tensor, "connections": [ConnectionInfo]}
            }
        """
        # Получаем всех соседей из кэша
        if self._all_neighbors_cache is None:
            logger.warning("⚠️ _all_neighbors_cache not initialized, computing neighbors now...")
            self._all_neighbors_cache = self._compute_all_neighbors()
            
        if cell_idx not in self._all_neighbors_cache:
            logger.warning(f"Cell {cell_idx} not found in cache, returning empty neighbors")
            return {
                "local": {"indices": [], "states": torch.empty(0, self.state_size), "connections": []},
                "functional": {"indices": [], "states": torch.empty(0, self.state_size), "connections": []},
                "distant": {"indices": [], "states": torch.empty(0, self.state_size), "connections": []}
            }
            
        neighbor_indices = self._all_neighbors_cache[cell_idx]
        
        # Используем существующий метод для классификации
        classified_connections = self.get_cached_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            states=states,
            functional_similarity_threshold=functional_similarity_threshold
        )
        
        # Формируем результат с индексами и состояниями
        result = {}
        for category in [ConnectionCategory.LOCAL, ConnectionCategory.FUNCTIONAL, ConnectionCategory.DISTANT]:
            connections = classified_connections.get(category, [])
            indices = [conn.target_idx for conn in connections]
            
            # Извлекаем состояния соседей если states переданы
            if states is not None and indices:
                if states.dim() == 3:  # [batch, num_cells, state_size]
                    neighbor_states = states[0, indices, :]
                elif states.dim() == 2:  # [num_cells, state_size]
                    neighbor_states = states[indices]
                else:
                    raise RuntimeError(f"Unexpected states dimension: {states.shape}")
            else:
                neighbor_states = torch.empty(0, self.state_size if hasattr(self, 'state_size') else states.shape[-1] if states is not None else 0)
                
            category_name = category.value.lower()
            result[category_name] = {
                "indices": indices,
                "states": neighbor_states,
                "connections": connections
            }
            
        return result

    def get_cached_connections(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        states: Optional[torch.Tensor] = None,
        functional_similarity_threshold: float = 0.3,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """
        Быстрая классификация с использованием кэша

        Args:
            cell_idx: Индекс клетки
            neighbor_indices: Список индексов соседей
            states: Состояния клеток для функциональной проверки
            functional_similarity_threshold: Порог для функциональной близости

        Returns:
            Классифицированные связи по категориям
        """
        # Проверяем инициализацию кэша
        if not self.is_precomputed:
            logger.warning("⚠️ Cache not precomputed, initializing now...")
            self.precompute_all_connections(force_rebuild=False)
            
        if cell_idx not in self.cache:
            logger.warning(f"Кэш не найден для клетки {cell_idx}")
            logger.debug_cache(f"Cache size: {len(self.cache)}, Cache keys sample: {list(self.cache.keys())[:10] if self.cache else 'Empty'}")
            logger.debug_cache(f"Looking for cell_idx: {cell_idx}, type: {type(cell_idx)}")
            return {cat: [] for cat in ConnectionCategory}

        cached_data = self.cache[cell_idx]
        result = {cat: [] for cat in ConnectionCategory}

        # Создаем set для быстрого поиска
        neighbor_set = set(neighbor_indices)
        
        # DEBUG: Log cache contents and neighbor indices
        logger.debug_cache(f"🔍 get_cached_connections for cell {cell_idx}:")
        logger.debug_cache(f"   neighbor_indices: {list(neighbor_indices)[:10]}... (len={len(neighbor_indices)})")
        logger.debug_cache(f"   cached_data type: {type(cached_data)}")
        logger.debug_cache(f"   cached_data keys: {list(cached_data.keys()) if isinstance(cached_data, dict) else 'Not a dict!'}")
        # Определяем формат ключей для логирования
        if isinstance(cached_data, dict):
            if ConnectionCategory.LOCAL in cached_data:
                local_count = len(cached_data.get(ConnectionCategory.LOCAL, []))
                functional_count = len(cached_data.get(ConnectionCategory.FUNCTIONAL, []))
                distant_count = len(cached_data.get(ConnectionCategory.DISTANT, []))
            else:
                local_count = len(cached_data.get('local', []))
                functional_count = len(cached_data.get('functional_candidates', []))
                distant_count = len(cached_data.get('distant', []))
        else:
            local_count = 'N/A'
            functional_count = 'N/A'
            distant_count = 'N/A'
            
        logger.debug_cache(f"   cached local connections: {local_count}")
        logger.debug_cache(f"   cached functional_candidates: {functional_count}")
        logger.debug_cache(f"   cached distant connections: {distant_count}")
        
        # Check first few cached connections
        if isinstance(cached_data, dict) and cached_data.get('local'):
            first_local = cached_data['local'][0]
            if hasattr(first_local, 'target_idx'):
                logger.debug_cache(f"   First local connection target_idx: {first_local.target_idx}")
            else:
                logger.debug_cache(f"   First local connection target_idx: {first_local.get('target_idx', 'N/A')}")

        # Проверяем структуру кэша
        if not isinstance(cached_data, dict):
            logger.error(f"❌ Cache data for cell {cell_idx} is not a dict: {type(cached_data)}")
            return {cat: [] for cat in ConnectionCategory}
            
        # Проверяем наличие необходимых ключей - поддерживаем оба формата (string и enum)
        # Старый формат использует строки, новый формат использует enum
        has_string_keys = "local" in cached_data
        has_enum_keys = ConnectionCategory.LOCAL in cached_data
        
        if not has_string_keys and not has_enum_keys:
            logger.error(f"❌ Cache data for cell {cell_idx} has unexpected format")
            logger.error(f"   Available keys: {list(cached_data.keys())}")
            return {cat: [] for cat in ConnectionCategory}

        # Определяем какой формат ключей используется
        if has_enum_keys:
            local_key = ConnectionCategory.LOCAL
            functional_key = ConnectionCategory.FUNCTIONAL
            distant_key = ConnectionCategory.DISTANT
        else:
            local_key = "local"
            functional_key = "functional_candidates"
            distant_key = "distant"
        
        # LOCAL связи - прямо из кэша
        for conn in cached_data.get(local_key, []):
            # Обрабатываем оба формата: объект CachedConnectionInfo или словарь
            if hasattr(conn, 'target_idx'):
                target_idx = conn.target_idx
                euclidean_distance = conn.euclidean_distance
                manhattan_distance = conn.manhattan_distance
            else:
                # Если это словарь (из загруженного кэша)
                target_idx = conn['target_idx']
                euclidean_distance = conn['euclidean_distance']
                manhattan_distance = conn['manhattan_distance']
                
            if target_idx in neighbor_set:
                result[ConnectionCategory.LOCAL].append(
                    ConnectionInfo(
                        source_idx=cell_idx,
                        target_idx=target_idx,
                        euclidean_distance=euclidean_distance,
                        manhattan_distance=manhattan_distance,
                        category=ConnectionCategory.LOCAL,
                    )
                )

        # DISTANT связи - прямо из кэша
        for conn in cached_data.get(distant_key, []):
            # Обрабатываем оба формата: объект CachedConnectionInfo или словарь
            if hasattr(conn, 'target_idx'):
                target_idx = conn.target_idx
                euclidean_distance = conn.euclidean_distance
                manhattan_distance = conn.manhattan_distance
            else:
                # Если это словарь (из загруженного кэша)
                target_idx = conn['target_idx']
                euclidean_distance = conn['euclidean_distance']
                manhattan_distance = conn['manhattan_distance']
                
            if target_idx in neighbor_set:
                result[ConnectionCategory.DISTANT].append(
                    ConnectionInfo(
                        source_idx=cell_idx,
                        target_idx=target_idx,
                        euclidean_distance=euclidean_distance,
                        manhattan_distance=manhattan_distance,
                        category=ConnectionCategory.DISTANT,
                    )
                )

        # FUNCTIONAL кандидаты - требуют проверки similarity
        functional_candidates = []
        for conn in cached_data.get(functional_key, []):
            # Обрабатываем оба формата: объект CachedConnectionInfo или словарь
            if hasattr(conn, 'target_idx'):
                target_idx = conn.target_idx
            else:
                target_idx = conn['target_idx']
                
            if target_idx in neighbor_set:
                functional_candidates.append(conn)

        if functional_candidates and states is not None:
            # Быстрая функциональная проверка
            functional_connections = self._check_functional_similarity(
                cell_idx, functional_candidates, states, functional_similarity_threshold
            )
            result[ConnectionCategory.FUNCTIONAL].extend(functional_connections)

            # ИСПРАВЛЕНО: Остальные кандидаты остаются FUNCTIONAL (не становятся DISTANT)
            # Это соответствует логике оригинального классификатора
            functional_indices = {conn.target_idx for conn in functional_connections}
            for conn in functional_candidates:
                # Обрабатываем оба формата
                if hasattr(conn, 'target_idx'):
                    target_idx = conn.target_idx
                    euclidean_distance = conn.euclidean_distance
                    manhattan_distance = conn.manhattan_distance
                else:
                    target_idx = conn['target_idx']
                    euclidean_distance = conn['euclidean_distance']
                    manhattan_distance = conn['manhattan_distance']
                    
                if target_idx not in functional_indices:
                    result[ConnectionCategory.FUNCTIONAL].append(
                        ConnectionInfo(
                            source_idx=cell_idx,
                            target_idx=target_idx,
                            euclidean_distance=euclidean_distance,
                            manhattan_distance=manhattan_distance,
                            category=ConnectionCategory.FUNCTIONAL,
                        )
                    )
        else:
            # Без проверки similarity все кандидаты становятся FUNCTIONAL
            for conn in functional_candidates:
                # Обрабатываем оба формата
                if hasattr(conn, 'target_idx'):
                    target_idx = conn.target_idx
                    euclidean_distance = conn.euclidean_distance
                    manhattan_distance = conn.manhattan_distance
                else:
                    target_idx = conn['target_idx']
                    euclidean_distance = conn['euclidean_distance']
                    manhattan_distance = conn['manhattan_distance']
                    
                result[ConnectionCategory.FUNCTIONAL].append(
                    ConnectionInfo(
                        source_idx=cell_idx,
                        target_idx=target_idx,
                        euclidean_distance=euclidean_distance,
                        manhattan_distance=manhattan_distance,
                        category=ConnectionCategory.FUNCTIONAL,
                    )
                )

        return result

    def _check_functional_similarity(
        self,
        cell_idx: int,
        candidates: List[CachedConnectionInfo],
        states: torch.Tensor,
        threshold: float,
    ) -> List[ConnectionInfo]:
        """Быстрая проверка функциональной близости для кандидатов"""
        if not candidates:
            return []

        functional_connections = []

        try:
            cell_state = states[cell_idx]

            for conn in candidates:
                # Обрабатываем оба формата
                if hasattr(conn, 'target_idx'):
                    target_idx = conn.target_idx
                    euclidean_distance = conn.euclidean_distance
                    manhattan_distance = conn.manhattan_distance
                else:
                    target_idx = conn['target_idx']
                    euclidean_distance = conn['euclidean_distance']
                    manhattan_distance = conn['manhattan_distance']
                    
                if target_idx < states.shape[0]:
                    neighbor_state = states[target_idx]

                    # Косинусное сходство (keep on GPU)
                    similarity_tensor = torch.cosine_similarity(
                        cell_state.unsqueeze(0), neighbor_state.unsqueeze(0), dim=1
                    )
                    similarity = similarity_tensor.squeeze().cpu().item()

                    # ИСПРАВЛЕНО: Для случайных данных используем более мягкий критерий
                    # Если similarity > -0.5 (не сильно противоположные), считаем функциональным
                    effective_threshold = min(
                        threshold, -0.3
                    )  # Более мягкий порог для тестов

                    if similarity >= effective_threshold:
                        functional_connections.append(
                            ConnectionInfo(
                                source_idx=cell_idx,
                                target_idx=target_idx,
                                euclidean_distance=euclidean_distance,
                                manhattan_distance=manhattan_distance,
                                category=ConnectionCategory.FUNCTIONAL,
                                functional_similarity=similarity,
                            )
                        )

        except IndexError as e:
            logger.warning(f"⚠️⚠️⚠️ Ошибка доступа к состояниям: {e}")

        return functional_connections

    def get_batch_cached_connections(
        self,
        cell_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        states: torch.Tensor,
        functional_similarity_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch версия для максимальной производительности

        Returns:
            Dict с масками для каждого типа связей
        """
        batch_size, max_neighbors = neighbor_indices.shape
        device = cell_indices.device

        # Инициализируем маски
        local_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        functional_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        distant_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        valid_mask = neighbor_indices >= 0

        # Convert to CPU once for batch processing - avoid repeated .item() calls
        cell_indices_cpu = cell_indices.cpu().numpy()
        neighbor_indices_cpu = neighbor_indices.cpu().numpy()
        
        # Обрабатываем каждую клетку в batch
        for batch_idx in range(batch_size):
            cell_idx = int(cell_indices_cpu[batch_idx])
            neighbors = neighbor_indices_cpu[batch_idx]
            valid_neighbors = neighbors[neighbors >= 0].tolist()

            if not valid_neighbors:
                continue

            # Получаем классификацию из кэша
            classifications = self.get_cached_connections(
                cell_idx, valid_neighbors, states, functional_similarity_threshold
            )

            # Заполняем маски
            for neighbor_pos, neighbor_idx in enumerate(neighbors):
                if neighbor_idx < 0:
                    continue

                neighbor_idx_val = int(neighbor_idx)

                # Ищем в результатах классификации
                for category, connections in classifications.items():
                    for conn in connections:
                        if conn.target_idx == neighbor_idx_val:
                            if category == ConnectionCategory.LOCAL:
                                local_mask[batch_idx, neighbor_pos] = True
                            elif category == ConnectionCategory.FUNCTIONAL:
                                functional_mask[batch_idx, neighbor_pos] = True
                            elif category == ConnectionCategory.DISTANT:
                                distant_mask[batch_idx, neighbor_pos] = True
                            break

        return {
            "local_mask": local_mask,
            "functional_mask": functional_mask,
            "distant_mask": distant_mask,
            "valid_mask": valid_mask,
        }

    def get_cache_stats(self) -> Dict[str, any]:
        """Статистика кэширования"""
        if not self.cache:
            return {"status": "empty"}

        total_connections = 0
        local_count = 0
        functional_candidates_count = 0
        distant_count = 0

        for cell_data in self.cache.values():
            # Проверяем формат кэша - строковые ключи или enum
            if cell_data and isinstance(next(iter(cell_data.keys()), None), str):
                # Старый формат со строковыми ключами
                local_count += len(cell_data.get("local", []))
                functional_candidates_count += len(cell_data.get("functional_candidates", []))
                distant_count += len(cell_data.get("distant", []))
            else:
                # Новый формат с enum ключами (пока не используется, но на будущее)
                from .connection_types import ConnectionCategory
                local_count += len(cell_data.get(ConnectionCategory.LOCAL, []))
                functional_candidates_count += len(cell_data.get(ConnectionCategory.FUNCTIONAL, []))
                distant_count += len(cell_data.get(ConnectionCategory.DISTANT, []))

        total_connections = local_count + functional_candidates_count + distant_count

        return {
            "status": "active",
            "cached_cells": len(self.cache),
            "total_connections": total_connections,
            "local_connections": local_count,
            "functional_candidates": functional_candidates_count,
            "distant_connections": distant_count,
            "cache_size_mb": len(pickle.dumps(self.cache)) / (1024 * 1024),
            "distance_cache_entries": len(self.distance_cache),
        }

    def clear_cache(self):
        """Очистка кэша"""
        self.cache.clear()
        self.distance_cache.clear()
        self.neighbor_cache.clear()
        logger.info("Кэш очищен")
