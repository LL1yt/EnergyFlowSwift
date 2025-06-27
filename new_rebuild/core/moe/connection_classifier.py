#!/usr/bin/env python3
"""
Connection Classifier - классификация связей между клетками
=========================================================

Основной классификатор связей, использующий модульную архитектуру
для определения типов связей в 3D нейронной решетке.

НОВАЯ АРХИТЕКТУРА:
- ConnectionCacheManager для pre-computed оптимизации
- Fallback к оригинальной логике при отсутствии кэша
- Автоматическое кэширование для повторного использования
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

from .connection_types import ConnectionCategory, ConnectionInfo
from .distance_calculator import DistanceCalculator
from .functional_similarity import FunctionalSimilarityAnalyzer
from .connection_cache import ConnectionCacheManager
from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedConnectionClassifier(nn.Module):
    """
    Унифицированный классификатор связей с pre-computed кэшированием

    ОПТИМИЗАЦИИ:
    - ConnectionCacheManager для pre-computed структур
    - Автоматическая инициализация кэша при первом запуске
    - Fallback к оригинальной логике при необходимости
    - Быстрая batch обработка через кэш

    Использует:
    - DistanceCalculator для пространственного анализа (fallback)
    - FunctionalSimilarityAnalyzer для функциональной близости (fallback)
    - Learnable пороги для адаптивной классификации
    """

    def __init__(
        self, lattice_dimensions: Tuple[int, int, int], enable_cache: bool = None
    ):
        super().__init__()

        config = get_project_config()

        self.lattice_dimensions = lattice_dimensions
        self.state_size = config.model.state_size

        # Получаем настройки кэширования из конфигурации
        cache_config = asdict(config.cache) if config.cache else {}
        self.enable_cache = (
            cache_config.get("enabled", True) if enable_cache is None else enable_cache
        )
        self.enable_performance_monitoring = cache_config.get(
            "enable_performance_monitoring", False
        )
        self.enable_detailed_stats = cache_config.get("enable_detailed_stats", False)

        # Модульные компоненты (для fallback)
        self.distance_calculator = DistanceCalculator(lattice_dimensions)
        self.similarity_analyzer = FunctionalSimilarityAnalyzer(self.state_size)

        # Pre-computed кэш менеджер
        if self.enable_cache:
            self.cache_manager = ConnectionCacheManager(
                lattice_dimensions, cache_config
            )
            # Автоматически инициализируем кэш
            self._initialize_cache()
        else:
            self.cache_manager = None

        # Learnable пороги для классификации
        self.local_distance_threshold = nn.Parameter(
            torch.tensor(config.model.local_distance_threshold)
        )
        self.functional_distance_threshold = nn.Parameter(
            torch.tensor(config.model.functional_distance_threshold)
        )
        self.distant_distance_threshold = nn.Parameter(
            torch.tensor(config.model.distant_distance_threshold)
        )
        self.functional_similarity_threshold = nn.Parameter(
            torch.tensor(config.model.functional_similarity_threshold)
        )

        # Целевые пропорции из конфига
        self.target_ratios = {
            "local": config.model.local_tier_ratio,
            "functional": config.model.functional_tier_ratio,
            "distant": config.model.distant_tier_ratio,
        }

        # Статистика использования
        self.reset_stats()

        # Статистика производительности (включается по настройкам)
        if self.enable_performance_monitoring:
            self.performance_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_classifications": 0,
                "avg_cache_speedup": 0.0,
                "total_fallback_time": 0.0,
                "total_cache_time": 0.0,
            }
        else:
            self.performance_stats = {}

        cache_status = "enabled" if self.enable_cache else "disabled"
        performance_status = (
            "enabled" if self.enable_performance_monitoring else "disabled"
        )
        logger.info(
            f"UnifiedConnectionClassifier initialized for {lattice_dimensions}, cache: {cache_status}, performance monitoring: {performance_status}"
        )

    def _initialize_cache(self):
        """Инициализация pre-computed кэша"""
        try:
            if self.cache_manager is not None:
                logger.info("🔄 Инициализация connection cache...")
                self.cache_manager.precompute_all_connections()

                # Логируем статистику кэша
                stats = self.cache_manager.get_cache_stats()
                if stats["status"] == "active":
                    logger.info(
                        f"✅ Cache готов: {stats['cached_cells']} клеток, {stats['total_connections']} связей, {stats['cache_size_mb']:.1f}MB"
                    )
                else:
                    logger.warning("⚠️ Cache пуст, используем fallback режим")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации кэша: {e}")
            logger.info("🔄 Переключаемся на fallback режим без кэша")
            self.cache_manager = None

    def classify_connections_batch(
        self,
        cell_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch классификация связей с оптимизацией через кэш

        Args:
            cell_indices: [batch] - индексы клеток
            neighbor_indices: [batch, max_neighbors] - индексы соседей
            states: [total_cells, state_size] - состояния всех клеток

        Returns:
            Dict с масками для каждого типа связей
        """
        if self.enable_performance_monitoring:
            self.performance_stats["total_classifications"] += 1

        # Пробуем использовать кэш
        if self.cache_manager is not None:
            try:
                if self.enable_performance_monitoring:
                    import time

                    start_time = time.time()

                result = self.cache_manager.get_batch_cached_connections(
                    cell_indices=cell_indices,
                    neighbor_indices=neighbor_indices,
                    states=states,
                    functional_similarity_threshold=self.functional_similarity_threshold.item(),
                )

                if self.enable_performance_monitoring:
                    cache_time = time.time() - start_time
                    self.performance_stats["cache_hits"] += 1
                    self.performance_stats["total_cache_time"] += cache_time

                    # Обновляем статистику speedup
                    self.performance_stats["avg_cache_speedup"] = (
                        self.performance_stats["avg_cache_speedup"] * 0.9
                        + cache_time * 0.1
                    )

                    if self.enable_detailed_stats:
                        logger.debug(
                            f"✅ Cache hit: {cache_time:.4f}s для batch_size={cell_indices.shape[0]}"
                        )
                return result

            except Exception as e:
                if self.enable_detailed_stats:
                    logger.warning(f"Cache miss, fallback: {e}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_misses"] += 1

        # Fallback к оригинальной логике
        if self.enable_performance_monitoring:
            import time

            start_time = time.time()

        result = self._classify_connections_batch_original(
            cell_indices, neighbor_indices, states
        )

        if self.enable_performance_monitoring:
            fallback_time = time.time() - start_time
            self.performance_stats["total_fallback_time"] += fallback_time

        return result

    def _classify_connections_batch_original(
        self,
        cell_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Оригинальная batch классификация (fallback режим)
        """
        try:
            logger.debug(
                f"🔍 classify_connections_batch_original: входные данные - cell_indices.shape={cell_indices.shape}, neighbor_indices.shape={neighbor_indices.shape}, states.shape={states.shape}"
            )

            batch_size, max_neighbors = neighbor_indices.shape
            device = cell_indices.device

            # Создаем валидную маску (исключаем -1 padding)
            valid_mask = neighbor_indices >= 0

        except Exception as e:
            import traceback

            logger.error(
                f"❌ ОШИБКА в classify_connections_batch_original (начало): {e}"
            )
            logger.error(f"📍 Traceback:\n{traceback.format_exc()}")
            raise

        if valid_mask.sum().item() == 0:
            return self._empty_classification_result(batch_size, max_neighbors, device)

        # Извлекаем валидные пары
        valid_cells = cell_indices.unsqueeze(1).expand(-1, max_neighbors)[valid_mask]
        valid_neighbors = neighbor_indices[valid_mask]

        # 1. Расчет расстояний
        euclidean_distances = self.distance_calculator.euclidean_distance_batch(
            valid_cells, valid_neighbors
        )

        # 2. Классификация по расстоянию
        local_mask_flat = euclidean_distances <= self.local_distance_threshold
        distant_mask_flat = euclidean_distances >= self.distant_distance_threshold
        # Функциональные связи: между local и functional_distance_threshold
        functional_candidate_mask = (
            euclidean_distances > self.local_distance_threshold
        ) * (euclidean_distances <= self.functional_distance_threshold)
        # Средние связи: между functional_distance и distant_threshold (будут проверены на similarity)
        middle_mask = (euclidean_distances > self.functional_distance_threshold) * (
            euclidean_distances < self.distant_distance_threshold
        )

        # 3. Уточнение функциональных связей
        # Прямые функциональные (близкие по расстоянию)
        functional_mask_flat = functional_candidate_mask.clone()

        # Проверяем средние связи на функциональную близость
        if middle_mask.sum().item() > 0:
            middle_cells = valid_cells[middle_mask]
            middle_neighbors = valid_neighbors[middle_mask]

            # Валидация индексов перед доступом к states
            max_index = states.shape[0] - 1
            valid_middle_cells = middle_cells <= max_index
            valid_middle_neighbors = middle_neighbors <= max_index

            if not (valid_middle_cells.all() and valid_middle_neighbors.all()):
                logger.warning(
                    f"⚠️ Неправильные индексы: cells max={middle_cells.max().item()}, neighbors max={middle_neighbors.max().item()}, states size={states.shape[0]}"
                )
                # Фильтруем только валидные индексы
                valid_pairs = valid_middle_cells & valid_middle_neighbors
                if valid_pairs.sum().item() == 0:
                    logger.warning("⚠️ Нет валидных пар для функциональной проверки")
                else:
                    middle_cells = middle_cells[valid_pairs]
                    middle_neighbors = middle_neighbors[valid_pairs]
                    cell_states = states[middle_cells]
                    neighbor_states = states[middle_neighbors]

                    similarities = self.similarity_analyzer(
                        cell_states, neighbor_states
                    )
                    high_similarity = (
                        similarities > self.functional_similarity_threshold
                    )

                    # Добавляем средние связи с высокой функциональной близостью к functional
                    middle_indices = torch.where(middle_mask)[0]
                    valid_middle_indices = middle_indices[valid_pairs]
                    functional_additions = valid_middle_indices[high_similarity]
                    functional_mask_flat[functional_additions] = True
            else:
                cell_states = states[middle_cells]
                neighbor_states = states[middle_neighbors]

                similarities = self.similarity_analyzer(cell_states, neighbor_states)
                high_similarity = similarities > self.functional_similarity_threshold

                # Добавляем средние связи с высокой функциональной близостью к functional
                middle_indices = torch.where(middle_mask)[0]
                functional_additions = middle_indices[high_similarity]
                functional_mask_flat[functional_additions] = True

        # 4. Восстанавливаем форму масок
        local_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        functional_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        distant_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )

        local_mask[valid_mask] = local_mask_flat
        functional_mask[valid_mask] = functional_mask_flat
        distant_mask[valid_mask] = distant_mask_flat

        # Обновляем статистику
        self._update_stats_batch(local_mask, functional_mask, distant_mask)

        return self._create_batch_classification_result(
            batch_size,
            max_neighbors,
            valid_mask,
            local_mask,
            functional_mask,
            distant_mask,
            device,
        )

    def classify_connections(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """Единичная классификация связей с кэшированием"""
        # Проверяем пустые neighbor_indices
        if torch.is_tensor(neighbor_indices):
            if neighbor_indices.numel() == 0:
                return {cat: [] for cat in ConnectionCategory}
        else:
            if not neighbor_indices:
                return {cat: [] for cat in ConnectionCategory}

        # Пробуем использовать кэш
        if self.cache_manager is not None:
            try:
                # Создаем полный тензор состояний для кэш менеджера
                cell_state_normalized = (
                    cell_state.unsqueeze(0) if cell_state.dim() == 1 else cell_state
                )
                neighbor_states_normalized = (
                    neighbor_states
                    if neighbor_states.dim() == 2
                    else neighbor_states.unsqueeze(0)
                )

                all_states = torch.cat(
                    [cell_state_normalized, neighbor_states_normalized], dim=0
                )

                result = self.cache_manager.get_cached_connections(
                    cell_idx=cell_idx,
                    neighbor_indices=(
                        neighbor_indices
                        if isinstance(neighbor_indices, list)
                        else neighbor_indices.tolist()
                    ),
                    states=all_states,
                    functional_similarity_threshold=self.functional_similarity_threshold.item(),
                )

                # ИСПРАВЛЕНО: Обновляем статистику при cache hit
                self._update_stats_from_result(result)

                self.performance_stats["cache_hits"] += 1
                logger.debug(f"✅ Cache hit для клетки {cell_idx}")
                logger.debug(
                    f"📊 Cache result: LOCAL={len(result.get(ConnectionCategory.LOCAL, []))}, FUNCTIONAL={len(result.get(ConnectionCategory.FUNCTIONAL, []))}, DISTANT={len(result.get(ConnectionCategory.DISTANT, []))}"
                )
                return result

            except Exception as e:
                logger.debug(f"Cache miss для клетки {cell_idx}: {e}")
                self.performance_stats["cache_misses"] += 1

        # Fallback к оригинальной логике
        return self._classify_connections_original(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )

    def _classify_connections_original(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """Оригинальная единичная классификация (fallback)"""
        # Конвертируем в batch формат
        cell_tensor = torch.tensor([cell_idx], device=cell_state.device)
        if torch.is_tensor(neighbor_indices):
            neighbor_tensor = neighbor_indices.unsqueeze(0)  # [1, num_neighbors]
        else:
            neighbor_tensor = torch.tensor([neighbor_indices], device=cell_state.device)

        # Создаем полный тензор состояний
        try:
            logger.debug(
                f"🔍 concat debug: cell_state.shape={cell_state.shape}, neighbor_states.shape={neighbor_states.shape}"
            )

            # Нормализуем размеры для корректной конкатенации
            if cell_state.dim() == 0:
                raise ValueError(
                    f"cell_state не может быть скаляром, получено: {cell_state.shape}"
                )
            elif cell_state.dim() == 1:
                cell_state_normalized = cell_state.unsqueeze(0)
            elif cell_state.dim() == 2 and cell_state.shape[0] == 1:
                cell_state_normalized = cell_state
            else:
                raise ValueError(
                    f"Неожиданная размерность cell_state: {cell_state.shape}, ожидалось [state_size] или [1, state_size]"
                )

            # neighbor_states должен быть [num_neighbors, state_size]
            if neighbor_states.dim() == 1:
                # [state_size] -> [1, state_size] (только один сосед)
                neighbor_states_normalized = neighbor_states.unsqueeze(0)
            elif neighbor_states.dim() == 2:
                # [num_neighbors, state_size] - уже правильный формат
                neighbor_states_normalized = neighbor_states
            else:
                raise ValueError(
                    f"Неожиданная размерность neighbor_states: {neighbor_states.shape}, ожидалось [state_size] или [num_neighbors, state_size]"
                )

            # Конкатенируем states: [1, state_size] + [num_neighbors, state_size] = [1+num_neighbors, state_size]
            all_states = torch.cat(
                [cell_state_normalized, neighbor_states_normalized], dim=0
            )
            logger.debug(f"🔍 all_states.shape после concat: {all_states.shape}")

        except Exception as e:
            logger.error(f"❌ concat error: {e}")
            logger.error(
                f"🔍 cell_state.shape={cell_state.shape}, neighbor_states.shape={neighbor_states.shape}"
            )
            logger.error(
                f"🔍 cell_state_normalized.shape={locals().get('cell_state_normalized', 'не определено')}"
            )
            logger.error(
                f"🔍 neighbor_states_normalized.shape={locals().get('neighbor_states_normalized', 'не определено')}"
            )
            raise

        # Вызываем batch версию
        batch_result = self._classify_connections_batch_original(
            cell_tensor, neighbor_tensor, all_states
        )

        # Конвертируем результат обратно
        result = {cat: [] for cat in ConnectionCategory}

        for i, neighbor_idx in enumerate(neighbor_indices):
            if batch_result["local_mask"][0, i].item():
                category = ConnectionCategory.LOCAL
            elif batch_result["functional_mask"][0, i].item():
                category = ConnectionCategory.FUNCTIONAL
            elif batch_result["distant_mask"][0, i].item():
                category = ConnectionCategory.DISTANT
            else:
                continue  # Исключенный сосед

            # Вычисляем расстояния
            euclidean_dist = self.distance_calculator.euclidean_distance(
                cell_idx, neighbor_idx
            )
            manhattan_dist = self.distance_calculator.manhattan_distance(
                cell_idx, neighbor_idx
            )

            connection_info = ConnectionInfo(
                source_idx=cell_idx,
                target_idx=neighbor_idx,
                euclidean_distance=euclidean_dist,
                manhattan_distance=manhattan_dist,
                category=category,
            )

            result[category].append(connection_info)

        return result

    def _empty_classification_result(
        self, batch_size: int, max_neighbors: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Пустой результат классификации"""
        return {
            "local_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "functional_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "distant_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "valid_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
        }

    def _create_batch_classification_result(
        self,
        batch_size: int,
        max_neighbors: int,
        valid_mask: torch.Tensor,
        local_mask: torch.Tensor,
        functional_mask: torch.Tensor,
        distant_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Создание результата batch классификации"""
        return {
            "local_mask": local_mask,
            "functional_mask": functional_mask,
            "distant_mask": distant_mask,
            "valid_mask": valid_mask,
        }

    def _update_stats_batch(
        self,
        local_mask: torch.Tensor,
        functional_mask: torch.Tensor,
        distant_mask: torch.Tensor,
    ):
        """Обновление статистики использования"""
        self.usage_stats["local_count"] += local_mask.sum().item()
        self.usage_stats["functional_count"] += functional_mask.sum().item()
        self.usage_stats["distant_count"] += distant_mask.sum().item()
        self.usage_stats["total_classifications"] += 1

    def get_classification_stats(self) -> Dict[str, Any]:
        """Получить статистику классификации включая производительность кэша"""
        total = max(
            1,
            self.usage_stats["local_count"]
            + self.usage_stats["functional_count"]
            + self.usage_stats["distant_count"],
        )

        stats = {
            "local_ratio": self.usage_stats["local_count"] / total,
            "functional_ratio": self.usage_stats["functional_count"] / total,
            "distant_ratio": self.usage_stats["distant_count"] / total,
            "total_connections": total,
            "total_classifications": self.usage_stats["total_classifications"],
            "thresholds": {
                "local_distance": self.local_distance_threshold.item(),
                "functional_distance": self.functional_distance_threshold.item(),
                "distant_distance": self.distant_distance_threshold.item(),
                "functional_similarity": self.functional_similarity_threshold.item(),
            },
        }

        # Добавляем статистику производительности кэша (если мониторинг включен)
        if self.cache_manager is not None:
            cache_stats = self.cache_manager.get_cache_stats()

            stats["cache_performance"] = {
                "cache_enabled": True,
                "performance_monitoring": self.enable_performance_monitoring,
                "detailed_stats": self.enable_detailed_stats,
                "cache_size_mb": cache_stats.get("cache_size_mb", 0),
                "cached_cells": cache_stats.get("cached_cells", 0),
            }

            # Добавляем подробную статистику только если мониторинг включен
            if self.enable_performance_monitoring and self.performance_stats:
                total_requests = self.performance_stats.get(
                    "cache_hits", 0
                ) + self.performance_stats.get("cache_misses", 0)
                hit_rate = self.performance_stats.get("cache_hits", 0) / max(
                    1, total_requests
                )

                stats["cache_performance"].update(
                    {
                        "cache_hit_rate": hit_rate,
                        "cache_hits": self.performance_stats.get("cache_hits", 0),
                        "cache_misses": self.performance_stats.get("cache_misses", 0),
                        "avg_cache_time": self.performance_stats.get(
                            "avg_cache_speedup", 0.0
                        ),
                        "total_cache_time": self.performance_stats.get(
                            "total_cache_time", 0.0
                        ),
                        "total_fallback_time": self.performance_stats.get(
                            "total_fallback_time", 0.0
                        ),
                        "total_classifications": self.performance_stats.get(
                            "total_classifications", 0
                        ),
                    }
                )

                # Вычисляем speedup если есть данные
                if (
                    self.performance_stats.get("total_fallback_time", 0) > 0
                    and self.performance_stats.get("total_cache_time", 0) > 0
                ):
                    avg_fallback_time = self.performance_stats[
                        "total_fallback_time"
                    ] / max(1, self.performance_stats.get("cache_misses", 1))
                    avg_cache_time = self.performance_stats["total_cache_time"] / max(
                        1, self.performance_stats.get("cache_hits", 1)
                    )
                    speedup = avg_fallback_time / max(0.001, avg_cache_time)
                    stats["cache_performance"]["speedup_ratio"] = speedup

        else:
            stats["cache_performance"] = {
                "cache_enabled": False,
                "fallback_mode": True,
                "performance_monitoring": self.enable_performance_monitoring,
                "detailed_stats": self.enable_detailed_stats,
                "cache_size_mb": 0,
                "cached_cells": 0,
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "avg_cache_time": 0.0,
                "total_cache_time": 0.0,
                "total_fallback_time": 0.0,
                "total_classifications": 0,
                "speedup_ratio": 1.0,
            }

        return stats

    def reset_stats(self):
        """Сброс статистики"""
        self.usage_stats = {
            "local_count": 0,
            "functional_count": 0,
            "distant_count": 0,
            "total_classifications": 0,
        }

        if self.enable_performance_monitoring:
            self.performance_stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "total_classifications": 0,
                "avg_cache_speedup": 0.0,
                "total_fallback_time": 0.0,
                "total_cache_time": 0.0,
            }
        else:
            self.performance_stats = {}

    def rebuild_cache(self, force: bool = True):
        """Принудительная перестройка кэша"""
        if self.cache_manager is not None:
            logger.info("🔄 Принудительная перестройка кэша...")
            self.cache_manager.clear_cache()
            self.cache_manager.precompute_all_connections(force_rebuild=force)
            logger.info("✅ Кэш перестроен")
        else:
            logger.warning("Cache manager не доступен")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получить подробную статистику кэша"""
        if self.cache_manager is not None:
            return self.cache_manager.get_cache_stats()
        else:
            return {"status": "disabled"}

    def _update_stats_from_result(
        self, result: Dict[ConnectionCategory, List[ConnectionInfo]]
    ):
        """Обновление статистики из результата классификации"""
        local_count = len(result.get(ConnectionCategory.LOCAL, []))
        functional_count = len(result.get(ConnectionCategory.FUNCTIONAL, []))
        distant_count = len(result.get(ConnectionCategory.DISTANT, []))

        self.usage_stats["local_count"] += local_count
        self.usage_stats["functional_count"] += functional_count
        self.usage_stats["distant_count"] += distant_count
        self.usage_stats["total_classifications"] += 1

        logger.debug(
            f"📊 Stats updated: LOCAL+{local_count}, FUNCTIONAL+{functional_count}, DISTANT+{distant_count}"
        )
