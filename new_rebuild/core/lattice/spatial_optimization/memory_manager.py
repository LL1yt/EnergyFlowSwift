#!/usr/bin/env python3
"""
Memory Pool Manager - Эффективное управление GPU памятью
========================================================

MemoryPoolManager обеспечивает эффективное использование GPU памяти
через pool allocation и переиспользование tensor'ов.
"""

import torch
import gc
from typing import Dict, List, Tuple
from ....config import get_project_config
from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager

logger = get_logger(__name__)


class MemoryPoolManager:
    """
    Менеджер memory pool для эффективного управления GPU памятью

    Использует pool allocation для минимизации фрагментации памяти
    и переиспользования tensor'ов.
    """

    def __init__(self, config: dict = None):
        if config is None:
            # Получаем конфигурацию из проекта - без fallback!
            project_config = get_project_config()
            if not hasattr(project_config, "spatial") or project_config.spatial is None:
                raise ValueError(
                    "MemoryPoolManager требует настроенную spatial конфигурацию! "
                    "Убедитесь, что config.spatial настроен в SimpleProjectConfig."
                )
            
            spatial_cfg = project_config.spatial
            self.config = {
                "garbage_collect_frequency": spatial_cfg.garbage_collect_frequency,
                "memory_pool_size_gb": spatial_cfg.memory_pool_size_gb,
                "chunk_size": spatial_cfg.chunk_size,
                "max_chunks_in_memory": spatial_cfg.max_chunks_in_memory,
                "enable_profiling": spatial_cfg.enable_profiling,
                "log_memory_usage": spatial_cfg.log_memory_usage,
            }
        else:
            self.config = config

        # Используем DeviceManager для консистентного управления устройствами
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Memory pools по типам tensor'ов
        self.tensor_pools: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self.allocated_tensors: List[torch.Tensor] = []
        self.allocation_count = 0

        # Статистика использования
        self.stats = {
            "total_allocations": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "memory_peak_mb": 0.0,
            "gc_calls": 0,
        }

        logger.info(
            f"[DISK] MemoryPoolManager инициализирован через DeviceManager для {self.device}"
        )

    def get_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Получает tensor из pool или создает новый

        Args:
            shape: форма tensor'а
            dtype: тип данных

        Returns:
            tensor готовый к использованию
        """
        self.stats["total_allocations"] += 1

        # Ищем в pool'е
        if shape in self.tensor_pools and self.tensor_pools[shape]:
            tensor = self.tensor_pools[shape].pop()
            tensor.zero_()  # Очищаем данные
            self.stats["pool_hits"] += 1
            return tensor

        # Создаем новый tensor через DeviceManager для безопасного выделения памяти
        tensor = self.device_manager.allocate_tensor(shape, dtype=dtype)
        self.stats["pool_misses"] += 1
        self._track_allocation(tensor)

        return tensor

    def return_tensor(self, tensor: torch.Tensor):
        """
        Возвращает tensor в pool для переиспользования

        Args:
            tensor: tensor для возврата в pool
        """
        if tensor.device != self.device:
            return  # Не добавляем tensor'ы с других устройств

        shape = tuple(tensor.shape)

        # Добавляем в соответствующий pool
        if shape not in self.tensor_pools:
            self.tensor_pools[shape] = []

        # Ограничиваем размер pool'а
        max_pool_size = 10  # Максимум 10 tensor'ов одного размера
        if len(self.tensor_pools[shape]) < max_pool_size:
            self.tensor_pools[shape].append(tensor.detach())

    def _track_allocation(self, tensor: torch.Tensor):
        """Отслеживает выделение памяти"""
        self.allocated_tensors.append(tensor)
        self.allocation_count += 1

        # Периодически запускаем garbage collection
        if self.allocation_count % self.config["garbage_collect_frequency"] == 0:
            self.garbage_collect()

        # Обновляем пиковое использование памяти
        if self.device.type == "cuda":
            current_memory_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            self.stats["memory_peak_mb"] = max(
                self.stats["memory_peak_mb"], current_memory_mb
            )

    def garbage_collect(self):
        """Принудительная очистка памяти через DeviceManager"""
        # Очищаем списки неиспользуемых tensor'ов
        self.allocated_tensors = [t for t in self.allocated_tensors if t.numel() > 0]

        # Очищаем pools от слишком старых tensor'ов
        for shape, pool in self.tensor_pools.items():
            if len(pool) > 5:  # Оставляем только 5 newest tensor'ов
                pool[:] = pool[-5:]

        # Используем DeviceManager для централизованной очистки памяти
        self.device_manager.cleanup()

        self.stats["gc_calls"] += 1
        logger.debug(
            f"   [CLEAN] Memory cleanup через DeviceManager: GC вызван #{self.stats['gc_calls']}"
        )

    def get_memory_stats(self) -> Dict[str, float]:
        """Получить статистику использования памяти"""
        stats = self.stats.copy()

        # Добавляем текущее использование памяти
        if self.device.type == "cuda":
            stats["current_memory_mb"] = torch.cuda.memory_allocated(self.device) / (
                1024**2
            )
            stats["reserved_memory_mb"] = torch.cuda.memory_reserved(self.device) / (
                1024**2
            )
        else:
            stats["current_memory_mb"] = 0.0
            stats["reserved_memory_mb"] = 0.0

        # Вычисляем эффективность pool'а
        total_requests = stats["pool_hits"] + stats["pool_misses"]
        if total_requests > 0:
            stats["pool_hit_rate"] = stats["pool_hits"] / total_requests
        else:
            stats["pool_hit_rate"] = 0.0

        # Информация о pool'ах
        stats["active_pools"] = len(self.tensor_pools)
        stats["pooled_tensors"] = sum(len(pool) for pool in self.tensor_pools.values())

        return stats

    def cleanup(self):
        """Полная очистка memory manager'а"""
        # Очищаем все pools
        for pool in self.tensor_pools.values():
            pool.clear()
        self.tensor_pools.clear()

        # Очищаем списки
        self.allocated_tensors.clear()

        # Финальная очистка памяти
        self.garbage_collect()

        logger.info("[CLEAN] MemoryPoolManager полностью очищен")


_memory_pool_manager_instance = None


def get_memory_pool_manager(config: dict = None) -> MemoryPoolManager:
    """
    Возвращает синглтон-экземпляр MemoryPoolManager.
    """
    global _memory_pool_manager_instance
    if _memory_pool_manager_instance is None:
        _memory_pool_manager_instance = MemoryPoolManager(config)
    return _memory_pool_manager_instance
