#!/usr/bin/env python3
"""
Централизованное управление устройствами и памятью
===============================================

DeviceManager обеспечивает:
1. Консистентное использование GPU/CPU во всех модулях
2. Автоматическое определение оптимального устройства
3. Управление памятью и предотвращение memory leaks
4. Централизованное логирование операций с устройствами
"""

import torch
import gc
from .logging import get_logger
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import psutil

logger = get_logger(__name__)


class MemoryMonitor:
    """Монитор памяти для отслеживания использования GPU/CPU памяти"""

    def __init__(self, device: torch.device):
        self.device = device
        self.allocation_count = 0
        self.peak_memory_mb = 0.0
        self._cleanup_threshold = None  # Ленивая инициализация

    @property 
    def cleanup_threshold(self) -> int:
        """Получает cleanup_threshold из централизованной конфигурации (ленивая инициализация)"""
        if self._cleanup_threshold is None:
            from ..config import get_project_config
            config = get_project_config()
            self._cleanup_threshold = config.memory_management.cleanup_threshold
        return self._cleanup_threshold

    def can_allocate(self, required_memory_bytes: int) -> bool:
        """
        Проверяет возможность выделения памяти

        Args:
            required_memory_bytes: Требуемый объем памяти в байтах

        Returns:
            True если память можно выделить
        """
        if self.device.type == "cuda":
            available_memory = torch.cuda.get_device_properties(
                self.device
            ).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            free_memory = available_memory - allocated_memory

            # Оставляем 15% буфер
            safe_memory = free_memory * 0.85
            return required_memory_bytes < safe_memory
        else:
            # CPU memory check
            available_memory = psutil.virtual_memory().available
            safe_memory = available_memory * 0.85
            return required_memory_bytes < safe_memory

    def cleanup(self):
        """Принудительная очистка памяти"""
        gc.collect()

        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                # Убираем автоматическую синхронизацию для лучшей производительности
                # torch.cuda.synchronize()  # Отключено для GPU оптимизации
            except Exception as e:
                # Игнорируем ошибки очистки при завершении программы
                logger.debug_memory(
                    f"⚠️ Игнорируем ошибки очистки при завершении программы {e}"
                )   
                pass

        logger.info(f"🧹 Memory cleanup выполнен для {self.device}")

    def get_memory_stats(self) -> Dict[str, float]:
        """Получить статистику использования памяти"""
        stats = {
            "device": str(self.device),
            "allocation_count": self.allocation_count,
            "peak_memory_mb": self.peak_memory_mb,
        }

        if self.device.type == "cuda":
            stats.update(
                {
                    "allocated_mb": torch.cuda.memory_allocated(self.device)
                    / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated(self.device)
                    / (1024**2),
                }
            )
        else:
            memory_info = psutil.virtual_memory()
            stats.update(
                {
                    "total_mb": memory_info.total / (1024**2),
                    "available_mb": memory_info.available / (1024**2),
                    "used_mb": memory_info.used / (1024**2),
                    "percent": memory_info.percent,
                }
            )

        return stats


class DeviceManager:
    """Централизованное управление устройствами и памятью"""

    def __init__(self, prefer_cuda: bool = True, debug_mode: bool = True, strict_gpu: bool = False):
        """
        Инициализация DeviceManager

        Args:
            prefer_cuda: Предпочитать CUDA если доступен
            debug_mode: Включить подробное логирование (передается из config.logging.debug_mode)
            strict_gpu: Строгий GPU-only режим (без CPU fallback)
        """
        self.prefer_cuda = prefer_cuda
        self.debug_mode = debug_mode
        self.strict_gpu = strict_gpu
        self.device = self._detect_optimal_device(prefer_cuda, strict_gpu)
        self.memory_monitor = MemoryMonitor(self.device)
        self._cleanup_on_del = False  # Отключаем cleanup в деструкторе для производительности

        # Счетчики для статистики
        self.tensor_transfers = 0
        self.allocations = 0

        if debug_mode:
            self._log_device_info()

    def _detect_optimal_device(self, prefer_cuda: bool, strict_gpu: bool = False) -> torch.device:
        """
        Определяет оптимальное устройство
        
        СТРОГАЯ ПОЛИТИКА: Без fallback на CPU при недостатке памяти GPU

        Args:
            prefer_cuda: Предпочитать CUDA
            strict_gpu: Строгий GPU-only режим (без CPU fallback)

        Returns:
            Оптимальное torch.device
            
        Raises:
            RuntimeError: При недостатке памяти GPU или других проблемах
        """
        if strict_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "❌ STRICT GPU MODE: CUDA required but not available. "
                "Install CUDA or disable strict_gpu mode."
            )
            
        if prefer_cuda or strict_gpu:
            if not torch.cuda.is_available():
                if strict_gpu:
                    raise RuntimeError(
                        "❌ STRICT GPU MODE: CUDA required but not available."
                    )
                else:
                    raise RuntimeError(
                        "❌ КРИТИЧЕСКАЯ ОШИБКА: Запрошено использование CUDA, но CUDA недоступен. "
                        "Установите CUDA или явно укажите prefer_cuda=False в конфигурации"
                    )
            
            # Проверяем количество устройств
            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: CUDA доступен, но GPU устройств не найдено. "
                    "Проверьте драйверы и подключение GPU"
                )

            # Проверяем доступную память GPU
            device_properties = torch.cuda.get_device_properties(0)
            gpu_memory_gb = device_properties.total_memory / (1024**3)

            if gpu_memory_gb < 8.0:  # Минимум 8GB для эффективной работы
                raise RuntimeError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА: Недостаточно памяти GPU "
                    f"({gpu_memory_gb:.1f}GB < 8GB минимум). "
                    f"Используйте GPU с большим объемом памяти или явно укажите prefer_cuda=False"
                )
            
            device = torch.device("cuda:0")
            if self.debug_mode:
                logger.info(
                    f"✅ CUDA device selected: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f}GB)"
                )
            return device
        else:
            # CPU явно запрошен - это нормально
            if self.debug_mode:
                logger.info("💻 CPU выбран по конфигурации")
            return torch.device("cpu")

    def _log_device_info(self):
        """Логирование информации об устройстве"""
        logger.info(f"🖥️ DeviceManager initialized:")
        logger.info(f"   Device: {self.device}")

        if self.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.device)
                logger.info(f"   GPU: {torch.cuda.get_device_name(self.device)}")
                logger.info(f"   Memory: {props.total_memory / (1024**3):.1f}GB")
                logger.info(f"   Compute Capability: {props.major}.{props.minor}")
            except (RuntimeError, AssertionError) as e:
                logger.warning(f"   ⚠️ Error getting GPU info: {str(e)}")
        else:
            try:
                memory_info = psutil.virtual_memory()
                logger.info(
                    f"   CPU Memory: {memory_info.total / (1024**3):.1f}GB total"
                )
                logger.info(f"   Available: {memory_info.available / (1024**3):.1f}GB")
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка получения информации CPU: {str(e)}")

    def get_available_memory_gb(self) -> float:
        """
        Возвращает доступную память в ГБ для текущего устройства.

        Returns:
            Доступная память в ГБ.
        """
        if self.device.type == "cuda":
            try:
                total_memory = torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                reserved_memory = torch.cuda.memory_reserved(self.device)
                available_memory_bytes = total_memory - reserved_memory
                return available_memory_bytes / (1024**3)
            except (RuntimeError, AssertionError) as e:
                logger.warning(f"⚠️ Не удалось получить память GPU: {e}")
                return 0.0
        else:
            # Для CPU используем psutil
            try:
                return psutil.virtual_memory().available / (1024**3)
            except Exception as e:
                logger.warning(f"⚠️ Не удалось получить память CPU: {e}")
                return 0.0

    def ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Гарантированно переносит tensor на правильное устройство

        Args:
            tensor: Исходный tensor

        Returns:
            Tensor на правильном устройстве
        """
        if tensor.device != self.device:
            self.tensor_transfers += 1
            if self.debug_mode and self.tensor_transfers <= 5:
                logger.debug_memory(
                    f"🔄 Перенос tensor {tensor.shape} с {tensor.device} на {self.device}"
                )

            return tensor.to(self.device, non_blocking=True)

        return tensor

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Безопасное выделение tensor'а с проверкой памяти

        Args:
            shape: Форма tensor'а
            dtype: Тип данных
            requires_grad: Требуется ли градиент

        Returns:
            Новый tensor на правильном устройстве
        """
        # Рассчитываем требуемую память
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_memory = torch.tensor(shape).prod().item() * element_size

        # Проверяем доступность памяти
        if not self.memory_monitor.can_allocate(required_memory):
            logger.warning(
                f"⚠️ Недостаточно памяти для tensor {shape}, выполняем cleanup"
            )
            self.memory_monitor.cleanup()

            # Повторная проверка после cleanup
            if not self.memory_monitor.can_allocate(required_memory):
                raise RuntimeError(
                    f"Невозможно выделить {required_memory / (1024**2):.1f}MB памяти для tensor {shape}"
                )

        # Создаем tensor
        tensor = torch.zeros(
            shape, device=self.device, dtype=dtype, requires_grad=requires_grad
        )

        self.allocations += 1
        self.memory_monitor.allocation_count += 1

        # Периодический cleanup (но не при первой аллокации)
        if (
            self.memory_monitor.allocation_count > 0 and
            self.memory_monitor.allocation_count % self.memory_monitor.cleanup_threshold == 0
        ):
            logger.info(f"� TRIGGER: Периодический cleanup сработал! allocation_count: {self.memory_monitor.allocation_count}, cleanup_threshold: {self.memory_monitor.cleanup_threshold}")
            logger.info(f"📊 Статистика тензоров: {self.tensor_transfers} переносов, {self.allocations} выделений")
            self.memory_monitor.cleanup()

        return tensor

    def allocate_like(self, reference_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Создает tensor похожий на reference_tensor но на правильном устройстве

        Args:
            reference_tensor: Референсный tensor
            **kwargs: Дополнительные параметры (dtype, requires_grad, etc.)

        Returns:
            Новый tensor на правильном устройстве
        """
        dtype = kwargs.get("dtype", reference_tensor.dtype)
        requires_grad = kwargs.get("requires_grad", reference_tensor.requires_grad)

        return self.allocate_tensor(
            shape=reference_tensor.shape, dtype=dtype, requires_grad=requires_grad
        )

    def transfer_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Переносит модель на правильное устройство

        Args:
            module: PyTorch модель

        Returns:
            Модель на правильном устройстве
        """
        if self.debug_mode:
            param_count = sum(p.numel() for p in module.parameters())
            logger.info(
                f"🔄 Перенос модели на {self.device} ({param_count:,} параметров)"
            )

        return module.to(self.device)

    def get_device(self) -> torch.device:
        """Получить текущее устройство"""
        return self.device

    def get_device_str(self) -> str:
        """Получить строковое представление устройства"""
        return str(self.device)

    def is_cuda(self) -> bool:
        """Проверяет, используется ли CUDA."""
        return self.device.type == "cuda"
    
    def enable_strict_gpu_mode(self):
        """Активирует строгий GPU-only режим"""
        if not self.is_cuda():
            raise RuntimeError("Cannot enable strict GPU mode: CUDA not available")
        self.strict_gpu = True
        logger.info("🔒 Strict GPU-only mode enabled - no CPU fallbacks allowed")
    
    def disable_strict_gpu_mode(self):
        """Деактивирует строгий GPU-only режим"""
        self.strict_gpu = False
        logger.info("🔓 Strict GPU-only mode disabled - CPU fallbacks allowed")

    def synchronize(self, force=False):
        """Синхронизация устройства (только когда необходимо)"""
        if force and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Получить полную статистику памяти и устройства"""
        stats = self.memory_monitor.get_memory_stats()
        stats.update(
            {
                "tensor_transfers": self.tensor_transfers,
                "total_allocations": self.allocations,
                "device_type": self.device.type,
                "prefer_cuda": self.prefer_cuda,
            }
        )

        return stats

    def cleanup(self):
        """Полная очистка памяти"""
        try:
            # Добавляем стек вызовов для отладки
            import traceback
            stack = traceback.format_stack()
            caller_info = "".join(stack[-3:-1])  # Берем 2 уровня вверх
            
            if hasattr(self, "memory_monitor") and self.memory_monitor:
                self.memory_monitor.cleanup()

            if hasattr(self, "debug_mode") and self.debug_mode:
                stats = self.get_memory_stats()
                logger.info(
                    f"🧹 DeviceManager cleanup: {stats['tensor_transfers']} переносов, {stats['total_allocations']} выделений"
                )
                logger.info(f"📍 Вызван из:\n{caller_info}")
        except Exception as e:
            # Игнорируем ошибки очистки при завершении программы
            logger.debug_memory(
                f"⚠️ Игнорируем ошибки очистки при завершении программы {e}. а если ошибка не связана с заверешением программы, то это баг, который нужно исправить, но мы игнорируем его просто так- добро пожаловать в мир багов и проблем с памятью!  и вайбкодинга  😅"
            )  
            pass

    def __del__(self):
        """Cleanup при удалении объекта - контролируется флагом"""
        if self._cleanup_on_del and hasattr(self, "memory_monitor"):
            logger.info(
                    f"🧹 DeviceManager cleanup: _cleanup_on_del"
                )
            self.cleanup()


# Глобальный экземпляр DeviceManager (singleton pattern)
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    prefer_cuda: bool = True, debug_mode: bool = True
) -> DeviceManager:
    """
    Получает глобальный instance DeviceManager (singleton pattern)

    Args:
        prefer_cuda: Предпочитать CUDA если доступен
        debug_mode: Режим отладки (передается из config.logging.debug_mode)

    Returns:
        DeviceManager instance
    """
    global _global_device_manager

    if _global_device_manager is None:
        _global_device_manager = DeviceManager(
            prefer_cuda=prefer_cuda, debug_mode=debug_mode
        )

    return _global_device_manager


def reset_device_manager():
    """Сброс глобального DeviceManager (для тестов)"""
    global _global_device_manager

    if _global_device_manager is not None:
        _global_device_manager.cleanup()
        _global_device_manager = None


# Удобные функции для быстрого доступа
def ensure_device(tensor: torch.Tensor) -> torch.Tensor:
    """Быстрый доступ к ensure_device"""
    return get_device_manager().ensure_device(tensor)


def allocate_tensor(shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
    """Быстрый доступ к allocate_tensor"""
    return get_device_manager().allocate_tensor(shape, **kwargs)


def transfer_module(module: torch.nn.Module) -> torch.nn.Module:
    """Быстрый доступ к transfer_module"""
    return get_device_manager().transfer_module(module)


def get_optimal_device() -> torch.device:
    """Быстрый доступ к устройству"""
    return get_device_manager().get_device()


def cleanup_memory():
    """Быстрый доступ к cleanup"""
    get_device_manager().cleanup()
