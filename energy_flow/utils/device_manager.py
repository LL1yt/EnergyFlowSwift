"""
Device Manager для энергетической архитектуры
=============================================

Адаптированная версия DeviceManager из new_rebuild.
Оптимизирован для работы с RTX 5090 32GB.
"""

import torch
import gc
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
import psutil
import GPUtil

from .logging import get_logger, log_memory_state

logger = get_logger(__name__)


class DeviceManager:
    """Централизованное управление GPU/CPU устройствами"""
    
    def __init__(self, device: Optional[str] = None, memory_fraction: float = 0.9):
        """
        Args:
            device: Устройство ('cuda', 'cpu' или None для автоопределения)
            memory_fraction: Доля GPU памяти для использования (0.9 = 90%)
        """
        self.device = self._setup_device(device)
        self.memory_fraction = memory_fraction
        self.is_cuda = self.device.type == 'cuda'
        
        if self.is_cuda:
            # Устанавливаем memory fraction
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Информация о GPU
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                logger.info(f"GPU: {gpu.name}, Memory: {gpu.memoryTotal}MB")
                logger.info(f"Using {memory_fraction*100}% of GPU memory")
        
        logger.info(f"DeviceManager initialized with device: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Настройка устройства"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return torch.device(device)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Получить информацию о памяти"""
        info = {}
        
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            info['gpu_allocated_gb'] = allocated
            info['gpu_reserved_gb'] = reserved
            info['gpu_total_gb'] = total
            info['gpu_free_gb'] = total - reserved
            info['gpu_usage_percent'] = (allocated / total) * 100
        
        # CPU memory
        cpu_info = psutil.virtual_memory()
        info['cpu_used_gb'] = cpu_info.used / 1024**3
        info['cpu_total_gb'] = cpu_info.total / 1024**3
        info['cpu_percent'] = cpu_info.percent
        
        return info
    
    def check_memory_availability(self, required_gb: float) -> bool:
        """Проверить доступность памяти"""
        if not self.is_cuda:
            return True  # Для CPU не проверяем
        
        info = self.get_memory_info()
        available = info['gpu_free_gb']
        
        if available < required_gb:
            logger.warning(
                f"Insufficient GPU memory: {available:.2f}GB available, "
                f"{required_gb:.2f}GB required"
            )
            return False
        
        return True
    
    def clear_cache(self):
        """Очистить кэш памяти"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.debug("Memory cache cleared")
    
    @contextmanager
    def memory_efficient_mode(self, threshold_gb: float = 28.0):
        """
        Контекст для работы в режиме экономии памяти
        
        Args:
            threshold_gb: Порог свободной памяти для активации режима
        """
        info = self.get_memory_info()
        efficient_mode = self.is_cuda and info.get('gpu_free_gb', 0) < threshold_gb
        
        if efficient_mode:
            logger.info(f"Entering memory efficient mode (free: {info['gpu_free_gb']:.2f}GB)")
            # Включаем gradient checkpointing и другие оптимизации
            torch.cuda.empty_cache()
            original_grad_enabled = torch.is_grad_enabled()
            
        try:
            yield efficient_mode
        finally:
            if efficient_mode:
                torch.cuda.empty_cache()
                if original_grad_enabled:
                    torch.set_grad_enabled(True)
    
    def allocate_tensor(self, shape: Tuple[int, ...], 
                       dtype: torch.dtype = torch.float32,
                       requires_grad: bool = False) -> torch.Tensor:
        """
        Безопасное выделение тензора с проверкой памяти
        
        Args:
            shape: Размерность тензора
            dtype: Тип данных
            requires_grad: Требуется ли градиент
            
        Returns:
            Allocated tensor
        """
        # Вычисляем требуемую память
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        bytes_per_element = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else \
                           torch.iinfo(dtype).bits // 8
        required_gb = (num_elements * bytes_per_element) / 1024**3
        
        # Проверяем доступность
        if not self.check_memory_availability(required_gb * 1.2):  # 20% запас
            self.clear_cache()
            if not self.check_memory_availability(required_gb):
                raise RuntimeError(f"Cannot allocate tensor of size {shape}, "
                                 f"requires {required_gb:.2f}GB")
        
        # Выделяем тензор
        tensor = torch.zeros(shape, dtype=dtype, device=self.device, 
                           requires_grad=requires_grad)
        
        logger.debug(f"Allocated tensor {shape} ({required_gb:.3f}GB)")
        return tensor
    
    def optimize_for_inference(self):
        """Оптимизация для инференса"""
        if self.is_cuda:
            # Отключаем cudnn benchmarking для стабильной памяти
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        # Отключаем градиенты
        torch.set_grad_enabled(False)
        
        # Очищаем кэш
        self.clear_cache()
        
        logger.info("Optimized for inference mode")
    
    def optimize_for_training(self):
        """Оптимизация для обучения"""
        if self.is_cuda:
            # Включаем cudnn benchmarking для скорости
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Устанавливаем mixed precision, если поддерживается
            if torch.cuda.get_device_capability()[0] >= 7:  # Compute capability 7.0+
                torch.set_float32_matmul_precision('high')
                logger.info("Enabled TensorFloat-32 for matmul operations")
        
        # Включаем градиенты
        torch.set_grad_enabled(True)
        
        logger.info("Optimized for training mode")
    
    def get_optimal_batch_size(self, model_size_gb: float, 
                             sample_size_mb: float) -> int:
        """
        Вычислить оптимальный размер батча
        
        Args:
            model_size_gb: Размер модели в GB
            sample_size_mb: Размер одного сэмпла в MB
            
        Returns:
            Рекомендуемый размер батча
        """
        if not self.is_cuda:
            return 32  # Дефолтный для CPU
        
        info = self.get_memory_info()
        available_gb = info['gpu_free_gb'] - 2.0  # Оставляем 2GB запас
        
        # Учитываем память для градиентов (примерно x2 от модели)
        available_for_data = available_gb - (model_size_gb * 2)
        
        if available_for_data <= 0:
            logger.warning("Insufficient memory for batching")
            return 1
        
        batch_size = int(available_for_data * 1024 / sample_size_mb)
        batch_size = max(1, min(batch_size, 128))  # Ограничиваем разумными пределами
        
        logger.info(f"Recommended batch size: {batch_size} "
                   f"(available: {available_gb:.1f}GB, model: {model_size_gb:.1f}GB)")
        
        return batch_size


# Глобальный device manager (создается при первом использовании)
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Получить глобальный DeviceManager"""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_device_manager(device_manager: DeviceManager):
    """Установить глобальный DeviceManager"""
    global _global_device_manager
    _global_device_manager = device_manager