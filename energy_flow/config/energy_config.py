"""
Конфигурация для энергетической архитектуры
===========================================

Централизованная конфигурация системы энергетических потоков.
Определяет размеры решетки, параметры моделей, пороги энергии и т.д.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

# Импорт нормализации (delayed для избежания циклических импортов)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.normalization import NormalizationManager

# Устанавливаем GPU как default device для всего проекта
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    print(f"🚀 Energy Flow: Default device set to CUDA ({torch.cuda.get_device_name()})")
else:
    print("⚠️ Energy Flow: CUDA not available, using CPU")


@dataclass
class EnergyConfig:
    """Основная конфигурация энергетической системы"""
    
    # Размеры решетки
    lattice_width: int
    lattice_height: int 
    lattice_depth: int
    
    # Параметры энергии (для скалярной энергии в диапазоне [-1, 1])
    max_active_flows: int = 1000
    energy_threshold: float = 0.05  # Минимальная энергия для продолжения (низкий для стабильности)
    spawn_threshold: float = 0.6    # Порог для создания новых потоков (умеренный)
    max_spawn_per_step: int = 3     # Ограниченное количество spawn'ов для стабильности
    
    # Параметры моделей
    # GRU (EnergyCarrier)
    carrier_hidden_size: int = 1024
    carrier_num_layers: int = 3
    carrier_dropout: float = 0.1
    
    # SimpleNeuron
    neuron_hidden_dim: int = 32
    neuron_output_dim: int = 64  # Должен совпадать с входом GRU
    
    # Размерности эмбеддингов
    input_embedding_dim_from_teacher: int = 768  # Стандартный размер от language models
    output_embedding_dim_to_teacher: int = input_embedding_dim_from_teacher # Размер выходного эмбеддинга
    # embedding_per_cell вычисляется автоматически в embedding_mapper на основе размеров решетки
    
    # Обучение
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100
    
    # Предварительное обучение (experimental features)
    initial_z_bias: float = 1.0  # Положительный bias для Z координаты (помощь необученной модели)
    use_forward_movement_bias: bool = True  # Включать ли bias для движения вперед
    progressive_z_multiplier: float = 2.1  # Множитель для динамического bias'а
    exploration_noise: float = 0.5  # Случайный шум для разнообразия путей
    use_exploration_noise: bool = True  # Включать ли exploration noise
    
    def __post_init__(self):
        """Валидация и вычисление производных параметров"""
        # Проверка размеров
        assert self.lattice_width > 0, "lattice_width должна быть > 0"
        assert self.lattice_height > 0, "lattice_height должна быть > 0"
        assert self.lattice_depth > 0, "lattice_depth должна быть > 0"
        
        # Вычисляем количество клеток на входной/выходной стороне
        self.input_cells = self.lattice_width * self.lattice_height
        self.output_cells = self.lattice_width * self.lattice_height
        
        # Размерности эмбеддингов определяются в embedding_mapper автоматически
        
        # Проверка параметров энергии
        assert 0 < self.energy_threshold < 1, "energy_threshold должен быть в (0, 1)"
        assert 0 < self.spawn_threshold <= 1, "spawn_threshold должен быть в (0, 1]"
        assert self.max_spawn_per_step > 0, "max_spawn_per_step должен быть > 0"
        
        # Проверка моделей
        assert self.carrier_hidden_size > 0, "carrier_hidden_size должен быть > 0"
        assert self.carrier_num_layers > 0, "carrier_num_layers должен быть > 0"
        assert self.neuron_hidden_dim > 0, "neuron_hidden_dim должен быть > 0"
        assert self.neuron_output_dim > 0, "neuron_output_dim должен быть > 0"
        
        # Создаем NormalizationManager
        self._normalization_manager = None  # Lazy initialization
    
    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.lattice_width * self.lattice_height * self.lattice_depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сохранения"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @property
    def normalization_manager(self) -> 'NormalizationManager':
        """Получение NormalizationManager (lazy initialization)"""
        if self._normalization_manager is None:
            from ..utils.normalization import create_normalization_manager
            self._normalization_manager = create_normalization_manager(
                self.lattice_width, self.lattice_height, self.lattice_depth
            )
        return self._normalization_manager


# Предустановленные конфигурации для разных режимов

def create_debug_config() -> EnergyConfig:
    """Минимальная конфигурация для отладки"""
    return EnergyConfig(
        lattice_width=20,
        lattice_height=20,
        lattice_depth=10,
        max_active_flows=1000,
        energy_threshold=0.01,  # Очень низкий для отладки (скалярная энергия)
        spawn_threshold=0.7,    # Немного выше базового для контроля spawn'ов
        max_spawn_per_step=2,   # Ограниченный spawn для отладки
        batch_size=8,
        carrier_hidden_size=256,  # Уменьшенный размер для отладки
        carrier_num_layers=2,
        carrier_dropout=0.05,   # Низкий dropout для отладки
        log_interval=1
    )


def create_experiment_config() -> EnergyConfig:
    """Сбалансированная конфигурация для экспериментов"""
    return EnergyConfig(
        lattice_width=50,
        lattice_height=50,
        lattice_depth=20,
        max_active_flows=500,
        batch_size=16,
        carrier_hidden_size=512,
        carrier_num_layers=2
    )


def create_optimized_config() -> EnergyConfig:
    """Полная конфигурация для RTX 5090"""
    return EnergyConfig(
        lattice_width=100,
        lattice_height=100,
        lattice_depth=50,
        max_active_flows=1000,
        batch_size=32,
        carrier_hidden_size=1024,
        carrier_num_layers=3
    )


# Глобальная конфигурация (опционально)
_global_config: Optional[EnergyConfig] = None


def set_energy_config(config: EnergyConfig):
    """Установить глобальную конфигурацию"""
    global _global_config
    _global_config = config


def get_energy_config() -> EnergyConfig:
    """Получить глобальную конфигурацию"""
    if _global_config is None:
        raise RuntimeError("Energy config not set. Call set_energy_config() first.")
    return _global_config