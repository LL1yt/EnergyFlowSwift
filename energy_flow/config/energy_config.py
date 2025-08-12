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
    
    # Параметры потоков
    max_active_flows: int = 100000  # Максимальное количество активных потоков для GPU
    max_spawn_per_step: int = 3     # Ограниченное количество spawn'ов для стабильности
    
    # Параметры моделей
    # GRU (EnergyCarrier)
    carrier_hidden_size: int = 1024
    carrier_num_layers: int = 3
    
    # SimpleNeuron
    neuron_hidden_dim: int = 32
    neuron_output_dim: int = 64  # Должен совпадать с входом GRU
    
    # Фильтрация потоков с маленькими смещениями (переосмысленный carrier_dropout)
    min_displacement_threshold: float = 0.5  # Минимальная длина смещения для сохранения потока
    enable_displacement_filtering: bool = False  # Отключить фильтрацию "топчущихся" потоков для полной проекционной архитектуры
    
    # Размерности эмбеддингов
    input_embedding_dim_from_teacher: int = 768  # Стандартный размер от language models
    output_embedding_dim_to_teacher: int = input_embedding_dim_from_teacher # Размер выходного эмбеддинга
    # embedding_per_cell вычисляется автоматически в embedding_mapper на основе размеров решетки
    
    # Обучение
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1  # Для RTX 5090 оптимизации
    max_steps_z: int = 1000
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # Mixed Precision Training (1.5x speedup, 50% memory saving)
    use_mixed_precision: bool = True                    # Включить mixed precision training
    mixed_precision_dtype: torch.dtype = torch.bfloat16  # bfloat16 для RTX 5090
    use_gradient_scaling: bool = True                   # Gradient scaling для стабильности
    gradient_scale_init: float = 2**16                  # Начальное значение gradient scaler
    
    # Text Bridge параметры (двунаправленное преобразование текст↔куб)
    text_bridge_enabled: bool = True           # Включить text bridge модуль
    text_cache_enabled: bool = False           # Включить LRU кэширование
    text_cache_size: int = 10000              # Размер LRU кэша
    text_cache_file: Optional[str] = None     # Файл для персистентного кэша (None = auto)
    text_loss_weight: float = 0.1             # Вес text loss в общем loss (0.0-1.0)
    iterative_correction_steps: int = 3       # Шаги итеративной коррекции для decoder
    text_generation_max_length: int = 64      # Максимальная длина генерируемого текста
    text_generation_num_beams: int = 4        # Количество beams для beam search
    text_generation_temperature: float = 1.0  # Температура для генерации текста
    
    # Adaptive max_steps (convergence detection)
    convergence_enabled: bool = True         # Включить адаптивное завершение
    convergence_threshold: float = 0.95      # Порог конвергенции (доля достигнутых выходов)
    convergence_min_steps: int = 5           # Минимальное количество шагов
    convergence_patience: int = 10           # Увеличенная терпеливость для полной проекционной архитектуры
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100
    
    # Эксплорация и шум (для относительных координат)
    exploration_noise: float = 0.1  # Небольшой шум для разнообразия движений
    use_exploration_noise: bool = True  # Включать exploration noise
    exploration_noise_apply_to_z: bool = True  # Применять ли шум по оси Z (по умолчанию нет)
    
    # Система масштабирования смещений (displacement scaling)
    displacement_scale: float = 1.0        # Временное масштабирование смещений для обучения
    displacement_warmup_steps: int = 100   # Количество шагов разогрева с полным масштабом
    displacement_scale_decay: float = 0.90 # Коэффициент убывания scale
    displacement_scale_min: float = 1.0    # Минимальный scale (натуральные смещения модели)
    displacement_scale_update_interval: int = 10  # Интервал обновления scale (в шагах)

    # Новая архитектура относительных координат
    relative_coordinates: bool = False  # Включить относительные координаты вместо абсолютных
    center_start_enabled: bool = False  # Стартовые позиции в центре куба (Z = depth/2)
    dual_output_planes: bool = False   # Две выходные плоскости (Z=0 и Z=depth)
    
    # Система spawn на основе длины смещения
    spawn_movement_threshold_ratio: float = 0.1  # depth/10 для threshold
    movement_based_spawn: bool = False  # Spawn на основе длины движения
    
    # Отражение границ X/Y
    boundary_reflection_enabled: bool = False  # Отражение от границ вместо завершения
    
    # Система важности выходных эмбеддингов (для dual output planes)
    proximity_weight: float = 0.7      # Вес близости к выходу
    path_length_weight: float = 0.3    # Вес длины пути
    safe_distance_minimum: float = 0.5 # Минимальное расстояние для безопасного деления
    
    # Тензорное хранилище потоков
    tensorized_storage_enabled: bool = True  # Включить TensorizedFlowStorage для активных потоков

    # Сбор/агрегация (оптимизация)
    collection_use_mixed_precision: bool = True
    collection_dtype: torch.dtype = torch.bfloat16
    cache_surface_indices_enabled: bool = True  # Кэшировать quantized surface_idx в TensorizedFlowStorage
    
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
        
        # Проверка параметров потоков
        assert self.max_active_flows > 0, "max_active_flows должен быть > 0"
        assert self.max_spawn_per_step > 0, "max_spawn_per_step должен быть > 0"
        
        # Проверка моделей
        assert self.carrier_hidden_size > 0, "carrier_hidden_size должен быть > 0"
        assert self.carrier_num_layers > 0, "carrier_num_layers должен быть > 0"
        assert self.neuron_hidden_dim > 0, "neuron_hidden_dim должен быть > 0"
        assert self.neuron_output_dim > 0, "neuron_output_dim должен быть > 0"
        
        # Проверка обучения
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps должен быть > 0"
        
        # Проверка Text Bridge параметров
        if self.text_bridge_enabled:
            assert 0.0 <= self.text_loss_weight <= 1.0, "text_loss_weight должен быть в [0.0, 1.0]"
            assert self.text_cache_size > 0, "text_cache_size должен быть > 0"
            assert self.iterative_correction_steps > 0, "iterative_correction_steps должен быть > 0"
            assert self.text_generation_max_length > 0, "text_generation_max_length должен быть > 0"
            assert self.text_generation_num_beams > 0, "text_generation_num_beams должен быть > 0"
            assert self.text_generation_temperature > 0, "text_generation_temperature должен быть > 0"
        
        # Проверка convergence параметров
        if self.convergence_enabled:
            assert 0.0 < self.convergence_threshold <= 1.0, "convergence_threshold должен быть в (0.0, 1.0]"
            assert self.convergence_min_steps > 0, "convergence_min_steps должен быть > 0"
            assert self.convergence_patience > 0, "convergence_patience должен быть > 0"
        
        # Создаем NormalizationManager
        self._normalization_manager = None  # Lazy initialization
    
    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.lattice_width * self.lattice_height * self.lattice_depth
    
    @property 
    def surface_dimension(self) -> int:
        """Размерность поверхности куба (для text_bridge)"""
        return self.lattice_width * self.lattice_height
    
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
            # Все нормализации теперь используют новую архитектуру относительных координат
        return self._normalization_manager


# Предустановленные конфигурации для разных режимов

def create_debug_config() -> EnergyConfig:
    """Минимальная конфигурация для отладки с включенным text_bridge"""
    return EnergyConfig(
        lattice_width=20,
        lattice_height=20,
        lattice_depth=10,
        max_active_flows=50000,
        max_spawn_per_step=1,   # Ограниченный spawn для отладки
        batch_size=8,
        carrier_hidden_size=256,  # Уменьшенный размер для отладки
        carrier_num_layers=2,
        log_interval=1,
        gradient_accumulation_steps=1,  # Без накопления для debug
        
        # Text Bridge включен для debug
        text_bridge_enabled=True,
        text_cache_enabled=False,  # Отключен для обучения
        text_cache_size=1000,          # Меньший кэш для debug
        text_loss_weight=0.2,          # Повышенный вес для обучения text bridge
        iterative_correction_steps=2,  # Меньше шагов для быстроты
        text_generation_max_length=32, # Короткие тексты для debug
        text_generation_num_beams=2,   # Меньше beams для скорости  
        text_generation_temperature=0.8,
        
        # Adaptive convergence для debug
        convergence_enabled=True,
        convergence_threshold=0.8,
        convergence_min_steps=3,
        convergence_patience=5,  # Увеличено для debug режима
        
        
        # НОВАЯ АРХИТЕКТУРА: Относительные координаты (включено для debug)
        relative_coordinates=True,      # Включить относительные координаты
        center_start_enabled=True,      # Старт из центра куба
        dual_output_planes=True,        # Две выходные плоскости
        movement_based_spawn=True,      # Spawn на основе длины движения
        boundary_reflection_enabled=True, # Отражение границ
        spawn_movement_threshold_ratio=0.15,  # 15% от depth для debug
        exploration_noise=0.05  # Маленький шум для debug
    )


def create_experiment_config() -> EnergyConfig:
    """RTX 5090 оптимизированная конфигурация для экспериментов"""
    return EnergyConfig(
        lattice_width=28,        # Оптимальный размер surface для 768D embeddings
        lattice_height=28,       # 50x50 = 2500 > 768, достаточное покрытие
        lattice_depth=40,        # Увеличено с 20 до 40 для более глубокой обработки
        batch_size=16,           # Увеличено с 16 до 32 для лучшей утилизации RTX 5090
        max_active_flows=200000, # Увеличено для поддержки больших batch_size
        carrier_hidden_size=512,
        carrier_num_layers=3,
        max_spawn_per_step=1,    # Контролируемый spawn
        
        # RTX 5090 память оптимизация
        gradient_accumulation_steps=4,  # Эффективный batch_size = 64*4 = 256
        
        # Text Bridge настройки для экспериментов
        text_bridge_enabled=True,
        text_cache_enabled=False,
        text_cache_size=5000,
        text_loss_weight=0.15,
        iterative_correction_steps=3,
        text_generation_max_length=48,
        text_generation_num_beams=3,
        text_generation_temperature=0.9,
        
        
        # НОВАЯ АРХИТЕКТУРА: Относительные координаты (включено для experiment)
        relative_coordinates=True,      # Включить относительные координаты
        center_start_enabled=True,      # Старт из центра куба
        dual_output_planes=True,        # Две выходные плоскости
        movement_based_spawn=True,      # Spawn на основе длины движения
        boundary_reflection_enabled=True, # Отражение границ для экспериментов
        spawn_movement_threshold_ratio=0.5,  # 50% от depth для experiment
        exploration_noise=0.05,  # Умеренный шум для экспериментов
        # Проекционная архитектура настройки
        enable_displacement_filtering=False,  # Отключить фильтрацию для полной проекции
        convergence_patience=8  # Увеличенная терпеливость для experiment
    )


def create_optimized_config() -> EnergyConfig:
    """Полная конфигурация для RTX 5090"""
    return EnergyConfig(
        lattice_width=100,
        lattice_height=100,
        lattice_depth=50,
        max_active_flows=200000,
        batch_size=32,
        carrier_hidden_size=1024,
        carrier_num_layers=3,
        max_spawn_per_step=3,    # Больше spawn для production
        
        # Text Bridge для производительной конфигурации
        text_bridge_enabled=True,
        text_cache_enabled=True,
        text_cache_size=10000,         # Максимальный кэш
        text_loss_weight=0.1,          # Базовый вес
        iterative_correction_steps=3,  # Полные шаги коррекции
        text_generation_max_length=64, # Полная длина текста
        text_generation_num_beams=4,   # Максимальное качество
        text_generation_temperature=1.0,
        
        
        # НОВАЯ АРХИТЕКТУРА: Относительные координаты (включено для optimized)
        relative_coordinates=True,     # Включить для оптимизированной конфигурации
        center_start_enabled=True,     # Включить для optimized
        dual_output_planes=True,       # Включить для optimized
        movement_based_spawn=True,     # Включить для optimized
        boundary_reflection_enabled=True, # Включить для optimized
        spawn_movement_threshold_ratio=0.1,
        
        # Проекционная архитектура настройки
        enable_displacement_filtering=False,  # Отключить фильтрацию для полной проекции
        convergence_patience=12  # Максимальная терпеливость для optimized
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