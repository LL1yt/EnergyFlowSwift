"""
[CONFIG] Universal Embedding Adapter
Универсальная система конвертации эмбедингов между любыми размерностями
Поддержка любых Teacher моделей и любых размеров куба
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalEmbeddingAdapter(nn.Module):
    """
    Универсальный адаптер для конвертации эмбедингов
    
    Поддерживает:
    - Любые input размеры (LLaMA, DistilBERT, GPT, etc.)
    - Любые output размеры (surface размеры куба)
    - Разные стратегии конвертации
    - Автоматическое определение размеров
    - Configuration-driven подход
    """
    
    def __init__(self, 
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 strategy: str = "learned_linear",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.strategy = strategy
        self.config = config or {}
        
        # Если размеры не заданы, создаем lazy инициализацию
        self.initialized = False
        
        if input_dim and output_dim:
            self._build_adapter()
            self.initialized = True
        
        logger.info(f"[CONFIG] UniversalEmbeddingAdapter создан: {input_dim}D → {output_dim}D, strategy={strategy}")
    
    def _build_adapter(self):
        """Построение адаптера после определения размеров"""
        if self.strategy == "learned_linear":
            self._build_learned_linear()
        elif self.strategy == "hierarchical":
            self._build_hierarchical()
        elif self.strategy == "attention_based":
            self._build_attention_based()
        elif self.strategy == "autoencoder":
            self._build_autoencoder()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _build_learned_linear(self):
        """Простая learned linear трансформация"""
        # Compression
        self.compressor = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU()
        )
        
        # Decompression (для reconstruction loss)
        self.decompressor = nn.Sequential(
            nn.Linear(self.output_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.GELU()
        )
        
        # Reconstruction loss
        self.reconstruction_loss = nn.MSELoss()
    
    def _build_hierarchical(self):
        """Hierarchical compression через intermediate layers"""
        # Определяем intermediate размеры
        intermediate_sizes = self._calculate_intermediate_sizes()
        
        # Encoder layers
        encoder_layers = []
        prev_size = self.input_dim
        
        for size in intermediate_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.LayerNorm(size),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        
        # Final compression
        encoder_layers.append(nn.Linear(prev_size, self.output_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse)
        decoder_layers = []
        prev_size = self.output_dim
        
        for size in reversed(intermediate_sizes):
            decoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.LayerNorm(size),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        
        # Final decompression
        decoder_layers.append(nn.Linear(prev_size, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.reconstruction_loss = nn.MSELoss()
    
    def _build_attention_based(self):
        """Attention-based selective compression"""
        # Если output меньше input, используем attention для выбора важных dims
        if self.output_dim < self.input_dim:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.input_dim,
                num_heads=min(8, self.input_dim // 64),
                batch_first=True
            )
            self.dimension_selector = nn.Linear(self.input_dim, self.output_dim)
        else:
            # Если output больше, используем expansion
            self.dimension_expander = nn.Linear(self.input_dim, self.output_dim)
        
        # Reconstruction path
        self.reconstructor = nn.Linear(self.output_dim, self.input_dim)
        self.reconstruction_loss = nn.MSELoss()
    
    def _build_autoencoder(self):
        """Autoencoder с bottleneck на output_dim"""
        bottleneck_dim = min(self.output_dim, self.input_dim // 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.LayerNorm(self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.output_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.output_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.input_dim // 2),
            nn.LayerNorm(self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, self.input_dim)
        )
        
        self.reconstruction_loss = nn.MSELoss()
    
    def _calculate_intermediate_sizes(self) -> List[int]:
        """Вычисление промежуточных размеров для hierarchical approach"""
        ratio = self.output_dim / self.input_dim
        
        if ratio > 0.5:
            # Небольшое сжатие - один intermediate layer
            return [int((self.input_dim + self.output_dim) / 2)]
        else:
            # Сильное сжатие - несколько layers
            sizes = []
            current = self.input_dim
            target = self.output_dim
            
            while current > target * 2:
                current = int(current * 0.7)  # Уменьшаем на 30% каждый шаг
                sizes.append(current)
            
            return sizes
    
    def initialize_from_data(self, sample_input: torch.Tensor, target_output_dim: int):
        """Автоматическая инициализация из примера данных"""
        if self.initialized:
            return
        
        if len(sample_input.shape) == 1:
            self.input_dim = sample_input.shape[0]
        else:
            self.input_dim = sample_input.shape[-1]
        
        self.output_dim = target_output_dim
        
        self._build_adapter()
        self.initialized = True
        
        logger.info(f"[CONFIG] Adapter инициализирован из данных: {self.input_dim}D → {self.output_dim}D")
    
    def forward(self, x: torch.Tensor, return_reconstruction: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass через адаптер
        
        Args:
            x: Input tensor [batch_size, input_dim] или [input_dim]
            return_reconstruction: Возвращать ли reconstruction для loss
            
        Returns:
            compressed: [batch_size, output_dim] или [output_dim]
            reconstruction: [batch_size, input_dim] (если запрошено)
        """
        if not self.initialized:
            raise RuntimeError("Adapter не инициализирован. Используйте initialize_from_data() или задайте размеры в конструкторе")
        
        # Обработка размерности и типов данных
        original_shape = x.shape
        
        # Приведение к float32 для совместимости (LLaMA может давать float16)
        if x.dtype == torch.float16:
            x = x.float()
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Добавляем batch dimension
            single_sample = True
        else:
            single_sample = False
        
        # Compression
        if self.strategy == "learned_linear":
            compressed = self.compressor(x)
            if return_reconstruction:
                reconstructed = self.decompressor(compressed)
        
        elif self.strategy == "hierarchical":
            compressed = self.encoder(x)
            if return_reconstruction:
                reconstructed = self.decoder(compressed)
        
        elif self.strategy == "attention_based":
            if self.output_dim < self.input_dim:
                # Attention-based selection
                attended, _ = self.attention(x, x, x)
                compressed = self.dimension_selector(attended)
            else:
                # Direct expansion
                compressed = self.dimension_expander(x)
            
            if return_reconstruction:
                reconstructed = self.reconstructor(compressed)
        
        elif self.strategy == "autoencoder":
            compressed = self.encoder(x)
            if return_reconstruction:
                reconstructed = self.decoder(compressed)
        
        # Возврат к original shape
        if single_sample:
            compressed = compressed.squeeze(0)
            if return_reconstruction:
                reconstructed = reconstructed.squeeze(0)
        
        if return_reconstruction:
            return compressed, reconstructed
        else:
            return compressed
    
    def compute_reconstruction_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Вычисление reconstruction loss"""
        return self.reconstruction_loss(original, reconstructed)
    
    def get_compression_ratio(self) -> float:
        """Получение коэффициента сжатия"""
        if not self.initialized:
            return 0.0
        return self.output_dim / self.input_dim
    
    def get_parameter_count(self) -> int:
        """Подсчет параметров адаптера"""
        return sum(p.numel() for p in self.parameters())
    
    def save_config(self, path: Union[str, Path]):
        """Сохранение конфигурации адаптера"""
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "strategy": self.strategy,
            "compression_ratio": self.get_compression_ratio(),
            "parameter_count": self.get_parameter_count(),
            "config": self.config
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"[SAVE] Adapter config сохранен: {path}")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'UniversalEmbeddingAdapter':
        """Загрузка адаптера из конфигурации"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            strategy=config["strategy"],
            config=config.get("config", {})
        )


class AdapterManager:
    """
    Менеджер для управления множественными адаптерами
    Автоматически создает нужные адаптеры для разных моделей/размеров
    """
    
    def __init__(self, config_dir: str = "config/adapters/"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.adapters: Dict[str, UniversalEmbeddingAdapter] = {}
        self.model_configs = {}
        
    def register_model(self, model_name: str, embedding_dim: int, **kwargs):
        """Регистрация новой модели"""
        self.model_configs[model_name] = {
            "embedding_dim": embedding_dim,
            "config": kwargs
        }
        logger.info(f"[WRITE] Модель зарегистрирована: {model_name} ({embedding_dim}D)")
    
    def get_adapter(self, 
                   source_model: str, 
                   target_surface_size: int,
                   strategy: str = "learned_linear") -> UniversalEmbeddingAdapter:
        """Получение/создание адаптера для конкретной пары модель→surface"""
        
        adapter_key = f"{source_model}→{target_surface_size}_{strategy}"
        
        if adapter_key in self.adapters:
            return self.adapters[adapter_key]
        
        # Создаем новый адаптер
        if source_model not in self.model_configs:
            raise ValueError(f"Модель {source_model} не зарегистрирована")
        
        source_dim = self.model_configs[source_model]["embedding_dim"]
        
        adapter = UniversalEmbeddingAdapter(
            input_dim=source_dim,
            output_dim=target_surface_size,
            strategy=strategy
        )
        
        self.adapters[adapter_key] = adapter
        
        # Сохраняем конфигурацию
        config_path = self.config_dir / f"{adapter_key}.yaml"
        adapter.save_config(config_path)
        
        logger.info(f"[CONFIG] Создан адаптер: {source_model} ({source_dim}D) → surface ({target_surface_size}D)")
        
        return adapter
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """Список всех доступных адаптеров"""
        result = {}
        for key, adapter in self.adapters.items():
            result[key] = {
                "input_dim": adapter.input_dim,
                "output_dim": adapter.output_dim,
                "strategy": adapter.strategy,
                "compression_ratio": adapter.get_compression_ratio(),
                "parameters": adapter.get_parameter_count()
            }
        return result


# Предустановленные конфигурации моделей
KNOWN_MODELS = {
    "Meta-Llama-3-8B": {"embedding_dim": 4096},  # LLaMA 3 8B
    "Meta-Llama-3-70B": {"embedding_dim": 8192}, # LLaMA 3 70B  
    "DistilBERT": {"embedding_dim": 768},         # DistilBERT
    "BERT-base": {"embedding_dim": 768},          # BERT base
    "BERT-large": {"embedding_dim": 1024},        # BERT large
    "RoBERTa-base": {"embedding_dim": 768},       # RoBERTa base
    "RoBERTa-large": {"embedding_dim": 1024},     # RoBERTa large
    "GPT-3.5": {"embedding_dim": 1536},          # GPT-3.5
    "text-embedding-ada-002": {"embedding_dim": 1536}, # OpenAI embedding
}


def create_adapter_for_cube(cube_dimensions: Tuple[int, int, int],
                           teacher_model: str = "Meta-Llama-3-8B",
                           strategy: str = "learned_linear") -> UniversalEmbeddingAdapter:
    """
    Удобная функция для создания адаптера для конкретного куба
    
    Args:
        cube_dimensions: (x, y, z) размеры куба
        teacher_model: Название teacher модели
        strategy: Стратегия конвертации
    
    Returns:
        UniversalEmbeddingAdapter: Готовый адаптер
    """
    # Вычисляем surface size (используем одну грань)
    surface_size = cube_dimensions[0] * cube_dimensions[1]
    
    # Получаем размер эмбедингов модели
    if teacher_model not in KNOWN_MODELS:
        raise ValueError(f"Unknown model: {teacher_model}. Known models: {list(KNOWN_MODELS.keys())}")
    
    source_dim = KNOWN_MODELS[teacher_model]["embedding_dim"]
    
    adapter = UniversalEmbeddingAdapter(
        input_dim=source_dim,
        output_dim=surface_size,
        strategy=strategy
    )
    
    logger.info(f"[TARGET] Адаптер создан для куба {cube_dimensions}: {teacher_model} ({source_dim}D) → surface ({surface_size}D)")
    
    return adapter


# Для удобства экспорта
__all__ = [
    "UniversalEmbeddingAdapter",
    "AdapterManager", 
    "KNOWN_MODELS",
    "create_adapter_for_cube"
] 