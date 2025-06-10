"""
AutoencoderDataset - Класс для подготовки данных к обучению куба в autoencoder режиме

Этот модуль реализует специализированный dataset для обучения 3D Cubic Core
на задачах реконструкции эмбедингов (autoencoder mode).

Ключевые возможности:
- Интеграция с EmbeddingLoader для загрузки различных типов эмбедингов
- Smart caching для эффективности обучения
- Batch processing с оптимальными размерами
- Поддержка различных источников данных (текстовые файлы, готовые эмбединги)
- Adaptive sampling для balanced training

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0 (Phase 3.1 - Stage 1.2)
Дата: 6 июня 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import pickle
import hashlib
from dataclasses import dataclass
import random

# Импорты готовых компонентов
try:
    from data.embedding_loader import EmbeddingLoader
    EMBEDDING_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING]  Warning: EmbeddingLoader not available: {e}")
    EMBEDDING_LOADER_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Конфигурация для AutoencoderDataset"""
    # Источники данных
    data_sources: List[str] = None  # Пути к файлам с данными
    embedding_format: str = "llm"   # word2vec | glove | bert | llm
    llm_model: str = "distilbert"   # Модель для генерации эмбедингов
    
    # Размеры и формат
    embedding_dim: int = 768        # Размерность эмбедингов
    max_samples: int = 10000       # Максимальное количество семплов
    min_samples: int = 100         # Минимальное количество семплов
    
    # Preprocessing
    normalize_embeddings: bool = True
    center_embeddings: bool = True
    add_noise: bool = False        # Добавлять ли шум для регуляризации
    noise_std: float = 0.01        # Стандартное отклонение шума
    
    # Caching
    cache_dir: str = "cache/autoencoder_dataset"
    use_cache: bool = True
    cache_embeddings: bool = True
    cache_batch_size: int = 1000   # Размер батча для кэширования
    
    # Validation split
    validation_split: float = 0.2  # Доля данных для валидации
    shuffle_data: bool = True      # Перемешивать ли данные
    random_seed: int = 42          # Seed для воспроизводимости
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        
        # Создание директории для кэша
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class AutoencoderDataset(Dataset):
    """
    Dataset для обучения 3D Cubic Core в autoencoder режиме
    
    Создает пары (embedding, embedding) для обучения реконструкции,
    с интеграцией EmbeddingLoader и smart caching системой.
    """
    
    def __init__(self, 
                 config: Optional[Union[DatasetConfig, Dict, str]] = None,
                 texts: Optional[List[str]] = None,
                 embeddings: Optional[torch.Tensor] = None):
        """
        Инициализация AutoencoderDataset
        
        Args:
            config: Конфигурация dataset (DatasetConfig, dict или путь к JSON)
            texts: Список текстов для генерации эмбедингов (опционально)
            embeddings: Готовые эмбединги (опционально)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("[START] Initializing AutoencoderDataset...")
        
        # Проверка зависимостей
        if not EMBEDDING_LOADER_AVAILABLE:
            raise ImportError("EmbeddingLoader is required for AutoencoderDataset. "
                            "Make sure data.embedding_loader is implemented.")
        
        # Загрузка конфигурации
        self.config = self._load_config(config)
        
        # Установка random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        # Инициализация EmbeddingLoader
        self.embedding_loader = EmbeddingLoader(
            cache_dir=str(Path(self.config.cache_dir) / "embedding_loader_cache")
        )
        
        # Данные
        self.embeddings: torch.Tensor = None
        self.train_embeddings: torch.Tensor = None
        self.val_embeddings: torch.Tensor = None
        self.is_validation_mode: bool = False
        
        # Метаданные
        self.dataset_info = {}
        self.cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_loads': 0
        }
        
        # Загрузка данных
        if embeddings is not None:
            self.logger.info("Using provided embeddings")
            self._load_from_embeddings(embeddings)
        elif texts is not None:
            self.logger.info("Generating embeddings from texts")
            self._load_from_texts(texts)
        elif self.config.data_sources:
            self.logger.info("Loading data from configured sources")
            self._load_from_sources()
        else:
            raise ValueError("No data source provided. Specify embeddings, texts, or data_sources in config.")
        
        # Создание train/val split
        self._create_train_val_split()
        
        self.logger.info(f"[OK] AutoencoderDataset initialized successfully")
        self.logger.info(f"   Total samples: {len(self.embeddings)}")
        self.logger.info(f"   Train samples: {len(self.train_embeddings)}")
        self.logger.info(f"   Val samples: {len(self.val_embeddings)}")
        self.logger.info(f"   Embedding dim: {self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 'Unknown'}")
    
    def _load_config(self, config: Optional[Union[DatasetConfig, Dict, str]]) -> DatasetConfig:
        """Загрузка и валидация конфигурации"""
        if config is None:
            return DatasetConfig()
        
        elif isinstance(config, DatasetConfig):
            return config
        
        elif isinstance(config, dict):
            return DatasetConfig(**config)
        
        elif isinstance(config, str):
            # Загрузка из JSON файла
            try:
                with open(config, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return DatasetConfig(**config_data)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config}: {e}")
                return DatasetConfig()
        
        else:
            self.logger.warning(f"Unknown config type: {type(config)}. Using default config.")
            return DatasetConfig()
    
    def _load_from_embeddings(self, embeddings: torch.Tensor):
        """Загрузка из готовых эмбедингов"""
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.from_numpy(np.array(embeddings)).float()
        
        # Проверка размерности
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings should be 2D tensor, got shape: {embeddings.shape}")
        
        if embeddings.shape[1] != self.config.embedding_dim:
            self.logger.warning(f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
                              f"expected {self.config.embedding_dim}")
            self.config.embedding_dim = embeddings.shape[1]
        
        self.embeddings = embeddings
        self._update_dataset_info("provided_embeddings", embeddings.shape[0])
    
    def _load_from_texts(self, texts: List[str]):
        """Генерация эмбедингов из текстов через EmbeddingLoader"""
        # Проверка кэша
        cache_key = self._create_cache_key_for_texts(texts)
        cached_embeddings = self._load_from_cache(cache_key)
        
        if cached_embeddings is not None and self.config.use_cache:
            self.logger.info("Loading embeddings from cache")
            self.embeddings = cached_embeddings
            self.cache_stats['cache_hits'] += 1
        else:
            self.logger.info(f"Generating embeddings from {len(texts)} texts using {self.config.llm_model}")
            
            # Ограничение количества текстов
            if len(texts) > self.config.max_samples:
                texts = texts[:self.config.max_samples]
                self.logger.info(f"Limited to {self.config.max_samples} texts")
            
            # Генерация эмбедингов
            embeddings = self.embedding_loader.batch_load_from_llm(
                texts=texts,
                model_key=self.config.llm_model,
                batch_size=self.config.cache_batch_size
            )
            
            # Preprocessing
            if self.config.normalize_embeddings or self.config.center_embeddings:
                embeddings = self.embedding_loader.preprocess_embeddings(
                    embeddings,
                    normalize=self.config.normalize_embeddings,
                    center=self.config.center_embeddings
                )
            
            self.embeddings = embeddings
            self.cache_stats['cache_misses'] += 1
            
            # Сохранение в кэш
            if self.config.cache_embeddings:
                self._save_to_cache(cache_key, embeddings)
        
        self._update_dataset_info("generated_from_texts", len(texts))
    
    def _load_from_sources(self):
        """Загрузка данных из конфигурированных источников"""
        all_embeddings = []
        
        for source_path in self.config.data_sources:
            source_path = Path(source_path)
            
            if not source_path.exists():
                self.logger.warning(f"Source file not found: {source_path}")
                continue
            
            try:
                if source_path.suffix in ['.txt']:
                    # Текстовый файл - читаем строки как тексты
                    with open(source_path, 'r', encoding='utf-8') as f:
                        texts = [line.strip() for line in f if line.strip()]
                    
                    # Генерируем эмбединги
                    embeddings = self.embedding_loader.batch_load_from_llm(
                        texts=texts,
                        model_key=self.config.llm_model,
                        batch_size=self.config.cache_batch_size
                    )
                    
                elif source_path.suffix in ['.pt', '.pth']:
                    # PyTorch тензор
                    embeddings = torch.load(source_path)
                    
                elif source_path.suffix in ['.npy']:
                    # NumPy массив
                    embeddings = torch.from_numpy(np.load(source_path)).float()
                    
                elif source_path.suffix in ['.pkl', '.pickle']:
                    # Pickle файл
                    with open(source_path, 'rb') as f:
                        data = pickle.load(f)
                    embeddings = torch.from_numpy(np.array(data)).float()
                    
                else:
                    # Пытаемся загрузить через EmbeddingLoader
                    embeddings = self.embedding_loader.load_embeddings(
                        path=source_path,
                        format_type=self.config.embedding_format,
                        preprocess=True
                    )
                
                all_embeddings.append(embeddings)
                self.logger.info(f"Loaded {embeddings.shape[0]} embeddings from {source_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load from {source_path}: {e}")
                continue
        
        if not all_embeddings:
            raise ValueError("No valid data sources found")
        
        # Объединение всех эмбедингов
        self.embeddings = torch.cat(all_embeddings, dim=0)
        self._update_dataset_info("loaded_from_sources", len(self.config.data_sources))
    
    def _create_train_val_split(self):
        """Создание разделения на train/validation"""
        total_samples = len(self.embeddings)
        
        if self.config.validation_split > 0:
            val_size = int(total_samples * self.config.validation_split)
            train_size = total_samples - val_size
            
            # Перемешивание если нужно
            indices = torch.arange(total_samples)
            if self.config.shuffle_data:
                indices = indices[torch.randperm(total_samples)]
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_embeddings = self.embeddings[train_indices]
            self.val_embeddings = self.embeddings[val_indices]
        else:
            # Нет валидации - все данные для обучения
            self.train_embeddings = self.embeddings
            self.val_embeddings = torch.empty(0, self.config.embedding_dim)
    
    def _create_cache_key_for_texts(self, texts: List[str]) -> str:
        """Создание ключа кэша для списка текстов"""
        # Создаем хэш от объединенных текстов + конфигурации
        text_content = "\n".join(texts[:100])  # Первые 100 текстов для хэша
        config_content = f"{self.config.llm_model}_{self.config.embedding_dim}_{len(texts)}"
        content = f"{text_content}_{config_content}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Загрузка эмбедингов из кэша"""
        if not self.config.use_cache:
            return None
        
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.pt"
        
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to load from cache {cache_path}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: torch.Tensor):
        """Сохранение эмбедингов в кэш"""
        if not self.config.cache_embeddings:
            return
        
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.pt"
        
        try:
            torch.save(embeddings, cache_path)
            self.logger.info(f"Cached embeddings to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache embeddings: {e}")
    
    def _update_dataset_info(self, source_type: str, sample_count: int):
        """Обновление метаданных dataset"""
        self.dataset_info.update({
            'source_type': source_type,
            'sample_count': sample_count,
            'embedding_dim': self.config.embedding_dim,
            'config': self.config.__dict__
        })
    
    def set_validation_mode(self, is_validation: bool = True):
        """Переключение между train/validation режимами"""
        self.is_validation_mode = is_validation
    
    def __len__(self) -> int:
        """Размер dataset"""
        if self.is_validation_mode:
            return len(self.val_embeddings)
        else:
            return len(self.train_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получение элемента dataset
        
        Args:
            idx: Индекс элемента
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_embedding, target_embedding)
            Для autoencoder режима input_embedding == target_embedding
        """
        if self.is_validation_mode:
            embedding = self.val_embeddings[idx]
        else:
            embedding = self.train_embeddings[idx]
        
        # Для autoencoder режима цель = вход
        input_embedding = embedding.clone()
        target_embedding = embedding.clone()
        
        # Добавление шума если нужно (только для входа)
        if self.config.add_noise and not self.is_validation_mode:
            noise = torch.randn_like(input_embedding) * self.config.noise_std
            input_embedding = input_embedding + noise
        
        return input_embedding, target_embedding
    
    def get_dataloader(self, 
                      batch_size: int = 32, 
                      shuffle: bool = True,
                      num_workers: int = 0,
                      validation: bool = False) -> DataLoader:
        """
        Создание DataLoader для dataset
        
        Args:
            batch_size: Размер батча
            shuffle: Перемешивать ли данные
            num_workers: Количество worker процессов
            validation: Использовать validation data
            
        Returns:
            DataLoader: Готовый DataLoader
        """
        # Устанавливаем режим
        original_mode = self.is_validation_mode
        self.set_validation_mode(validation)
        
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle and not validation,  # Не перемешиваем validation данные
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Восстанавливаем исходный режим
        self.set_validation_mode(original_mode)
        
        return dataloader
    
    def get_sample_embeddings(self, n_samples: int = 5) -> Dict[str, torch.Tensor]:
        """
        Получение примеров эмбедингов для анализа
        
        Args:
            n_samples: Количество примеров
            
        Returns:
            Dict с примерами train и validation эмбедингов
        """
        samples = {}
        
        if len(self.train_embeddings) > 0:
            train_indices = torch.randperm(len(self.train_embeddings))[:n_samples]
            samples['train'] = self.train_embeddings[train_indices]
        
        if len(self.val_embeddings) > 0:
            val_indices = torch.randperm(len(self.val_embeddings))[:n_samples]
            samples['validation'] = self.val_embeddings[val_indices]
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики dataset"""
        stats = {
            'total_samples': len(self.embeddings),
            'train_samples': len(self.train_embeddings),
            'val_samples': len(self.val_embeddings),
            'embedding_dim': self.config.embedding_dim,
            'validation_split': self.config.validation_split,
            'cache_stats': self.cache_stats.copy(),
            'dataset_info': self.dataset_info.copy()
        }
        
        if len(self.embeddings) > 0:
            stats.update({
                'embedding_mean': self.embeddings.mean().item(),
                'embedding_std': self.embeddings.std().item(),
                'embedding_min': self.embeddings.min().item(),
                'embedding_max': self.embeddings.max().item()
            })
        
        return stats
    
    def save_dataset_info(self, path: str):
        """Сохранение информации о dataset"""
        info = {
            'statistics': self.get_statistics(),
            'config': self.config.__dict__,
            'creation_time': str(Path().cwd()),  # Placeholder для времени создания
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Dataset info saved to {path}")
    
    def __repr__(self):
        return (f"AutoencoderDataset("
                f"samples={len(self.embeddings)}, "
                f"dim={self.config.embedding_dim}, "
                f"train={len(self.train_embeddings)}, "
                f"val={len(self.val_embeddings)}, "
                f"mode={'validation' if self.is_validation_mode else 'train'})")


def create_text_dataset(texts: List[str], 
                       llm_model: str = "distilbert",
                       validation_split: float = 0.2,
                       **kwargs) -> AutoencoderDataset:
    """
    Удобная функция для создания dataset из текстов
    
    Args:
        texts: Список текстов
        llm_model: Модель для генерации эмбедингов
        validation_split: Доля для validation
        **kwargs: Дополнительные параметры для DatasetConfig
        
    Returns:
        AutoencoderDataset: Настроенный dataset
    """
    config = DatasetConfig(
        llm_model=llm_model,
        validation_split=validation_split,
        **kwargs
    )
    
    return AutoencoderDataset(config=config, texts=texts)


def create_file_dataset(file_paths: List[str],
                       embedding_format: str = "llm",
                       llm_model: str = "distilbert",
                       validation_split: float = 0.2,
                       **kwargs) -> AutoencoderDataset:
    """
    Удобная функция для создания dataset из файлов
    
    Args:
        file_paths: Пути к файлам с данными
        embedding_format: Формат эмбедингов
        llm_model: LLM модель (если format=llm)
        validation_split: Доля для validation
        **kwargs: Дополнительные параметры для DatasetConfig
        
    Returns:
        AutoencoderDataset: Настроенный dataset
    """
    config = DatasetConfig(
        data_sources=file_paths,
        embedding_format=embedding_format,
        llm_model=llm_model,
        validation_split=validation_split,
        **kwargs
    )
    
    return AutoencoderDataset(config=config)