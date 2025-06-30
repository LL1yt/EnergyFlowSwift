#!/usr/bin/env python3
"""
Unified Dataset Loader для реального обучения 3D Cellular Neural Network
======================================================================

Объединяет все источники эмбеддингов:
- dialogue datasets (cache/dialogue_dataset/)
- prepared embeddings (data/embeddings/)
- cache embeddings (cache/llm_*.pt)

Использует централизованную конфигурацию из new_rebuild.config
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import random
from dataclasses import dataclass
import gc

from ....config import SimpleProjectConfig
from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager

logger = get_logger(__name__)


class GPUMemoryEstimator:
    """Оценка и управление GPU памятью для датасетов"""
    
    def __init__(self):
        self.device_manager = get_device_manager()
        self.embedding_size_mb = 768 * 4 / (1024**2)  # float32, 768 dim
    
    def estimate_dataset_memory_mb(self, num_samples: int) -> float:
        """Оценка памяти для датасета в MB"""
        return num_samples * self.embedding_size_mb
    
    def get_safe_sample_limit(self, reserve_for_training_gb: float = 20.0) -> Optional[int]:
        """
        Вычисляет безопасный лимит сэмплов, оставляя память для обучения
        
        Args:
            reserve_for_training_gb: Сколько GB оставить для обучения
        """
        if not self.device_manager.is_cuda():
            return None  # CPU режим - без ограничений
            
        total_memory_gb = self.device_manager.get_available_memory_gb()
        available_for_dataset_gb = total_memory_gb - reserve_for_training_gb
        
        if available_for_dataset_gb <= 0:
            logger.warning(f"⚠️ Недостаточно GPU памяти для датасета. Total: {total_memory_gb:.1f}GB, Reserved: {reserve_for_training_gb}GB")
            return 100  # Минимальный fallback
            
        available_for_dataset_mb = available_for_dataset_gb * 1024
        safe_samples = int(available_for_dataset_mb / self.embedding_size_mb * 0.8)  # 80% безопасность
        
        logger.info(f"🧮 GPU Memory Planning:")
        logger.info(f"  Total GPU: {total_memory_gb:.1f}GB")
        logger.info(f"  Reserved for training: {reserve_for_training_gb}GB")
        logger.info(f"  Available for dataset: {available_for_dataset_gb:.1f}GB")
        logger.info(f"  Safe sample limit: {safe_samples:,}")
        
        return safe_samples


@dataclass 
class DatasetStats:
    """Статистика датасета"""
    total_samples: int
    embedding_dim: int
    source_distribution: Dict[str, int]
    type_distribution: Dict[str, int]


class UnifiedEmbeddingDataset(Dataset):
    """
    Unified Dataset для всех доступных источников эмбеддингов
    Использует централизованную конфигурацию
    """
    
    def __init__(self, config: SimpleProjectConfig, max_total_samples: Optional[int] = None):
        self.config = config
        self.device_manager = get_device_manager()
        self.memory_estimator = GPUMemoryEstimator()
        
        # Умное планирование памяти
        self.max_total_samples = self._plan_memory_usage(max_total_samples)
        
        self.embeddings: List[torch.Tensor] = []
        self.metadata: List[Dict] = []
        self.use_gpu_acceleration = self.device_manager.is_cuda()
        
        logger.info("🔄 Initializing GPU-accelerated UnifiedEmbeddingDataset...")
        logger.info(f"🚀 GPU acceleration: {'✅ Enabled' if self.use_gpu_acceleration else '❌ Disabled'}")
        if self.max_total_samples is not None:
            logger.info(f"📊 Smart memory limit: {self.max_total_samples:,} samples")
        
        # Загружаем данные из всех источников
        self._load_all_sources()
        
        # Применяем общий лимит если задан
        self._apply_total_limit()
        
        # GPU-ускоренная фильтрация и валидация
        if self.use_gpu_acceleration:
            self._gpu_filter_and_validate()
        else:
            self._filter_and_validate()
        
        logger.info(f"✅ Dataset ready: {len(self.embeddings)} samples")
        self._log_memory_usage()
    
    def _plan_memory_usage(self, requested_limit: Optional[int]) -> Optional[int]:
        """Планирование использования памяти с учетом обучения"""
        
        # Получаем резерв для обучения из конфигурации или используем по умолчанию
        training_reserve_gb = getattr(self.config.training_embedding, 'gpu_memory_reserve_gb', 20.0)
        
        # Вычисляем безопасный лимит исходя из доступной памяти
        safe_limit = self.memory_estimator.get_safe_sample_limit(training_reserve_gb)
        
        if requested_limit is None:
            return safe_limit  # Используем автоматический лимит
        
        if safe_limit is None:
            return requested_limit  # CPU режим - без ограничений
            
        # Используем минимум из запрошенного и безопасного
        final_limit = min(requested_limit, safe_limit)
        
        if final_limit < requested_limit:
            logger.warning(f"⚠️ Requested {requested_limit:,} samples, but GPU memory allows only {final_limit:,}")
            
        return final_limit
    
    def _log_memory_usage(self):
        """Логирование использования памяти"""
        if self.use_gpu_acceleration:
            stats = self.device_manager.get_memory_stats()
            estimated_mb = self.memory_estimator.estimate_dataset_memory_mb(len(self.embeddings))
            
            logger.info(f"📊 GPU Memory Usage:")
            logger.info(f"  Dataset estimated: {estimated_mb:.1f}MB")
            logger.info(f"  GPU allocated: {stats.get('allocated_mb', 0):.1f}MB")
            logger.info(f"  GPU available: {self.device_manager.get_available_memory_gb():.1f}GB")
    
    def _load_all_sources(self):
        """Загружаем данные из всех доступных источников"""
        
        # Всегда загружаем основные источники
        self._load_dialogue_cache()
        self._load_prepared_embeddings()
        self._load_cache_embeddings()
    
    def _apply_total_limit(self):
        """Применяем общий лимит на количество сэмплов"""
        if self.max_total_samples is None:
            return
            
        current_total = len(self.embeddings)
        logger.info(f"📊 Before limit: {current_total} samples")
        
        if current_total > self.max_total_samples:
            # Случайным образом выбираем сэмплы, чтобы сохранить разнообразие
            indices = list(range(current_total))
            random.shuffle(indices)
            selected_indices = indices[:self.max_total_samples]
            
            # Применяем выбор
            self.embeddings = [self.embeddings[i] for i in selected_indices]
            self.metadata = [self.metadata[i] for i in selected_indices]
            
            logger.info(f"📊 Applied total limit: {len(self.embeddings)} samples (reduced from {current_total})")
    
    def _load_dialogue_cache(self):
        """Загружаем dialogue datasets из cache/dialogue_dataset/"""
        cache_dir = Path("cache/dialogue_dataset")
        files = list(cache_dir.glob("*.pt"))
        
        logger.info(f"📂 Loading dialogue cache: {len(files)} files")
        
        # Определяем устройство для загрузки
        map_location = 'cuda' if self.use_gpu_acceleration else 'cpu'
        
        loaded_count = 0
        for file in files:
            try:
                data = torch.load(file, map_location=map_location)
                
                # Обрабатываем структуру из анализа: questions [4, 768], answers [4, 768]
                embeddings = []
                
                if 'questions' in data and isinstance(data['questions'], torch.Tensor):
                    questions = data['questions']
                    if questions.dim() == 2:  # [batch, 768]
                        embeddings.extend([questions[i] for i in range(questions.size(0))])
                
                if 'answers' in data and isinstance(data['answers'], torch.Tensor):
                    answers = data['answers'] 
                    if answers.dim() == 2:  # [batch, 768]
                        embeddings.extend([answers[i] for i in range(answers.size(0))])
                
                for emb in embeddings:
                    if self._is_valid_embedding(emb):
                        self.embeddings.append(emb)
                        self.metadata.append({
                            "source": "dialogue_cache",
                            "file": file.name,
                            "type": "dialogue"
                        })
                        loaded_count += 1
                            
            except Exception as e:
                logger.warning(f"Failed to load dialogue file {file}: {e}")
                
        logger.info(f"✅ Loaded {loaded_count} embeddings from dialogue cache")
    
    def _load_prepared_embeddings(self):
        """Загружаем prepared embeddings из data/embeddings/"""
        embeddings_dir = Path("data/embeddings")
        files = list(embeddings_dir.glob("*.pt"))
        
        logger.info(f"📂 Loading prepared embeddings: {len(files)} files")
        
        # Определяем устройство для загрузки
        map_location = 'cuda' if self.use_gpu_acceleration else 'cpu'
        
        loaded_count = 0
        for file in files:
            try:
                data = torch.load(file, map_location=map_location)
                
                # Обрабатываем структуру из анализа: question_embeddings, answer_embeddings
                embeddings = []
                
                if isinstance(data, dict):
                    for key in ['question_embeddings', 'answer_embeddings']:
                        if key in data:
                            emb_data = data[key]
                            if isinstance(emb_data, torch.Tensor) and emb_data.dim() == 2:
                                # [num_samples, 768] -> list of [768] tensors
                                embeddings.extend([emb_data[i] for i in range(emb_data.size(0))])
                
                elif isinstance(data, torch.Tensor):
                    if data.dim() == 2:  # [batch, dim]
                        embeddings.extend([data[i] for i in range(data.size(0))])
                    elif data.dim() == 1:  # [dim]
                        embeddings.append(data)
                
                for emb in embeddings:
                    if self._is_valid_embedding(emb):
                        self.embeddings.append(emb)
                        self.metadata.append({
                            "source": "prepared_embeddings",
                            "file": file.name,
                            "type": "prepared"
                        })
                        loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load prepared embedding {file}: {e}")
                
        logger.info(f"✅ Loaded {loaded_count} embeddings from prepared files")
    
    def _load_cache_embeddings(self):
        """Загружаем cache эмбеддинги из cache/llm_*.pt"""
        cache_files = list(Path("cache").glob("llm_*.pt"))
        
        logger.info(f"📂 Loading cache embeddings: {len(cache_files)} files")
        
        # Определяем устройство для загрузки
        map_location = 'cuda' if self.use_gpu_acceleration else 'cpu'
        
        loaded_count = 0
        for file in cache_files:
            try:
                data = torch.load(file, map_location=map_location)
                
                # Структура из анализа: [94, 4096] torch.float16
                if isinstance(data, torch.Tensor):
                    # Эти файлы имеют размерность 4096, нам нужна 768
                    # Пропускаем их пока или используем проекцию
                    if data.shape[-1] != self.config.embedding.input_dim:
                        logger.debug(f"Skipping {file.name}: wrong dimension {data.shape[-1]} != {self.config.embedding.input_dim}")
                        continue
                    
                    if data.dim() == 2:  # [batch, dim]
                        for i in range(min(data.size(0), 100)):  # Ограничиваем количество
                            emb = data[i].float()  # Конвертируем из float16
                            if self._is_valid_embedding(emb):
                                self.embeddings.append(emb)
                                self.metadata.append({
                                    "source": "cache_embeddings",
                                    "file": file.name,
                                    "type": "cache"
                                })
                                loaded_count += 1
                    elif data.dim() == 1:  # [dim]
                        emb = data.float()
                        if self._is_valid_embedding(emb):
                            self.embeddings.append(emb)
                            self.metadata.append({
                                "source": "cache_embeddings",
                                "file": file.name,
                                "type": "cache"
                            })
                            loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load cache embedding {file}: {e}")
                
        logger.info(f"✅ Loaded {loaded_count} embeddings from cache files")
    
    def _is_valid_embedding(self, emb: torch.Tensor) -> bool:
        """Проверяем валидность эмбеддинга используя конфиг"""
        if not isinstance(emb, torch.Tensor):
            return False
            
        if emb.dim() != 1:
            return False
            
        if emb.size(0) != self.config.embedding.input_dim:
            return False
            
        # Проверяем норму (базовые пределы)
        norm = torch.norm(emb).item()
        if norm < 0.1 or norm > 100.0:
            return False
            
        # Проверяем на NaN/Inf
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            return False
            
        return True
    
    def _filter_and_validate(self):
        """Финальная фильтрация и валидация данных"""
        logger.info("🔍 Filtering and validating dataset...")
        
        # Shuffle
        combined = list(zip(self.embeddings, self.metadata))
        random.shuffle(combined)
        self.embeddings, self.metadata = zip(*combined)
        self.embeddings = list(self.embeddings)
        self.metadata = list(self.metadata)
        
        # Конвертируем в нужный тип
        self.embeddings = [emb.float() for emb in self.embeddings]
        
        # Статистика по источникам
        source_stats = {}
        for meta in self.metadata:
            source = meta['source']
            source_stats[source] = source_stats.get(source, 0) + 1
            
        logger.info("📊 Dataset statistics:")
        for source, count in source_stats.items():
            logger.info(f"  {source}: {count} samples")
    
    def _gpu_filter_and_validate(self):
        """GPU-ускоренная фильтрация и валидация данных"""
        logger.info("🚀 GPU-accelerated filtering and validation...")
        
        if not self.embeddings:
            return
            
        # Конвертируем в batch tensor для GPU обработки
        try:
            # Создаем батч тензор на GPU
            embeddings_batch = torch.stack([emb.float() for emb in self.embeddings])
            
            # Векторизованная валидация на GPU
            valid_mask = self._gpu_validate_batch(embeddings_batch)
            
            # Применяем маску
            valid_embeddings_batch = embeddings_batch[valid_mask]
            valid_metadata = [meta for i, meta in enumerate(self.metadata) if valid_mask[i]]
            
            # Преобразуем обратно в список
            self.embeddings = [valid_embeddings_batch[i] for i in range(valid_embeddings_batch.size(0))]
            self.metadata = valid_metadata
            
            # Shuffle на GPU
            if len(self.embeddings) > 0:
                indices = torch.randperm(len(self.embeddings), device=embeddings_batch.device)
                self.embeddings = [self.embeddings[i] for i in indices.cpu()]
                self.metadata = [self.metadata[i] for i in indices.cpu()]
            
            logger.info(f"✅ GPU validation completed: {len(self.embeddings)} valid samples")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU validation failed, falling back to CPU: {e}")
            self._filter_and_validate()
            return
            
        # Статистика по источникам
        source_stats = {}
        for meta in self.metadata:
            source = meta['source']
            source_stats[source] = source_stats.get(source, 0) + 1
            
        logger.info("📊 GPU Dataset statistics:")
        for source, count in source_stats.items():
            logger.info(f"  {source}: {count} samples")
            
        # Очистка GPU памяти
        if 'embeddings_batch' in locals():
            del embeddings_batch
        if 'valid_embeddings_batch' in locals():
            del valid_embeddings_batch
        gc.collect()
        torch.cuda.empty_cache()
    
    def _gpu_validate_batch(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Векторизованная валидация батча эмбеддингов на GPU
        
        Args:
            embeddings_batch: [N, 768] tensor on GPU
            
        Returns:
            valid_mask: [N] boolean tensor
        """
        N = embeddings_batch.size(0)
        device = embeddings_batch.device
        
        # Проверка размерности (все должны быть 768)
        dim_valid = embeddings_batch.size(1) == self.config.embedding.input_dim
        
        # Векторизованная проверка норм
        norms = torch.norm(embeddings_batch, dim=1)  # [N]
        norm_valid = (norms > 0.1) & (norms < 100.0)  # [N]
        
        # Проверка на NaN/Inf
        nan_valid = ~torch.isnan(embeddings_batch).any(dim=1)  # [N]
        inf_valid = ~torch.isinf(embeddings_batch).any(dim=1)  # [N]
        
        # Комбинируем все условия
        valid_mask = norm_valid & nan_valid & inf_valid
        
        # Если размерность неправильная, отклоняем все
        if not dim_valid:
            valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
            
        logger.info(f"🔍 GPU validation: {valid_mask.sum().item()}/{N} samples passed")
        
        return valid_mask
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        return {
            'embedding': self.embeddings[idx],
            'metadata': self.metadata[idx]
        }
    
    def get_stats(self) -> DatasetStats:
        """Получить статистику по датасету"""
        source_stats = {}
        type_stats = {}
        
        for meta in self.metadata:
            source = meta['source']
            type_name = meta.get('type', 'unknown')
            
            source_stats[source] = source_stats.get(source, 0) + 1
            type_stats[type_name] = type_stats.get(type_name, 0) + 1
        
        return DatasetStats(
            total_samples=len(self.embeddings),
            embedding_dim=self.config.embedding.input_dim,
            source_distribution=source_stats,
            type_distribution=type_stats
        )


def create_training_dataloader(
    config: SimpleProjectConfig,
    max_total_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DatasetStats]:
    """
    Создает DataLoader для реального обучения используя центральную конфигурацию
    
    Args:
        config: Центральная конфигурация проекта
        max_total_samples: Общий лимит на количество сэмплов (переопределяет config)
        shuffle: Перемешивать данные
        num_workers: Количество воркеров для загрузки
        
    Returns:
        Tuple[DataLoader, DatasetStats]: DataLoader и статистика датасета
    """
    
    # Используем общий лимит из конфигурации, если он задан
    if config.training_embedding.max_total_samples is not None:
        # Приоритет: max_total_samples из конфигурации
        effective_max_samples = config.training_embedding.max_total_samples
        logger.info(f"📊 Using max_total_samples from config: {effective_max_samples}")
    else:
        # Fallback на параметр функции
        effective_max_samples = max_total_samples
        if effective_max_samples is not None:
            logger.info(f"📊 Using max_total_samples parameter: {effective_max_samples}")
        else:
            logger.info("📊 No sample limit - using full dataset")
    
    dataset = UnifiedEmbeddingDataset(config, effective_max_samples)
    stats = dataset.get_stats()
    
    # Используем batch_size из конфигурации
    batch_size = config.training_embedding.embedding_batch_size
    
    # Оптимизации для RTX 5090
    device_manager = get_device_manager()
    is_cuda = device_manager.is_cuda()
    
    # Оптимальные параметры для RTX 5090
    optimal_num_workers = 8 if is_cuda else num_workers  # 4*2 GPU cores
    prefetch_factor = 4 if is_cuda else 2
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=optimal_num_workers,
        pin_memory=is_cuda,
        drop_last=True,  # Для стабильности размеров батчей
        persistent_workers=True if optimal_num_workers > 0 else False,  # PyTorch 2.x оптимизация
        prefetch_factor=prefetch_factor if optimal_num_workers > 0 else None
    )
    
    logger.info(f"🚀 Optimized DataLoader for RTX 5090:")
    logger.info(f"  Samples: {len(dataset):,}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Workers: {optimal_num_workers}")
    logger.info(f"  Pin memory: {is_cuda}")
    logger.info(f"  Persistent workers: {optimal_num_workers > 0}")
    logger.info(f"  Prefetch factor: {prefetch_factor}")
    
    logger.info(f"🚀 DataLoader created: {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader, stats


def main():
    """Тестируем unified dataset loader"""
    from ....utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("🧪 TESTING UNIFIED DATASET LOADER")
    logger.info("=" * 50)
    
    # Используем центральную конфигурацию
    from ....config import SimpleProjectConfig
    config = SimpleProjectConfig()
    
    # Ограничиваем для быстрого теста
    max_samples = 50
    
    # Создаем DataLoader
    dataloader, stats = create_training_dataloader(
        config=config,
        max_total_samples=max_samples,
        shuffle=True
    )
    
    logger.info(f"\n📊 DATASET STATISTICS:")
    logger.info(f"Total samples: {stats.total_samples}")
    logger.info(f"Embedding dim: {stats.embedding_dim}")
    logger.info(f"Source distribution: {stats.source_distribution}")
    logger.info(f"Type distribution: {stats.type_distribution}")
    
    # Тестируем загрузку нескольких батчей
    logger.info(f"\n🔄 TESTING BATCH LOADING:")
    for i, batch in enumerate(dataloader):
        embeddings = batch['embedding']  # [batch_size, embedding_dim]
        metadata = batch['metadata']  # List of dicts
        
        logger.info(f"Batch {i+1}:")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Embeddings dtype: {embeddings.dtype}")
        logger.info(f"  Metadata samples: {len(metadata)}")
        
        if i >= 2:  # Тестируем только первые 3 батча
            break
    
    logger.info(f"\n✅ Unified Dataset Loader test completed!")
    logger.info(f"📈 Ready for real training with {stats.total_samples} samples")


if __name__ == "__main__":
    main()