#!/usr/bin/env python3
"""
Unified Dataset Loader для реального обучения
Объединяет dialogue datasets, prepared embeddings, и SNLI данные
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import random
from dataclasses import dataclass

from generate_snli_embedding_dataset import SNLIEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Конфигурация для unified dataset loader"""
    # Источники данных
    use_dialogue_cache: bool = True
    use_prepared_embeddings: bool = True  
    use_snli_generator: bool = False  # Включаем только если нужно
    use_cache_embeddings: bool = True
    
    # Параметры загрузки
    max_samples_per_source: Optional[int] = None
    shuffle_sources: bool = True
    embedding_dim: int = 768  # DistilBERT dimension
    
    # Фильтрация
    min_embedding_norm: float = 0.1
    max_embedding_norm: float = 100.0


class UnifiedEmbeddingDataset(Dataset):
    """
    Unified Dataset для всех доступных источников эмбеддингов
    Поддерживает: dialogue cache, prepared embeddings, cache files, SNLI
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.embeddings: List[torch.Tensor] = []
        self.metadata: List[Dict] = []
        
        logger.info("[SYNC] Initializing UnifiedEmbeddingDataset...")
        
        # Загружаем данные из всех источников
        self._load_all_sources()
        
        # Фильтруем и валидируем
        self._filter_and_validate()
        
        logger.info(f"[OK] Dataset ready: {len(self.embeddings)} samples")
    
    def _load_all_sources(self):
        """Загружаем данные из всех доступных источников"""
        
        if self.config.use_dialogue_cache:
            self._load_dialogue_cache()
            
        if self.config.use_prepared_embeddings:
            self._load_prepared_embeddings()
            
        if self.config.use_cache_embeddings:
            self._load_cache_embeddings()
            
        if self.config.use_snli_generator:
            self._load_snli_data()
    
    def _load_dialogue_cache(self):
        """Загружаем dialogue datasets из cache/dialogue_dataset/"""
        cache_dir = Path("cache/dialogue_dataset")
        files = list(cache_dir.glob("*.pt"))
        
        logger.info(f"[DIRECTORY] Loading dialogue cache: {len(files)} files")
        
        loaded_count = 0
        for file in files:
            try:
                data = torch.load(file, map_location='cpu')
                
                # Извлекаем эмбеддинги из dialogue data
                embeddings = self._extract_embeddings_from_dialogue(data)
                
                for emb in embeddings:
                    if self._is_valid_embedding(emb):
                        self.embeddings.append(emb)
                        self.metadata.append({
                            "source": "dialogue_cache",
                            "file": file.name,
                            "type": "dialogue"
                        })
                        loaded_count += 1
                        
                        if (self.config.max_samples_per_source and 
                            loaded_count >= self.config.max_samples_per_source):
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to load dialogue file {file}: {e}")
                
        logger.info(f"[OK] Loaded {loaded_count} embeddings from dialogue cache")
    
    def _extract_embeddings_from_dialogue(self, data: Dict) -> List[torch.Tensor]:
        """Извлекаем эмбеддинги из dialogue data структуры"""
        embeddings = []
        
        # Проверяем разные возможные ключи
        for key in ['embeddings', 'input_embeddings', 'teacher_embeddings', 'premise_embedding', 'hypothesis_embedding']:
            if key in data:
                emb_data = data[key]
                
                if isinstance(emb_data, torch.Tensor):
                    if emb_data.dim() == 2:  # [batch, dim]
                        embeddings.extend([emb_data[i] for i in range(emb_data.size(0))])
                    elif emb_data.dim() == 1:  # [dim]
                        embeddings.append(emb_data)
                elif isinstance(emb_data, list):
                    for emb in emb_data:
                        if isinstance(emb, torch.Tensor):
                            embeddings.append(emb)
        
        return embeddings
    
    def _load_prepared_embeddings(self):
        """Загружаем prepared embeddings из data/embeddings/"""
        embeddings_dir = Path("data/embeddings")
        files = list(embeddings_dir.glob("*.pt"))
        
        logger.info(f"[DIRECTORY] Loading prepared embeddings: {len(files)} files")
        
        loaded_count = 0
        for file in files:
            try:
                data = torch.load(file, map_location='cpu')
                
                if isinstance(data, torch.Tensor):
                    if data.dim() == 2:  # [batch, dim]
                        for i in range(data.size(0)):
                            emb = data[i]
                            if self._is_valid_embedding(emb):
                                self.embeddings.append(emb)
                                self.metadata.append({
                                    "source": "prepared_embeddings",
                                    "file": file.name,
                                    "type": "prepared"
                                })
                                loaded_count += 1
                    elif data.dim() == 1:  # [dim]
                        if self._is_valid_embedding(data):
                            self.embeddings.append(data)
                            self.metadata.append({
                                "source": "prepared_embeddings", 
                                "file": file.name,
                                "type": "prepared"
                            })
                            loaded_count += 1
                            
                elif isinstance(data, dict):
                    # Извлекаем эмбеддинги из dict структуры
                    embeddings = self._extract_embeddings_from_dialogue(data)
                    for emb in embeddings:
                        if self._is_valid_embedding(emb):
                            self.embeddings.append(emb)
                            self.metadata.append({
                                "source": "prepared_embeddings",
                                "file": file.name,
                                "type": "prepared_dict"
                            })
                            loaded_count += 1
                
                if (self.config.max_samples_per_source and 
                    loaded_count >= self.config.max_samples_per_source):
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load prepared embedding {file}: {e}")
                
        logger.info(f"[OK] Loaded {loaded_count} embeddings from prepared files")
    
    def _load_cache_embeddings(self):
        """Загружаем cache эмбеддинги из cache/llm_*.pt"""
        cache_files = list(Path("cache").glob("llm_*.pt"))
        
        logger.info(f"[DIRECTORY] Loading cache embeddings: {len(cache_files)} files")
        
        loaded_count = 0
        for file in cache_files:
            try:
                data = torch.load(file, map_location='cpu')
                
                if isinstance(data, torch.Tensor):
                    if data.dim() == 2:  # [batch, dim]
                        for i in range(data.size(0)):
                            emb = data[i]
                            if self._is_valid_embedding(emb):
                                self.embeddings.append(emb)
                                self.metadata.append({
                                    "source": "cache_embeddings",
                                    "file": file.name,
                                    "type": "cache"
                                })
                                loaded_count += 1
                    elif data.dim() == 1:  # [dim]
                        if self._is_valid_embedding(data):
                            self.embeddings.append(data)
                            self.metadata.append({
                                "source": "cache_embeddings",
                                "file": file.name, 
                                "type": "cache"
                            })
                            loaded_count += 1
                
                if (self.config.max_samples_per_source and 
                    loaded_count >= self.config.max_samples_per_source):
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load cache embedding {file}: {e}")
                
        logger.info(f"[OK] Loaded {loaded_count} embeddings from cache files")
    
    def _load_snli_data(self):
        """Генерируем SNLI эмбеддинги при необходимости"""
        logger.info("[TOOL] Generating SNLI embeddings...")
        
        try:
            generator = SNLIEmbeddingGenerator()
            snli_data = generator.load_snli_dataset(size_fraction=0.1)  # 10% для начала
            
            loaded_count = 0
            for item in snli_data:
                if loaded_count >= (self.config.max_samples_per_source or 1000):
                    break
                    
                # Генерируем эмбеддинги для premise и hypothesis
                try:
                    premise_emb = generator.loader.get_text_embedding(item['premise'])
                    hypothesis_emb = generator.loader.get_text_embedding(item['hypothesis'])
                    
                    for emb, text_type in [(premise_emb, 'premise'), (hypothesis_emb, 'hypothesis')]:
                        if self._is_valid_embedding(emb):
                            self.embeddings.append(emb)
                            self.metadata.append({
                                "source": "snli_generator",
                                "type": text_type,
                                "label": item.get('label', 'unknown')
                            })
                            loaded_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to generate SNLI embedding: {e}")
                    continue
                    
            logger.info(f"[OK] Generated {loaded_count} SNLI embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load SNLI data: {e}")
    
    def _is_valid_embedding(self, emb: torch.Tensor) -> bool:
        """Проверяем валидность эмбеддинга"""
        if not isinstance(emb, torch.Tensor):
            return False
            
        if emb.dim() != 1:
            return False
            
        if emb.size(0) != self.config.embedding_dim:
            return False
            
        # Проверяем норму
        norm = torch.norm(emb).item()
        if norm < self.config.min_embedding_norm or norm > self.config.max_embedding_norm:
            return False
            
        # Проверяем на NaN/Inf
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            return False
            
        return True
    
    def _filter_and_validate(self):
        """Финальная фильтрация и валидация данных"""
        logger.info("[SEARCH] Filtering and validating dataset...")
        
        # Shuffle если нужно
        if self.config.shuffle_sources:
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
            
        logger.info("[DATA] Dataset statistics:")
        for source, count in source_stats.items():
            logger.info(f"  {source}: {count} samples")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        return {
            'embedding': self.embeddings[idx],
            'metadata': self.metadata[idx]
        }
    
    def get_stats(self) -> Dict:
        """Получить статистику по датасету"""
        source_stats = {}
        type_stats = {}
        
        for meta in self.metadata:
            source = meta['source']
            type_name = meta.get('type', 'unknown')
            
            source_stats[source] = source_stats.get(source, 0) + 1
            type_stats[type_name] = type_stats.get(type_name, 0) + 1
        
        return {
            "total_samples": len(self.embeddings),
            "embedding_dim": self.config.embedding_dim,
            "source_distribution": source_stats,
            "type_distribution": type_stats
        }


def create_training_dataloader(
    config: Optional[DatasetConfig] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, Dict]:
    """
    Создает DataLoader для реального обучения
    
    Returns:
        Tuple[DataLoader, Dict]: DataLoader и статистика датасета
    """
    if config is None:
        config = DatasetConfig()
    
    dataset = UnifiedEmbeddingDataset(config)
    stats = dataset.get_stats()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"[START] DataLoader created: {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader, stats


def main():
    """Тестируем unified dataset loader"""
    print("[TEST] TESTING UNIFIED DATASET LOADER")
    print("=" * 50)
    
    # Создаем конфигурацию для тестирования
    config = DatasetConfig(
        use_dialogue_cache=True,
        use_prepared_embeddings=True,
        use_cache_embeddings=True,
        use_snli_generator=False,  # Отключаем для быстрого теста
        max_samples_per_source=100,  # Ограничиваем для теста
        shuffle_sources=True
    )
    
    # Создаем DataLoader
    dataloader, stats = create_training_dataloader(
        config=config,
        batch_size=8,
        shuffle=True
    )
    
    print(f"\n[DATA] DATASET STATISTICS:")
    print(json.dumps(stats, indent=2))
    
    # Тестируем загрузку нескольких батчей
    print(f"\n[SYNC] TESTING BATCH LOADING:")
    for i, batch in enumerate(dataloader):
        embeddings = batch['embedding']  # [batch_size, embedding_dim]
        metadata = batch['metadata']  # List of dicts
        
        print(f"Batch {i+1}:")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Embeddings dtype: {embeddings.dtype}")
        print(f"  Metadata samples: {len(metadata)}")
        
        if i >= 2:  # Тестируем только первые 3 батча
            break
    
    print(f"\n[OK] Unified Dataset Loader test completed!")
    print(f"[UP] Ready for real training with {stats['total_samples']} samples")


if __name__ == "__main__":
    main()