"""
SNLIProvider - провайдер для SNLI датасета
==========================================

Адаптация legacy generate_snli_embedding_dataset.py под новую архитектуру:
- Загрузка SNLI датасета из HuggingFace
- Генерация эмбеддингов через TeacherModelProvider
- Кэширование и валидация данных
"""

import torch
from typing import List, Tuple, Optional, Dict, Any
import random
from datasets import load_dataset
from pathlib import Path
import json

from .base_provider import BaseDataProvider
from .teacher_model import TeacherModelProvider
from ...utils.logging import get_logger, DEBUG_INIT

logger = get_logger(__name__)


class SNLIProvider(BaseDataProvider):
    """
    Провайдер для Stanford Natural Language Inference (SNLI) датасета
    
    Генерирует пары premise-hypothesis в формате question-answer
    для обучения energy_flow архитектуры
    """
    
    def __init__(self, config, teacher_provider: Optional[TeacherModelProvider] = None):
        super().__init__("SNLI", config)
        
        self.teacher_provider = teacher_provider
        self.snli_fraction = config.snli_fraction
        self.min_text_length = getattr(config, 'snli_min_text_length', 10)
        # Кэш загруженных данных (в памяти)
        self._snli_pairs = None
        self._embeddings_cache = None

        # Локальный файловый кэш
        self._cache_dir = Path(getattr(config, 'snli_cache_dir', 'cache/snli'))
        self._cache_enabled = getattr(config, 'snli_cache_enabled', True)
        self._cache_file = self._cache_dir / self._build_cache_filename()

        logger.log(DEBUG_INIT, f"SNLIProvider: fraction={self.snli_fraction}, "
                               f"min_length={self.min_text_length}")
    
    def is_available(self) -> bool:
        """Проверка доступности SNLI датасета"""
        try:
            # Проверяем подключение к HuggingFace datasets
            from datasets import load_dataset
            
            # Быстрая проверка без полной загрузки
            logger.info("🔍 Checking SNLI dataset availability...")
            dataset = load_dataset("snli", split="train[:1%]")  # Загружаем 1% для проверки
            
            if len(dataset) > 0:
                logger.info(f"✅ SNLI dataset available: {len(dataset)} samples in test load")
                return True
            else:
                logger.warning("❌ SNLI dataset appears empty")
                return False
                
        except Exception as e:
            logger.error(f"❌ SNLI dataset not available: {e}")
            return False
    
    def initialize(self) -> bool:
        """Инициализация провайдера (без полной загрузки данных)"""
        logger.log(DEBUG_INIT, "Initializing SNLI provider...")
        
        try:
            # Проверяем teacher provider если передан
            if self.teacher_provider and not self.teacher_provider.ensure_initialized():
                logger.warning("❌ Teacher provider not available, SNLI will work without embeddings")
            
            logger.info(f"✅ SNLI provider initialized (fraction={self.snli_fraction})")
            return True
            
        except Exception as e:
            logger.error(f"❌ SNLI provider initialization failed: {e}")
            return False
    
    def _load_snli_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Загрузка и фильтрация SNLI данных"""
        if self._snli_pairs is not None and max_samples is None:
            return self._snli_pairs  # Используем кэш если нет ограничений
        
        logger.info(f"📥 Loading SNLI dataset (fraction={self.snli_fraction})")
        # Попытка чтения из локального кэша (только когда нет max_samples ограничения)
        if self._cache_enabled and max_samples is None and self._cache_file.exists():
            try:
                logger.info(f"🗄️  Loading SNLI pairs from local cache: {self._cache_file}")
                with self._cache_file.open('r', encoding='utf-8') as f:
                    pairs = [json.loads(line) for line in f]
                self._snli_pairs = pairs
                logger.info(f"✅ Loaded {len(pairs):,} SNLI pairs from cache")
                return pairs
            except Exception as e:
                logger.warning(f"⚠️ Failed to load SNLI cache ({e}), rebuilding ...")

        try:
            dataset = load_dataset("snli")
            train_data = dataset["train"]
            total_size = len(train_data)
            target_size = int(total_size * self.snli_fraction)
            if max_samples:
                target_size = min(target_size, max_samples)
            logger.info(f"📊 SNLI total size: {total_size:,}, will use: {target_size:,}")

            indices = random.sample(range(total_size), target_size)
            pairs = []
            valid_labels = {0, 1, 2}
            for idx in indices:
                example = train_data[idx]
                if (
                    example["label"] in valid_labels and
                    example["premise"] and example["hypothesis"] and
                    len(example["premise"].strip()) >= self.min_text_length and
                    len(example["hypothesis"].strip()) >= self.min_text_length
                ):
                    pairs.append({
                        "input_text": example["premise"],
                        "target_text": example["hypothesis"],
                        "label": example["label"],
                        "snli_id": idx,
                        "source": "snli"
                    })

            # Лог статистики
            label_counts = {}
            for pair in pairs:
                label_counts[pair['label']] = label_counts.get(pair['label'], 0) + 1
            logger.info(f"✅ SNLI data loaded: {len(pairs):,} valid pairs")
            label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
            for lid, cnt in label_counts.items():
                logger.info(f"   {label_names.get(lid,'unknown')}: {cnt:,} ({cnt/len(pairs)*100:.1f}%)")

            # Сохраняем в локальный кэш (только полный набор без max_samples)
            if self._cache_enabled and max_samples is None:
                try:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    with self._cache_file.open('w', encoding='utf-8') as f:
                        for rec in pairs:
                            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    logger.info(f"💾 Cached SNLI pairs to {self._cache_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to write SNLI cache: {e}")
                self._snli_pairs = pairs
            return pairs
        except Exception as e:
            logger.error(f"❌ SNLI data loading failed: {e}")
            return []

    def _build_cache_filename(self) -> str:
        """Имя файла кэша зависит от ключевых параметров фильтрации"""
        # Пример: snli_frac0.2_min10.jsonl
        frac = f"{self.snli_fraction}".replace('.', 'p')
        return f"snli_frac{frac}_min{self.min_text_length}.jsonl"
    
    def get_text_pairs(self, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
        """Получить пары текстов из SNLI"""
        if not self.ensure_initialized():
            return []
        
        # Применяем ограничение из конфигурации
        effective_max = max_samples
        if self.config.max_samples_per_source:
            if effective_max:
                effective_max = min(effective_max, self.config.max_samples_per_source)
            else:
                effective_max = self.config.max_samples_per_source
        
        snli_data = self._load_snli_data(effective_max)
        
        # Конвертируем в формат (input, target)
        text_pairs = [(pair["input_text"], pair["target_text"]) for pair in snli_data]
        
        logger.debug(f"📝 SNLI text pairs: {len(text_pairs)} samples")
        return text_pairs
    
    def get_embeddings(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить эмбеддинги для SNLI пар"""
        if not self.ensure_initialized():
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
        
        if not self.teacher_provider:
            logger.warning("❌ No teacher provider available for embedding generation")
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
        
        # Получаем текстовые пары
        text_pairs = self.get_text_pairs(max_samples)
        if not text_pairs:
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
        
        # Разделяем на input и target тексты
        input_texts = [pair[0] for pair in text_pairs]
        target_texts = [pair[1] for pair in text_pairs]
        
        logger.info(f"🔄 Generating SNLI embeddings for {len(text_pairs)} pairs...")
        
        try:
            # Генерируем эмбеддинги через teacher model
            input_embeddings = self.teacher_provider.encode_texts(input_texts)
            target_embeddings = self.teacher_provider.encode_texts(target_texts)
            
            # Валидация
            if not self.validate_embeddings(input_embeddings, "SNLI_input"):
                logger.warning("❌ SNLI input embeddings validation failed")
            if not self.validate_embeddings(target_embeddings, "SNLI_target"):
                logger.warning("❌ SNLI target embeddings validation failed")
            
            # Нормализация
            input_embeddings = self.normalize_embeddings(input_embeddings)
            target_embeddings = self.normalize_embeddings(target_embeddings)
            
            logger.info(f"✅ SNLI embeddings generated: {input_embeddings.shape}")
            return input_embeddings, target_embeddings
            
        except Exception as e:
            logger.error(f"❌ SNLI embedding generation failed: {e}")
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
    
    def get_mixed_data(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Получить SNLI данные с дополнительными метаданными"""
        base_data = super().get_mixed_data(max_samples)
        
        # Добавляем SNLI-специфичную информацию
        if base_data['text_pairs']:
            snli_data = self._load_snli_data(max_samples)
            
            # Добавляем метаданные
            base_data.update({
                'snli_labels': [pair['label'] for pair in snli_data],
                'snli_ids': [pair['snli_id'] for pair in snli_data],
                'dataset_fraction': self.snli_fraction,
                'label_distribution': self._get_label_distribution(snli_data)
            })
        
        return base_data
    
    def _get_label_distribution(self, snli_data: List[Dict]) -> Dict[str, int]:
        """Получить распределение labels в данных"""
        label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
        distribution = {}
        
        for pair in snli_data:
            label_name = label_names.get(pair["label"], "unknown")
            distribution[label_name] = distribution.get(label_name, 0) + 1
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика для SNLI"""
        base_stats = super().get_statistics()
        
        if self._is_initialized:
            # Быстрая выборка для статистики
            sample_data = self._load_snli_data(max_samples=100)
            
            if sample_data:
                base_stats.update({
                    'dataset_fraction': self.snli_fraction,
                    'min_text_length': self.min_text_length,
                    'label_distribution': self._get_label_distribution(sample_data),
                    'estimated_total_pairs': int(len(sample_data) / min(100 / len(sample_data), 1)),
                    'teacher_provider_available': self.teacher_provider is not None
                })
        
        return base_stats


def create_snli_provider(config, teacher_provider: Optional[TeacherModelProvider] = None) -> SNLIProvider:
    """Фабричная функция для создания SNLIProvider"""
    return SNLIProvider(config, teacher_provider)