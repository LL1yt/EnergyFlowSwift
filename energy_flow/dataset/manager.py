"""
DatasetManager - центральный класс для управления датасетами
===========================================================

Объединяет все провайдеры данных и обеспечивает единый интерфейс для:
- Проверки наличия локальной модели учителя
- Загрузки и подготовки датасетов из различных источников
- Генерации эмбеддингов от модели-учителя
- Создания DataLoader для обучения
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import time

from .config import DatasetConfig
from .providers import (
    BaseDataProvider, 
    TeacherModelProvider, 
    SNLIProvider, 
    PrecomputedProvider,
    create_teacher_model_provider,
    create_snli_provider,
    create_precomputed_provider
)
from ..config import EnergyConfig
from ..utils.logging import get_logger, DEBUG_INIT, DEBUG_TRAINING

logger = get_logger(__name__)


class EnergyFlowDataset(Dataset):
    """
    Dataset класс для energy_flow архитектуры
    
    Обеспечивает унифицированный доступ к парам (input_text, target_text)
    и соответствующим эмбеддингам от модели-учителя
    """
    
    def __init__(self, text_pairs: List[Tuple[str, str]], 
                 input_embeddings: torch.Tensor, 
                 target_embeddings: torch.Tensor,
                 metadata: Optional[List[Dict]] = None):
        """
        Args:
            text_pairs: Список пар (input_text, target_text)
            input_embeddings: Входные эмбеддинги [N, embed_dim]
            target_embeddings: Целевые эмбеддинги [N, embed_dim]
            metadata: Дополнительные метаданные для каждой пары
        """
        assert len(text_pairs) == len(input_embeddings) == len(target_embeddings), \
            f"Size mismatch: {len(text_pairs)} texts vs {len(input_embeddings)} vs {len(target_embeddings)} embeddings"
        
        self.text_pairs = text_pairs
        self.input_embeddings = input_embeddings
        self.target_embeddings = target_embeddings
        self.metadata = metadata or [{}] * len(text_pairs)
        
    def __len__(self) -> int:
        return len(self.text_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'input_text': self.text_pairs[idx][0],
            'target_text': self.text_pairs[idx][1],
            'input_embedding': self.input_embeddings[idx],
            'target_embedding': self.target_embeddings[idx],
            'metadata': self.metadata[idx]
        }


class DatasetManager:
    """
    Центральный менеджер для управления датасетами energy_flow
    
    Функционал:
    - Проверка и загрузка модели-учителя
    - Управление провайдерами данных
    - Подготовка унифицированных датасетов
    - Создание DataLoader для обучения
    """
    
    def __init__(self, config: DatasetConfig, energy_config: Optional[EnergyConfig] = None):
        """
        Args:
            config: Конфигурация датасета
            energy_config: Основная конфигурация energy_flow (для интеграции)
        """
        self.config = config
        self.energy_config = energy_config
        
        # Обновляем config из energy_config если передан
        if energy_config:
            self.config.update_from_energy_config(energy_config)
        
        # Провайдеры данных
        self.teacher_provider: Optional[TeacherModelProvider] = None
        self.providers: Dict[str, BaseDataProvider] = {}
        
        # Кэш подготовленных данных
        self._prepared_dataset: Optional[EnergyFlowDataset] = None
        self._dataset_statistics: Optional[Dict[str, Any]] = None
        
        logger.log(DEBUG_INIT, f"DatasetManager initialized: sources={config.dataset_sources}")
    
    def ensure_teacher_model(self) -> bool:
        """
        Убеждается что модель-учитель готова к работе
        
        Returns:
            True если модель доступна и инициализирована
        """
        logger.info("🔍 Checking teacher model availability...")
        
        if self.teacher_provider is None:
            self.teacher_provider = create_teacher_model_provider(self.config)
        
        # Проверяем доступность
        if not self.teacher_provider.is_available():
            logger.info("📥 Teacher model not available locally, attempting download...")
            
            # Пытаемся загрузить модель
            if not self.teacher_provider.download_model_if_needed():
                logger.error("❌ Failed to download teacher model")
                return False
        
        # Инициализируем провайдер
        if not self.teacher_provider.ensure_initialized():
            logger.error("❌ Failed to initialize teacher model")
            return False
        
        logger.info(f"✅ Teacher model ready: {self.config.teacher_model}")
        return True
    
    def initialize_providers(self) -> Dict[str, bool]:
        """
        Инициализация всех провайдеров данных
        
        Returns:
            Словарь с результатами инициализации для каждого источника
        """
        logger.info(f"🔧 Initializing data providers: {self.config.dataset_sources}")
        
        results = {}
        
        for source in self.config.dataset_sources:
            try:
                if source == "precomputed":
                    provider = create_precomputed_provider(self.config)
                elif source == "snli":
                    provider = create_snli_provider(self.config, self.teacher_provider)
                else:
                    logger.warning(f"❌ Unknown data source: {source}")
                    results[source] = False
                    continue
                
                # Проверяем доступность и инициализируем
                if provider.is_available() and provider.ensure_initialized():
                    self.providers[source] = provider
                    results[source] = True
                    logger.info(f"✅ {source} provider ready")
                else:
                    logger.warning(f"⚠️ {source} provider not available")
                    results[source] = False
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize {source} provider: {e}")
                results[source] = False
        
        available_count = sum(results.values())
        logger.info(f"📊 Provider initialization: {available_count}/{len(self.config.dataset_sources)} available")
        
        return results
    
    def prepare_dataset(self, force_reload: bool = False) -> Optional[EnergyFlowDataset]:
        """
        Подготовка унифицированного датасета из всех источников
        
        Args:
            force_reload: Принудительная перезагрузка даже если есть кэш
            
        Returns:
            EnergyFlowDataset или None при ошибке
        """
        if self._prepared_dataset is not None and not force_reload:
            logger.info("📋 Using cached dataset")
            return self._prepared_dataset
        
        logger.info("🔄 Preparing unified dataset...")
        start_time = time.time()
        
        # Убеждаемся что teacher model готов
        if not self.ensure_teacher_model():
            logger.error("❌ Teacher model not available, cannot prepare dataset")
            return None
        
        # Инициализируем провайдеры
        provider_results = self.initialize_providers()
        available_providers = [name for name, available in provider_results.items() if available]
        
        if not available_providers:
            logger.error("❌ No data providers available")
            return None
        
        # Собираем данные из всех доступных провайдеров
        all_text_pairs = []
        all_input_embeddings = []
        all_target_embeddings = []
        all_metadata = []
        
        for provider_name in available_providers:
            provider = self.providers[provider_name]
            
            try:
                logger.info(f"📥 Loading data from {provider_name}...")
                
                # Получаем данные из провайдера
                provider_data = provider.get_mixed_data(self.config.max_samples_per_source)
                
                if provider_data['count'] > 0:
                    # Добавляем к общим данным
                    all_text_pairs.extend(provider_data['text_pairs'])
                    all_input_embeddings.append(provider_data['input_embeddings'])
                    all_target_embeddings.append(provider_data['target_embeddings'])
                    
                    # Метаданные с указанием источника
                    provider_metadata = [{'source': provider_name} for _ in range(provider_data['count'])]
                    all_metadata.extend(provider_metadata)
                    
                    logger.info(f"✅ {provider_name}: {provider_data['count']} samples loaded")
                else:
                    logger.warning(f"⚠️ {provider_name}: no data available")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load data from {provider_name}: {e}")
                continue
        
        if not all_text_pairs:
            logger.error("❌ No data loaded from any provider")
            return None
        
        # Объединяем эмбеддинги
        combined_input_embeddings = torch.cat(all_input_embeddings, dim=0)
        combined_target_embeddings = torch.cat(all_target_embeddings, dim=0)
        
        # Перемешиваем если включено
        if self.config.shuffle_data:
            indices = torch.randperm(len(all_text_pairs))
            all_text_pairs = [all_text_pairs[i] for i in indices]
            combined_input_embeddings = combined_input_embeddings[indices]
            combined_target_embeddings = combined_target_embeddings[indices]
            all_metadata = [all_metadata[i] for i in indices]
        
        # Создаем датасет
        dataset = EnergyFlowDataset(
            text_pairs=all_text_pairs,
            input_embeddings=combined_input_embeddings,
            target_embeddings=combined_target_embeddings,
            metadata=all_metadata
        )
        
        preparation_time = time.time() - start_time
        
        # Кэшируем результат
        self._prepared_dataset = dataset
        
        # Собираем статистику
        self._dataset_statistics = self._compute_dataset_statistics(dataset, available_providers, preparation_time)
        
        logger.info(f"✅ Dataset prepared: {len(dataset)} samples in {preparation_time:.2f}s")
        logger.log(DEBUG_TRAINING, f"Dataset embedding shapes: input={combined_input_embeddings.shape}, "
                                  f"target={combined_target_embeddings.shape}")
        
        return dataset
    
    def create_dataloader(self, batch_size: Optional[int] = None, 
                         shuffle: Optional[bool] = None,
                         num_workers: int = 0) -> Optional[DataLoader]:
        """
        Создание DataLoader для обучения
        
        Args:
            batch_size: Размер батча (по умолчанию из config)
            shuffle: Перемешивание (по умолчанию из config)
            num_workers: Количество worker процессов
            
        Returns:
            DataLoader или None при ошибке
        """
        dataset = self.prepare_dataset()
        if dataset is None:
            return None
        
        effective_batch_size = batch_size or self.config.batch_size
        effective_shuffle = shuffle if shuffle is not None else self.config.shuffle_data
        
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Для стабильности обучения
        )
        
        logger.info(f"📦 DataLoader created: batch_size={effective_batch_size}, "
                   f"shuffle={effective_shuffle}, batches={len(dataloader)}")
        
        return dataloader
    
    def get_teacher_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Получение эмбеддингов от модели-учителя для произвольных текстов
        
        Args:
            texts: Список текстов для кодирования
            
        Returns:
            Тензор эмбеддингов [len(texts), embed_dim]
        """
        if not self.ensure_teacher_model():
            raise RuntimeError("Teacher model not available")
        
        return self.teacher_provider.encode_texts(texts)
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Полная диагностика готовности системы
        
        Returns:
            Словарь с результатами проверки всех компонентов
        """
        logger.info("🔍 Running comprehensive setup validation...")
        
        validation_results = {
            'teacher_model': False,
            'providers': {},
            'dataset_preparation': False,
            'overall_status': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 1. Проверка teacher model
            if self.ensure_teacher_model():
                validation_results['teacher_model'] = True
                logger.info("✅ Teacher model validation passed")
            else:
                validation_results['errors'].append("Teacher model not available")
            
            # 2. Проверка провайдеров
            provider_results = self.initialize_providers()
            validation_results['providers'] = provider_results
            
            available_providers = sum(provider_results.values())
            if available_providers > 0:
                logger.info(f"✅ Data providers validation: {available_providers} available")
            else:
                validation_results['errors'].append("No data providers available")
            
            # 3. Проверка подготовки датасета
            if validation_results['teacher_model'] and available_providers > 0:
                dataset = self.prepare_dataset()
                if dataset is not None and len(dataset) > 0:
                    validation_results['dataset_preparation'] = True
                    logger.info(f"✅ Dataset preparation successful: {len(dataset)} samples")
                else:
                    validation_results['errors'].append("Dataset preparation failed")
            
            # 4. Общий статус
            validation_results['overall_status'] = (
                validation_results['teacher_model'] and
                available_providers > 0 and
                validation_results['dataset_preparation']
            )
            
            if validation_results['overall_status']:
                logger.info("🎉 Setup validation PASSED - system ready for training")
            else:
                logger.warning("⚠️ Setup validation FAILED - issues need to be resolved")
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            validation_results['errors'].append(f"Validation exception: {e}")
        
        return validation_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику по подготовленному датасету"""
        if self._dataset_statistics is None:
            self.prepare_dataset()  # Подготавливаем если еще не готово
        
        return self._dataset_statistics or {'error': 'No statistics available'}
    
    def _compute_dataset_statistics(self, dataset: EnergyFlowDataset, 
                                   providers: List[str], preparation_time: float) -> Dict[str, Any]:
        """Вычисление подробной статистики по датасету"""
        stats = {
            'total_samples': len(dataset),
            'providers_used': providers,
            'preparation_time_seconds': preparation_time,
            'embedding_dimension': dataset.input_embeddings.shape[1],
            'config': self.config.to_dict()
        }
        
        # Статистика по источникам данных
        source_counts = {}
        for metadata in dataset.metadata:
            source = metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        stats['source_distribution'] = source_counts
        
        # Статистика по длинам текстов
        input_lengths = [len(pair[0].split()) for pair in dataset.text_pairs[:1000]]  # Выборка
        target_lengths = [len(pair[1].split()) for pair in dataset.text_pairs[:1000]]
        
        if input_lengths:
            stats['text_statistics'] = {
                'avg_input_length': sum(input_lengths) / len(input_lengths),
                'avg_target_length': sum(target_lengths) / len(target_lengths),
                'max_input_length': max(input_lengths),
                'max_target_length': max(target_lengths)
            }
        
        # Статистика по эмбеддингам
        input_norms = torch.norm(dataset.input_embeddings[:1000], dim=1)  # Выборка
        target_norms = torch.norm(dataset.target_embeddings[:1000], dim=1)
        
        stats['embedding_statistics'] = {
            'input_norm_mean': input_norms.mean().item(),
            'input_norm_std': input_norms.std().item(),
            'target_norm_mean': target_norms.mean().item(),
            'target_norm_std': target_norms.std().item()
        }
        
        return stats
    
    def clear_cache(self):
        """Очистка всех кэшей"""
        self._prepared_dataset = None
        self._dataset_statistics = None
        
        if self.teacher_provider:
            self.teacher_provider.clear_cache()
        
        for provider in self.providers.values():
            if hasattr(provider, 'clear_cache'):
                provider.clear_cache()
        
        logger.info("🧹 All caches cleared")


def create_dataset_manager(config: DatasetConfig, 
                          energy_config: Optional[EnergyConfig] = None) -> DatasetManager:
    """
    Фабричная функция для создания DatasetManager
    
    Args:
        config: Конфигурация датасета
        energy_config: Основная конфигурация energy_flow
        
    Returns:
        Настроенный DatasetManager
    """
    return DatasetManager(config, energy_config)