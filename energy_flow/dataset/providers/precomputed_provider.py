"""
PrecomputedProvider - провайдер для готовых эмбеддингов
======================================================

Адаптация legacy precomputed_embedding_loader.py под новую архитектуру:
- Загрузка готовых .pt файлов с эмбеддингами
- Поддержка различных форматов данных
- Кэширование и валидация
"""

import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from .base_provider import BaseDataProvider
from ...utils.logging import get_logger, DEBUG_INIT

logger = get_logger(__name__)


class PrecomputedProvider(BaseDataProvider):
    """
    Провайдер для загрузки готовых эмбеддингов из .pt файлов
    
    Поддерживает форматы:
    - Файлы созданные generate_snli_embedding_dataset.py
    - Unified dataset loader файлы
    - Простые тензоры эмбеддингов
    """
    
    def __init__(self, config):
        super().__init__("Precomputed", config)
        
        self.embeddings_dir = config.get_absolute_embeddings_dir()
        self.cache_dir = config.get_absolute_cache_dir()
        
        # Кэш загруженных файлов
        self._loaded_files = {}
        self._file_metadata = {}
        
        logger.log(DEBUG_INIT, f"PrecomputedProvider: embeddings_dir={self.embeddings_dir}")
    
    def is_available(self) -> bool:
        """Проверка наличия готовых файлов эмбеддингов"""
        try:
            # Ищем .pt файлы в директории эмбеддингов
            embedding_files = list(self.embeddings_dir.glob("*.pt"))
            
            # Также проверяем кэш директорию
            cache_files = []
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.glob("*.pt"))
            
            total_files = len(embedding_files) + len(cache_files)
            
            if total_files > 0:
                logger.info(f"✅ Found {total_files} precomputed files: "
                           f"{len(embedding_files)} in embeddings, {len(cache_files)} in cache")
                return True
            else:
                logger.info("📁 No precomputed embedding files found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Precomputed availability check failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """Инициализация провайдера"""
        logger.log(DEBUG_INIT, "Initializing precomputed provider...")
        
        try:
            # Сканируем доступные файлы и их метаданные
            self._scan_available_files()
            
            if self._file_metadata:
                logger.info(f"✅ Precomputed provider initialized: {len(self._file_metadata)} files available")
                return True
            else:
                logger.warning("⚠️ No valid precomputed files found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Precomputed provider initialization failed: {e}")
            return False
    
    def _scan_available_files(self):
        """Сканирование доступных файлов и извлечение метаданных"""
        self._file_metadata.clear()
        
        # Сканируем основную директорию эмбеддингов
        if self.embeddings_dir.exists():
            self._scan_directory(self.embeddings_dir, "embeddings")
        
        # Сканируем кэш директорию
        if self.cache_dir.exists():
            self._scan_directory(self.cache_dir, "cache")
        
        logger.info(f"📊 Scanned files: {len(self._file_metadata)} valid precomputed files")
    
    def _scan_directory(self, directory: Path, source_type: str):
        """Сканирование конкретной директории"""
        for file_path in directory.glob("*.pt"):
            try:
                # Быстрая загрузка метаданных без полной загрузки данных
                metadata = self._extract_file_metadata(file_path, source_type)
                if metadata:
                    self._file_metadata[str(file_path)] = metadata
                    
            except Exception as e:
                logger.warning(f"❌ Failed to scan {file_path.name}: {e}")
    
    def _extract_file_metadata(self, file_path: Path, source_type: str) -> Optional[Dict[str, Any]]:
        """Извлечение метаданных из файла без полной загрузки"""
        try:
            # Загружаем только для анализа структуры
            data = torch.load(file_path, map_location='cpu')
            
            metadata = {
                'file_path': str(file_path),
                'filename': file_path.name,
                'source_type': source_type,
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'format': 'unknown'
            }
            
            # Определяем формат файла
            if isinstance(data, dict):
                # Формат generate_snli_embedding_dataset.py
                if 'question_embeddings' in data and 'answer_embeddings' in data:
                    metadata.update({
                        'format': 'snli_dataset',
                        'size': data.get('size', 'unknown'),
                        'teacher_model': data.get('teacher_model', 'unknown'),
                        'embedding_dim': data['question_embeddings'].shape[1] if data['question_embeddings'].dim() == 2 else None,
                        'sample_count': len(data['question_embeddings'])
                    })
                
                # Формат unified_dataset_loader.py  
                elif 'input_embeddings' in data and 'target_embeddings' in data:
                    metadata.update({
                        'format': 'unified_dataset',
                        'embedding_dim': data['input_embeddings'].shape[1] if data['input_embeddings'].dim() == 2 else None,
                        'sample_count': len(data['input_embeddings'])
                    })
                
                # Другие dict форматы
                else:
                    metadata['format'] = 'dict_format'
                    metadata['keys'] = list(data.keys())
            
            elif isinstance(data, torch.Tensor):
                # Простой тензор
                metadata.update({
                    'format': 'tensor',
                    'tensor_shape': list(data.shape),
                    'embedding_dim': data.shape[1] if data.dim() == 2 else None,
                    'sample_count': data.shape[0] if data.dim() >= 1 else 1
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"❌ Metadata extraction failed for {file_path.name}: {e}")
            return None
    
    def _load_file_data(self, file_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Загрузка данных из файла с кэшированием"""
        if file_path in self._loaded_files:
            return self._loaded_files[file_path]
        
        try:
            logger.info(f"📥 Loading precomputed file: {Path(file_path).name}")
            
            data = torch.load(file_path, map_location=self.device)
            metadata = self._file_metadata.get(file_path, {})
            format_type = metadata.get('format', 'unknown')
            
            # Конвертируем в единый формат
            converted_data = None
            
            if format_type == 'snli_dataset':
                # Формат generate_snli_embedding_dataset.py
                converted_data = {
                    'input_embeddings': data['question_embeddings'],
                    'target_embeddings': data['answer_embeddings'],
                    'metadata': {
                        'teacher_model': data.get('teacher_model'),
                        'dataset_info': data.get('dataset_info', {}),
                        'sample_pairs': data.get('sample_pairs', [])
                    }
                }
                
            elif format_type == 'unified_dataset':
                # Формат unified_dataset_loader.py
                converted_data = {
                    'input_embeddings': data['input_embeddings'],
                    'target_embeddings': data['target_embeddings'],
                    'metadata': data.get('metadata', {})
                }
                
            elif format_type == 'tensor':
                # Простой тензор - создаем копии для input и target
                converted_data = {
                    'input_embeddings': data,
                    'target_embeddings': data.clone(),  # Используем копию
                    'metadata': {'source': 'tensor_file'}
                }
                
            elif isinstance(data, dict):
                # Пытаемся извлечь эмбеддинги из произвольного dict
                input_emb = self._extract_embeddings_from_dict(data, 'input')
                target_emb = self._extract_embeddings_from_dict(data, 'target')
                
                if input_emb is not None and target_emb is not None:
                    converted_data = {
                        'input_embeddings': input_emb,
                        'target_embeddings': target_emb,
                        'metadata': data.get('metadata', {})
                    }
            
            if converted_data:
                # Валидация данных
                if self._validate_loaded_data(converted_data):
                    self._loaded_files[file_path] = converted_data
                    logger.info(f"✅ Loaded {Path(file_path).name}: "
                               f"{converted_data['input_embeddings'].shape[0]} samples")
                    return converted_data
                else:
                    logger.warning(f"❌ Data validation failed for {Path(file_path).name}")
            else:
                logger.warning(f"❌ Unknown format for {Path(file_path).name}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load {Path(file_path).name}: {e}")
            return None
    
    def _extract_embeddings_from_dict(self, data: Dict, emb_type: str) -> Optional[torch.Tensor]:
        """Извлечение эмбеддингов из dict структуры"""
        # Возможные ключи для input эмбеддингов
        input_keys = ['input_embeddings', 'question_embeddings', 'premise_embeddings', 'embeddings']
        target_keys = ['target_embeddings', 'answer_embeddings', 'hypothesis_embeddings', 'labels']
        
        keys_to_check = input_keys if emb_type == 'input' else target_keys
        
        for key in keys_to_check:
            if key in data and isinstance(data[key], torch.Tensor):
                tensor = data[key]
                if tensor.dim() == 2:  # [samples, embedding_dim]
                    return tensor
        
        return None
    
    def _validate_loaded_data(self, data: Dict[str, torch.Tensor]) -> bool:
        """Валидация загруженных данных"""
        try:
            input_emb = data['input_embeddings']
            target_emb = data['target_embeddings']
            
            # Проверка размерностей
            if input_emb.dim() != 2 or target_emb.dim() != 2:
                return False
            
            # Проверка совпадения количества образцов
            if input_emb.shape[0] != target_emb.shape[0]:
                return False
            
            # Проверка размерности эмбеддингов
            if input_emb.shape[1] != target_emb.shape[1]:
                return False
            
            # Валидация через базовый класс
            return (self.validate_embeddings(input_emb, "precomputed_input") and 
                   self.validate_embeddings(target_emb, "precomputed_target"))
            
        except Exception as e:
            logger.error(f"❌ Data validation error: {e}")
            return False
    
    def get_text_pairs(self, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
        """Получить текстовые пары (если доступны в метаданных)"""
        if not self.ensure_initialized():
            return []
        
        text_pairs = []
        
        # Ищем файлы с текстовыми парами в метаданных
        for file_path, metadata in self._file_metadata.items():
            if metadata.get('format') == 'snli_dataset':
                # Загружаем файл для получения sample_pairs
                data = self._load_file_data(file_path)
                if data and 'metadata' in data:
                    sample_pairs = data['metadata'].get('sample_pairs', [])
                    for pair in sample_pairs:
                        if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                            text_pairs.append((pair['question'], pair['answer']))
        
        # Применяем ограничения
        if max_samples:
            text_pairs = text_pairs[:max_samples]
        
        if self.config.max_samples_per_source:
            text_pairs = text_pairs[:self.config.max_samples_per_source]
        
        logger.debug(f"📝 Precomputed text pairs: {len(text_pairs)} samples")
        return text_pairs
    
    def get_embeddings(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получить эмбеддинги из всех доступных файлов"""
        if not self.ensure_initialized():
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
        
        all_input_embeddings = []
        all_target_embeddings = []
        
        # Загружаема эмбеддинги из всех файлов
        for file_path, metadata in self._file_metadata.items():
            data = self._load_file_data(file_path)
            if data:
                input_emb = data['input_embeddings']
                target_emb = data['target_embeddings']
                
                # Нормализация если включена
                input_emb = self.normalize_embeddings(input_emb)
                target_emb = self.normalize_embeddings(target_emb)
                
                all_input_embeddings.append(input_emb)
                all_target_embeddings.append(target_emb)
        
        if not all_input_embeddings:
            logger.warning("❌ No valid embeddings found in precomputed files")
            empty_tensor = torch.empty(0, 768)
            return empty_tensor, empty_tensor
        
        # Объединяем все эмбеддинги
        input_embeddings = torch.cat(all_input_embeddings, dim=0)
        target_embeddings = torch.cat(all_target_embeddings, dim=0)
        
        # Применяем ограничения
        if max_samples:
            input_embeddings = input_embeddings[:max_samples]
            target_embeddings = target_embeddings[:max_samples]
        
        if self.config.max_samples_per_source:
            limit = self.config.max_samples_per_source
            input_embeddings = input_embeddings[:limit]
            target_embeddings = target_embeddings[:limit]
        
        # Перемешиваем если включено
        if self.config.shuffle_data and len(input_embeddings) > 1:
            indices = torch.randperm(len(input_embeddings))
            input_embeddings = input_embeddings[indices]
            target_embeddings = target_embeddings[indices]
        
        logger.info(f"✅ Precomputed embeddings loaded: {input_embeddings.shape}")
        return input_embeddings, target_embeddings
    
    def get_file_list(self) -> List[Dict[str, Any]]:
        """Получить список доступных файлов с метаданными"""
        if not self.ensure_initialized():
            return []
        
        return list(self._file_metadata.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика для precomputed provider"""
        base_stats = super().get_statistics()
        
        if self._is_initialized:
            total_samples = 0
            file_formats = {}
            
            for metadata in self._file_metadata.values():
                samples = metadata.get('sample_count', 0)
                total_samples += samples
                
                format_type = metadata.get('format', 'unknown')
                file_formats[format_type] = file_formats.get(format_type, 0) + 1
            
            base_stats.update({
                'total_files': len(self._file_metadata),
                'total_samples': total_samples,
                'file_formats': file_formats,
                'loaded_files_count': len(self._loaded_files),
                'embeddings_dir': str(self.embeddings_dir),
                'cache_dir': str(self.cache_dir)
            })
        
        return base_stats


def create_precomputed_provider(config) -> PrecomputedProvider:
    """Фабричная функция для создания PrecomputedProvider"""
    return PrecomputedProvider(config)