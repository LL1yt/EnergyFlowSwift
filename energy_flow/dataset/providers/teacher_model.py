"""
TeacherModelProvider - управление моделью-учителем (DistilBERT)
==============================================================

Оптимизированный провайдер для управления моделью-учителем:
- Автоматическая проверка локальной модели
- Загрузка из HuggingFace при необходимости  
- GPU оптимизация и кэширование эмбеддингов
- Интеграция с legacy download_distilbert.py
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import os
import time

from .base_provider import BaseDataProvider
from ...utils.logging import get_logger, DEBUG_INIT, DEBUG_MEMORY

logger = get_logger(__name__)


class TeacherModelProvider(BaseDataProvider):
    """
    Провайдер для управления моделью-учителем (DistilBERT)
    
    Обеспечивает:
    - Проверку наличия локальной модели
    - Автоматическую загрузку при необходимости
    - Генерацию эмбеддингов для текстов
    - Кэширование для оптимизации производительности
    """
    
    def __init__(self, config):
        super().__init__("TeacherModel", config)
        
        self.model_name = config.teacher_model
        self.local_model_path = config.get_absolute_local_model_path()
        self.use_local_model = config.use_local_model
        
        self.tokenizer = None
        self.model = None
        self.embedding_cache = {}  # Простой cache для повторяющихся текстов
        
        logger.log(DEBUG_INIT, f"TeacherModelProvider: model={self.model_name}, "
                              f"local_path={self.local_model_path}, use_local={self.use_local_model}")
    
    def is_available(self) -> bool:
        """Проверка доступности модели"""
        try:
            # Сначала проверяем локальную модель если включена
            if self.use_local_model and self._check_local_model():
                logger.info(f"✅ Local model available: {self.local_model_path}")
                return True
            
            # Проверяем доступность в HuggingFace
            logger.info(f"🌐 Checking HuggingFace availability for {self.model_name}")
            return self._check_huggingface_model()
            
        except Exception as e:
            logger.error(f"❌ Model availability check failed: {e}")
            return False
    
    def _check_local_model(self) -> bool:
        """Проверка локальной модели"""
        if not self.local_model_path.exists():
            logger.info(f"📁 Local model path not found: {self.local_model_path}")
            return False
        
        # Проверяем наличие ключевых файлов
        required_files = ["config.json", "tokenizer.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]  # Один из этих файлов должен быть
        missing_files = []
        
        # Проверяем обязательные файлы
        for file in required_files:
            file_path = self.local_model_path / file
            if not file_path.exists():
                missing_files.append(file)
        
        # Проверяем что есть хотя бы один файл модели
        model_file_exists = any((self.local_model_path / model_file).exists() for model_file in model_files)
        if not model_file_exists:
            missing_files.append("model file (pytorch_model.bin or model.safetensors)")
        
        if missing_files:
            logger.warning(f"📁 Local model incomplete, missing: {missing_files}")
            return False
        
        logger.info(f"✅ Local model complete: {self.local_model_path}")
        return True
    
    def _check_huggingface_model(self) -> bool:
        """Проверка доступности модели в HuggingFace"""
        try:
            # Простая проверка без загрузки
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name)
            return config is not None
        except Exception as e:
            logger.warning(f"🌐 HuggingFace model check failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """Инициализация модели и токенизатора"""
        logger.log(DEBUG_INIT, f"Initializing teacher model: {self.model_name}")
        
        try:
            # Определяем источник модели
            model_source = str(self.local_model_path) if (self.use_local_model and self._check_local_model()) else self.model_name
            
            logger.info(f"🔄 Loading model from: {model_source}")
            start_time = time.time()
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели
            self.model = AutoModel.from_pretrained(model_source)
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            
            # Подсчет параметров
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"✅ Teacher model loaded successfully:")
            logger.info(f"   Source: {model_source}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Load time: {load_time:.2f}s")
            logger.info(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
            logger.log(DEBUG_MEMORY, f"Model memory: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize teacher model: {e}")
            return False
    
    def download_model_if_needed(self) -> bool:
        """
        Загрузка модели в локальный кэш если необходимо
        
        Интегрируется с legacy download_distilbert.py логикой
        """
        if not self.use_local_model:
            return True  # Локальная модель не требуется
        
        if self._check_local_model():
            return True  # Модель уже есть
        
        logger.info(f"📥 Downloading model to local cache: {self.local_model_path}")
        
        try:
            # Создаем директорию
            self.local_model_path.mkdir(parents=True, exist_ok=True)
            
            # Загружаем и сохраняем токенизатор
            logger.info("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.local_model_path)
            
            # Загружаем и сохраняем модель
            logger.info("📥 Downloading model...")
            model = AutoModel.from_pretrained(self.model_name)
            model.save_pretrained(self.local_model_path)
            
            # Проверяем что все сохранилось
            if self._check_local_model():
                file_size_mb = sum(f.stat().st_size for f in self.local_model_path.rglob('*') if f.is_file()) / 1024 / 1024
                logger.info(f"✅ Model downloaded successfully: {file_size_mb:.1f} MB")
                
                # Логируем что именно было сохранено
                saved_files = [f.name for f in self.local_model_path.iterdir() if f.is_file()]
                logger.info(f"   Files saved: {', '.join(saved_files)}")
                return True
            else:
                logger.error("❌ Model download verification failed")
                # Покажем что реально есть для отладки
                if self.local_model_path.exists():
                    actual_files = [f.name for f in self.local_model_path.iterdir() if f.is_file()]
                    logger.error(f"   Found files: {actual_files}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model download failed: {e}")
            return False
    
    def encode_texts(self, texts: List[str], use_cache: bool = True) -> torch.Tensor:
        """
        Кодирование списка текстов в эмбеддинги
        
        Args:
            texts: Список текстов
            use_cache: Использовать кэш для повторяющихся текстов
            
        Returns:
            Тензор эмбеддингов [len(texts), embedding_dim]
        """
        if not self.ensure_initialized():
            raise RuntimeError("Teacher model not initialized")
        
        if not texts:
            raise ValueError("Empty text list")
        
        # Проверка кэша
        embeddings = []
        texts_to_process = []
        cache_hits = 0
        
        for text in texts:
            if use_cache and self.config.embedding_cache_enabled and text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
                cache_hits += 1
            else:
                embeddings.append(None)
                texts_to_process.append((len(embeddings) - 1, text))
        
        # Обрабатываем некэшированные тексты
        if texts_to_process:
            logger.log(DEBUG_MEMORY, f"Processing {len(texts_to_process)} texts, cache hits: {cache_hits}")
            
            # Батчевая обработка для эффективности
            batch_size = self.config.cache_batch_size
            for i in range(0, len(texts_to_process), batch_size):
                batch_items = texts_to_process[i:i + batch_size]
                batch_texts = [item[1] for item in batch_items]
                
                # Генерируем эмбеддинги для батча
                batch_embeddings = self._generate_embeddings_batch(batch_texts)
                
                # Сохраняем в результат и кэш
                for j, (orig_idx, text) in enumerate(batch_items):
                    embedding = batch_embeddings[j]
                    embeddings[orig_idx] = embedding
                    
                    # Кэшируем если включено
                    if use_cache and self.config.embedding_cache_enabled:
                        self.embedding_cache[text] = embedding
        
        # Конвертируем в тензор
        result = torch.stack(embeddings)
        
        # Валидация и нормализация
        if not self.validate_embeddings(result, "teacher_embeddings"):
            logger.warning("❌ Generated embeddings failed validation")
        
        result = self.normalize_embeddings(result)
        
        logger.debug(f"✅ Generated embeddings: {result.shape}, cache size: {len(self.embedding_cache)}")
        return result
    
    def _generate_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """Генерация эмбеддингов для батча текстов"""
        # Токенизация
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Генерация эмбеддингов
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
            
            # Mean pooling с учетом attention mask
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return embeddings.cpu()
    
    def clear_cache(self):
        """Очистка кэша эмбеддингов"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"🧹 Embedding cache cleared: {cache_size} entries removed")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Информация о кэше эмбеддингов"""
        if not self.embedding_cache:
            return {'size': 0, 'memory_mb': 0}
        
        # Приблизительная оценка памяти
        sample_embedding = next(iter(self.embedding_cache.values()))
        embedding_size_bytes = sample_embedding.numel() * 4  # float32
        total_memory_mb = len(self.embedding_cache) * embedding_size_bytes / 1024 / 1024
        
        return {
            'size': len(self.embedding_cache),
            'memory_mb': total_memory_mb,
            'embedding_dim': sample_embedding.shape[0] if sample_embedding.dim() == 1 else sample_embedding.shape[1]
        }
    
    # Методы BaseDataProvider (не используются напрямую для TeacherModel)
    def get_text_pairs(self, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
        """TeacherModel не предоставляет текстовые пары"""
        return []
    
    def get_embeddings(self, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """TeacherModel не предоставляет готовые эмбеддинги"""
        empty_tensor = torch.empty(0, 768)  # DistilBERT dimension
        return empty_tensor, empty_tensor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика для TeacherModel"""
        base_stats = super().get_statistics()
        
        if self._is_initialized:
            cache_info = self.get_cache_info()
            base_stats.update({
                'model_name': self.model_name,
                'local_model_available': self._check_local_model(),
                'cache_info': cache_info,
                'embedding_dimension': 768 if self.model else None
            })
        
        return base_stats


def create_teacher_model_provider(config) -> TeacherModelProvider:
    """Фабричная функция для создания TeacherModelProvider"""
    return TeacherModelProvider(config)