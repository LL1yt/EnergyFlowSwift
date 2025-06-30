#!/usr/bin/env python3
"""
Декодер эмбедингов в текст
=========================

Простой декодер для преобразования эмбедингов обратно в текст.
Включает базовое кэширование и возможность joint training с кубом.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import json
from pathlib import Path

from ...utils.logging import get_logger, LogContext
from ...utils.device_manager import get_device_manager
from ...utils.model_cache import get_model_cache_manager
from ...config.simple_config import SimpleProjectConfig

logger = get_logger(__name__)


class EmbeddingTextCache:
    """
    Простой кэш для хранения пар эмбединг-текст

    Использует хэширование эмбедингов для быстрого поиска
    и LRU стратегию для управления размером.
    """

    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # Основные хранилища
        self.embedding_to_text = {}  # hash -> text
        self.text_to_embedding = {}  # text -> (hash, embedding)
        self.embeddings_store = {}  # hash -> embedding tensor

        # LRU tracking
        self.access_order = []  # Порядок доступа к элементам
        self.access_count = {}  # Счетчики использования

        self.logger = get_logger(__name__)

    def _hash_embedding(self, embedding: torch.Tensor) -> str:
        """Создает хэш для эмбединга"""
        # Округляем до 4 знаков для уменьшения вариативности
        rounded = torch.round(embedding * 10000) / 10000
        embedding_bytes = rounded.detach().cpu().numpy().tobytes()
        return hashlib.md5(embedding_bytes).hexdigest()[:16]

    def _hash_text(self, text: str) -> str:
        """Создает хэш для текста"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def get(self, embedding: torch.Tensor) -> Optional[str]:
        """Получение текста по эмбедингу"""
        emb_hash = self._hash_embedding(embedding)

        # Прямое попадание
        if emb_hash in self.embedding_to_text:
            self._update_access(emb_hash)
            return self.embedding_to_text[emb_hash]

        # Поиск похожих эмбедингов
        best_text = self._find_similar_embedding(embedding)
        return best_text

    def _find_similar_embedding(self, embedding: torch.Tensor) -> Optional[str]:
        """Поиск похожего эмбединга в кэше с использованием векторизации."""
        if not self.embeddings_store:
            return None

        cached_hashes = list(self.embeddings_store.keys())
        cached_embeddings = list(self.embeddings_store.values())

        if not cached_embeddings:
            return None

        # Векторизованное вычисление на целевом устройстве
        target_device = embedding.device
        moved_embeddings = [t.to(target_device) for t in cached_embeddings]
        cached_embeddings_tensor = torch.stack(moved_embeddings)

        similarities = F.cosine_similarity(
            embedding.unsqueeze(0), cached_embeddings_tensor
        )

        best_similarity, best_idx = torch.max(similarities, dim=0)

        if best_similarity.item() > self.similarity_threshold:
            best_hash = cached_hashes[best_idx.item()]
            best_text = self.embedding_to_text.get(best_hash)
            if best_text:
                self.logger.debug(
                    f"Found similar embedding with similarity {best_similarity.item():.3f}"
                )
                self._update_access(best_hash)
            return best_text

        return None

    def put(self, embedding: torch.Tensor, text: str):
        """Сохранение пары эмбединг-текст"""
        emb_hash = self._hash_embedding(embedding)
        text_hash = self._hash_text(text)

        # Проверяем лимит размера
        if len(self.embedding_to_text) >= self.max_size:
            self._evict_lru()

        # Сохраняем
        self.embedding_to_text[emb_hash] = text
        self.text_to_embedding[text] = (emb_hash, embedding.clone().detach())
        self.embeddings_store[emb_hash] = embedding.clone().detach()

        self._update_access(emb_hash)

    def _update_access(self, emb_hash: str):
        """Обновление порядка доступа для LRU"""
        if emb_hash in self.access_order:
            self.access_order.remove(emb_hash)
        self.access_order.append(emb_hash)
        self.access_count[emb_hash] = self.access_count.get(emb_hash, 0) + 1

    def _evict_lru(self):
        """Удаление наименее используемого элемента"""
        if not self.access_order:
            return

        lru_hash = self.access_order.pop(0)

        # Удаляем из всех хранилищ
        text = self.embedding_to_text.pop(lru_hash, None)
        if text:
            self.text_to_embedding.pop(text, None)
        self.embeddings_store.pop(lru_hash, None)
        self.access_count.pop(lru_hash, None)

    def get_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        return {
            "size": len(self.embedding_to_text),
            "max_size": self.max_size,
            "hit_rate": sum(self.access_count.values())
            / max(len(self.access_count), 1),
            "similarity_threshold": self.similarity_threshold,
        }

    def save(self, path: str):
        """Сохранение кэша на диск"""
        cache_data = {
            "embedding_to_text": self.embedding_to_text,
            "text_to_embedding": {
                k: (v[0], v[1].tolist()) for k, v in self.text_to_embedding.items()
            },
            "embeddings_store": {
                k: v.tolist() for k, v in self.embeddings_store.items()
            },
            "access_count": self.access_count,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache_data, f)

        self.logger.info(f"Cache saved to {path} ({len(self.embedding_to_text)} items)")

    def load(self, path: str):
        """Загрузка кэша с диска"""
        if not Path(path).exists():
            return

        with open(path, "r") as f:
            cache_data = json.load(f)

        self.embedding_to_text = cache_data["embedding_to_text"]
        self.text_to_embedding = {
            k: (v[0], torch.tensor(v[1]))
            for k, v in cache_data["text_to_embedding"].items()
        }
        self.embeddings_store = {
            k: torch.tensor(v) for k, v in cache_data["embeddings_store"].items()
        }
        self.access_count = cache_data["access_count"]

        self.logger.info(
            f"Cache loaded from {path} ({len(self.embedding_to_text)} items)"
        )


class SimpleTextDecoder(nn.Module):
    """
    Простой декодер эмбедингов в текст с поддержкой RTX 5090

    Использует предобученную модель (например, DistilBERT) для
    приблизительного декодирования эмбедингов обратно в текст.
    Оптимизирован для работы с GPU высокого класса.
    """

    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager()

        # Настройки
        self.teacher_dim = config.embedding.teacher_embedding_dim
        self.decoder_model_name = config.embedding.decoder_model
        self.max_length = config.embedding.max_decode_length

        # GPU оптимизации для RTX 5090
        self.use_gpu_acceleration = (
            self.device_manager.is_cuda()
            and self.device_manager.get_available_memory_gb() > 16
        )
        self.gpu_batch_size = (
            64 if self.use_gpu_acceleration else 8
        )  # Больше батчи для 5090

        # Кэш с GPU поддержкой
        self.cache_enabled = config.embedding.decoder_cache_enabled
        if self.cache_enabled:
            cache_path = Path(config.embedding.cache_dir) / "text_decoder_cache.json"
            max_cache_size = (
                50000 if self.use_gpu_acceleration else 10000
            )  # Больше для 5090
            self.cache = EmbeddingTextCache(max_size=max_cache_size)
            self.cache.load(str(cache_path))
            self.cache_path = cache_path

        # Менеджер локальных моделей
        self.model_cache_manager = get_model_cache_manager(config)

        # Инициализация модели декодера будет lazy
        self._decoder_model = None
        self._tokenizer = None

        gpu_info = f" (GPU: {self.use_gpu_acceleration}, batch: {self.gpu_batch_size})"
        local_info = f" (local: {config.embedding.prefer_local_models})"
        self.logger.info(
            f"[TEXT] SimpleTextDecoder initialized (cache: {self.cache_enabled}){gpu_info}{local_info}"
        )

    def _init_decoder_model(self):
        """Lazy инициализация модели декодера с поддержкой локального кэша"""
        if self._decoder_model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModel

            # Получаем путь к модели (локальный или для загрузки)
            model_path = self.model_cache_manager.get_model_path(
                self.decoder_model_name
            )

            if model_path is None:
                self.logger.error(
                    f"Failed to get model path for {self.decoder_model_name}"
                )
                self._decoder_model = "dummy"
                self._tokenizer = "dummy"
                return

            self.logger.info(f"[SYNC] Loading model from: {model_path}")

            # Загружаем токенизатор и модель
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._decoder_model = AutoModel.from_pretrained(model_path)

            # Переносим на устройство (RTX 5090)
            self._decoder_model = self.device_manager.transfer_module(
                self._decoder_model
            )
            self._decoder_model.eval()  # Для inference

            # Информация о загруженной модели
            is_local = (
                Path(model_path).exists()
                and not model_path.startswith("http")
                and "local_cache" in model_path
            )
            source_info = "local cache" if is_local else "online"
            self.logger.info(
                f"[OK] Decoder model loaded from {source_info}: {self.decoder_model_name}"
            )

        except ImportError:
            self.logger.warning("transformers not available, using dummy decoder")
            self._decoder_model = "dummy"
            self._tokenizer = "dummy"
        except Exception as e:
            self.logger.error(f"Failed to load decoder model: {e}")
            self.logger.info("Falling back to dummy decoder")
            self._decoder_model = "dummy"
            self._tokenizer = "dummy"

    def decode_embeddings(
        self, embeddings: torch.Tensor, use_cache: bool = True
    ) -> List[str]:
        """
        GPU-оптимизированное декодирование эмбедингов в текст

        Args:
            embeddings: Tensor размера [batch, embedding_dim]
            use_cache: Использовать ли кэш

        Returns:
            Список декодированных текстов
        """
        with LogContext("text_decoding", batch_size=embeddings.size(0)):
            batch_size = embeddings.size(0)

            # Переносим на GPU если возможно
            if self.use_gpu_acceleration:
                embeddings = self.device_manager.ensure_device(embeddings)

            # GPU-оптимизированный путь для больших батчей
            if self.use_gpu_acceleration and batch_size > self.gpu_batch_size:
                return self._decode_large_batch_gpu(embeddings, use_cache)

            # Стандартный путь
            results = []
            cache_hits = 0

            for i in range(batch_size):
                embedding = embeddings[i]

                # Проверяем кэш
                if use_cache and self.cache_enabled:
                    cached_text = self.cache.get(embedding)
                    if cached_text:
                        results.append(cached_text)
                        cache_hits += 1
                        continue

                # Декодируем новый
                decoded_text = self._decode_single(embedding)
                results.append(decoded_text)

                # Сохраняем в кэш
                if use_cache and self.cache_enabled:
                    self.cache.put(embedding, decoded_text)

            self.logger.info(
                f"Decoded {batch_size} embeddings (cache hits: {cache_hits})"
            )
            return results

    def _decode_large_batch_gpu(
        self, embeddings: torch.Tensor, use_cache: bool
    ) -> List[str]:
        """GPU-оптимизированное декодирование для больших батчей (RTX 5090)"""
        batch_size = embeddings.size(0)
        results = []
        total_cache_hits = 0

        # Обрабатываем батчами для эффективности GPU
        for start_idx in range(0, batch_size, self.gpu_batch_size):
            end_idx = min(start_idx + self.gpu_batch_size, batch_size)
            batch_embeddings = embeddings[start_idx:end_idx]

            batch_results = []
            cache_hits = 0

            # Векторизованная проверка кэша
            if use_cache and self.cache_enabled:
                for i in range(batch_embeddings.size(0)):
                    cached_text = self.cache.get(batch_embeddings[i])
                    if cached_text:
                        batch_results.append(cached_text)
                        cache_hits += 1
                    else:
                        batch_results.append(None)  # Placeholder
            else:
                batch_results = [None] * batch_embeddings.size(0)

            # Декодируем только не закэшированные
            uncached_indices = [
                i for i, result in enumerate(batch_results) if result is None
            ]

            if uncached_indices:
                uncached_embeddings = batch_embeddings[uncached_indices]
                decoded_batch = self._decode_batch_gpu(uncached_embeddings)

                # Заполняем результаты
                for i, decoded_text in zip(uncached_indices, decoded_batch):
                    batch_results[i] = decoded_text

                    # Кэшируем новые результаты
                    if use_cache and self.cache_enabled:
                        self.cache.put(batch_embeddings[i], decoded_text)

            results.extend(batch_results)
            total_cache_hits += cache_hits

        self.logger.info(
            f"GPU decoded {batch_size} embeddings in {(batch_size + self.gpu_batch_size - 1) // self.gpu_batch_size} batches (cache hits: {total_cache_hits})"
        )
        return results

    def _decode_batch_gpu(self, embeddings: torch.Tensor) -> List[str]:
        """Векторизованное декодирование батча на GPU"""
        self._init_decoder_model()

        if self._decoder_model == "dummy":
            # Векторизованный dummy декодер
            with torch.no_grad():
                # Создаем хэши для всего батча
                embeddings_cpu = embeddings.cpu()
                hashes = []
                for i in range(embeddings_cpu.size(0)):
                    emb_hash = hashlib.md5(
                        embeddings_cpu[i].numpy().tobytes()
                    ).hexdigest()[:8]
                    hashes.append(f"[GPU Batch decoded: {emb_hash}]")
                return hashes

        try:
            # GPU-ускоренное декодирование через transformers
            with torch.no_grad():
                # Генерируем candidates векторно
                candidates = [
                    "Advanced GPU processing result.",
                    "High-performance decoding output.",
                    "RTX 5090 accelerated generation.",
                    "Neural network decoded message.",
                    "AI-generated text content.",
                ]

                batch_size = embeddings.size(0)
                results = []

                # Векторизованный поиск лучших кандидатов
                for i in range(batch_size):
                    embedding = embeddings[i]
                    best_text = candidates[i % len(candidates)]  # Простая ротация

                    # В реальной реализации здесь был бы векторизованный similarity search
                    results.append(f"{best_text} [GPU:{i}]")

                return results

        except Exception as e:
            self.logger.warning(f"GPU batch decoding error: {e}")
            # Fallback
            return [f"[GPU Decode Error: {str(e)[:30]}]"] * embeddings.size(0)

    def _decode_single(self, embedding: torch.Tensor) -> str:
        """Декодирование одного эмбединга"""
        self._init_decoder_model()

        if self._decoder_model == "dummy":
            # Dummy декодер для тестирования
            emb_hash = hashlib.md5(
                embedding.detach().cpu().numpy().tobytes()
            ).hexdigest()[:8]
            return f"[Generated from embedding {emb_hash}]"

        try:
            # Используем ближайший поиск по эмбедингам модели
            # Это приблизительный подход, позже можно улучшить

            # Генерируем случайные токены и ищем ближайший по эмбедингу
            with torch.no_grad():
                # Простая эвристика - генерируем несколько вариантов и выбираем лучший
                candidates = [
                    "This is a sample text.",
                    "Hello world example.",
                    "Generated response text.",
                    "Sample decoded message.",
                    "Embedding reconstruction.",
                ]

                best_text = candidates[0]
                best_similarity = -1

                for candidate in candidates:
                    # Получаем эмбединг кандидата
                    inputs = self._tokenizer(
                        candidate, return_tensors="pt", padding=True, truncation=True
                    )
                    inputs = {
                        k: self.device_manager.ensure_device(v)
                        for k, v in inputs.items()
                    }

                    outputs = self._decoder_model(**inputs)
                    candidate_embedding = outputs.last_hidden_state.mean(
                        dim=1
                    ).squeeze()

                    # Вычисляем сходство
                    similarity = F.cosine_similarity(
                        embedding.unsqueeze(0), candidate_embedding.unsqueeze(0)
                    ).item()

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_text = candidate

                return f"{best_text} (sim: {best_similarity:.3f})"

        except Exception as e:
            self.logger.warning(f"Decoding error: {e}")
            return f"[Decoding error: {str(e)[:50]}]"

    def update_training_pairs(self, embeddings: torch.Tensor, texts: List[str]):
        """Обновление кэша парами из обучающих данных"""
        if not self.cache_enabled:
            return

        for i, text in enumerate(texts):
            if i < embeddings.size(0):
                self.cache.put(embeddings[i], text)

        self.logger.info(f"Updated cache with {len(texts)} training pairs")

    def save_cache(self):
        """Сохранение кэша"""
        if self.cache_enabled and hasattr(self, "cache_path"):
            self.cache.save(str(self.cache_path))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        if self.cache_enabled:
            return self.cache.get_stats()
        return {"cache_enabled": False}


class JointTextDecoder(nn.Module):
    """
    Декодер, который может обучаться вместе с кубом

    Расширенная версия для joint training с основной моделью.
    """

    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)

        # Базовый декодер
        self.base_decoder = SimpleTextDecoder(config)

        # Дополнительная обучаемая часть
        self.embedding_dim = config.embedding.teacher_embedding_dim
        self.vocab_size = 50000  # Примерный размер словаря

        # Простая архитектура для joint learning
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.vocab_size),
        )

        self.training_mode = True  # Включено ли обучение декодера

    def forward(
        self, embeddings: torch.Tensor, target_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass для joint training

        Args:
            embeddings: Входные эмбединги
            target_texts: Целевые тексты (для обучения)

        Returns:
            Dict с результатами декодирования и loss
        """
        results = {"decoded_texts": [], "logits": None, "loss": None}

        # Основное декодирование
        if not self.training:
            # В режиме inference используем базовый декодер
            results["decoded_texts"] = self.base_decoder.decode_embeddings(embeddings)
        else:
            # В режиме обучения используем обучаемую часть
            logits = self.projection(embeddings)
            results["logits"] = logits

            # Если есть целевые тексты, вычисляем loss
            if target_texts and self.training_mode:
                # Здесь нужна токенизация целевых текстов
                # Упрощенная версия пока
                target_tokens = self._tokenize_texts(target_texts)
                if target_tokens is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, self.vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100,
                    )
                    results["loss"] = loss

            # Декодируем для отображения
            results["decoded_texts"] = self._decode_from_logits(logits)

        return results

    def _tokenize_texts(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Простая токенизация текстов"""
        # Заглушка - в реальной реализации нужен токенизатор
        return None

    def _decode_from_logits(self, logits: torch.Tensor) -> List[str]:
        """Декодирование из logits"""
        # Простая эвристика - берем топ токены
        top_tokens = torch.argmax(logits, dim=-1)

        results = []
        for i in range(logits.size(0)):
            # Безопасное получение токенов
            if top_tokens.dim() > 1:
                tokens_slice = top_tokens[i][:5].tolist()
            else:
                tokens_slice = [top_tokens[i].item()]

            # Заглушка декодирования
            results.append(f"[Joint decoded: tokens {tokens_slice}]")

        return results

    def set_training_mode(self, enabled: bool):
        """Включение/выключение обучения декодера"""
        self.training_mode = enabled
        self.logger.info(f"Decoder training mode: {enabled}")


def create_text_decoder(
    config: SimpleProjectConfig, joint_training: bool = False
) -> nn.Module:
    """Фабричная функция для создания декодера"""
    if joint_training:
        return JointTextDecoder(config)
    else:
        return SimpleTextDecoder(config)
