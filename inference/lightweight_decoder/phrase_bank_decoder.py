"""
🔤 PHRASE BANK DECODER - Декодирование через поиск фраз

Реализует декодирование эмбедингов в текст через поиск наиболее семантически
близких фраз в предобученном phrase bank.

Phase 2.7.1 - PhraseBankDecoder Implementation
Phase 2.7.2 - STAGE 1.2 OPTIMIZATION ✨
Phase 2.7.3 - STAGE 1.3 PRODUCTION READINESS [START]
"""

import torch
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple, Set, Union
from dataclasses import dataclass, field
import re
from collections import defaultdict, OrderedDict
import hashlib
import json
from pathlib import Path
import time

from .phrase_bank import PhraseBank, PhraseEntry

@dataclass
class DecodingConfig:
    """Конфигурация для декодирования"""
    max_candidates: int = 10          # Максимум кандидатов для поиска
    similarity_threshold: float = 0.8  # Минимальный threshold similarity
    assembly_method: str = "context_aware"  # weighted, greedy, beam_search, context_aware
    coherence_weight: float = 0.3     # Вес когерентности
    relevance_weight: float = 0.7     # Вес релевантности
    context_weight: float = 0.4       # Вес контекстной релевантности
    min_phrase_length: int = 3        # Минимальная длина фразы (слова)
    max_phrase_length: int = 50       # Максимальная длина фразы (слова)
    
    # Context-aware parameters
    context_window_size: int = 5      # Размер контекстного окна для анализа
    category_bonus: float = 0.1       # Бонус за совпадающую категорию
    length_preference: str = "medium" # short, medium, long, adaptive
    
    # Post-processing parameters  
    enable_grammar_fix: bool = True   # Грамматические исправления
    enable_coherence_boost: bool = True  # Повышение когерентности
    enable_redundancy_removal: bool = True  # Удаление избыточности
    
    # 🆕 Stage 1.3: Production readiness parameters
    enable_caching: bool = True       # Кэширование результатов
    cache_size: int = 1000           # Размер кэша
    enable_fallbacks: bool = True     # Резервные стратегии
    max_retry_attempts: int = 3       # Максимум попыток повтора
    timeout_seconds: float = 5.0      # Таймаут для операций
    enable_performance_monitoring: bool = True  # Мониторинг производительности
    
    # Error handling parameters
    strict_mode: bool = False         # Строгий режим (исключения vs fallbacks)
    default_fallback_text: str = "Unable to decode."  # Текст по умолчанию
    log_errors: bool = True          # Логирование ошибок
    
    # Configuration validation
    validate_on_init: bool = True     # Валидация при инициализации
    
    def __post_init__(self):
        """Валидация конфигурации после инициализации"""
        if self.validate_on_init:
            self.validate()
    
    def validate(self):
        """🆕 Валидация конфигурации"""
        errors = []
        
        # Проверка диапазонов
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.context_weight <= 1.0:
            errors.append("context_weight must be between 0.0 and 1.0")
            
        if self.max_candidates <= 0:
            errors.append("max_candidates must be positive")
            
        if self.cache_size <= 0:
            errors.append("cache_size must be positive")
        
        # Проверка строковых параметров
        valid_assembly_methods = ["weighted", "greedy", "beam_search", "context_aware"]
        if self.assembly_method not in valid_assembly_methods:
            errors.append(f"assembly_method must be one of {valid_assembly_methods}")
        
        valid_length_preferences = ["short", "medium", "long", "adaptive"]
        if self.length_preference not in valid_length_preferences:
            errors.append(f"length_preference must be one of {valid_length_preferences}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

class PatternCache:
    """🆕 Кэширование для повторяющихся паттернов"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_embedding(self, embedding: torch.Tensor) -> str:
        """Создание хэша для эмбединга"""
        # Округляем до 4 знаков для стабильного хэширования
        rounded = torch.round(embedding * 10000) / 10000
        return hashlib.md5(rounded.numpy().tobytes()).hexdigest()
    
    def get(self, embedding: torch.Tensor) -> Optional[Dict]:
        """Получение результата из кэша"""
        key = self._hash_embedding(embedding)
        
        if key in self.cache:
            # Перемещаем в конец (LRU)
            result = self.cache.pop(key)
            self.cache[key] = result
            self.hit_count += 1
            return result
        
        self.miss_count += 1
        return None
    
    def put(self, embedding: torch.Tensor, result: Dict):
        """Сохранение результата в кэш"""
        key = self._hash_embedding(embedding)
        
        # Удаляем старые записи если кэш полон
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Удаляем самую старую запись
        
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict:
        """Статистика кэша"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests
        }

class ErrorHandler:
    """🆕 Продвинутая обработка ошибок"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
        self.error_counts = defaultdict(int)
        self.recent_errors: List[Dict] = []
        self.max_recent_errors = 50
    
    def handle_error(self, error: Exception, context: str, 
                    fallback_fn: Optional[callable] = None) -> Union[str, None]:
        """Обработка ошибки с резервными стратегиями"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time()
        }
        
        # Логирование
        if self.config.log_errors:
            logging.error(f"Error in {context}: {error}")
        
        # Статистика ошибок
        self.error_counts[error_info['error_type']] += 1
        self.recent_errors.append(error_info)
        
        # Ограничиваем количество недавних ошибок
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
        
        # Выбор стратегии
        if self.config.strict_mode:
            raise error
        
        if self.config.enable_fallbacks and fallback_fn:
            try:
                return fallback_fn()
            except Exception as fallback_error:
                if self.config.log_errors:
                    logging.error(f"Fallback failed: {fallback_error}")
        
        return self.config.default_fallback_text
    
    def get_error_stats(self) -> Dict:
        """Статистика ошибок"""
        return {
            'error_counts': dict(self.error_counts),
            'recent_errors_count': len(self.recent_errors),
            'total_errors': sum(self.error_counts.values())
        }

class PerformanceMonitor:
    """🆕 Мониторинг производительности"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
    
    def time_operation(self, operation_name: str):
        """Контекстный менеджер для измерения времени операций"""
        return self._OperationTimer(self, operation_name) if self.enabled else self._NoOpTimer()
    
    class _OperationTimer:
        def __init__(self, monitor, operation_name: str):
            self.monitor = monitor
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            self.monitor.operation_times[self.operation_name].append(duration)
            self.monitor.operation_counts[self.operation_name] += 1
    
    class _NoOpTimer:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def get_stats(self) -> Dict:
        """Статистика производительности"""
        if not self.enabled:
            return {'monitoring_disabled': True}
        
        stats = {}
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'total_time_ms': sum(times) * 1000
                }
        
        return stats

class ContextAnalyzer:
    """🆕 Анализатор контекста для улучшенного выбора фраз"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
        self.phrase_history: List[PhraseEntry] = []
        self.category_frequencies = defaultdict(int)
        self.length_preferences = {
            "short": (1, 5),
            "medium": (3, 15), 
            "long": (10, 50),
            "adaptive": (1, 50)
        }
    
    def analyze_context(self, candidates: List[Tuple[PhraseEntry, float]], 
                       embedding: torch.Tensor) -> List[Tuple[PhraseEntry, float]]:
        """Контекстный анализ кандидатов"""
        if not candidates:
            return candidates
        
        # [TARGET] Context-aware scoring
        scored_candidates = []
        
        for phrase, base_similarity in candidates:
            context_score = self._calculate_context_score(phrase, embedding)
            
            # Комбинированный score
            final_score = (
                base_similarity * (1 - self.config.context_weight) +
                context_score * self.config.context_weight
            )
            
            scored_candidates.append((phrase, final_score))
        
        # Сортировка по итоговому score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def _calculate_context_score(self, phrase: PhraseEntry, embedding: torch.Tensor) -> float:
        """Расчет контекстного score для фразы"""
        context_score = 0.0
        
        # 1. Категориальная совместимость
        if phrase.category in self.category_frequencies:
            category_boost = min(0.2, self.category_frequencies[phrase.category] * 0.05)
            context_score += category_boost
        
        # 2. Длина фразы по предпочтениям
        length_score = self._score_phrase_length(phrase.length)
        context_score += length_score * 0.3
        
        # 3. История использования (избегаем повторов)
        if phrase in self.phrase_history[-self.config.context_window_size:]:
            context_score -= 0.15  # Штраф за недавнее использование
        
        # 4. Семантическая связность с предыдущими фразами
        if self.phrase_history:
            coherence_score = self._calculate_semantic_coherence(phrase)
            context_score += coherence_score * 0.25
        
        return max(0.0, min(1.0, context_score))
    
    def _score_phrase_length(self, length: int) -> float:
        """Оценка длины фразы согласно предпочтениям"""
        pref_range = self.length_preferences[self.config.length_preference]
        min_len, max_len = pref_range
        
        if min_len <= length <= max_len:
            # В предпочитаемом диапазоне
            return 1.0
        elif length < min_len:
            # Слишком короткая
            return max(0.0, 0.5 - (min_len - length) * 0.1)
        else:
            # Слишком длинная
            return max(0.0, 0.7 - (length - max_len) * 0.05)
    
    def _calculate_semantic_coherence(self, phrase: PhraseEntry) -> float:
        """Семантическая связность с предыдущими фразами"""
        if not self.phrase_history:
            return 0.0
        
        # Берем последние фразы для анализа связности
        recent_phrases = self.phrase_history[-3:]
        
        coherence_scores = []
        for prev_phrase in recent_phrases:
            # Простая эвристика: совпадение категорий повышает связность
            if phrase.category == prev_phrase.category:
                coherence_scores.append(0.8)
            else:
                # Можно добавить более сложную логику сравнения эмбедингов
                coherence_scores.append(0.3)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def update_context(self, selected_phrase: PhraseEntry):
        """Обновление контекста после выбора фразы"""
        self.phrase_history.append(selected_phrase)
        self.category_frequencies[selected_phrase.category] += 1
        
        # Ограничиваем размер истории
        if len(self.phrase_history) > self.config.context_window_size * 2:
            self.phrase_history = self.phrase_history[-self.config.context_window_size:]
    
    def reset_context(self):
        """Сброс контекста для новой сессии"""
        self.phrase_history.clear()
        self.category_frequencies.clear()

class TextPostProcessor:
    """🆕 Постобработчик текста для повышения качества"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
    
    def process_text(self, raw_text: str, confidence: float = 0.0) -> str:
        """Главный метод постобработки"""
        processed_text = raw_text
        
        if self.config.enable_grammar_fix:
            processed_text = self._fix_basic_grammar(processed_text)
        
        if self.config.enable_coherence_boost:
            processed_text = self._boost_coherence(processed_text, confidence)
        
        if self.config.enable_redundancy_removal:
            processed_text = self._remove_redundancy(processed_text)
        
        return processed_text.strip()
    
    def _fix_basic_grammar(self, text: str) -> str:
        """Базовые грамматические исправления"""
        # Капитализация начала предложений
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Исправление пунктуации
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        
        return text
    
    def _boost_coherence(self, text: str, confidence: float) -> str:
        """Повышение когерентности текста"""
        # При низкой уверенности добавляем смягчающие слова
        if confidence < 0.6:
            coherence_boosters = ["perhaps", "possibly", "it seems", "likely"]
            if not any(booster in text.lower() for booster in coherence_boosters):
                text = f"It seems {text.lower()}"
        
        return text
    
    def _remove_redundancy(self, text: str) -> str:
        """Удаление избыточности"""
        words = text.split()
        
        # Удаление повторяющихся слов подряд
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()
        
        return ' '.join(cleaned_words)

class TextAssembler:
    """Сборщик текста из найденных фраз"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
        self.context_analyzer = ContextAnalyzer(config)  # 🆕
        self.post_processor = TextPostProcessor(config)  # 🆕
    
    def assemble_weighted(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Взвешенная сборка на основе similarity scores"""
        if not candidates:
            return "No suitable phrases found."
        
        # Простая стратегия: берем лучшую фразу
        best_phrase, best_similarity = candidates[0]
        
        # Проверка качества
        if best_similarity < self.config.similarity_threshold:
            return "Low confidence phrase match."
        
        # 🆕 Обновляем контекст
        self.context_analyzer.update_context(best_phrase)
        
        return best_phrase.text
    
    def assemble_greedy(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Жадная сборка - просто лучший кандидат"""
        if not candidates:
            return "No phrases available."
        
        best_phrase, _ = candidates[0]
        self.context_analyzer.update_context(best_phrase)
        return best_phrase.text
    
    def assemble_beam_search(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Beam search сборка (упрощенная версия)"""
        if not candidates:
            return "No beam candidates."
        
        # Пока простая реализация - можно расширить для multi-phrase assembly
        filtered_candidates = [
            (phrase, similarity) for phrase, similarity in candidates
            if self.config.min_phrase_length <= phrase.length <= self.config.max_phrase_length
        ]
        
        if not filtered_candidates:
            # Fallback к первому кандидату
            return candidates[0][0].text
        
        # Возьмем лучший отфильтрованный
        best_phrase, _ = filtered_candidates[0]
        self.context_analyzer.update_context(best_phrase)
        return best_phrase.text
    
    def assemble_context_aware(self, candidates: List[Tuple[PhraseEntry, float]], 
                             embedding: torch.Tensor) -> str:
        """🆕 Контекстно-осведомленная сборка"""
        if not candidates:
            return "No context-aware candidates."
        
        # Контекстный анализ кандидатов
        context_candidates = self.context_analyzer.analyze_context(candidates, embedding)
        
        if not context_candidates:
            return "Context analysis failed."
        
        # Выбираем лучший контекстный кандидат
        best_phrase, confidence = context_candidates[0]
        
        # Обновляем контекст
        self.context_analyzer.update_context(best_phrase)
        
        # Постобработка текста
        processed_text = self.post_processor.process_text(best_phrase.text, confidence)
        
        return processed_text
    
    def assemble(self, candidates: List[Tuple[PhraseEntry, float]], 
                embedding: Optional[torch.Tensor] = None) -> str:
        """Главный метод сборки"""
        if self.config.assembly_method == "weighted":
            return self.assemble_weighted(candidates)
        elif self.config.assembly_method == "greedy":
            return self.assemble_greedy(candidates)
        elif self.config.assembly_method == "beam_search":
            return self.assemble_beam_search(candidates)
        elif self.config.assembly_method == "context_aware":
            if embedding is not None:
                return self.assemble_context_aware(candidates, embedding)
            else:
                logging.warning("Context-aware assembly requires embedding, falling back to weighted")
                return self.assemble_weighted(candidates)
        else:
            logging.warning(f"Unknown assembly method: {self.config.assembly_method}")
            return self.assemble_greedy(candidates)
    
    def reset_session(self):
        """🆕 Сброс сессии для нового диалога"""
        self.context_analyzer.reset_context()

class QualityAssessor:
    """Оценщик качества декодирования"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
    
    def assess_candidates(self, candidates: List[Tuple[PhraseEntry, float]]) -> Dict:
        """Оценка качества кандидатов"""
        if not candidates:
            return {
                'quality_score': 0.0,
                'confidence': 0.0,
                'coherence': 0.0,
                'diversity': 0.0
            }
        
        # Базовые метрики
        similarities = [similarity for _, similarity in candidates]
        
        quality_score = np.mean(similarities)
        confidence = max(similarities)
        coherence = self._assess_coherence(candidates)
        diversity = self._assess_diversity(candidates)
        
        return {
            'quality_score': float(quality_score),
            'confidence': float(confidence),
            'coherence': float(coherence),
            'diversity': float(diversity)
        }
    
    def _assess_coherence(self, candidates: List[Tuple[PhraseEntry, float]]) -> float:
        """Оценка когерентности кандидатов"""
        if len(candidates) <= 1:
            return 1.0
        
        # Простая метрика: схожесть категорий
        categories = [phrase.category for phrase, _ in candidates]
        unique_categories = set(categories)
        
        # Больше одинаковых категорий = выше когерентность
        coherence = 1.0 - (len(unique_categories) - 1) / len(candidates)
        return max(0.0, coherence)
    
    def _assess_diversity(self, candidates: List[Tuple[PhraseEntry, float]]) -> float:
        """Оценка разнообразия кандидатов"""
        if len(candidates) <= 1:
            return 0.0
        
        # Простая метрика: разнообразие длин фраз
        lengths = [phrase.length for phrase, _ in candidates]
        diversity = np.std(lengths) / max(lengths) if max(lengths) > 0 else 0.0
        
        return min(1.0, diversity)

class PhraseBankDecoder:
    """[START] Production-ready декодер на основе phrase bank (Stage 1.3)"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 phrase_bank_size: int = 50000,
                 similarity_threshold: float = 0.8,
                 config: Optional[DecodingConfig] = None):
        
        self.embedding_dim = embedding_dim
        self.phrase_bank_size = phrase_bank_size
        self.similarity_threshold = similarity_threshold
        
        # Конфигурация с валидацией
        self.config = config or DecodingConfig(similarity_threshold=similarity_threshold)
        
        # 🆕 Production-ready компоненты
        self.cache = PatternCache(max_size=self.config.cache_size) if self.config.enable_caching else None
        self.error_handler = ErrorHandler(self.config)
        self.performance_monitor = PerformanceMonitor(enabled=self.config.enable_performance_monitoring)
        
        # Основные компоненты
        self.phrase_bank = PhraseBank(
            embedding_dim=embedding_dim,
            similarity_threshold=similarity_threshold,
            max_phrases=phrase_bank_size
        )
        
        self.assembler = TextAssembler(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        
        # Расширенная статистика
        self.stats = {
            'total_decodings': 0,
            'successful_decodings': 0,
            'cache_hits': 0,
            'fallback_uses': 0,
            'error_count': 0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0,
            'avg_decode_time_ms': 0.0
        }
        
        self.ready = False
        
        logging.info(f"PhraseBankDecoder (Stage 1.3) initialized: dim={embedding_dim}, " +
                    f"threshold={similarity_threshold}, caching={self.config.enable_caching}")
    
    def load_phrase_bank(self, embedding_loader=None, bank_path: Optional[str] = None):
        """Загрузка phrase bank"""
        if bank_path:
            # Загрузка из файла
            self.phrase_bank.load_bank(bank_path)
        elif embedding_loader:
            # Создание sample bank для тестирования
            self.phrase_bank.load_sample_bank(embedding_loader)
        else:
            raise ValueError("Either embedding_loader or bank_path must be provided")
        
        self.ready = True
        logging.info("Phrase bank loaded successfully")
    
    def decode(self, embedding: torch.Tensor) -> str:
        """[START] Production-ready декодирование с Stage 1.3 оптимизациями"""
        if not self.ready:
            error_msg = "Phrase bank not loaded. Call load_phrase_bank() first."
            return self.error_handler.handle_error(
                ValueError(error_msg), 
                "decode_readiness_check",
                fallback_fn=lambda: self.config.default_fallback_text
            )
        
        if embedding.dim() != 1 or embedding.size(0) != self.embedding_dim:
            error_msg = f"Expected embedding shape ({self.embedding_dim},), got {embedding.shape}"
            return self.error_handler.handle_error(
                ValueError(error_msg),
                "decode_input_validation",
                fallback_fn=lambda: self.config.default_fallback_text
            )
        
        # 🆕 Проверка кэша
        if self.cache:
            cached_result = self.cache.get(embedding)
            if cached_result:
                self.stats['cache_hits'] += 1
                self.stats['total_decodings'] += 1  # Учитываем кэшированные вызовы
                return cached_result['result']['decoded_text']
        
        try:
            with self.performance_monitor.time_operation("full_decode"):
                # Поиск кандидатов с мониторингом
                with self.performance_monitor.time_operation("phrase_search"):
                    candidates = self.phrase_bank.search_phrases(
                        embedding, 
                        k=self.config.max_candidates,
                        min_similarity=self.config.similarity_threshold
                    )
                
                # Оценка качества
                with self.performance_monitor.time_operation("quality_assessment"):
                    quality_metrics = self.quality_assessor.assess_candidates(candidates)
                
                # Сборка текста
                with self.performance_monitor.time_operation("text_assembly"):
                    decoded_text = self.assembler.assemble(candidates, embedding)
                
                # 🆕 Сохранение в кэш
                if self.cache:
                    cache_data = {
                        'decoded_text': decoded_text,
                        'quality_metrics': quality_metrics
                    }
                    self.cache.put(embedding, cache_data)
                
                # Обновление статистики
                self._update_stats(quality_metrics, len(candidates) > 0, decode_time=0.0)
                
                logging.debug(f"Decoded: {decoded_text} (confidence: {quality_metrics['confidence']:.3f})")
                
                return decoded_text
            
        except Exception as e:
            self.stats['error_count'] += 1
            
            # Резервная стратегия
            def fallback_decode():
                self.stats['fallback_uses'] += 1
                # Простое декодирование без дополнительных возможностей
                try:
                    basic_candidates = self.phrase_bank.search_phrases(embedding, k=3, min_similarity=0.5)
                    if basic_candidates:
                        return basic_candidates[0][0].text
                    return "Fallback: basic phrase match failed."
                except:
                    return self.config.default_fallback_text
            
            return self.error_handler.handle_error(e, "decode_main", fallback_fn=fallback_decode)
    
    def decode_with_metrics(self, embedding: torch.Tensor) -> Tuple[str, Dict]:
        """Декодирование с подробными метриками"""
        if not self.ready:
            raise ValueError("Phrase bank not loaded.")
        
        # Поиск кандидатов
        candidates = self.phrase_bank.search_phrases(
            embedding, 
            k=self.config.max_candidates,
            min_similarity=self.config.similarity_threshold
        )
        
        # Оценка качества
        quality_metrics = self.quality_assessor.assess_candidates(candidates)
        
        # Сборка текста
        decoded_text = self.assembler.assemble(candidates)
        
        # Дополнительные метрики
        detailed_metrics = {
            **quality_metrics,
            'num_candidates': len(candidates),
            'top_similarity': candidates[0][1] if candidates else 0.0,
            'phrase_categories': [phrase.category for phrase, _ in candidates[:3]],
            'phrase_bank_stats': self.phrase_bank.get_statistics()
        }
        
        self._update_stats(quality_metrics, len(candidates) > 0)
        
        return decoded_text, detailed_metrics
    
    def batch_decode(self, embeddings: torch.Tensor) -> List[str]:
        """Batch декодирование для эффективности"""
        if embeddings.dim() != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"Expected embeddings shape (N, {self.embedding_dim}), got {embeddings.shape}")
        
        results = []
        for i, embedding in enumerate(embeddings):
            try:
                result = self.decode(embedding)
                results.append(result)
            except Exception as e:
                logging.warning(f"Batch decode failed for item {i}: {e}")
                results.append("Batch decode error.")
        
        return results
    
    def get_statistics(self) -> Dict:
        """[START] Production-ready статистика декодера"""
        success_rate = (
            self.stats['successful_decodings'] / max(self.stats['total_decodings'], 1) * 100
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['total_decodings'], 1) * 100
        )
        
        base_stats = {
            **self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'phrase_bank_stats': self.phrase_bank.get_statistics(),
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'assembly_method': self.config.assembly_method,
                'max_candidates': self.config.max_candidates,
                'caching_enabled': self.config.enable_caching,
                'fallbacks_enabled': self.config.enable_fallbacks
            }
        }
        
        # 🆕 Добавляем статистику production компонентов
        if self.cache:
            base_stats['cache_stats'] = self.cache.get_stats()
        
        base_stats['error_stats'] = self.error_handler.get_error_stats()
        base_stats['performance_stats'] = self.performance_monitor.get_stats()
        
        return base_stats
    
    def _update_stats(self, quality_metrics: Dict, success: bool, decode_time: float = 0.0):
        """[START] Обновление расширенной статистики"""
        self.stats['total_decodings'] += 1
        
        if success:
            self.stats['successful_decodings'] += 1
        
        # Скользящее среднее для метрик
        total = self.stats['total_decodings']
        
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + quality_metrics.get('confidence', 0.0)) / total
        )
        
        self.stats['avg_quality'] = (
            (self.stats['avg_quality'] * (total - 1) + quality_metrics.get('quality_score', 0.0)) / total
        )
        
        # 🆕 Обновление времени декодирования
        if decode_time > 0:
            self.stats['avg_decode_time_ms'] = (
                (self.stats['avg_decode_time_ms'] * (total - 1) + decode_time * 1000) / total
            )
    
    def set_config(self, **kwargs):
        """Обновление конфигурации"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logging.info(f"Updated config: {key} = {value}")
            else:
                logging.warning(f"Unknown config parameter: {key}")
        
        # Пересоздание компонентов при необходимости
        self.assembler = TextAssembler(self.config)
        self.quality_assessor = QualityAssessor(self.config)
    
    def start_new_session(self):
        """🆕 Начало новой сессии декодирования"""
        self.assembler.reset_session()
        logging.info("Started new decoding session")
    
    def get_context_info(self) -> Dict:
        """🆕 Получение информации о текущем контексте"""
        return {
            'phrase_history_length': len(self.assembler.context_analyzer.phrase_history),
            'category_frequencies': dict(self.assembler.context_analyzer.category_frequencies),
            'current_length_preference': self.config.length_preference,
            'assembly_method': self.config.assembly_method
        }
    
    def decode_with_context_reset(self, embedding: torch.Tensor, reset_context: bool = False) -> str:
        """🆕 Декодирование с опциональным сбросом контекста"""
        if reset_context:
            self.start_new_session()
        
        return self.decode(embedding)
    
    def batch_decode_with_sessions(self, embeddings: torch.Tensor, 
                                 session_boundaries: Optional[List[int]] = None) -> List[str]:
        """🆕 Batch декодирование с учетом границ сессий"""
        if embeddings.dim() != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"Expected embeddings shape (N, {self.embedding_dim}), got {embeddings.shape}")
        
        results = []
        session_boundaries = session_boundaries or []
        
        for i, embedding in enumerate(embeddings):
            # Проверяем, нужно ли сбросить контекст
            if i in session_boundaries:
                self.start_new_session()
                logging.debug(f"Reset context at position {i}")
            
            try:
                result = self.decode(embedding)
                results.append(result)
            except Exception as e:
                logging.warning(f"Batch decode failed for item {i}: {e}")
                results.append("Batch decode error.")
        
        return results
    
    def clear_cache(self):
        """🆕 Очистка кэша"""
        if self.cache:
            self.cache.clear()
            logging.info("Cache cleared")
    
    def save_config(self, filepath: str):
        """🆕 Сохранение конфигурации"""
        config_dict = {
            'embedding_dim': self.embedding_dim,
            'phrase_bank_size': self.phrase_bank_size,
            'similarity_threshold': self.similarity_threshold,
            'assembly_method': self.config.assembly_method,
            'enable_caching': self.config.enable_caching,
            'cache_size': self.config.cache_size,
            'enable_fallbacks': self.config.enable_fallbacks,
            'strict_mode': self.config.strict_mode,
            'default_fallback_text': self.config.default_fallback_text
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logging.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """🆕 Загрузка конфигурации"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Обновляем конфигурацию
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Пересоздаем компоненты
        self.assembler = TextAssembler(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        
        if self.config.enable_caching and not self.cache:
            self.cache = PatternCache(max_size=self.config.cache_size)
        elif not self.config.enable_caching and self.cache:
            self.cache = None
        
        logging.info(f"Configuration loaded from {filepath}")
    
    def get_health_status(self) -> Dict:
        """🆕 Проверка состояния системы"""
        health = {
            'status': 'healthy',
            'ready': self.ready,
            'components': {
                'phrase_bank': self.phrase_bank is not None,
                'assembler': self.assembler is not None,
                'quality_assessor': self.quality_assessor is not None,
                'cache': self.cache is not None,
                'error_handler': self.error_handler is not None,
                'performance_monitor': self.performance_monitor is not None
            },
            'error_rate': 0.0,
            'cache_efficiency': 0.0
        }
        
        # Проверка частоты ошибок
        total_ops = self.stats['total_decodings']
        if total_ops > 0:
            error_rate = (self.stats['error_count'] / total_ops) * 100
            health['error_rate'] = error_rate
            
            if error_rate > 10:  # Больше 10% ошибок
                health['status'] = 'degraded'
            elif error_rate > 25:  # Больше 25% ошибок
                health['status'] = 'unhealthy'
        
        # Проверка эффективности кэша
        if self.cache and total_ops > 0:
            cache_efficiency = (self.stats['cache_hits'] / total_ops) * 100
            health['cache_efficiency'] = cache_efficiency
        
        return health
    
    def optimize_for_production(self):
        """[START] Оптимизация для продакшн режима"""
        optimizations = []
        
        # Включаем кэширование если отключено
        if not self.config.enable_caching:
            self.config.enable_caching = True
            self.cache = PatternCache(max_size=self.config.cache_size)
            optimizations.append("Enabled caching")
        
        # Включаем fallbacks
        if not self.config.enable_fallbacks:
            self.config.enable_fallbacks = True
            optimizations.append("Enabled fallbacks")
        
        # Отключаем strict mode
        if self.config.strict_mode:
            self.config.strict_mode = False
            optimizations.append("Disabled strict mode")
        
        # Оптимизируем размер кэша
        if self.config.cache_size < 500:
            self.config.cache_size = 1000
            if self.cache:
                self.cache.max_size = 1000
            optimizations.append("Increased cache size to 1000")
        
        # Включаем мониторинг производительности
        if not self.config.enable_performance_monitoring:
            self.config.enable_performance_monitoring = True
            self.performance_monitor = PerformanceMonitor(enabled=True)
            optimizations.append("Enabled performance monitoring")
        
        logging.info(f"Production optimizations applied: {', '.join(optimizations)}")
        return optimizations

# Логирование
logger = logging.getLogger(__name__) 