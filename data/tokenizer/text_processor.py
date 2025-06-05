"""
TextProcessor - 3D Cellular Neural Network

Класс для предобработки и очистки текста перед токенизацией.
Выполняет нормализацию Unicode, очистку, обработку пунктуации
и другие операции предобработки.

Автор: 3D CNN Team
Дата: Декабрь 2025
"""

import re
import unicodedata
import logging
from typing import List, Dict, Optional, Union, Any
import string

# Опциональные зависимости для продвинутой обработки
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextProcessor:
    """
    Класс для предобработки текста перед токенизацией.
    
    Поддерживает:
    - Нормализация Unicode
    - Очистка от лишних символов
    - Обработка пунктуации
    - Удаление стоп-слов (опционально)
    - Stemming/Lemmatization (опционально)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация TextProcessor.
        
        Args:
            config: Словарь конфигурации для настройки обработки
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Настройки из конфигурации
        self.lowercase = self.config.get('lowercase', True)
        self.strip_whitespace = self.config.get('strip_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.remove_punctuation = self.config.get('remove_punctuation', False)
        self.normalize_punctuation = self.config.get('normalize_punctuation', True)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.language = self.config.get('language', 'english')
        
        # Компиляция регулярных выражений для производительности
        self._compile_patterns()
        
        # Инициализация NLTK компонентов
        self._init_nltk_components()
    
    def _get_default_config(self) -> Dict:
        """Получение конфигурации по умолчанию."""
        return {
            'lowercase': True,
            'strip_whitespace': True,
            'normalize_unicode': True,
            'remove_punctuation': False,
            'normalize_punctuation': True,
            'remove_stopwords': False,
            'remove_numbers': False,
            'remove_urls': True,
            'remove_emails': True,
            'language': 'english'
        }
    
    def _compile_patterns(self) -> None:
        """Компиляция регулярных выражений для оптимизации."""
        # URL паттерн
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email паттерн
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Паттерн для множественных пробелов
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Паттерн для чисел
        self.numbers_pattern = re.compile(r'\b\d+\b')
        
        # Паттерн для нормализации пунктуации
        self.punctuation_patterns = [
            (re.compile(r'[""„"]'), '"'),  # Различные кавычки
            (re.compile(r'[''`]'), "'"),   # Различные апострофы
            (re.compile(r'[–—]'), '-'),    # Различные тире
            (re.compile(r'[…]'), '...'),   # Многоточие
        ]
    
    def _init_nltk_components(self) -> None:
        """Инициализация NLTK компонентов."""
        self.stopwords_set = set()
        self.stemmer = None
        self.lemmatizer = None
        
        if NLTK_AVAILABLE and self.remove_stopwords:
            try:
                # Попытка загрузить стоп-слова
                self.stopwords_set = set(stopwords.words(self.language))
                self.logger.info(f"Loaded {len(self.stopwords_set)} stopwords for {self.language}")
            except Exception as e:
                self.logger.warning(f"Failed to load stopwords: {str(e)}")
                # Простой набор английских стоп-слов
                self.stopwords_set = {
                    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                    'to', 'was', 'will', 'with'
                }
        
        if NLTK_AVAILABLE:
            try:
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize stemmer/lemmatizer: {str(e)}")
    
    def preprocess(self, text: str) -> str:
        """
        Основная функция предобработки текста.
        
        Args:
            text: Входной текст для обработки
            
        Returns:
            Обработанный текст
        """
        if not text or not isinstance(text, str):
            return ""
        
        processed_text = text
        
        # Последовательная обработка
        if self.remove_urls:
            processed_text = self.remove_url_patterns(processed_text)
        
        if self.remove_emails:
            processed_text = self.remove_email_patterns(processed_text)
        
        if self.normalize_unicode:
            processed_text = self.normalize_unicode_chars(processed_text)
        
        if self.normalize_punctuation:
            processed_text = self.normalize_punctuation_marks(processed_text)
        
        if self.remove_numbers:
            processed_text = self.remove_number_patterns(processed_text)
        
        if self.remove_punctuation:
            processed_text = self.remove_punctuation_marks(processed_text)
        
        if self.strip_whitespace:
            processed_text = self.clean_whitespace(processed_text)
        
        if self.lowercase:
            processed_text = processed_text.lower()
        
        if self.remove_stopwords and self.stopwords_set:
            processed_text = self.remove_stopword_tokens(processed_text)
        
        return processed_text.strip()
    
    def normalize_unicode_chars(self, text: str) -> str:
        """
        Нормализация Unicode символов.
        
        Args:
            text: Входной текст
            
        Returns:
            Нормализованный текст
        """
        # NFD нормализация (разложение) + удаление диакритических знаков
        normalized = unicodedata.normalize('NFD', text)
        # Удаляем комбинирующие символы (диакритические знаки)
        no_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # NFC нормализация (композиция)
        return unicodedata.normalize('NFC', no_accents)
    
    def clean_whitespace(self, text: str) -> str:
        """
        Очистка множественных пробелов и специальных символов.
        
        Args:
            text: Входной текст
            
        Returns:
            Очищенный текст
        """
        # Замена всех видов пробелов на обычный пробел
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        # Замена множественных пробелов на один
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def remove_punctuation_marks(self, text: str) -> str:
        """
        Удаление знаков пунктуации.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст без пунктуации
        """
        # Создание таблицы транслитерации для удаления пунктуации
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def normalize_punctuation_marks(self, text: str) -> str:
        """
        Нормализация знаков пунктуации.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст с нормализованной пунктуацией
        """
        processed_text = text
        
        # Применение паттернов нормализации
        for pattern, replacement in self.punctuation_patterns:
            processed_text = pattern.sub(replacement, processed_text)
        
        return processed_text
    
    def remove_url_patterns(self, text: str) -> str:
        """
        Удаление URL из текста.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст без URL
        """
        return self.url_pattern.sub('', text)
    
    def remove_email_patterns(self, text: str) -> str:
        """
        Удаление email адресов из текста.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст без email адресов
        """
        return self.email_pattern.sub('', text)
    
    def remove_number_patterns(self, text: str) -> str:
        """
        Удаление чисел из текста.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст без чисел
        """
        return self.numbers_pattern.sub('', text)
    
    def remove_stopword_tokens(self, text: str) -> str:
        """
        Удаление стоп-слов из текста.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст без стоп-слов
        """
        if not self.stopwords_set:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords_set]
        return ' '.join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """
        Применение stemming к тексту.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст с примененным stemming
        """
        if not self.stemmer:
            return text
        
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Применение lemmatization к тексту.
        
        Args:
            text: Входной текст
            
        Returns:
            Текст с примененным lemmatization
        """
        if not self.lemmatizer:
            return text
        
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Batch обработка списка текстов.
        
        Args:
            texts: Список входных текстов
            
        Returns:
            Список обработанных текстов
        """
        if not texts or not isinstance(texts, list):
            return []
        
        processed_texts = []
        for text in texts:
            try:
                processed = self.preprocess(text)
                processed_texts.append(processed)
            except Exception as e:
                self.logger.warning(f"Failed to preprocess text: {str(e)}")
                processed_texts.append("")  # Пустая строка для проблемных текстов
        
        return processed_texts
    
    def get_processing_stats(self, original_text: str, processed_text: str) -> Dict[str, Any]:
        """
        Получение статистики обработки текста.
        
        Args:
            original_text: Оригинальный текст
            processed_text: Обработанный текст
            
        Returns:
            Словарь со статистикой
        """
        return {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'original_words': len(original_text.split()),
            'processed_words': len(processed_text.split()),
            'reduction_ratio': 1.0 - (len(processed_text) / len(original_text)) if original_text else 0.0,
            'config': self.config.copy()
        }
    
    def validate_text(self, text: str, max_length: int = 1000000) -> bool:
        """
        Валидация входного текста.
        
        Args:
            text: Текст для валидации
            max_length: Максимальная допустимая длина
            
        Returns:
            True если текст валиден, False иначе
        """
        if not isinstance(text, str):
            return False
        
        if len(text) > max_length:
            return False
        
        # Проверка на минимальную длину
        if len(text.strip()) < 1:
            return False
        
        return True
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Базовая детекция языка (заглушка для будущего расширения).
        
        Args:
            text: Входной текст
            
        Returns:
            Код языка или None
        """
        # Простая эвристика для английского языка
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that']
        words = text.lower().split()
        english_count = sum(1 for word in words if word in english_indicators)
        
        if len(words) > 0 and english_count / len(words) > 0.1:
            return 'english'
        
        return None
    
    def __repr__(self) -> str:
        return f"TextProcessor(language='{self.language}', stopwords={bool(self.stopwords_set)})"
    
    def __str__(self) -> str:
        features = []
        if self.lowercase:
            features.append("lowercase")
        if self.normalize_unicode:
            features.append("unicode_norm")
        if self.remove_punctuation:
            features.append("no_punct")
        if self.remove_stopwords:
            features.append("no_stopwords")
        if self.remove_urls:
            features.append("no_urls")
        
        return f"TextProcessor: {', '.join(features) if features else 'minimal'}" 