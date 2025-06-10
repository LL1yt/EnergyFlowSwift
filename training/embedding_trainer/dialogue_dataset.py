"""
DialogueDataset - Класс для подготовки данных к обучению куба в dialogue режиме

Этот модуль реализует специализированный dataset для обучения 3D Cubic Core
на задачах диалога (question_embedding → answer_embedding).

Ключевые возможности:
- Интеграция с Teacher LLM (LLaMA 3, Mistral, etc.) для генерации эмбедингов Q&A
- Conversation pairs: (question_embedding, answer_embedding)
- Smart caching для эффективности обучения
- Multi-turn dialogue support
- Quality filtering для диалоговых пар
- Context-aware training подготовка

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0 (Phase 3.1 - Stage 1.3)
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
import re

# Импорты готовых компонентов
try:
    from data.embedding_loader import EmbeddingLoader
    from data.embedding_loader.format_handlers import SUPPORTED_LLM_MODELS
    EMBEDDING_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING]  Warning: EmbeddingLoader not available: {e}")
    EMBEDDING_LOADER_AVAILABLE = False


@dataclass
class DialogueConfig:
    """Конфигурация для DialogueDataset"""
    # Teacher LLM настройки
    teacher_model: str = "llama3-8b"    # Модель для генерации эмбедингов
    fallback_model: str = "distilbert"   # Запасная модель если основная недоступна
    embedding_dim: int = 768             # Размерность эмбедингов
    
    # Источники данных
    data_sources: List[str] = None       # Пути к файлам с диалогами
    dialogue_format: str = "qa_pairs"    # qa_pairs | conversation | json
    max_conversations: int = 5000        # Максимальное количество диалогов
    min_conversations: int = 50          # Минимальное количество диалогов
    
    # Multi-turn диалоги
    support_multiturn: bool = True       # Поддержка многоходовых диалогов
    max_turns_per_conversation: int = 10 # Максимум реплик в диалоге
    context_window: int = 2              # Количество предыдущих реплик для контекста
    
    # Quality filtering
    enable_quality_filter: bool = True   # Включить фильтрацию качества
    min_question_length: int = 5         # Минимальная длина вопроса (символы)
    min_answer_length: int = 10          # Минимальная длина ответа (символы)
    max_question_length: int = 512       # Максимальная длина вопроса
    max_answer_length: int = 1024        # Максимальная длина ответа
    semantic_similarity_threshold: float = 0.3  # Минимальная семантическая связность Q&A
    
    # Preprocessing
    normalize_embeddings: bool = True    # Нормализация эмбедингов
    center_embeddings: bool = True       # Центрирование эмбедингов
    add_context_noise: bool = False      # Добавлять контекстный шум
    context_noise_std: float = 0.02     # Стандартное отклонение контекстного шума
    
    # Caching
    cache_dir: str = "cache/dialogue_dataset"
    use_cache: bool = True
    cache_embeddings: bool = True
    cache_batch_size: int = 500          # Размер батча для кэширования
    
    # Validation split
    validation_split: float = 0.2        # Доля данных для валидации
    shuffle_conversations: bool = True    # Перемешивать ли диалоги
    random_seed: int = 42                # Seed для воспроизводимости
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        
        # Создание директории для кэша
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Интеграция с центральной системой конфигурации
        self._load_from_central_config()
    
    def _load_from_central_config(self):
        """Загрузка настроек из центральной системы конфигурации"""
        try:
            from utils.config_loader import config_manager
            
            # Загружаем teacher models из конфига
            teacher_config = config_manager.get_teacher_models_config()
            if teacher_config:
                # Берем первую доступную модель как основную
                if 'models' in teacher_config and teacher_config['models']:
                    available_models = teacher_config['models']
                    self.teacher_model = available_models[0]
                    if len(available_models) > 1:
                        self.fallback_model = available_models[1]
                    
                    print(f"[INFO] Loaded teacher models from central config:")
                    print(f"   Primary: {self.teacher_model}")
                    print(f"   Fallback: {self.fallback_model}")
            
            # Загружаем общие настройки
            general_config = config_manager.get_config()
            if general_config:
                # Настройки качества данных
                if 'dialogue_dataset' in general_config:
                    dialogue_settings = general_config['dialogue_dataset']
                    
                    if 'quality_filter' in dialogue_settings:
                        quality_settings = dialogue_settings['quality_filter']
                        self.min_question_length = quality_settings.get('min_question_length', self.min_question_length)
                        self.min_answer_length = quality_settings.get('min_answer_length', self.min_answer_length)
                        self.semantic_similarity_threshold = quality_settings.get('semantic_similarity_threshold', self.semantic_similarity_threshold)
                    
                    # Настройки кэширования
                    if 'caching' in dialogue_settings:
                        cache_settings = dialogue_settings['caching']
                        self.use_cache = cache_settings.get('enabled', self.use_cache)
                        self.cache_batch_size = cache_settings.get('batch_size', self.cache_batch_size)
                    
                    # Настройки валидации
                    if 'validation' in dialogue_settings:
                        val_settings = dialogue_settings['validation']
                        self.validation_split = val_settings.get('split', self.validation_split)
                        self.random_seed = val_settings.get('seed', self.random_seed)
                
                print(f"[OK] DialogueConfig integrated with central configuration")
            
        except Exception as e:
            print(f"[WARNING] Could not load from central config ({e}), using defaults")


def map_model_name_to_key(model_name: str) -> str:
    """
    Преобразование полного имени модели в ключ для SUPPORTED_LLM_MODELS
    
    Args:
        model_name: Полное имя модели (например, "distilbert-base-uncased")
        
    Returns:
        Ключ модели для SUPPORTED_LLM_MODELS (например, "distilbert")
    """
    # Создаем обратное отображение value -> key
    name_to_key = {v: k for k, v in SUPPORTED_LLM_MODELS.items()}
    
    # Прямое совпадение
    if model_name in name_to_key:
        return name_to_key[model_name]
    
    # Если это уже ключ
    if model_name in SUPPORTED_LLM_MODELS:
        return model_name
    
    # Поиск по частичному совпадению
    for model_value, model_key in name_to_key.items():
        if model_name in model_value or model_value in model_name:
            return model_key
    
    # Специальные mappings для часто используемых имен
    common_mappings = {
        "distilbert-base-uncased": "distilbert",
        "distilbert": "distilbert",
        "roberta-base": "roberta", 
        "roberta": "roberta",
        "gpt2": "gpt2",
        "sentence-transformers/all-MiniLM-L6-v2": "distilbert",  # fallback
        "sentence-transformers/all-mpnet-base-v2": "distilbert", # fallback
    }
    
    if model_name in common_mappings:
        return common_mappings[model_name]
    
    # Fallback - если ничего не найдено, возвращаем distilbert
    print(f"[WARNING] Model '{model_name}' not found in SUPPORTED_LLM_MODELS, using 'distilbert' as fallback")
    return "distilbert"


class DialogueDataset(Dataset):
    """
    Dataset для обучения 3D Cubic Core в dialogue режиме
    
    Создает пары (question_embedding, answer_embedding) для обучения
    трансформации вопрос → ответ через Teacher LLM эмбединги.
    """
    
    def __init__(self, 
                 config: Optional[Union[DialogueConfig, Dict, str]] = None,
                 dialogue_pairs: Optional[List[Dict]] = None,
                 conversations: Optional[List[List[Dict]]] = None,
                 question_embeddings: Optional[torch.Tensor] = None,
                 answer_embeddings: Optional[torch.Tensor] = None):
        """
        Инициализация DialogueDataset
        
        Args:
            config: Конфигурация dataset (DialogueConfig, dict или путь к JSON)
            dialogue_pairs: Список Q&A пар [{"question": str, "answer": str}, ...]
            conversations: Список многоходовых диалогов [[{role, text}, ...], ...]
            question_embeddings: Готовые эмбединги вопросов (опционально)
            answer_embeddings: Готовые эмбединги ответов (опционально)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("[START] Initializing DialogueDataset for Stage 1.3...")
        
        # Проверка зависимостей
        if not EMBEDDING_LOADER_AVAILABLE:
            raise ImportError("EmbeddingLoader is required for DialogueDataset. "
                            "Make sure data.embedding_loader is implemented.")
        
        # Загрузка конфигурации
        self.config = self._load_config(config)
        
        # Установка random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        # Инициализация Teacher LLM Encoder
        self.embedding_loader = EmbeddingLoader(
            cache_dir=str(Path(self.config.cache_dir) / "embedding_loader_cache")
        )
        
        # Проверка доступности teacher модели
        self._validate_teacher_model()
        
        # Данные
        self.question_embeddings: torch.Tensor = None
        self.answer_embeddings: torch.Tensor = None
        self.train_questions: torch.Tensor = None
        self.train_answers: torch.Tensor = None
        self.val_questions: torch.Tensor = None
        self.val_answers: torch.Tensor = None
        self.is_validation_mode: bool = False
        
        # Метаданные
        self.dialogue_metadata = []
        self.dataset_info = {}
        self.cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_loads': 0,
            'quality_filtered': 0
        }
        
        # Загрузка данных
        if question_embeddings is not None and answer_embeddings is not None:
            self.logger.info("Using provided Q&A embeddings")
            self._load_from_embeddings(question_embeddings, answer_embeddings)
        elif dialogue_pairs is not None:
            self.logger.info("Generating embeddings from dialogue pairs")
            self._load_from_dialogue_pairs(dialogue_pairs)
        elif conversations is not None:
            self.logger.info("Processing multi-turn conversations")
            self._load_from_conversations(conversations)
        elif self.config.data_sources:
            self.logger.info("Loading dialogues from configured sources")
            self._load_from_sources()
        else:
            raise ValueError("No dialogue data provided. Specify dialogue_pairs, conversations, "
                           "embeddings, or data_sources in config.")
        
        # Создание train/val split
        self._create_train_val_split()
        
        self.logger.info(f"[OK] DialogueDataset initialized successfully")
        self.logger.info(f"   Total conversation pairs: {len(self.question_embeddings)}")
        self.logger.info(f"   Train pairs: {len(self.train_questions)}")
        self.logger.info(f"   Val pairs: {len(self.val_questions)}")
        self.logger.info(f"   Embedding dim: {self.question_embeddings.shape[1]}")
        self.logger.info(f"   Teacher model: {self.config.teacher_model}")
        self.logger.info(f"   Quality filtered: {self.cache_stats['quality_filtered']} pairs")
    
    def _load_config(self, config: Optional[Union[DialogueConfig, Dict, str]]) -> DialogueConfig:
        """Загрузка и валидация конфигурации"""
        if config is None:
            return DialogueConfig()
        
        elif isinstance(config, DialogueConfig):
            return config
        
        elif isinstance(config, dict):
            return DialogueConfig(**config)
        
        elif isinstance(config, str):
            # Загрузка из JSON файла
            try:
                with open(config, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return DialogueConfig(**config_data)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config}: {e}")
                return DialogueConfig()
        
        else:
            self.logger.warning(f"Unknown config type: {type(config)}. Using default config.")
            return DialogueConfig()
    
    def _validate_teacher_model(self):
        """Проверка доступности teacher модели"""
        try:
            # Преобразуем имя модели в ключ
            teacher_model_key = map_model_name_to_key(self.config.teacher_model)
            
            # Тестируем основную модель
            test_embedding = self.embedding_loader.load_from_llm(
                texts=["Test message"],
                model_key=teacher_model_key
            )
            self.logger.info(f"[OK] Teacher model {self.config.teacher_model} (key: {teacher_model_key}) is available")
            
            # Обновляем config с правильным ключом
            self.config.teacher_model = teacher_model_key
            
        except Exception as e:
            self.logger.warning(f"[WARNING]  Teacher model {self.config.teacher_model} not available: {e}")
            
            # Пробуем fallback
            fallback_key = map_model_name_to_key(self.config.fallback_model)
            self.logger.info(f"Switching to fallback model: {self.config.fallback_model} (key: {fallback_key})")
            self.config.teacher_model = fallback_key
    
    def _load_from_embeddings(self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor):
        """Загрузка из готовых Q&A эмбедингов"""
        if not isinstance(question_embeddings, torch.Tensor):
            question_embeddings = torch.from_numpy(np.array(question_embeddings)).float()
        if not isinstance(answer_embeddings, torch.Tensor):
            answer_embeddings = torch.from_numpy(np.array(answer_embeddings)).float()
        
        # Проверка размерностей
        if len(question_embeddings.shape) != 2 or len(answer_embeddings.shape) != 2:
            raise ValueError(f"Embeddings should be 2D tensors, got shapes: "
                           f"{question_embeddings.shape}, {answer_embeddings.shape}")
        
        if question_embeddings.shape[0] != answer_embeddings.shape[0]:
            raise ValueError(f"Number of questions and answers must match: "
                           f"{question_embeddings.shape[0]} vs {answer_embeddings.shape[0]}")
        
        if question_embeddings.shape[1] != self.config.embedding_dim:
            self.logger.warning(f"Question embedding dimension mismatch: got {question_embeddings.shape[1]}, "
                              f"expected {self.config.embedding_dim}")
            self.config.embedding_dim = question_embeddings.shape[1]
        
        self.question_embeddings = question_embeddings
        self.answer_embeddings = answer_embeddings
        self._update_dataset_info("provided_embeddings", question_embeddings.shape[0])
    
    def _load_from_dialogue_pairs(self, dialogue_pairs: List[Dict]):
        """Генерация эмбедингов из диалоговых пар через Teacher LLM"""
        # Фильтрация качества
        if self.config.enable_quality_filter:
            dialogue_pairs = self._filter_dialogue_quality(dialogue_pairs)
        
        # Ограничение количества диалогов
        if len(dialogue_pairs) > self.config.max_conversations:
            dialogue_pairs = dialogue_pairs[:self.config.max_conversations]
            self.logger.info(f"Limited to {self.config.max_conversations} dialogue pairs")
        
        # Проверка кэша
        cache_key = self._create_cache_key_for_dialogues(dialogue_pairs)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None and self.config.use_cache:
            self.logger.info("Loading dialogue embeddings from cache")
            self.question_embeddings = cached_data['questions']
            self.answer_embeddings = cached_data['answers']
            self.dialogue_metadata = cached_data.get('metadata', [])
            self.cache_stats['cache_hits'] += 1
        else:
            self.logger.info(f"Generating embeddings from {len(dialogue_pairs)} dialogue pairs "
                           f"using Teacher LLM: {self.config.teacher_model}")
            
            # Извлечение текстов
            questions = [pair['question'] for pair in dialogue_pairs]
            answers = [pair['answer'] for pair in dialogue_pairs]
            
            # Генерация эмбедингов через Teacher LLM
            question_embeddings = self.embedding_loader.batch_load_from_llm(
                texts=questions,
                model_key=self.config.teacher_model,
                batch_size=self.config.cache_batch_size
            )
            
            answer_embeddings = self.embedding_loader.batch_load_from_llm(
                texts=answers,
                model_key=self.config.teacher_model,
                batch_size=self.config.cache_batch_size
            )
            
            # Preprocessing
            if self.config.normalize_embeddings or self.config.center_embeddings:
                question_embeddings = self.embedding_loader.preprocess_embeddings(
                    question_embeddings,
                    normalize=self.config.normalize_embeddings,
                    center=self.config.center_embeddings
                )
                answer_embeddings = self.embedding_loader.preprocess_embeddings(
                    answer_embeddings,
                    normalize=self.config.normalize_embeddings,
                    center=self.config.center_embeddings
                )
            
            # Контекстный шум для регуляризации
            if self.config.add_context_noise:
                noise_std = self.config.context_noise_std
                question_noise = torch.randn_like(question_embeddings) * noise_std
                answer_noise = torch.randn_like(answer_embeddings) * noise_std
                question_embeddings += question_noise
                answer_embeddings += answer_noise
            
            self.question_embeddings = question_embeddings
            self.answer_embeddings = answer_embeddings
            
            # Сохранение метаданных
            self.dialogue_metadata = [
                {
                    'question': pair['question'],
                    'answer': pair['answer'],
                    'question_length': len(pair['question']),
                    'answer_length': len(pair['answer'])
                }
                for pair in dialogue_pairs
            ]
            
            self.cache_stats['cache_misses'] += 1
            
            # Сохранение в кэш
            if self.config.cache_embeddings:
                cache_data = {
                    'questions': question_embeddings,
                    'answers': answer_embeddings,
                    'metadata': self.dialogue_metadata
                }
                self._save_to_cache(cache_key, cache_data)
        
        self._update_dataset_info("generated_from_dialogue_pairs", len(dialogue_pairs))
    
    def _load_from_conversations(self, conversations: List[List[Dict]]):
        """Обработка многоходовых диалогов"""
        self.logger.info(f"Processing {len(conversations)} multi-turn conversations")
        
        dialogue_pairs = []
        
        for conversation in conversations:
            # Извлечение Q&A пар из многоходового диалога
            pairs = self._extract_qa_pairs_from_conversation(conversation)
            dialogue_pairs.extend(pairs)
        
        self.logger.info(f"Extracted {len(dialogue_pairs)} Q&A pairs from conversations")
        
        # Обработка как обычные диалоговые пары
        self._load_from_dialogue_pairs(dialogue_pairs)
        self._update_dataset_info("generated_from_conversations", len(conversations))
    
    def _extract_qa_pairs_from_conversation(self, conversation: List[Dict]) -> List[Dict]:
        """Извлечение Q&A пар из одного многоходового диалога"""
        pairs = []
        
        # Простейшая стратегия: каждая реплика пользователя → следующая реплика ассистента
        for i in range(len(conversation) - 1):
            current_turn = conversation[i]
            next_turn = conversation[i + 1]
            
            # Ищем пары пользователь → ассистент
            if (current_turn.get('role', '').lower() in ['user', 'human', 'question'] and
                next_turn.get('role', '').lower() in ['assistant', 'bot', 'answer']):
                
                pairs.append({
                    'question': current_turn.get('text', current_turn.get('content', '')),
                    'answer': next_turn.get('text', next_turn.get('content', ''))
                })
        
        return pairs
    
    def _load_from_sources(self):
        """Загрузка диалогов из конфигурированных источников"""
        all_dialogue_pairs = []
        
        for source_path in self.config.data_sources:
            source_path = Path(source_path)
            
            if not source_path.exists():
                self.logger.warning(f"Source file not found: {source_path}")
                continue
            
            try:
                if source_path.suffix in ['.json', '.jsonl']:
                    # JSON файл с диалогами
                    pairs = self._load_dialogues_from_json(source_path)
                    
                elif source_path.suffix in ['.txt']:
                    # Текстовый файл (простой формат Q: A:)
                    pairs = self._load_dialogues_from_text(source_path)
                    
                elif source_path.suffix in ['.csv']:
                    # CSV файл с колонками question, answer
                    pairs = self._load_dialogues_from_csv(source_path)
                    
                else:
                    self.logger.warning(f"Unsupported file format: {source_path}")
                    continue
                
                all_dialogue_pairs.extend(pairs)
                self.logger.info(f"Loaded {len(pairs)} dialogue pairs from {source_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load from {source_path}: {e}")
                continue
        
        if not all_dialogue_pairs:
            raise ValueError("No valid dialogue sources found")
        
        # Обработка всех загруженных диалогов
        self._load_from_dialogue_pairs(all_dialogue_pairs)
        self._update_dataset_info("loaded_from_sources", len(self.config.data_sources))
    
    def _load_dialogues_from_json(self, file_path: Path) -> List[Dict]:
        """Загрузка диалогов из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                # JSONL формат (строка = JSON объект)
                dialogues = [json.loads(line) for line in f if line.strip()]
            else:
                # Обычный JSON
                data = json.load(f)
                if isinstance(data, list):
                    dialogues = data
                else:
                    dialogues = data.get('dialogues', data.get('conversations', []))
        
        # Конвертация в стандартный формат
        pairs = []
        for dialogue in dialogues:
            if 'question' in dialogue and 'answer' in dialogue:
                # Простой Q&A формат
                pairs.append({
                    'question': dialogue['question'],
                    'answer': dialogue['answer']
                })
            elif 'conversations' in dialogue:
                # Многоходовой формат
                conversation_pairs = self._extract_qa_pairs_from_conversation(dialogue['conversations'])
                pairs.extend(conversation_pairs)
        
        return pairs
    
    def _load_dialogues_from_text(self, file_path: Path) -> List[Dict]:
        """Загрузка диалогов из текстового файла (формат Q: A:)"""
        pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разбор формата Q: ... A: ...
        qa_pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for question, answer in matches:
            pairs.append({
                'question': question.strip(),
                'answer': answer.strip()
            })
        
        return pairs
    
    def _load_dialogues_from_csv(self, file_path: Path) -> List[Dict]:
        """Загрузка диалогов из CSV файла"""
        import csv
        pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Поддержка различных названий колонок
                question = (row.get('question') or row.get('Q') or 
                          row.get('input') or row.get('user_message', ''))
                answer = (row.get('answer') or row.get('A') or 
                        row.get('response') or row.get('assistant_message', ''))
                
                if question and answer:
                    pairs.append({
                        'question': question.strip(),
                        'answer': answer.strip()
                    })
        
        return pairs
    
    def _filter_dialogue_quality(self, dialogue_pairs: List[Dict]) -> List[Dict]:
        """Фильтрация диалогов по качеству"""
        filtered_pairs = []
        filtered_count = 0
        
        for pair in dialogue_pairs:
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            
            # Проверка длины
            if (len(question) < self.config.min_question_length or 
                len(answer) < self.config.min_answer_length or
                len(question) > self.config.max_question_length or
                len(answer) > self.config.max_answer_length):
                filtered_count += 1
                continue
            
            # Проверка на пустые строки и повторы
            if (not question.strip() or not answer.strip() or
                question.strip() == answer.strip()):
                filtered_count += 1
                continue
            
            # TODO: Семантическая проверка связности (можно добавить позже)
            # if self._check_semantic_relevance(question, answer) < self.config.semantic_similarity_threshold:
            #     filtered_count += 1
            #     continue
            
            filtered_pairs.append(pair)
        
        self.cache_stats['quality_filtered'] = filtered_count
        self.logger.info(f"Quality filter: kept {len(filtered_pairs)}, filtered {filtered_count}")
        
        return filtered_pairs
    
    def _create_train_val_split(self):
        """Создание разделения на train/validation"""
        total_pairs = len(self.question_embeddings)
        
        if self.config.validation_split > 0:
            val_size = int(total_pairs * self.config.validation_split)
            train_size = total_pairs - val_size
            
            # Перемешивание если нужно
            indices = torch.arange(total_pairs)
            if self.config.shuffle_conversations:
                indices = indices[torch.randperm(total_pairs)]
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_questions = self.question_embeddings[train_indices]
            self.train_answers = self.answer_embeddings[train_indices]
            self.val_questions = self.question_embeddings[val_indices]
            self.val_answers = self.answer_embeddings[val_indices]
        else:
            # Нет валидации - все данные для обучения
            self.train_questions = self.question_embeddings
            self.train_answers = self.answer_embeddings
            self.val_questions = torch.empty(0, self.config.embedding_dim)
            self.val_answers = torch.empty(0, self.config.embedding_dim)
    
    def _create_cache_key_for_dialogues(self, dialogue_pairs: List[Dict]) -> str:
        """Создание ключа кэша для диалоговых пар"""
        # Создаем хэш от части диалогов + конфигурации
        sample_dialogues = dialogue_pairs[:50]  # Первые 50 для хэша
        dialogue_content = "\n".join([f"Q:{d['question']} A:{d['answer']}" for d in sample_dialogues])
        config_content = f"{self.config.teacher_model}_{self.config.embedding_dim}_{len(dialogue_pairs)}"
        content = f"{dialogue_content}_{config_content}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Загрузка диалоговых эмбедингов из кэша"""
        if not self.config.use_cache:
            return None
        
        cache_path = Path(self.config.cache_dir) / f"dialogue_{cache_key}.pt"
        
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to load from cache {cache_path}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, cache_data: Dict):
        """Сохранение диалоговых эмбедингов в кэш"""
        if not self.config.cache_embeddings:
            return
        
        cache_path = Path(self.config.cache_dir) / f"dialogue_{cache_key}.pt"
        
        try:
            torch.save(cache_data, cache_path)
            self.logger.info(f"Cached dialogue embeddings to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache dialogue embeddings: {e}")
    
    def _update_dataset_info(self, source_type: str, sample_count: int):
        """Обновление метаданных dataset"""
        self.dataset_info.update({
            'source_type': source_type,
            'dialogue_pairs_count': sample_count,
            'embedding_dim': self.config.embedding_dim,
            'teacher_model': self.config.teacher_model,
            'config': self.config.__dict__
        })
    
    def set_validation_mode(self, is_validation: bool = True):
        """Переключение между train/validation режимами"""
        self.is_validation_mode = is_validation
    
    def __len__(self) -> int:
        """Размер dataset"""
        if self.is_validation_mode:
            return len(self.val_questions)
        else:
            return len(self.train_questions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получение элемента dataset
        
        Returns:
            Tuple[question_embedding, answer_embedding] для обучения 3D Cubic Core
        """
        if self.is_validation_mode:
            question_emb = self.val_questions[idx]
            answer_emb = self.val_answers[idx]
        else:
            question_emb = self.train_questions[idx]
            answer_emb = self.train_answers[idx]
        
        return question_emb, answer_emb
    
    def get_dataloader(self, 
                      batch_size: int = 32, 
                      shuffle: bool = True,
                      num_workers: int = 0,
                      validation: bool = False) -> DataLoader:
        """
        Создание DataLoader для dialogue обучения
        
        Args:
            batch_size: Размер батча
            shuffle: Перемешивать ли данные
            num_workers: Количество воркеров для загрузки
            validation: Использовать validation набор
            
        Returns:
            DataLoader для dialogue training
        """
        # Временно переключаемся в нужный режим
        original_mode = self.is_validation_mode
        self.set_validation_mode(validation)
        
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Восстанавливаем режим
        self.set_validation_mode(original_mode)
        
        return dataloader
    
    def get_sample_dialogues(self, n_samples: int = 5) -> Dict[str, Any]:
        """Получение примеров диалогов для анализа"""
        if not self.dialogue_metadata:
            return {"error": "No dialogue metadata available"}
        
        n_samples = min(n_samples, len(self.dialogue_metadata))
        sample_indices = random.sample(range(len(self.dialogue_metadata)), n_samples)
        
        samples = []
        for idx in sample_indices:
            metadata = self.dialogue_metadata[idx]
            question_emb = self.question_embeddings[idx]
            answer_emb = self.answer_embeddings[idx]
            
            # Вычисление семантической близости Q&A
            cosine_similarity = torch.cosine_similarity(
                question_emb.unsqueeze(0), 
                answer_emb.unsqueeze(0)
            ).item()
            
            samples.append({
                'question': metadata['question'],
                'answer': metadata['answer'],
                'question_length': metadata['question_length'],
                'answer_length': metadata['answer_length'],
                'qa_similarity': cosine_similarity,
                'question_embedding_shape': question_emb.shape,
                'answer_embedding_shape': answer_emb.shape
            })
        
        return {
            'samples': samples,
            'dataset_size': len(self.dialogue_metadata),
            'teacher_model': self.config.teacher_model
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики dataset"""
        if len(self.question_embeddings) == 0:
            return {"error": "Dataset is empty"}
        
        # Базовая статистика
        stats = {
            'total_dialogue_pairs': len(self.question_embeddings),
            'train_pairs': len(self.train_questions),
            'validation_pairs': len(self.val_questions),
            'embedding_dimension': self.question_embeddings.shape[1],
            'teacher_model': self.config.teacher_model,
            'cache_stats': self.cache_stats,
            'config_summary': {
                'quality_filtering': self.config.enable_quality_filter,
                'multiturn_support': self.config.support_multiturn,
                'normalization': self.config.normalize_embeddings,
                'context_noise': self.config.add_context_noise
            }
        }
        
        # Статистика качества эмбедингов
        question_norms = torch.norm(self.question_embeddings, dim=1)
        answer_norms = torch.norm(self.answer_embeddings, dim=1)
        
        qa_similarities = torch.cosine_similarity(
            self.question_embeddings, 
            self.answer_embeddings, 
            dim=1
        )
        
        stats.update({
            'embedding_quality': {
                'question_norm_mean': question_norms.mean().item(),
                'question_norm_std': question_norms.std().item(),
                'answer_norm_mean': answer_norms.mean().item(),
                'answer_norm_std': answer_norms.std().item(),
                'qa_similarity_mean': qa_similarities.mean().item(),
                'qa_similarity_std': qa_similarities.std().item(),
                'qa_similarity_range': [qa_similarities.min().item(), qa_similarities.max().item()]
            }
        })
        
        # Статистика длины текстов (если доступна)
        if self.dialogue_metadata:
            question_lengths = [m['question_length'] for m in self.dialogue_metadata]
            answer_lengths = [m['answer_length'] for m in self.dialogue_metadata]
            
            stats.update({
                'text_statistics': {
                    'question_length_mean': np.mean(question_lengths),
                    'question_length_std': np.std(question_lengths),
                    'answer_length_mean': np.mean(answer_lengths),
                    'answer_length_std': np.std(answer_lengths)
                }
            })
        
        return stats
    
    def save_dataset_info(self, path: str):
        """Сохранение информации о dataset"""
        info = {
            'dataset_info': self.dataset_info,
            'statistics': self.get_statistics(),
            'sample_dialogues': self.get_sample_dialogues(3)
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Dataset info saved to: {path}")
    
    def __repr__(self):
        return (f"DialogueDataset(pairs={len(self.question_embeddings)}, "
                f"train={len(self.train_questions)}, val={len(self.val_questions)}, "
                f"dim={self.config.embedding_dim}, teacher={self.config.teacher_model})")


# ================================
# HELPER FUNCTIONS
# ================================

def create_dialogue_dataset(dialogue_pairs: List[Dict], 
                          teacher_model: str = "llama3-8b",
                          validation_split: float = 0.2,
                          **kwargs) -> DialogueDataset:
    """
    Удобная функция для создания DialogueDataset из диалоговых пар
    
    Args:
        dialogue_pairs: Список Q&A пар [{"question": str, "answer": str}, ...]
        teacher_model: Teacher LLM модель для генерации эмбедингов
        validation_split: Доля данных для валидации
        **kwargs: Дополнительные параметры конфигурации
        
    Returns:
        Готовый DialogueDataset
    """
    config = DialogueConfig(
        teacher_model=teacher_model,
        validation_split=validation_split,
        **kwargs
    )
    
    return DialogueDataset(
        config=config,
        dialogue_pairs=dialogue_pairs
    )


def create_conversation_dataset(conversations: List[List[Dict]],
                              teacher_model: str = "llama3-8b", 
                              validation_split: float = 0.2,
                              **kwargs) -> DialogueDataset:
    """
    Удобная функция для создания DialogueDataset из многоходовых диалогов
    
    Args:
        conversations: Список диалогов [[{role, text}, ...], ...]
        teacher_model: Teacher LLM модель для генерации эмбедингов
        validation_split: Доля данных для валидации
        **kwargs: Дополнительные параметры конфигурации
        
    Returns:
        Готовый DialogueDataset
    """
    # Устанавливаем support_multiturn=True, но позволяем override через kwargs
    kwargs.setdefault('support_multiturn', True)
    
    config = DialogueConfig(
        teacher_model=teacher_model,
        validation_split=validation_split,
        **kwargs
    )
    
    return DialogueDataset(
        config=config,
        conversations=conversations
    )


def load_dialogue_dataset_from_files(file_paths: List[str],
                                   dialogue_format: str = "qa_pairs",
                                   teacher_model: str = "llama3-8b",
                                   validation_split: float = 0.2,
                                   **kwargs) -> DialogueDataset:
    """
    Удобная функция для загрузки DialogueDataset из файлов
    
    Args:
        file_paths: Пути к файлам с диалогами
        dialogue_format: Формат диалогов (qa_pairs, conversation, json)
        teacher_model: Teacher LLM модель для генерации эмбедингов
        validation_split: Доля данных для валидации
        **kwargs: Дополнительные параметры конфигурации
        
    Returns:
        Готовый DialogueDataset
    """
    config = DialogueConfig(
        data_sources=file_paths,
        dialogue_format=dialogue_format,
        teacher_model=teacher_model,
        validation_split=validation_split,
        **kwargs
    )
    
    return DialogueDataset(config=config)