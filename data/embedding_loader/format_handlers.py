"""
Format handlers for different embedding file formats.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List

import torch
import numpy as np
from torch import Tensor


logger = logging.getLogger(__name__)


class FormatHandler(ABC):
    """Базовый абстрактный класс для обработчиков форматов эмбедингов."""
    
    @abstractmethod
    def load(self, path: str) -> Union[Tensor, np.ndarray]:
        """
        Загрузка эмбедингов из файла.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Загруженные эмбединги
        """
        pass
    
    @abstractmethod
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """
        Получение словаря токен -> индекс.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Словарь токенов
        """
        pass


class TextFormatHandler(FormatHandler):
    """Базовый обработчик для текстовых форматов."""
    
    def _parse_text_line(self, line: str) -> tuple:
        """
        Парсинг строки из текстового файла.
        
        Args:
            line: Строка для парсинга
            
        Returns:
            (token, vector) кортеж
        """
        parts = line.strip().split()
        if len(parts) < 2:
            return None, None
        
        token = parts[0]
        try:
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            return token, vector
        except ValueError:
            logger.warning(f"Failed to parse line: {line[:50]}...")
            return None, None


class Word2VecHandler(TextFormatHandler):
    """Обработчик Word2Vec формата (.txt и .bin)."""
    
    def load(self, path: str) -> np.ndarray:
        """
        Загрузка Word2Vec эмбедингов.
        
        Args:
            path: Путь к файлу (.txt или .bin)
            
        Returns:
            numpy.ndarray: Матрица эмбедингов
        """
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.bin':
            return self._load_binary(path)
        elif file_ext == '.txt':
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported Word2Vec file extension: {file_ext}")
    
    def _load_text(self, path: str) -> np.ndarray:
        """Загрузка из текстового формата."""
        embeddings = []
        vocab = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            # Первая строка может содержать размеры
            first_line = f.readline().strip()
            if len(first_line.split()) == 2:
                vocab_size, embed_dim = map(int, first_line.split())
                logger.info(f"Word2Vec info: vocab_size={vocab_size}, embed_dim={embed_dim}")
            else:
                # Если первая строка - это эмбединг, вернемся к началу
                f.seek(0)
            
            for idx, line in enumerate(f):
                token, vector = self._parse_text_line(line)
                if token is not None and vector is not None:
                    vocab[token] = idx
                    embeddings.append(vector)
        
        if not embeddings:
            raise ValueError("No valid embeddings found in file")
        
        self._vocabulary = vocab
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Loaded Word2Vec embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def _load_binary(self, path: str) -> np.ndarray:
        """Загрузка из бинарного формата (требует gensim)."""
        try:
            from gensim.models import KeyedVectors
            
            logger.info(f"Loading Word2Vec binary file: {path}")
            model = KeyedVectors.load_word2vec_format(path, binary=True)
            
            # Сохраняем словарь
            self._vocabulary = {word: idx for idx, word in enumerate(model.index_to_key)}
            
            embeddings = model.vectors
            logger.info(f"Loaded Word2Vec binary embeddings: {embeddings.shape}")
            
            return embeddings
            
        except ImportError:
            raise ImportError("gensim is required for loading binary Word2Vec files. "
                            "Install with: pip install gensim")
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """Получение словаря Word2Vec."""
        if not hasattr(self, '_vocabulary'):
            # Загружаем эмбединги, чтобы построить словарь
            self.load(path)
        return self._vocabulary


class GloVeHandler(TextFormatHandler):
    """Обработчик GloVe формата (.txt)."""
    
    def load(self, path: str) -> np.ndarray:
        """
        Загрузка GloVe эмбедингов.
        
        Args:
            path: Путь к .txt файлу
            
        Returns:
            numpy.ndarray: Матрица эмбедингов
        """
        embeddings = []
        vocab = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token, vector = self._parse_text_line(line)
                if token is not None and vector is not None:
                    vocab[token] = idx
                    embeddings.append(vector)
        
        if not embeddings:
            raise ValueError("No valid embeddings found in GloVe file")
        
        self._vocabulary = vocab
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Loaded GloVe embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """Получение словаря GloVe."""
        if not hasattr(self, '_vocabulary'):
            self.load(path)
        return self._vocabulary


class BertHandler(FormatHandler):
    """Обработчик BERT эмбедингов (.pt, .pkl)."""
    
    def load(self, path: str) -> Union[Tensor, np.ndarray]:
        """
        Загрузка BERT эмбедингов.
        
        Args:
            path: Путь к файлу (.pt или .pkl)
            
        Returns:
            torch.Tensor или numpy.ndarray: Эмбединги
        """
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.pt':
            return self._load_pytorch(path)
        elif file_ext == '.pkl':
            return self._load_pickle(path)
        else:
            raise ValueError(f"Unsupported BERT file extension: {file_ext}")
    
    def _load_pytorch(self, path: str) -> Tensor:
        """Загрузка из PyTorch формата."""
        try:
            embeddings = torch.load(path, map_location='cpu')
            logger.info(f"Loaded BERT PyTorch embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch embeddings: {e}")
    
    def _load_pickle(self, path: str) -> np.ndarray:
        """Загрузка из pickle формата."""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Данные могут быть в разных форматах
            if isinstance(data, dict):
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                elif 'vectors' in data:
                    embeddings = data['vectors']
                else:
                    # Берем первое tensor/array значение
                    for key, value in data.items():
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            embeddings = value
                            break
                    else:
                        raise ValueError("No embeddings found in pickle file")
            else:
                embeddings = data
            
            # Конвертируем в numpy если нужно
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            
            logger.info(f"Loaded BERT pickle embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise ValueError(f"Failed to load pickle embeddings: {e}")
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """
        Получение словаря BERT.
        
        Note: BERT обычно использует словарь токенайзера,
        который не всегда доступен в файле эмбедингов.
        """
        logger.warning("BERT vocabulary not available from embedding file. "
                      "Use tokenizer vocabulary instead.")
        return {} 


class LLMHandler(FormatHandler):
    """
    Обработчик для извлечения эмбедингов из открытых LLM моделей.
    
    Поддерживает Knowledge Distillation через извлечение эмбедингов
    из предобученных языковых моделей (LLaMA, Mistral, GPT и др.)
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Инициализация LLM handler.
        
        Args:
            model_name: Имя модели на HuggingFace Hub
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Определяем устройство из центрального конфига
        self._device = self._get_device_from_config()
        logger.info(f"Initialized LLM handler for {model_name} on {self._device}")
    
    def _get_device_from_config(self) -> str:
        """Получить устройство из центрального конфига"""
        try:
            # Проверяем доступность torch.cuda
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                return "cpu"
            
            # Загружаем конфиг
            try:
                from utils.config_loader import config_manager
                if config_manager.should_use_gpu():
                    device = config_manager.get_gpu_device()
                    logger.info(f"Using GPU from config: {device}")
                    return device
                else:
                    logger.info("GPU disabled in config, using CPU")
                    return "cpu"
            except Exception as e:
                logger.warning(f"Could not load config, defaulting to GPU: {e}")
                return "cuda:0"  # Fallback to GPU if config fails
                
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"
    
    def _is_large_model(self) -> bool:
        """Определить, является ли модель большой (требует device_map)"""
        large_model_patterns = [
            "llama", "Llama", "LLaMA", "LLAMA",
            "mistral", "Mistral",
            "codellama", "CodeLlama",
            # Локальные пути тоже считаем большими моделями
            "Meta-Llama", "meta-llama"
        ]
        
        model_name_lower = self.model_name.lower()
        for pattern in large_model_patterns:
            if pattern.lower() in model_name_lower:
                return True
        return False
    
    def load_model(self):
        """Ленивая загрузка LLM модели и токенайзера."""
        if self.model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                
                logger.info(f"Loading LLM model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Для больших моделей используем device_map
                if self._device.startswith("cuda") and self._is_large_model():
                    logger.info(f"Loading large model with device_map auto")
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.float16  # Используем FP16 для экономии памяти
                    )
                else:
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.to(self._device)
                
                self.model.eval()
                
                logger.info(f"Successfully loaded {self.model_name}")
                
            except ImportError:
                raise ImportError("transformers library is required for LLM support. "
                                "Install with: pip install transformers")
            except Exception as e:
                raise ValueError(f"Failed to load LLM model {self.model_name}: {e}")
    
    def load(self, path: str) -> np.ndarray:
        """
        Загрузка заранее сохраненных LLM эмбедингов.
        
        Args:
            path: Путь к файлу с эмбедингами (.pt, .npy)
            
        Returns:
            numpy.ndarray: Матрица эмбедингов
        """
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.pt':
            embeddings = torch.load(path, map_location='cpu')
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
        elif file_ext == '.npy':
            embeddings = np.load(path)
        else:
            raise ValueError(f"Unsupported LLM embedding file format: {file_ext}")
        
        logger.info(f"Loaded LLM embeddings from {path}: {embeddings.shape}")
        return embeddings
    
    def generate_embeddings(self, texts: List[str], 
                          pooling_strategy: str = "mean") -> torch.Tensor:
        """
        Генерация эмбедингов из текстов в реальном времени.
        
        Args:
            texts: Список текстов для обработки
            pooling_strategy: Стратегия агрегации токенов ("mean", "cls", "max")
            
        Returns:
            torch.Tensor: Эмбединги текстов [batch_size, hidden_size]
        """
        self.load_model()
        
        if not texts:
            raise ValueError("Empty text list provided")
        
        # Добавляем padding token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Токенизация с padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Извлечение эмбедингов
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
            
            # Применяем стратегию агрегации
            if pooling_strategy == "mean":
                # Учитываем attention mask для корректного усреднения
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling_strategy == "cls":
                # Берем первый токен (обычно [CLS])
                embeddings = hidden_states[:, 0, :]
            elif pooling_strategy == "max":
                # Максимальное значение по sequence dimension
                embeddings = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        embeddings = embeddings.cpu()
        logger.info(f"Generated embeddings for {len(texts)} texts: {embeddings.shape}")
        
        return embeddings
    
    def batch_generate_embeddings(self, texts: List[str], 
                                batch_size: int = 16) -> torch.Tensor:
        """
        Генерация эмбедингов батчами для больших объемов текста.
        
        Args:
            texts: Список текстов
            batch_size: Размер батча
            
        Returns:
            torch.Tensor: Объединенные эмбединги
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_embeddings(self, embeddings: torch.Tensor, save_path: str):
        """
        Сохранение сгенерированных эмбедингов для кэширования.
        
        Args:
            embeddings: Эмбединги для сохранения
            save_path: Путь для сохранения
        """
        file_ext = os.path.splitext(save_path)[1].lower()
        
        if file_ext == '.pt':
            torch.save(embeddings, save_path)
        elif file_ext == '.npy':
            np.save(save_path, embeddings.numpy())
        else:
            raise ValueError(f"Unsupported save format: {file_ext}")
        
        logger.info(f"Saved LLM embeddings to {save_path}")
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """
        Получение словаря LLM (если применимо).
        
        Note: Для LLM обычно используется токенайзер
        """
        self.load_model()
        if hasattr(self.tokenizer, 'get_vocab'):
            return self.tokenizer.get_vocab()
        else:
            logger.warning("LLM tokenizer does not support vocabulary extraction")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о загруженной модели."""
        self.load_model()
        
        return {
            "model_name": self.model_name,
            "device": self._device,
            "hidden_size": self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else "unknown",
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else "unknown",
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', "unknown")
        }


# Поддерживаемые LLM модели для Knowledge Distillation
SUPPORTED_LLM_MODELS = {
    # Локальные модели (приоритет)
    "llama3-8b-local": r"C:\Users\n0n4a\Meta-Llama-3-8B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",  # Fallback для онлайн версии
    
    # Открытые модели
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1", 
    "codellama-7b": "codellama/CodeLlama-7b-hf",
    
    # Более легкие модели
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "gpt2": "gpt2",
    
    # Для тестирования
    "dialogpt": "microsoft/DialoGPT-medium",
}


def create_llm_handler(model_key: str) -> LLMHandler:
    """
    Фабричная функция для создания LLM handler.
    
    Args:
        model_key: Ключ модели из SUPPORTED_LLM_MODELS
        
    Returns:
        LLMHandler: Настроенный handler
    """
    if model_key not in SUPPORTED_LLM_MODELS:
        raise ValueError(f"Unsupported model key: {model_key}. "
                        f"Supported: {list(SUPPORTED_LLM_MODELS.keys())}")
    
    model_name = SUPPORTED_LLM_MODELS[model_key]
    return LLMHandler(model_name) 