#!/usr/bin/env python3
"""
Утилита для кэширования моделей локально
=======================================

Управляет загрузкой и кэшированием моделей для работы без интернета.
Оптимизировано для RTX 5090 с быстрой загрузкой с локального диска.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import hashlib
import json

from .logging import get_logger
from ..config.simple_config import SimpleProjectConfig

logger = get_logger(__name__)


class ModelCacheManager:
    """
    Менеджер локального кэша моделей

    Обеспечивает быструю загрузку моделей с локального диска
    и автоматическую загрузку при необходимости.
    """

    def __init__(self, config: SimpleProjectConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Пути
        self.local_models_dir = Path(config.embedding.local_models_dir)
        self.auto_download = config.embedding.auto_download_models
        self.prefer_local = config.embedding.prefer_local_models

        # Создаем директорию если не существует
        self.local_models_dir.mkdir(parents=True, exist_ok=True)

        # Метаданные кэша
        self.cache_metadata_file = self.local_models_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        self.logger.info(f"🗄️ ModelCacheManager initialized: {self.local_models_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Загрузка метаданных кэша"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")

        return {"models": {}, "created": None, "version": "1.0"}

    def _save_metadata(self):
        """Сохранение метаданных кэша"""
        try:
            with open(self.cache_metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")

    def _get_model_path(self, model_name: str) -> Path:
        """Получение пути к локальной модели"""
        # Очищаем имя модели для использования в пути
        clean_name = model_name.replace("/", "_").replace("-", "_")
        return self.local_models_dir / clean_name

    def is_model_cached(self, model_name: str) -> bool:
        """Проверка наличия модели в кэше"""
        model_path = self._get_model_path(model_name)

        self.logger.debug(f"🔍 Checking cache for '{model_name}' at: {model_path}")

        # Проверяем основные файлы
        required_files = ["config.json"]

        if model_name.startswith("distilbert"):
            required_files.extend(["tokenizer.json"])
            # Проверяем наличие модели в любом из поддерживаемых форматов
            model_files = ["pytorch_model.bin", "model.safetensors"]

        missing_files = []
        for file in required_files:
            file_path = model_path / file
            if not file_path.exists():
                missing_files.append(file)
                self.logger.debug(f"  ❌ Missing: {file}")
            else:
                self.logger.debug(f"  ✅ Found: {file}")

        # Для distilbert проверяем наличие файла модели в любом формате
        if model_name.startswith("distilbert"):
            model_file_found = False
            for model_file in model_files:
                model_file_path = model_path / model_file
                if model_file_path.exists():
                    self.logger.debug(f"  ✅ Found model file: {model_file}")
                    model_file_found = True
                    break
                else:
                    self.logger.debug(f"  ❌ Missing model file: {model_file}")

            if not model_file_found:
                missing_files.append(
                    "model file (pytorch_model.bin or model.safetensors)"
                )

        is_cached = len(missing_files) == 0

        if is_cached:
            self.logger.debug(f"✅ Model '{model_name}' is cached at: {model_path}")
        else:
            self.logger.debug(
                f"❌ Model '{model_name}' not cached. Missing files: {missing_files}"
            )

        return is_cached

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Получение пути к модели с улучшенной логикой кэширования.

        Args:
            model_name: Имя модели (например, 'distilbert-base-uncased')

        Returns:
            Путь к модели для загрузки
        """
        self.logger.debug(f"🔍 get_model_path called for '{model_name}'")
        self.logger.debug(f"  prefer_local: {self.prefer_local}")
        self.logger.debug(f"  auto_download: {self.auto_download}")

        local_path = self._get_model_path(model_name)
        is_cached = self.is_model_cached(model_name)

        self.logger.debug(f"  local_path: {local_path}")
        self.logger.debug(f"  is_cached: {is_cached}")

        # 1. Если предпочитаем локальную версию
        if self.prefer_local:
            if is_cached:
                self.logger.info(f"📁 Using cached model: {local_path}")
                return str(local_path)

            # Локальная версия предпочтительна, но её нет. Пытаемся скачать.
            if self.auto_download:
                self.logger.info(
                    f"🔄 Model '{model_name}' not in cache, attempting to download."
                )
                if self._download_model(model_name):
                    self.logger.info(
                        f"✅ Download successful. Now using cached model: {local_path}"
                    )
                    return str(local_path)
                else:
                    self.logger.warning(
                        f"⚠️ Failed to download '{model_name}'. Falling back to online version."
                    )
            else:
                self.logger.warning(
                    f"Local model preferred but not found, and auto-download is off. Falling back to online."
                )

        # 2. Если локальная версия не предпочтительна или скачать не удалось, используем онлайн
        self.logger.info(f"🌐 Using online model: {model_name}")
        return model_name

    def _download_model(self, model_name: str) -> bool:
        """
        Загрузка модели в локальный кэш

        Args:
            model_name: Имя модели для загрузки

        Returns:
            True если загрузка успешна
        """
        try:
            from transformers import AutoModel, AutoTokenizer

            local_path = self._get_model_path(model_name)
            local_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"📥 Downloading {model_name} to {local_path}")

            # Загружаем токенизатор
            self.logger.info("  Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(local_path))

            # Загружаем модель
            self.logger.info("  Downloading model...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(str(local_path))

            # Обновляем метаданные
            self.metadata["models"][model_name] = {
                "path": str(local_path),
                "downloaded_at": None,  # timestamp
                "size_mb": self._get_directory_size_mb(local_path),
            }
            self._save_metadata()

            self.logger.info(f"✅ Successfully cached {model_name}")
            return True

        except ImportError:
            self.logger.error("transformers library not available for model download")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False

    def _get_directory_size_mb(self, path: Path) -> float:
        """Вычисляет размер директории в MB"""
        total_size = 0
        try:
            for file in path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    def download_distilbert(self) -> bool:
        """Загрузка DistilBERT (совместимость с legacy)"""
        return self._download_model("distilbert-base-uncased")

    def clear_cache(self, model_name: Optional[str] = None):
        """Очистка кэша модели или всех моделей"""
        if model_name:
            model_path = self._get_model_path(model_name)
            if model_path.exists():
                shutil.rmtree(model_path)
                self.metadata["models"].pop(model_name, None)
                self._save_metadata()
                self.logger.info(f"🗑️ Cleared cache for {model_name}")
        else:
            # Очищаем весь кэш
            for path in self.local_models_dir.iterdir():
                if path.is_dir() and path.name != "__pycache__":
                    shutil.rmtree(path)
            self.metadata["models"] = {}
            self._save_metadata()
            self.logger.info("🗑️ Cleared all model cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Информация о кэше моделей"""
        total_size = 0
        model_count = 0

        for model_name, info in self.metadata["models"].items():
            if "size_mb" in info:
                total_size += info["size_mb"]
                model_count += 1

        return {
            "models_count": model_count,
            "total_size_mb": total_size,
            "cache_dir": str(self.local_models_dir),
            "models": list(self.metadata["models"].keys()),
        }

    def verify_model_integrity(self, model_name: str) -> bool:
        """Проверка целостности модели"""
        if not self.is_model_cached(model_name):
            return False

        model_path = self._get_model_path(model_name)

        try:
            # Пытаемся загрузить модель для проверки
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModel.from_pretrained(str(model_path))

            self.logger.info(f"✅ Model {model_name} integrity verified")
            return True

        except Exception as e:
            self.logger.warning(f"❌ Model {model_name} integrity check failed: {e}")
            return False


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===

_global_cache_manager: Optional[ModelCacheManager] = None


def get_model_cache_manager(
    config: Optional[SimpleProjectConfig] = None,
) -> ModelCacheManager:
    """Получение глобального менеджера кэша моделей"""
    global _global_cache_manager

    if _global_cache_manager is None:
        if config is None:
            from ..config.simple_config import get_project_config

            config = get_project_config()
        logger.debug("🔧 Creating new global ModelCacheManager")
        _global_cache_manager = ModelCacheManager(config)
    else:
        logger.debug("♻️ Reusing existing global ModelCacheManager")

    return _global_cache_manager


# === LEGACY COMPATIBILITY ===


def download_distilbert() -> Optional[str]:
    """Legacy функция для загрузки DistilBERT"""
    manager = get_model_cache_manager()
    if manager.download_distilbert():
        return str(manager._get_model_path("distilbert-base-uncased"))
    return None


def get_distilbert_path() -> Optional[str]:
    """Получение пути к DistilBERT (локального или для загрузки)"""
    manager = get_model_cache_manager()
    return manager.get_model_path("distilbert-base-uncased")


# === UTILITY FUNCTIONS ===


def setup_model_cache(models: list = None) -> Dict[str, bool]:
    """
    Настройка кэша моделей

    Args:
        models: Список моделей для загрузки. По умолчанию ['distilbert-base-uncased']

    Returns:
        Словарь результатов загрузки
    """
    if models is None:
        models = ["distilbert-base-uncased"]

    manager = get_model_cache_manager()
    results = {}

    for model in models:
        logger.info(f"🔄 Setting up cache for {model}")
        results[model] = manager._download_model(model)

    # Показываем статистику
    cache_info = manager.get_cache_info()
    logger.info(f"📊 Cache setup complete:")
    logger.info(f"  Models: {cache_info['models_count']}")
    logger.info(f"  Total size: {cache_info['total_size_mb']:.1f} MB")

    return results


def check_model_cache_status() -> Dict[str, Any]:
    """Проверка статуса кэша моделей"""
    manager = get_model_cache_manager()
    info = manager.get_cache_info()

    # Добавляем статус доступности моделей
    info["model_status"] = {}
    for model in ["distilbert-base-uncased"]:
        info["model_status"][model] = {
            "cached": manager.is_model_cached(model),
            "path": manager.get_model_path(model),
        }

    return info
