#!/usr/bin/env python3
"""
Загрузчик готовых эмбеддингов из предварительно сгенерированного файла
Используется для быстрого обучения без пересчета эмбеддингов
"""

import torch
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingDataset(Dataset):
    """Датасет для предварительно вычисленных эмбеддингов"""

    def __init__(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ):
        assert len(question_embeddings) == len(
            answer_embeddings
        ), f"Mismatch: {len(question_embeddings)} questions vs {len(answer_embeddings)} answers"

        self.question_embeddings = question_embeddings
        self.answer_embeddings = answer_embeddings

    def __len__(self):
        return len(self.question_embeddings)

    def __getitem__(self, idx):
        return self.question_embeddings[idx], self.answer_embeddings[idx]


class PrecomputedEmbeddingLoader:
    """Загрузчик готовых эмбеддингов из файла"""

    def __init__(self):
        self.dataset_cache = {}  # Кэш для загруженных датасетов

    def load_dataset(
        self, embeddings_file: str, use_cache: bool = True
    ) -> PrecomputedEmbeddingDataset:
        """
        Загружает датасет эмбеддингов из файла

        Args:
            embeddings_file: Путь к файлу с эмбеддингами (.pt)
            use_cache: Использовать кэш для повторных загрузок

        Returns:
            PrecomputedEmbeddingDataset
        """
        embeddings_path = Path(embeddings_file)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embeddings_file}")

        # Проверяем кэш
        cache_key = str(embeddings_path.absolute())
        if use_cache and cache_key in self.dataset_cache:
            logger.info(f"[CACHE] Loading dataset from cache: {embeddings_path.name}")
            return self.dataset_cache[cache_key]

        logger.info(f"[LOAD] Loading embedding dataset from: {embeddings_path.name}")

        try:
            # Загружаем данные
            # Используем GPU если доступно
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            data = torch.load(embeddings_file, map_location=device)

            # Проверяем структуру данных
            required_keys = ["question_embeddings", "answer_embeddings"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Missing keys in embedding file: {missing_keys}")

            question_embeddings = data["question_embeddings"]
            answer_embeddings = data["answer_embeddings"]

            # Проверяем размерности
            if question_embeddings.shape[0] != answer_embeddings.shape[0]:
                raise ValueError(
                    f"Size mismatch: {question_embeddings.shape[0]} questions vs {answer_embeddings.shape[0]} answers"
                )

            if question_embeddings.shape[1] != answer_embeddings.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: {question_embeddings.shape[1]} vs {answer_embeddings.shape[1]}"
                )

            # Создаем датасет
            dataset = PrecomputedEmbeddingDataset(
                question_embeddings, answer_embeddings
            )

            # Информация о датасете
            logger.info(f"✅ Dataset loaded successfully:")
            logger.info(f"   Size: {len(dataset):,} pairs")
            logger.info(f"   Embedding dimension: {question_embeddings.shape[1]}")
            logger.info(f"   Teacher model: {data.get('teacher_model', 'unknown')}")
            logger.info(
                f"   File size: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB"
            )

            # Проверяем нормализацию
            q_norms = question_embeddings.norm(dim=1)
            a_norms = answer_embeddings.norm(dim=1)
            logger.info(
                f"   Question embeddings norm: mean={q_norms.mean():.4f}, std={q_norms.std():.4f}"
            )
            logger.info(
                f"   Answer embeddings norm: mean={a_norms.mean():.4f}, std={a_norms.std():.4f}"
            )

            # Сохраняем в кэш
            if use_cache:
                self.dataset_cache[cache_key] = dataset

            return dataset

        except Exception as e:
            logger.error(f"❌ Failed to load embedding dataset: {e}")
            raise

    def list_available_datasets(self, data_dir: str = "data/embeddings") -> list:
        """Список доступных датасетов эмбеддингов"""
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return []

        # Ищем .pt файлы
        embedding_files = list(data_path.glob("*.pt"))

        datasets_info = []
        for file_path in embedding_files:
            try:
                # Загружаем метаданные без полной загрузки
                # Используем GPU если доступно
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                data = torch.load(file_path, map_location=device)

                info = {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "size": data.get("size", "unknown"),
                    "teacher_model": data.get("teacher_model", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                    "file_size_mb": file_path.stat().st_size / 1024 / 1024,
                }

                datasets_info.append(info)

            except Exception as e:
                logger.warning(f"Failed to read metadata from {file_path.name}: {e}")

        # Сортируем по времени создания (новые первыми)
        datasets_info.sort(key=lambda x: x["timestamp"], reverse=True)

        return datasets_info

    def get_latest_dataset(self, data_dir: str = "data/embeddings") -> Optional[str]:
        """Возвращает путь к самому новому датасету"""
        datasets = self.list_available_datasets(data_dir)

        if not datasets:
            return None

        return datasets[0]["file_path"]

    def clear_cache(self):
        """Очищает кэш загруженных датасетов"""
        self.dataset_cache.clear()
        logger.info("[CLEAR] Dataset cache cleared")


def create_precomputed_dataset(embeddings_file: str) -> PrecomputedEmbeddingDataset:
    """
    Удобная функция для создания датасета из файла эмбеддингов
    Совместима с интерфейсом simple_embedding_fallback
    """
    loader = PrecomputedEmbeddingLoader()
    return loader.load_dataset(embeddings_file)


def test_precomputed_loader():
    """Тест загрузчика готовых эмбеддингов"""
    print("🧪 Testing PrecomputedEmbeddingLoader")

    loader = PrecomputedEmbeddingLoader()

    # 1. Список доступных датасетов
    print("\n[LIST] Available datasets:")
    datasets = loader.list_available_datasets()

    if not datasets:
        print("   No datasets found. Run generate_large_embedding_dataset.py first!")
        return

    for i, dataset_info in enumerate(datasets):
        print(f"   {i+1}. {dataset_info['filename']}")
        print(f"      Size: {dataset_info['size']:,} pairs")
        print(f"      Model: {dataset_info['teacher_model']}")
        print(f"      File size: {dataset_info['file_size_mb']:.1f} MB")
        print(f"      Created: {dataset_info['timestamp']}")
        print()

        # 2. Загружаем самый новый датасет
    latest_file = loader.get_latest_dataset()
    if latest_file:
        print(f"[LOAD] Loading latest dataset: {Path(latest_file).name}")
        dataset = loader.load_dataset(latest_file)

        # 3. Проверяем данные
        sample_q, sample_a = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Question embedding shape: {sample_q.shape}")
        print(f"   Answer embedding shape: {sample_a.shape}")
        print(f"   Question norm: {sample_q.norm().item():.6f}")
        print(f"   Answer norm: {sample_a.norm().item():.6f}")

        # 4. Тест повторной загрузки (кэш)
        print(f"\n[CACHE] Testing cache...")
        dataset2 = loader.load_dataset(latest_file)
        print(f"   Cache working: {dataset is dataset2}")


if __name__ == "__main__":
    test_precomputed_loader()
