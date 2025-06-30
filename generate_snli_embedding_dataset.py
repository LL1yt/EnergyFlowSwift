#!/usr/bin/env python3
"""
Генератор эмбеддингов из SNLI датасета для обучения 3D куба
Использует 1/5 часть SNLI (Stanford Natural Language Inference) датасета
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import random
from datasets import load_dataset

from simple_embedding_fallback import SimpleFallbackEmbeddingLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNLIEmbeddingGenerator:
    """Генератор эмбеддингов из SNLI датасета"""

    def __init__(self, teacher_model: str = "distilbert-base-uncased"):
        self.teacher_model = teacher_model
        self.loader = SimpleFallbackEmbeddingLoader(teacher_model)

    def load_snli_dataset(self, size_fraction: float = 0.2) -> List[Dict[str, str]]:
        """
        Загружает SNLI датасет и извлекает premise-hypothesis пары

        Args:
            size_fraction: Какую часть датасета использовать (0.2 = 1/5)

        Returns:
            Список словарей с парами premise-hypothesis
        """
        logger.info(
            f"[SNLI] Загружаем SNLI датасет (используем {size_fraction*100:.0f}% данных)"
        )

        try:
            # Загружаем SNLI датасет
            dataset = load_dataset("snli")

            # Используем только train split для максимального размера
            train_data = dataset["train"]
            total_size = len(train_data)

            # Вычисляем нужный размер
            target_size = int(total_size * size_fraction)

            logger.info(f"[SNLI] Полный размер: {total_size:,} примеров")
            logger.info(f"[SNLI] Будем использовать: {target_size:,} примеров")

            # Случайная выборка для разнообразия
            indices = random.sample(range(total_size), target_size)

            # Извлекаем данные
            pairs = []
            valid_labels = {
                "entailment",
                "contradiction",
                "neutral",
            }  # Исключаем -1 (invalid)

            for idx in indices:
                example = train_data[idx]

                # Проверяем валидность данных
                if (
                    example["label"] != -1  # Валидный label
                    and example["premise"]  # Не пустой premise
                    and example["hypothesis"]  # Не пустой hypothesis
                    and len(example["premise"].strip()) > 10  # Достаточно длинный
                    and len(example["hypothesis"].strip()) > 10
                ):

                    # Создаем пару в формате question-answer
                    pair = {
                        "question": example["premise"],
                        "answer": example["hypothesis"],
                        "label": example["label"],  # Сохраняем label для анализа
                        "snli_id": idx,
                    }
                    pairs.append(pair)

            logger.info(f"[SNLI] Извлечено {len(pairs):,} валидных пар")

            # Статистика по labels
            label_counts = {}
            for pair in pairs:
                label = pair["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            logger.info(f"[SNLI] Распределение по labels:")
            label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
            for label_id, count in label_counts.items():
                logger.info(
                    f"   {label_names.get(label_id, 'unknown')}: {count:,} ({count/len(pairs)*100:.1f}%)"
                )

            return pairs

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке SNLI: {e}")
            raise

    def create_embedding_dataset(
        self,
        snli_pairs: List[Dict[str, str]],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Создает тензоры эмбеддингов из SNLI пар

        Args:
            snli_pairs: Список пар premise-hypothesis
            normalize: Нормализовать эмбеддинги
            batch_size: Размер батча для обработки

        Returns:
            Tuple (question_embeddings, answer_embeddings)
        """
        logger.info(f"[EMBED] Создаем эмбеддинги для {len(snli_pairs):,} пар")

        # Извлекаем тексты
        premises = [pair["question"] for pair in snli_pairs]  # premise как question
        hypotheses = [pair["answer"] for pair in snli_pairs]  # hypothesis как answer

        # Создаем эмбеддинги батчами для экономии памяти
        premise_embeddings = []
        hypothesis_embeddings = []

        num_batches = (len(premises) + batch_size - 1) // batch_size

        for i in range(0, len(premises), batch_size):
            batch_end = min(i + batch_size, len(premises))
            batch_premises = premises[i:batch_end]
            batch_hypotheses = hypotheses[i:batch_end]

            logger.info(f"[EMBED] Обрабатываем батч {i//batch_size + 1}/{num_batches}")

            # Получаем эмбеддинги
            premise_batch_emb = self.loader.encode_texts(batch_premises)
            hypothesis_batch_emb = self.loader.encode_texts(batch_hypotheses)

            premise_embeddings.append(premise_batch_emb)
            hypothesis_embeddings.append(hypothesis_batch_emb)

            # Очищаем память
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Объединяем батчи
        premise_embeddings = torch.cat(premise_embeddings, dim=0)
        hypothesis_embeddings = torch.cat(hypothesis_embeddings, dim=0)

        # Нормализация
        if normalize:
            premise_embeddings = torch.nn.functional.normalize(
                premise_embeddings, dim=1
            )
            hypothesis_embeddings = torch.nn.functional.normalize(
                hypothesis_embeddings, dim=1
            )
            logger.info("[EMBED] Эмбеддинги нормализованы")

        logger.info(f"[EMBED] Создано эмбеддингов:")
        logger.info(f"   Premise embeddings: {premise_embeddings.shape}")
        logger.info(f"   Hypothesis embeddings: {hypothesis_embeddings.shape}")
        logger.info(f"   Размерность: {premise_embeddings.shape[1]}")

        return premise_embeddings, hypothesis_embeddings

    def save_dataset(
        self,
        premise_embeddings: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
        snli_pairs: List[Dict[str, str]],
        output_dir: str = "data/embeddings",
    ):
        """Сохраняет эмбеддинги в формате совместимом с precomputed_embedding_loader"""

        # Создаем директорию если нужно
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Создаем имя файла с информацией о датасете
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snli_embeddings_{len(snli_pairs)}pairs_{self.teacher_model.replace('/', '_')}_{timestamp}.pt"
        file_path = output_path / filename

        # Подготавливаем данные для сохранения
        save_data = {
            "question_embeddings": premise_embeddings,
            "answer_embeddings": hypothesis_embeddings,
            "teacher_model": self.teacher_model,
            "timestamp": timestamp,
            "size": len(snli_pairs),
            "dataset_info": {
                "source": "SNLI (Stanford Natural Language Inference)",
                "pairs_count": len(snli_pairs),
                "embedding_dim": premise_embeddings.shape[1],
                "data_format": "premise -> hypothesis pairs",
                "fraction_used": "1/5 of SNLI dataset",
                "labels_distribution": self._get_label_distribution(snli_pairs),
            },
            "sample_pairs": snli_pairs[:10],  # Сохраняем несколько примеров
        }

        # Сохраняем
        logger.info(f"[SAVE] Сохраняем датасет: {filename}")
        torch.save(save_data, file_path)

        # Информация о сохраненном файле
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        logger.info(f"[SAVE] Датасет сохранен:")
        logger.info(f"   Файл: {file_path}")
        logger.info(f"   Размер: {file_size_mb:.1f} MB")
        logger.info(f"   Пар: {len(snli_pairs):,}")
        logger.info(f"   Модель: {self.teacher_model}")

        return str(file_path)

    def _get_label_distribution(self, pairs: List[Dict[str, str]]) -> Dict[str, int]:
        """Получает распределение labels в датасете"""
        label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
        distribution = {}

        for pair in pairs:
            label_name = label_names.get(pair["label"], "unknown")
            distribution[label_name] = distribution.get(label_name, 0) + 1

        return distribution


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Генерация эмбеддингов из SNLI датасета"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Какую часть SNLI использовать (0.2 = 1/5)",
    )
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="Teacher модель для эмбеддингов",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Размер батча для обработки"
    )
    parser.add_argument(
        "--output-dir", default="data/embeddings", help="Директория для сохранения"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Не нормализовать эмбеддинги"
    )

    args = parser.parse_args()

    try:
        # Создаем генератор
        generator = SNLIEmbeddingGenerator(args.model)

        # Загружаем SNLI данные
        snli_pairs = generator.load_snli_dataset(size_fraction=args.fraction)

        # Создаем эмбеддинги
        premise_embeddings, hypothesis_embeddings = generator.create_embedding_dataset(
            snli_pairs, normalize=not args.no_normalize, batch_size=args.batch_size
        )

        # Сохраняем
        saved_file = generator.save_dataset(
            premise_embeddings, hypothesis_embeddings, snli_pairs, args.output_dir
        )

        logger.info(f"✅ SNLI эмбеддинги готовы: {saved_file}")
        logger.info(f"   Теперь можно использовать в run_dynamic_training.py")

        # Тест загрузки через precomputed_embedding_loader
        logger.info(f"\n🧪 Тестируем загрузку через PrecomputedEmbeddingLoader...")
        from precomputed_embedding_loader import PrecomputedEmbeddingLoader

        loader = PrecomputedEmbeddingLoader()
        dataset = loader.load_dataset(saved_file)

        sample_q, sample_a = dataset[0]
        logger.info(f"🧪 Тест успешен:")
        logger.info(f"   Sample premise embedding: {sample_q.shape}")
        logger.info(f"   Sample hypothesis embedding: {sample_a.shape}")
        logger.info(f"   Готов для обучения!")

    except KeyboardInterrupt:
        logger.info("🛑 Остановлено пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
