#!/usr/bin/env python3
"""
Генератор большого датасета эмбеддингов для обучения 3D куба
Создает тысячи пар question-answer и сохраняет готовые эмбеддинги
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import random

from simple_embedding_fallback import SimpleFallbackEmbeddingLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LargeEmbeddingDatasetGenerator:
    """Генератор больших датасетов эмбеддингов"""

    def __init__(self, teacher_model: str = "distilbert-base-uncased"):
        self.teacher_model = teacher_model
        self.loader = SimpleFallbackEmbeddingLoader(teacher_model)
        self.dialogue_templates = self._load_dialogue_templates()

    def _load_dialogue_templates(self) -> List[Dict[str, str]]:
        """Загружает шаблоны диалогов и расширяет их"""

        # Базовые шаблоны по разным темам
        base_templates = [
            # AI и ML
            {
                "question": "What is artificial intelligence?",
                "answer": "AI is the simulation of human intelligence in machines.",
            },
            {
                "question": "How do neural networks work?",
                "answer": "Neural networks process data through interconnected layers of nodes.",
            },
            {
                "question": "What is machine learning?",
                "answer": "ML enables computers to learn from data without explicit programming.",
            },
            {
                "question": "Explain deep learning.",
                "answer": "Deep learning uses multi-layered neural networks for complex pattern recognition.",
            },
            {
                "question": "What are transformers in AI?",
                "answer": "Transformers are attention-based models for sequence processing.",
            },
            {
                "question": "How does natural language processing work?",
                "answer": "NLP enables computers to understand and generate human language.",
            },
            {
                "question": "What is computer vision?",
                "answer": "Computer vision enables machines to interpret visual information.",
            },
            {
                "question": "Explain reinforcement learning.",
                "answer": "RL trains agents through interaction with an environment.",
            },
            {
                "question": "What is supervised learning?",
                "answer": "Supervised learning uses labeled data to train models.",
            },
            {
                "question": "How do convolutional networks work?",
                "answer": "CNNs use filters to detect features in spatial data.",
            },
            # Программирование
            {
                "question": "What is Python?",
                "answer": "Python is a high-level programming language known for simplicity.",
            },
            {
                "question": "How does object-oriented programming work?",
                "answer": "OOP organizes code into classes and objects with properties and methods.",
            },
            {
                "question": "What is a function?",
                "answer": "A function is a reusable block of code that performs specific tasks.",
            },
            {
                "question": "Explain recursion.",
                "answer": "Recursion is when a function calls itself to solve smaller subproblems.",
            },
            {
                "question": "What are data structures?",
                "answer": "Data structures organize and store data for efficient access and modification.",
            },
            {
                "question": "How do algorithms work?",
                "answer": "Algorithms are step-by-step procedures for solving computational problems.",
            },
            {
                "question": "What is debugging?",
                "answer": "Debugging is the process of finding and fixing errors in code.",
            },
            {
                "question": "Explain version control.",
                "answer": "Version control tracks changes to code over time for collaboration.",
            },
            {
                "question": "What is an API?",
                "answer": "API is an interface that allows different software applications to communicate.",
            },
            {
                "question": "How does database work?",
                "answer": "Databases store and organize data for efficient retrieval and management.",
            },
            # Математика
            {
                "question": "What is calculus?",
                "answer": "Calculus studies continuous change through derivatives and integrals.",
            },
            {
                "question": "Explain linear algebra.",
                "answer": "Linear algebra studies vectors, matrices, and linear transformations.",
            },
            {
                "question": "What is probability?",
                "answer": "Probability measures the likelihood of events occurring.",
            },
            {
                "question": "How does statistics work?",
                "answer": "Statistics analyzes data to discover patterns and make inferences.",
            },
            {
                "question": "What is geometry?",
                "answer": "Geometry studies shapes, sizes, and properties of space.",
            },
            {
                "question": "Explain trigonometry.",
                "answer": "Trigonometry studies relationships between angles and sides in triangles.",
            },
            {
                "question": "What is algebra?",
                "answer": "Algebra uses symbols and equations to represent mathematical relationships.",
            },
            {
                "question": "How does logic work?",
                "answer": "Logic studies valid reasoning and argument structures.",
            },
            {
                "question": "What is number theory?",
                "answer": "Number theory studies properties and relationships of integers.",
            },
            {
                "question": "Explain set theory.",
                "answer": "Set theory studies collections of objects and their relationships.",
            },
            # Наука
            {
                "question": "What is physics?",
                "answer": "Physics studies matter, energy, and their interactions in the universe.",
            },
            {
                "question": "How does chemistry work?",
                "answer": "Chemistry studies composition, structure, and properties of substances.",
            },
            {
                "question": "What is biology?",
                "answer": "Biology studies living organisms and their life processes.",
            },
            {
                "question": "Explain evolution.",
                "answer": "Evolution describes how species change over time through natural selection.",
            },
            {
                "question": "What is genetics?",
                "answer": "Genetics studies heredity and variation in living organisms.",
            },
            {
                "question": "How does the brain work?",
                "answer": "The brain processes information through networks of interconnected neurons.",
            },
            {
                "question": "What is quantum mechanics?",
                "answer": "Quantum mechanics describes behavior of matter and energy at atomic scales.",
            },
            {
                "question": "Explain thermodynamics.",
                "answer": "Thermodynamics studies heat, work, and energy transfer in systems.",
            },
            {
                "question": "What is electromagnetism?",
                "answer": "Electromagnetism studies electric and magnetic fields and their interactions.",
            },
            {
                "question": "How does DNA work?",
                "answer": "DNA stores genetic information in sequences of nucleotide bases.",
            },
        ]

        return base_templates

    def _generate_variations(
        self, template: Dict[str, str], num_variations: int = 10
    ) -> List[Dict[str, str]]:
        """Генерирует вариации базового шаблона"""
        variations = [template]  # Включаем оригинал

        question = template["question"]
        answer = template["answer"]

        # Вариации вопросов
        question_starters = [
            "Can you explain",
            "Tell me about",
            "What do you know about",
            "How would you describe",
            "Could you clarify",
            "I want to understand",
            "Please explain",
            "Help me understand",
            "What exactly is",
            "Could you tell me about",
            "I need to know about",
            "Describe",
        ]

        # Вариации ответов
        answer_starters = [
            "Simply put,",
            "In essence,",
            "Basically,",
            "To put it simply,",
            "In other words,",
            "Essentially,",
            "Fundamentally,",
            "At its core,",
            "The key point is that",
            "Generally speaking,",
            "To summarize,",
        ]

        answer_endings = [
            "This is fundamental to understanding the concept.",
            "This forms the basis of the field.",
            "This is essential knowledge in this area.",
            "This concept is widely used in practice.",
            "Understanding this is crucial for further learning.",
            "This principle underlies many applications.",
            "This knowledge is foundational to the subject.",
        ]

        for i in range(min(num_variations - 1, 9)):  # -1 так как оригинал уже включен
            # Модифицируем вопрос
            if "What is" in question:
                new_question = question.replace(
                    "What is", random.choice(question_starters)
                )
            elif "How do" in question or "How does" in question:
                new_question = (
                    random.choice(question_starters)
                    + " "
                    + question.lower().replace("how does ", "").replace("how do ", "")
                )
            else:
                new_question = random.choice(question_starters) + " " + question.lower()

            # Модифицируем ответ
            new_answer = random.choice(answer_starters) + " " + answer.lower()
            if random.random() > 0.5:  # 50% шанс добавить ending
                new_answer += " " + random.choice(answer_endings)

            variations.append({"question": new_question, "answer": new_answer})

        return variations

    def generate_large_dataset(self, target_size: int = 10000) -> List[Dict[str, str]]:
        """Генерирует большой датасет диалогов"""
        logger.info(f"🎯 Generating dataset with {target_size} samples...")

        all_dialogues = []

        # Генерируем вариации для каждого шаблона
        variations_per_template = max(1, target_size // len(self.dialogue_templates))

        for i, template in enumerate(self.dialogue_templates):
            variations = self._generate_variations(template, variations_per_template)
            all_dialogues.extend(variations)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"   Processed {i + 1}/{len(self.dialogue_templates)} templates, "
                    f"generated {len(all_dialogues)} samples so far"
                )

        # Ограничиваем до целевого размера
        if len(all_dialogues) > target_size:
            all_dialogues = random.sample(all_dialogues, target_size)

        logger.info(f"✅ Generated {len(all_dialogues)} dialogue samples")
        return all_dialogues

    def create_embedding_dataset(
        self,
        dialogue_pairs: List[Dict[str, str]],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Создает датасет эмбеддингов из диалогов"""
        logger.info(f"🔄 Converting {len(dialogue_pairs)} dialogues to embeddings...")

        questions = [pair["question"] for pair in dialogue_pairs]
        answers = [pair["answer"] for pair in dialogue_pairs]

        # Обрабатываем батчами для экономии памяти
        question_embeddings = []
        answer_embeddings = []

        total_batches = (len(questions) + batch_size - 1) // batch_size

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers = answers[i : i + batch_size]

            # Генерируем эмбеддинги для батча
            q_emb = self.loader.encode_texts(batch_questions)
            a_emb = self.loader.encode_texts(batch_answers)

            question_embeddings.append(q_emb)
            answer_embeddings.append(a_emb)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"   Processed batch {i // batch_size + 1}/{total_batches}")

        # Объединяем все батчи
        question_embeddings = torch.cat(question_embeddings, dim=0)
        answer_embeddings = torch.cat(answer_embeddings, dim=0)

        # Нормализация
        if normalize:
            question_embeddings = torch.nn.functional.normalize(
                question_embeddings, p=2, dim=1
            )
            answer_embeddings = torch.nn.functional.normalize(
                answer_embeddings, p=2, dim=1
            )
            logger.info("✅ Embeddings normalized")

        logger.info(f"📊 Final embeddings shape:")
        logger.info(f"   Questions: {question_embeddings.shape}")
        logger.info(f"   Answers: {answer_embeddings.shape}")

        return question_embeddings, answer_embeddings

    def save_dataset(
        self,
        question_embeddings: torch.Tensor,
        answer_embeddings: torch.Tensor,
        dialogue_pairs: List[Dict[str, str]],
        output_dir: str = "data/embeddings",
    ):
        """Сохраняет датасет эмбеддингов"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Сохраняем эмбеддинги
        embeddings_file = output_path / f"large_embedding_dataset_{timestamp}.pt"
        torch.save(
            {
                "question_embeddings": question_embeddings,
                "answer_embeddings": answer_embeddings,
                "teacher_model": self.teacher_model,
                "timestamp": timestamp,
                "size": len(question_embeddings),
            },
            embeddings_file,
        )

        # Сохраняем метаданные
        metadata_file = output_path / f"dataset_metadata_{timestamp}.json"
        metadata = {
            "teacher_model": self.teacher_model,
            "timestamp": timestamp,
            "size": len(question_embeddings),
            "embedding_dim": question_embeddings.shape[1],
            "normalized": True,
            "sample_dialogues": dialogue_pairs[:10],  # Первые 10 для примера
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Dataset saved:")
        logger.info(f"   Embeddings: {embeddings_file}")
        logger.info(f"   Metadata: {metadata_file}")

        return embeddings_file, metadata_file


def main():
    """Главная функция для генерации большого датасета"""
    print("🚀 ГЕНЕРАТОР БОЛЬШОГО ДАТАСЕТА ЭМБЕДДИНГОВ")
    print("=" * 60)

    # Параметры
    target_size = 10000  # 10K samples
    teacher_model = "distilbert-base-uncased"

    # Создаем генератор
    generator = LargeEmbeddingDatasetGenerator(teacher_model)

    try:
        # 1. Генерируем диалоги
        start_time = time.time()
        dialogue_pairs = generator.generate_large_dataset(target_size)
        generation_time = time.time() - start_time

        logger.info(f"⏰ Dialogue generation time: {generation_time:.1f}s")

        # 2. Создаем эмбеддинги
        start_time = time.time()
        question_embeddings, answer_embeddings = generator.create_embedding_dataset(
            dialogue_pairs, normalize=True, batch_size=32
        )
        embedding_time = time.time() - start_time

        logger.info(f"⏰ Embedding generation time: {embedding_time:.1f}s")

        # 3. Сохраняем датасет
        embeddings_file, metadata_file = generator.save_dataset(
            question_embeddings, answer_embeddings, dialogue_pairs
        )

        # 4. Статистика
        print("\n📊 СТАТИСТИКА ДАТАСЕТА:")
        print(f"   Размер: {len(question_embeddings):,} пар")
        print(f"   Размерность эмбеддингов: {question_embeddings.shape[1]}")
        print(f"   Размер файла: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Учительская модель: {teacher_model}")
        print(f"   Время генерации: {generation_time + embedding_time:.1f}s")

        # 5. Тест загрузки
        print("\n🧪 ТЕСТ ЗАГРУЗКИ:")
        test_data = torch.load(embeddings_file)
        print(f"   ✅ Файл успешно загружен")
        print(f"   ✅ Размер: {test_data['size']}")
        print(f"   ✅ Форма эмбеддингов: {test_data['question_embeddings'].shape}")

        return 0

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
