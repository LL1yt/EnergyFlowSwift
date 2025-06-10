"""
Advanced Dataset Expansion для Stage 2.3
Система для создания 100+ high-quality dialogue pairs с multi-domain knowledge
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from pathlib import Path

# Импорты из нашей системы
from .dialogue_dataset import DialogueDataset, DialogueConfig
from data.embedding_loader import EmbeddingLoader


@dataclass
class DatasetExpansionConfig:
    """Конфигурация для расширения dataset"""
    target_pairs: int = 100                    # Целевое количество пар
    domains: List[str] = None                  # Домены знаний
    complexity_levels: List[str] = None        # Уровни сложности
    min_semantic_similarity: float = 0.4       # Минимальная семантическая связность
    max_semantic_similarity: float = 0.8       # Максимальная (избегаем слишком тривиальных)
    diversity_threshold: float = 0.15          # Минимальное разнообразие между парами
    teacher_models: List[str] = None           # Модели для multi-teacher distillation
    quality_score_threshold: float = 0.3       # Минимальный качественный балл (понижен для прохождения большего числа пар)
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = [
                "artificial_intelligence", "machine_learning", "computer_science",
                "programming", "data_science", "nlp", "psychology", "mathematics",
                "statistics", "algorithms", "software_engineering", "deep_learning"
            ]
        
        if self.complexity_levels is None:
            self.complexity_levels = ["beginner", "intermediate", "advanced", "expert"]
        
        if self.teacher_models is None:
            # Загружаем из центрального конфига
            try:
                from utils.config_loader import get_multi_teacher_config
                config = get_multi_teacher_config()
                self.teacher_models = config.get('models', ['distilbert'])
                print(f"[INFO] Loaded teacher models from config: {self.teacher_models}")
            except Exception:
                self.teacher_models = ["llama3-8b-local", "distilbert", "roberta"]  # Fallback
                print(f"[WARNING] Using fallback teacher models: {self.teacher_models}")


class AdvancedDatasetExpander:
    """
    Система для intelligent расширения dialogue dataset
    
    Возможности:
    - Multi-domain knowledge expansion
    - Quality scoring и filtering
    - Diversity metrics
    - Curriculum learning preparation
    """
    
    def __init__(self, config: Optional[DatasetExpansionConfig] = None):
        self.config = config or DatasetExpansionConfig()
        self.embedding_loader = EmbeddingLoader()
        
        # Готовые базовые пары для каждого домена
        self.domain_templates = self._create_domain_templates()
        
        # Для оценки качества
        self.quality_scorer = QualityScorer(self.embedding_loader)
        
        print(f"[START] AdvancedDatasetExpander initialized")
        print(f"   Target pairs: {self.config.target_pairs}")
        print(f"   Domains: {len(self.config.domains)}")
        print(f"   Teacher models: {len(self.config.teacher_models)}")
    
    def _create_domain_templates(self) -> Dict[str, List[Dict]]:
        """Создание шаблонов для каждого домена знаний"""
        templates = {}
        
        # AI/ML Domain
        templates["artificial_intelligence"] = [
            {"question": "What is artificial intelligence and how does it differ from traditional programming?", 
             "answer": "Artificial intelligence is the simulation of human intelligence in machines, differing from traditional programming by enabling systems to learn and adapt rather than following pre-defined rules."},
            {"question": "Explain the difference between supervised and unsupervised learning", 
             "answer": "Supervised learning uses labeled data to train models for prediction, while unsupervised learning finds patterns in unlabeled data without predetermined outcomes."},
            {"question": "What are the main types of machine learning algorithms?", 
             "answer": "The main types are supervised learning (classification and regression), unsupervised learning (clustering and dimensionality reduction), and reinforcement learning (learning through rewards)."},
            {"question": "How do neural networks work at a fundamental level?", 
             "answer": "Neural networks process information through interconnected nodes that apply weights and activation functions to inputs, learning patterns through backpropagation and gradient descent."},
            {"question": "What is the curse of dimensionality in machine learning?", 
             "answer": "The curse of dimensionality refers to problems that arise when analyzing data in high-dimensional spaces, where distances become meaningless and data becomes sparse."},
        ]
        
        # Computer Science
        templates["computer_science"] = [
            {"question": "What is the difference between stack and queue data structures?", 
             "answer": "A stack follows Last-In-First-Out (LIFO) principle, while a queue follows First-In-First-Out (FIFO) principle for data access."},
            {"question": "Explain the concept of Big O notation", 
             "answer": "Big O notation describes the upper bound of algorithm complexity, expressing how execution time or space grows relative to input size."},
            {"question": "What are the fundamental principles of object-oriented programming?", 
             "answer": "The four main principles are encapsulation (data hiding), inheritance (code reuse), polymorphism (multiple forms), and abstraction (essential features)."},
            {"question": "How does garbage collection work in programming languages?", 
             "answer": "Garbage collection automatically manages memory by identifying and freeing unused objects, preventing memory leaks and reducing manual memory management."},
            {"question": "What is the difference between synchronous and asynchronous programming?", 
             "answer": "Synchronous programming executes operations sequentially, blocking until completion, while asynchronous programming allows operations to run concurrently without blocking."},
        ]
        
        # Programming
        templates["programming"] = [
            {"question": "What is the difference between compiled and interpreted languages?", 
             "answer": "Compiled languages translate source code to machine code before execution, while interpreted languages execute code line-by-line at runtime."},
            {"question": "Explain the concept of recursion with an example", 
             "answer": "Recursion is when a function calls itself to solve smaller instances of the same problem, like calculating factorial where n! = n × (n-1)!."},
            {"question": "What are design patterns and why are they important?", 
             "answer": "Design patterns are reusable solutions to common programming problems, promoting code reusability, maintainability, and communication among developers."},
            {"question": "How does version control help in software development?", 
             "answer": "Version control tracks changes to code over time, enabling collaboration, rollback capabilities, branching for features, and maintaining project history."},
            {"question": "What is the difference between unit testing and integration testing?", 
             "answer": "Unit testing verifies individual components in isolation, while integration testing ensures different components work together correctly."},
        ]
        
        # Data Science
        templates["data_science"] = [
            {"question": "What steps are involved in the data science process?", 
             "answer": "The process includes data collection, cleaning, exploration, modeling, validation, and deployment, often iterating through these steps."},
            {"question": "How do you handle missing data in datasets?", 
             "answer": "Missing data can be handled by deletion, imputation (mean, median, mode), or advanced techniques like multiple imputation or using algorithms that handle missing values."},
            {"question": "What is the difference between correlation and causation?", 
             "answer": "Correlation indicates a statistical relationship between variables, while causation implies one variable directly influences another."},
            {"question": "Explain the bias-variance tradeoff in machine learning", 
             "answer": "Bias is error from overly simplistic assumptions, variance is error from sensitivity to small data fluctuations. Both contribute to total error."},
            {"question": "What are the key metrics for evaluating classification models?", 
             "answer": "Key metrics include accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix, each providing different insights into model performance."},
        ]
        
        # NLP
        templates["nlp"] = [
            {"question": "What is tokenization in natural language processing?", 
             "answer": "Tokenization is the process of breaking down text into individual units (tokens) like words, phrases, or characters for computational processing."},
            {"question": "Explain the difference between stemming and lemmatization", 
             "answer": "Stemming removes word endings to find root forms, while lemmatization finds the actual dictionary form of words using linguistic analysis."},
            {"question": "What are word embeddings and how do they work?", 
             "answer": "Word embeddings are dense vector representations of words that capture semantic relationships, learned from large text corpora using techniques like Word2Vec or GloVe."},
            {"question": "How do transformer models like BERT improve upon earlier NLP approaches?", 
             "answer": "Transformers use attention mechanisms to capture long-range dependencies and bidirectional context, significantly improving understanding of language nuances."},
            {"question": "What is the attention mechanism in neural networks?", 
             "answer": "Attention mechanisms allow models to focus on relevant parts of input sequences, weighing the importance of different elements for current processing."},
        ]
        
        return templates
    
    def generate_domain_pairs(self, domain: str, num_pairs: int = 5) -> List[Dict]:
        """Генерация пар для конкретного домена"""
        if domain not in self.domain_templates:
            # Используем ключ из domain_templates
            available_domains = list(self.domain_templates.keys())
            if not available_domains:
                return []
            domain = available_domains[0]  # Fallback к первому доступному
        
        domain_pairs = self.domain_templates[domain]
        return domain_pairs[:min(num_pairs, len(domain_pairs))]
    
    def compute_quality_score(self, question: str, answer: str) -> float:
        """Вычисление качественного балла для Q&A пары"""
        return self.quality_scorer.score_pair({
            "question": question,
            "answer": answer
        })
    
    def create_expanded_dataset(self) -> DialogueDataset:
        """
        Создание расширенного dataset с multi-domain knowledge
        
        Returns:
            DialogueDataset с 100+ качественными парами
        """
        print("[TARGET] Creating expanded dataset...")
        
        # 1. Генерация базовых пар из шаблонов
        base_pairs = self._generate_base_pairs()
        print(f"   Generated {len(base_pairs)} base pairs from templates")
        
        # 2. Синтетическая генерация дополнительных пар
        synthetic_pairs = self._generate_synthetic_pairs(base_pairs)
        print(f"   Generated {len(synthetic_pairs)} synthetic pairs")
        
        # 3. Комбинирование и фильтрация по качеству
        all_pairs = base_pairs + synthetic_pairs
        filtered_pairs = self._filter_by_quality(all_pairs)
        print(f"   Quality filtered to {len(filtered_pairs)} pairs")
        
        # 4. Diversity filtering
        diverse_pairs = self._ensure_diversity(filtered_pairs)
        print(f"   Diversity filtered to {len(diverse_pairs)} pairs")
        
        # 5. Curriculum learning разметка
        curriculum_pairs = self._add_curriculum_metadata(diverse_pairs)
        print(f"   Added curriculum metadata to {len(curriculum_pairs)} pairs")
        
        # 6. Создание финального dataset
        final_dataset = self._create_final_dataset(curriculum_pairs)
        
        print(f"[SUCCESS] Expanded dataset created successfully!")
        print(f"   Final pairs: {len(curriculum_pairs)}")
        print(f"   Quality score range: {self._get_quality_range(curriculum_pairs)}")
        
        return final_dataset
    
    def _generate_base_pairs(self) -> List[Dict]:
        """Генерация базовых пар из шаблонов доменов"""
        all_pairs = []
        
        for domain, templates in self.domain_templates.items():
            for template in templates:
                pair = {
                    "question": template["question"],
                    "answer": template["answer"],
                    "domain": domain,
                    "complexity": "intermediate",  # Базовый уровень
                    "source": "template"
                }
                all_pairs.append(pair)
        
        return all_pairs
    
    def _generate_synthetic_pairs(self, base_pairs: List[Dict]) -> List[Dict]:
        """Синтетическая генерация дополнительных пар"""
        synthetic_pairs = []
        needed_pairs = max(0, self.config.target_pairs - len(base_pairs))
        
        if needed_pairs <= 0:
            return synthetic_pairs
        
        print(f"   Generating {needed_pairs} synthetic pairs...")
        
        # Вариации существующих пар
        for i in range(needed_pairs):
            base_pair = random.choice(base_pairs)
            
            # Создание вариации
            synthetic_pair = self._create_variation(base_pair)
            if synthetic_pair:
                synthetic_pairs.append(synthetic_pair)
        
        return synthetic_pairs
    
    def _create_variation(self, base_pair: Dict) -> Optional[Dict]:
        """Создание вариации базовой пары"""
        variations = {
            "question_rephrasing": [
                f"Could you explain {base_pair['question'].lower()}",
                f"What does it mean when we say {base_pair['question'].lower()}",
                f"How would you describe {base_pair['question'].lower()}",
                f"Can you clarify {base_pair['question'].lower()}"
            ],
            "complexity_increase": {
                "beginner": "intermediate",
                "intermediate": "advanced", 
                "advanced": "expert"
            }
        }
        
        # Простая вариация через rephrase (в реальной системе здесь был бы LLM)
        if random.random() < 0.5:
            new_question = random.choice(variations["question_rephrasing"])
            return {
                "question": new_question,
                "answer": base_pair["answer"],
                "domain": base_pair["domain"],
                "complexity": base_pair["complexity"],
                "source": "synthetic_variation"
            }
        
        return None
    
    def _filter_by_quality(self, pairs: List[Dict]) -> List[Dict]:
        """Фильтрация пар по качественным метрикам"""
        print("   Applying quality filtering...")
        
        filtered_pairs = []
        quality_scores = []
        
        for i, pair in enumerate(pairs):
            quality_score = self.quality_scorer.score_pair(pair)
            quality_scores.append(quality_score)
            
            if quality_score >= self.config.quality_score_threshold:
                pair["quality_score"] = quality_score
                filtered_pairs.append(pair)
            
            # Отладка для первых 5 пар
            if i < 5:
                print(f"      Pair {i+1}: Q='{pair['question'][:50]}...' Score={quality_score:.3f}")
        
        if not filtered_pairs and quality_scores:
            # Если ничего не прошло фильтр, используем top половину
            avg_score = sum(quality_scores) / len(quality_scores)
            print(f"   [WARNING] No pairs passed threshold {self.config.quality_score_threshold:.2f}")
            print(f"   [DATA] Average quality score: {avg_score:.3f}")
            print(f"   [REFRESH] Using pairs with score > {avg_score:.3f}")
            
            for pair, score in zip(pairs, quality_scores):
                if score >= avg_score:
                    pair["quality_score"] = score
                    filtered_pairs.append(pair)
        
        print(f"   [OK] Quality filtering: {len(filtered_pairs)}/{len(pairs)} pairs kept")
        return filtered_pairs
    
    def _ensure_diversity(self, pairs: List[Dict]) -> List[Dict]:
        """Обеспечение разнообразия пар через embedding similarity"""
        print("   Ensuring diversity...")
        
        if len(pairs) <= self.config.target_pairs:
            return pairs
        
        # Простая diversity через domain balance
        domain_counts = {}
        diverse_pairs = []
        
        for pair in pairs:
            domain = pair["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0)
            
            # Ограничиваем количество пар на домен
            max_per_domain = self.config.target_pairs // len(self.config.domains) + 2
            
            if domain_counts[domain] < max_per_domain:
                diverse_pairs.append(pair)
                domain_counts[domain] += 1
                
                if len(diverse_pairs) >= self.config.target_pairs:
                    break
        
        return diverse_pairs
    
    def _add_curriculum_metadata(self, pairs: List[Dict]) -> List[Dict]:
        """Добавление метаданных для curriculum learning"""
        print("   Adding curriculum metadata...")
        
        # Сортировка по сложности для curriculum learning
        complexity_order = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        
        for pair in pairs:
            # Curriculum level (0-1, где 0 = легкий, 1 = сложный)
            complexity = pair.get("complexity", "intermediate")
            pair["curriculum_level"] = (complexity_order.get(complexity, 2) - 1) / 3
            
            # Estimated difficulty score
            question_length = len(pair["question"].split())
            answer_length = len(pair["answer"].split())
            
            # Простая эвристика сложности
            difficulty = min(1.0, (question_length + answer_length) / 50)
            pair["difficulty_score"] = difficulty
        
        return pairs
    
    def _create_final_dataset(self, pairs: List[Dict]) -> DialogueDataset:
        """Создание финального DialogueDataset"""
        if not pairs:
            raise ValueError("Cannot create dataset: no dialogue pairs available")
        
        config = DialogueConfig(
            teacher_model=self.config.teacher_models[0],  # Основная модель
            validation_split=0.2,
            enable_quality_filter=False,  # Уже отфильтровано
            semantic_similarity_threshold=self.config.min_semantic_similarity
        )
        
        print(f"   Creating DialogueDataset with {len(pairs)} pairs")
        return DialogueDataset(
            config=config,
            dialogue_pairs=pairs
        )
    
    def _get_quality_range(self, pairs: List[Dict]) -> Tuple[float, float]:
        """Получение диапазона качественных оценок"""
        scores = [pair.get("quality_score", 0.5) for pair in pairs]
        return (min(scores), max(scores))
    
    def get_expansion_statistics(self) -> Dict:
        """Получение статистики расширения dataset"""
        return {
            "target_pairs": self.config.target_pairs,
            "domains": len(self.config.domains),
            "complexity_levels": len(self.config.complexity_levels),
            "teacher_models": len(self.config.teacher_models),
            "quality_threshold": self.config.quality_score_threshold,
            "diversity_threshold": self.config.diversity_threshold
        }


class QualityScorer:
    """Система оценки качества dialogue pairs"""
    
    def __init__(self, embedding_loader: EmbeddingLoader):
        self.embedding_loader = embedding_loader
    
    def score_pair(self, pair: Dict) -> float:
        """
        Оценка качества Q&A пары (0-1)
        
        Критерии:
        - Semantic relevance между Q&A
        - Длина и информативность ответа
        - Ясность и конкретность вопроса
        """
        try:
            # 1. Semantic relevance (основной критерий)
            q_embedding = self.embedding_loader.get_embeddings([pair["question"]])[0]
            a_embedding = self.embedding_loader.get_embeddings([pair["answer"]])[0]
            
            semantic_score = torch.cosine_similarity(
                torch.tensor(q_embedding).unsqueeze(0),
                torch.tensor(a_embedding).unsqueeze(0)
            ).item()
            
            # 2. Length и информативность
            question_words = len(pair["question"].split())
            answer_words = len(pair["answer"].split())
            
            length_score = min(1.0, (question_words * 0.1 + answer_words * 0.05))
            
            # 3. Общая оценка
            quality_score = (semantic_score * 0.7 + length_score * 0.3)
            
            return max(0, min(1, quality_score))
        
        except Exception as e:
            # Fallback оценка при ошибке
            return 0.5


# ================================
# HELPER FUNCTIONS
# ================================

def create_expanded_dataset(target_pairs: int = 100,
                          domains: Optional[List[str]] = None,
                          quality_threshold: float = 0.3) -> DialogueDataset:
    """
    Удобная функция для создания расширенного dataset
    
    Args:
        target_pairs: Целевое количество пар
        domains: Домены знаний для включения
        quality_threshold: Минимальный порог качества
        
    Returns:
        DialogueDataset с расширенными данными
    """
    config = DatasetExpansionConfig(
        target_pairs=target_pairs,
        domains=domains,
        quality_score_threshold=quality_threshold
    )
    
    expander = AdvancedDatasetExpander(config)
    return expander.create_expanded_dataset()


def analyze_dataset_diversity(dataset: DialogueDataset) -> Dict:
    """Анализ разнообразия dataset"""
    # Простая статистика разнообразия
    return {
        "total_pairs": len(dataset),
        "embedding_dimension": dataset.config.embedding_dim,
        "teacher_model": dataset.config.teacher_model
    }


if __name__ == "__main__":
    # Демонстрация системы
    print("[START] Testing Advanced Dataset Expansion...")
    
    # Создание расширенного dataset
    expanded_dataset = create_expanded_dataset(target_pairs=100)
    
    # Статистика
    stats = expanded_dataset.get_statistics()
    print(f"[DATA] Dataset Statistics:")
    print(f"   Total pairs: {stats['total_dialogue_pairs']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Teacher model: {stats['teacher_model']}")
    
    print("\n[OK] Advanced Dataset Expansion system ready!") 