#!/usr/bin/env python3
"""
🧪 Test Model Inference - Task 1.1

Тестирование обученной 3D Cellular Neural Network на новых фразах.
Сравнение выходных эмбедингов с baseline DistilBERT.

Цель:
- Проверить, генерирует ли модель осмысленные ответы на нестандартные вопросы
- Сравнить выходы с baseline (DistilBERT напрямую)
- Определить качество semantic transformations

Автор: 3D Cellular Neural Network Project
Дата: 2025-01-09
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json

# Импорты нашей системы
from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, 
    EmergentTrainingConfig,
    create_emergent_trainer
)
from data.embedding_loader import EmbeddingLoader
from inference.lightweight_decoder.generative_decoder import (
    GenerativeDecoder,
    create_generative_decoder
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/inference_testing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceTestConfig:
    """Конфигурация для тестирования inference"""
    model_checkpoint: str = "checkpoints/versioned/milestone_overnight_fixed_final_1290/trainer_milestone_overnight_fixed_final_1290.pt"
    baseline_model: str = "distilbert-base-uncased"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test settings
    similarity_threshold: float = 0.7  # Порог для считания ответа "хорошим"
    top_k_similar: int = 3            # Сколько топ-похожих показывать
    
    # Output settings
    save_results: bool = True
    results_file: str = "results/inference_test_results.json"
    detailed_analysis: bool = True


class ModelInferenceTester:
    """Тестер для inference обученной модели"""
    
    def __init__(self, config: InferenceTestConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Инициализация модели и baseline
        self.model = None
        self.baseline_model = None
        self.baseline_tokenizer = None
        self.embedding_loader = None
        self.decoder = None
        
        # Результаты тестов
        self.test_results = []
        
        logger.info(f"[INIT] Initializing InferenceTester on {self.device}")
        
    def load_trained_model(self):
        """Загрузка обученной 3D модели"""
        try:
            logger.info(f"[LOAD] Loading trained model from {self.config.model_checkpoint}")
            
            # Создаем trainer с конфигурацией, соответствующей обученной модели
            config = EmergentTrainingConfig(
                teacher_model="distilbert-base-uncased",
                cube_dimensions=(15, 15, 11),
                enable_full_cube_gradient=True,
                spatial_propagation_depth=11,
                emergent_specialization=True
            )
            
            self.model = EmergentCubeTrainer(config, device=str(self.device))
            
            # Загружаем веса
            checkpoint = torch.load(self.config.model_checkpoint, map_location=self.device)
            
            # Проверяем структуру checkpoint'а
            if 'model_state_dict' in checkpoint:
                # Новый формат с отдельными ключами
                model_state_dict = checkpoint['model_state_dict']
                logger.info(f"[LOAD] Loading from new checkpoint format")
            else:
                # Старый формат - весь checkpoint это state_dict
                model_state_dict = checkpoint
                logger.info(f"[LOAD] Loading from legacy checkpoint format")
            
            # Загружаем веса с strict=False, чтобы игнорировать несовпадающие ключи
            missing_keys, unexpected_keys =             self.model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"[LOAD] Missing keys in model: {len(missing_keys)} keys")
                logger.debug(f"Missing: {missing_keys[:5]}...")  # Показываем первые 5
            
            if unexpected_keys:
                logger.warning(f"[LOAD] Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected: {unexpected_keys[:5]}...")  # Показываем первые 5
            
            # Принудительно перемещаем все параметры на правильное устройство
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("[SUCCESS] Trained model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load trained model: {e}")
            return False
    
    def load_baseline_model(self):
        """Загрузка baseline DistilBERT модели"""
        try:
            logger.info(f"[LOAD] Loading baseline model: {self.config.baseline_model}")
            
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.config.baseline_model)
            self.baseline_model = AutoModel.from_pretrained(self.config.baseline_model)
            self.baseline_model.to(self.device)
            self.baseline_model.eval()
            
            logger.info("[SUCCESS] Baseline model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load baseline model: {e}")
            return False
    
    def setup_embedding_loader(self):
        """Настройка embedding loader для преобразования текста в эмбединги"""
        try:
            logger.info("[SETUP] Setting up embedding loader")
            
            self.embedding_loader = EmbeddingLoader(
                cache_dir="cache/embeddings"
            )
            
            logger.info("[SUCCESS] Embedding loader setup complete")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup embedding loader: {e}")
            return False
    
    def setup_decoder(self):
        """Настройка generative decoder для human-readable output"""
        try:
            logger.info("[SETUP] Setting up generative decoder")
            
            self.decoder = create_generative_decoder(
                architecture="resource_efficient_v21",
                embedding_dim=768,
                verbose_logging=False,
                fallback_enabled=True
            )
            
            # Перемещаем decoder на правильное устройство
            self.decoder.to(self.device)
            
            logger.info("[SUCCESS] Generative decoder setup complete")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup decoder: {e}")
            return False
    
    def prepare_test_questions(self) -> List[Dict[str, str]]:
        """Подготовка разнообразных тестовых вопросов"""
        
        test_questions = [
            # AI/ML темы (схожие с обучающими)
            {
                "category": "AI/ML",
                "question": "What is the difference between supervised and unsupervised learning?",
                "expected_topic": "machine learning fundamentals"
            },
            {
                "category": "AI/ML", 
                "question": "How does gradient descent work in neural networks?",
                "expected_topic": "neural network optimization"
            },
            {
                "category": "AI/ML",
                "question": "What are the advantages of transformer architecture?",
                "expected_topic": "deep learning architectures"
            },
            {
                "category": "AI/ML",
                "question": "Explain the concept of overfitting in machine learning",
                "expected_topic": "model generalization"
            },
            
            # Общие вопросы (out-of-domain)
            {
                "category": "General",
                "question": "What is the capital of France?",
                "expected_topic": "geography"
            },
            {
                "category": "General",
                "question": "How do plants perform photosynthesis?",
                "expected_topic": "biology"
            },
            {
                "category": "General",
                "question": "What is the meaning of life?",
                "expected_topic": "philosophy"
            },
            
            # Сложные технические вопросы
            {
                "category": "Technical",
                "question": "How does quantum computing differ from classical computing?",
                "expected_topic": "quantum physics and computing"
            },
            {
                "category": "Technical",
                "question": "What are the principles of distributed systems consensus?",
                "expected_topic": "computer systems"
            },
            
            # Простые житейские вопросы
            {
                "category": "Simple",
                "question": "How to make a good cup of coffee?",
                "expected_topic": "cooking and beverages"
            },
            {
                "category": "Simple",
                "question": "What should I wear today?",
                "expected_topic": "daily life"
            }
        ]
        
        logger.info(f"[PREP] Prepared {len(test_questions)} test questions across {len(set(q['category'] for q in test_questions))} categories")
        return test_questions
    
    def get_model_response(self, question: str) -> torch.Tensor:
        """Получение ответа от обученной модели"""
        try:
            # Преобразуем вопрос в эмбединг через LLM loader
            question_embedding = self.embedding_loader.load_from_llm([question], model_key="distilbert")
            question_embedding = question_embedding[0]  # Берем первый (и единственный) эмбединг
            
            if question_embedding is None:
                logger.error(f"Failed to get embedding for question: {question}")
                return None
            
            # Преобразуем в тензор и добавляем batch dimension
            question_tensor = torch.tensor(question_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Получаем ответ от модели
            with torch.no_grad():
                outputs = self.model(question_tensor)
                # Берем выходную поверхность как ответ
                response_surface = outputs['output_surface']  # [batch, 225]
                
                # Проецируем обратно в размерность DistilBERT (768)
                # Используем linear слой из loss function модели
                response_embedding = self.model.loss_function.surface_to_embedding(response_surface)  # [batch, 768]
                
            return response_embedding.squeeze(0)  # Убираем batch dimension
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return None
    
    def get_baseline_response(self, question: str) -> torch.Tensor:
        """Получение baseline ответа от DistilBERT"""
        try:
            # Tokenize и encode
            inputs = self.baseline_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Получаем эмбединги
            with torch.no_grad():
                outputs = self.baseline_model(**inputs)
                # Используем [CLS] token embedding
                baseline_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
                
            return baseline_embedding.squeeze(0)  # Убираем batch dimension
            
        except Exception as e:
            logger.error(f"Error getting baseline response: {e}")
            return None
    
    def calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Расчет cosine similarity между эмбедингами"""
        try:
            # Переводим в numpy и нормализуем
            emb1 = embedding1.cpu().numpy().reshape(1, -1)  
            emb2 = embedding2.cpu().numpy().reshape(1, -1)
            
            # Cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0, 0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def run_single_test(self, question_data: Dict[str, str]) -> Dict:
        """Запуск одного теста"""
        question = question_data["question"]
        category = question_data["category"]
        expected_topic = question_data["expected_topic"]
        
        logger.info(f"[TEST] Testing [{category}]: {question}")
        
        # Засекаем время
        start_time = time.time()
        
        # Получаем ответы от обеих моделей
        model_response = self.get_model_response(question)
        baseline_response = self.get_baseline_response(question)
        
        inference_time = time.time() - start_time
        
        if model_response is None or baseline_response is None:
            logger.error(f"Failed to get responses for question: {question}")
            return None
        
        # Логируем размерности для отладки
        logger.debug(f"Model response shape: {model_response.shape}")
        logger.debug(f"Baseline response shape: {baseline_response.shape}")
        
        # Рассчитываем similarity
        similarity = self.calculate_similarity(model_response, baseline_response)
        
        # Декодируем ответ модели в читаемый текст
        decoded_text = "N/A"
        decode_time = 0
        if self.decoder:
            try:
                decode_start = time.time()
                # Убеждаемся что tensor на правильном device
                model_response_device = model_response.to(self.device)
                decode_result = self.decoder.generate(model_response_device)
                decode_time = (time.time() - decode_start) * 1000
                decoded_text = decode_result['text']
                logger.info(f"   [DECODED]: '{decoded_text}'")
            except Exception as e:
                logger.warning(f"   [DECODE_ERROR]: {e}")
                decoded_text = f"Decode error: {str(e)}"
        
        # Формируем результат
        result = {
            "question": question,
            "category": category,
            "expected_topic": expected_topic,
            "similarity_to_baseline": similarity,
            "inference_time_ms": inference_time * 1000,
            "decode_time_ms": decode_time,
            "decoded_answer": decoded_text,
            "model_embedding_shape": list(model_response.shape),
            "baseline_embedding_shape": list(baseline_response.shape),
            "is_good_response": similarity >= self.config.similarity_threshold
        }
        
        # Логируем результат
        status = "GOOD" if result["is_good_response"] else "POOR"
        logger.info(f"   [{status}] | Similarity: {similarity:.3f} | Time: {inference_time*1000:.1f}ms | Decode: {decode_time:.1f}ms")
        
        return result
    
    def run_comprehensive_test(self) -> Dict:
        """Запуск полного набора тестов"""
        logger.info("[START] Starting comprehensive inference testing")
        
        # Подготавливаем тестовые вопросы
        test_questions = self.prepare_test_questions()
        
        # Запускаем тесты
        results = []
        successful_tests = 0
        
        for question_data in test_questions:
            result = self.run_single_test(question_data)
            if result:
                results.append(result)
                if result["is_good_response"]:
                    successful_tests += 1
        
        # Анализируем результаты
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_similarity = np.mean([r["similarity_to_baseline"] for r in results]) if results else 0
        avg_inference_time = np.mean([r["inference_time_ms"] for r in results]) if results else 0
        avg_decode_time = np.mean([r["decode_time_ms"] for r in results]) if results else 0
        
        # Статистика по категориям
        category_stats = {}
        for result in results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "good": 0, "similarities": []}
            
            category_stats[category]["total"] += 1
            category_stats[category]["similarities"].append(result["similarity_to_baseline"])
            if result["is_good_response"]:
                category_stats[category]["good"] += 1
        
        # Финальный отчет
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_similarity": avg_similarity,
            "average_inference_time_ms": avg_inference_time,
            "average_decode_time_ms": avg_decode_time,
            "similarity_threshold": self.config.similarity_threshold,
            "category_statistics": {},
            "detailed_results": results,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Добавляем статистику по категориям
        for category, stats in category_stats.items():
            summary["category_statistics"][category] = {
                "total": stats["total"],
                "good": stats["good"],
                "success_rate": (stats["good"] / stats["total"]) * 100,
                "avg_similarity": np.mean(stats["similarities"])
            }
        
        self.test_results = summary
        
        # Логируем итоги
        logger.info("=" * 60)
        logger.info("[COMPLETE] INFERENCE TESTING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"[STATS] Total Tests: {total_tests}")
        logger.info(f"[STATS] Successful: {successful_tests} ({success_rate:.1f}%)")
        logger.info(f"[STATS] Average Similarity: {avg_similarity:.3f}")
        logger.info(f"[STATS] Average Inference: {avg_inference_time:.1f}ms")
        logger.info(f"[STATS] Average Decode: {avg_decode_time:.1f}ms")
        logger.info("")
        
        logger.info("[BREAKDOWN] Category Breakdown:")
        for category, stats in summary["category_statistics"].items():
            logger.info(f"   {category}: {stats['good']}/{stats['total']} ({stats['success_rate']:.1f}%) | Avg: {stats['avg_similarity']:.3f}")
        
        return summary
    
    def save_results(self):
        """Сохранение результатов тестирования"""
        if not self.config.save_results or not self.test_results:
            return
        
        try:
            # Создаем директорию если не существует
            results_path = Path(self.config.results_file)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем результаты
            with open(self.config.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SAVE] Results saved to {self.config.results_file}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")
    
    def run_full_test_suite(self) -> bool:
        """Запуск полного набора тестов"""
        try:
            # Загружаем модели
            if not self.load_trained_model():
                return False
                
            if not self.load_baseline_model():
                return False
                
            if not self.setup_embedding_loader():
                return False
                
            if not self.setup_decoder():
                return False
            
            # Запускаем тесты
            self.run_comprehensive_test()
            
            # Сохраняем результаты
            self.save_results()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Test suite failed: {e}")
            return False


def main():
    """Основная функция запуска тестирования"""
    logger.info("[MAIN] Starting Model Inference Testing")
    
    # Создаем конфигурацию
    config = InferenceTestConfig()
    
    # Проверяем существование модели
    if not Path(config.model_checkpoint).exists():
        logger.error(f"[ERROR] Model checkpoint not found: {config.model_checkpoint}")
        return False
    
    # Создаем директории для логов и результатов
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Запускаем тестирование
    tester = ModelInferenceTester(config)
    success = tester.run_full_test_suite()
    
    if success:
        logger.info("[SUCCESS] Model inference testing completed successfully!")
        
        # Показываем краткий summary
        if tester.test_results:
            results = tester.test_results
            print("\n" + "="*60)
            print("[FINAL] RESULTS SUMMARY")
            print("="*60)
            print(f"Success Rate: {results['success_rate']:.1f}% ({results['successful_tests']}/{results['total_tests']})")
            print(f"Average Similarity: {results['average_similarity']:.3f}")
            print(f"Average Inference Time: {results['average_inference_time_ms']:.1f}ms")
            print(f"Average Decode Time: {results['average_decode_time_ms']:.1f}ms")
            print("")
            print("Category Performance:")
            for category, stats in results['category_statistics'].items():
                print(f"  {category}: {stats['success_rate']:.1f}% (similarity: {stats['avg_similarity']:.3f})")
            print("")
            print("Sample Decoded Answers:")
            for i, result in enumerate(results['detailed_results'][:3]):  # Показываем первые 3
                print(f"  Q{i+1}: {result['question'][:50]}...")
                print(f"      A: {result['decoded_answer']}")
            print("="*60)
        
        return True
    else:
        logger.error("[ERROR] Model inference testing failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 