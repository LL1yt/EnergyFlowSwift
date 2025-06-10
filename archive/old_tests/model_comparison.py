"""
[BOT] Model Comparison Suite - Stage 3.1.3
Инфраструктура для тестирования и сравнения различных Teacher models
Автоматический выбор оптимальных конфигураций для каждой модели
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
import json
from collections import defaultdict

# Импорты системы
from .adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from data.embedding_adapter.universal_adapter import KNOWN_MODELS, UniversalEmbeddingAdapter
from .cube_trainer import EmbeddingMetrics
from data.embedding_loader import EmbeddingLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelTestConfig:
    """Конфигурация тестирования для конкретной модели"""
    model_name: str
    embedding_dim: int
    adapter_strategies: List[str] = field(default_factory=lambda: ["learned_linear", "hierarchical"])
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.0005, 0.0001])
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8])
    test_epochs: int = 5  # Для быстрого тестирования
    quality_threshold: float = 0.8  # Минимальный cosine similarity


@dataclass 
class ModelTestResult:
    """Результат тестирования модели"""
    model_name: str
    adapter_strategy: str
    learning_rate: float
    batch_size: int
    
    # Качественные метрики
    final_loss: float
    reconstruction_loss: float
    cosine_similarity: float
    semantic_preservation: float
    
    # Performance метрики
    training_time: float
    memory_usage: float
    convergence_epochs: int
    parameter_count: int
    compression_ratio: float
    
    # Статус
    converged: bool
    quality_passed: bool
    recommended: bool = False


class ModelDetectionSystem:
    """
    Система автоматического определения и конфигурации моделей
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Расширенная база знаний о моделях
        self.model_database = KNOWN_MODELS.copy()
        
        # Добавляем дополнительные модели
        self.model_database.update({
            "Claude-3": {"embedding_dim": 2048},
            "GPT-4": {"embedding_dim": 1536},
            "Gemini": {"embedding_dim": 3072},
        })
        
        # Рекомендуемые конфигурации по compression ratio
        self.strategy_recommendations = {
            # High compression (>10x) - нужны продвинутые стратегии
            "high_compression": {
                "ratio_threshold": 10.0,
                "recommended_strategies": ["hierarchical", "attention_based"],
                "learning_rates": [0.0005, 0.0001],
                "batch_sizes": [4, 8]
            },
            # Medium compression (3-10x) - balance между качеством и скоростью  
            "medium_compression": {
                "ratio_threshold": 3.0,
                "recommended_strategies": ["learned_linear", "hierarchical"],
                "learning_rates": [0.001, 0.0005],
                "batch_sizes": [8, 16]
            },
            # Low compression (<3x) - можно использовать простые стратегии
            "low_compression": {
                "ratio_threshold": 1.0,
                "recommended_strategies": ["learned_linear"],
                "learning_rates": [0.001, 0.002],
                "batch_sizes": [16, 32]
            }
        }
    
    def detect_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Определение конфигурации модели по имени"""
        # Точное совпадение
        if model_name in self.model_database:
            return self.model_database[model_name]
        
        # Fuzzy matching для вариантов имен
        for known_model in self.model_database:
            if known_model.lower() in model_name.lower() or model_name.lower() in known_model.lower():
                self.logger.info(f"[MAGNIFY] Model '{model_name}' matched to '{known_model}'")
                return self.model_database[known_model]
        
        return None
    
    def get_recommended_config(self, model_name: str, target_surface_size: int = 225) -> ModelTestConfig:
        """Получение рекомендуемой конфигурации для модели"""
        model_info = self.detect_model(model_name)
        
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        embedding_dim = model_info["embedding_dim"]
        compression_ratio = embedding_dim / target_surface_size
        
        # Выбираем категорию compression
        if compression_ratio >= self.strategy_recommendations["high_compression"]["ratio_threshold"]:
            category = "high_compression"
        elif compression_ratio >= self.strategy_recommendations["medium_compression"]["ratio_threshold"]:
            category = "medium_compression"
        else:
            category = "low_compression"
        
        config_template = self.strategy_recommendations[category]
        
        self.logger.info(f"[TARGET] Model {model_name}: {embedding_dim}D → {target_surface_size}D")
        self.logger.info(f"   Compression: {compression_ratio:.1f}x ({category})")
        self.logger.info(f"   Recommended strategies: {config_template['recommended_strategies']}")
        
        return ModelTestConfig(
            model_name=model_name,
            embedding_dim=embedding_dim,
            adapter_strategies=config_template["recommended_strategies"],
            learning_rates=config_template["learning_rates"],
            batch_sizes=config_template["batch_sizes"]
        )
    
    def list_supported_models(self) -> List[str]:
        """Список всех поддерживаемых моделей"""
        return list(self.model_database.keys())


class ModelComparisonSuite:
    """
    Комплексная система сравнения Teacher models
    
    Возможности:
    - Автоматическое тестирование multiple models
    - Comparison метрик качества и производительности
    - Рекомендации optimal configurations
    - Benchmarking suite для production readiness
    """
    
    def __init__(self, 
                 cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                 surface_strategy: str = "single",
                 output_dir: str = "results/model_comparison",
                 device: str = "cpu"):
        """
        Инициализация Model Comparison Suite
        
        Args:
            cube_dimensions: Размеры куба для тестирования
            surface_strategy: Стратегия surface processing 
            output_dir: Директория для результатов
            device: Устройство для вычислений
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("[BOT] Initializing ModelComparisonSuite...")
        
        self.cube_dimensions = cube_dimensions
        self.surface_strategy = surface_strategy
        self.surface_size = cube_dimensions[0] * cube_dimensions[1]  # 15*15 = 225
        self.device = torch.device(device)
        
        # Директория результатов
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты
        self.model_detector = ModelDetectionSystem()
        self.test_results: List[ModelTestResult] = []
        self.comparison_metrics = {}
        
        # Загрузка тестовых данных (будем использовать простые embeddings)
        self.test_embeddings = self._prepare_test_data()
        
        self.logger.info(f"[OK] ModelComparisonSuite готов:")
        self.logger.info(f"   Cube: {cube_dimensions} → Surface: {self.surface_size}D")
        self.logger.info(f"   Supported models: {len(self.model_detector.list_supported_models())}")
        self.logger.info(f"   Output: {self.output_dir}")
    
    def _prepare_test_data(self) -> Dict[str, torch.Tensor]:
        """Подготовка тестовых данных для каждой модели"""
        test_data = {}
        
        for model_name, model_info in self.model_detector.model_database.items():
            embedding_dim = model_info["embedding_dim"]
            
            # Создаем синтетические embeddings для тестирования
            # В реальном случае здесь будет загрузка из EmbeddingLoader
            test_data[model_name] = {
                "questions": torch.randn(20, embedding_dim),  # 20 вопросов
                "answers": torch.randn(20, embedding_dim)     # 20 ответов
            }
        
        self.logger.info(f"[DATA] Test data prepared for {len(test_data)} models")
        return test_data
    
    def test_single_model(self, 
                         model_name: str, 
                         config: Optional[ModelTestConfig] = None) -> List[ModelTestResult]:
        """
        Тестирование одной модели с разными конфигурациями
        
        Args:
            model_name: Название модели для тестирования
            config: Конфигурация тестирования (или auto-detect)
            
        Returns:
            List[ModelTestResult]: Результаты всех тестов для модели
        """
        self.logger.info(f"🧪 Testing model: {model_name}")
        
        # Получаем конфигурацию
        if config is None:
            config = self.model_detector.get_recommended_config(model_name, self.surface_size)
        
        results = []
        test_data = self.test_embeddings[model_name]
        
        # Тестируем все комбинации параметров
        total_tests = len(config.adapter_strategies) * len(config.learning_rates) * len(config.batch_sizes)
        current_test = 0
        
        for strategy in config.adapter_strategies:
            for lr in config.learning_rates:
                for batch_size in config.batch_sizes:
                    current_test += 1
                    self.logger.info(f"   Test {current_test}/{total_tests}: {strategy}, lr={lr}, batch={batch_size}")
                    
                    try:
                        result = self._run_single_test(
                            model_name=model_name,
                            embedding_dim=config.embedding_dim,
                            adapter_strategy=strategy,
                            learning_rate=lr,
                            batch_size=batch_size,
                            test_epochs=config.test_epochs,
                            test_data=test_data,
                            quality_threshold=config.quality_threshold
                        )
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"[ERROR] Test failed: {e}")
                        # Создаем failed result
                        failed_result = ModelTestResult(
                            model_name=model_name,
                            adapter_strategy=strategy,
                            learning_rate=lr,
                            batch_size=batch_size,
                            final_loss=999.0,
                            reconstruction_loss=999.0,
                            cosine_similarity=0.0,
                            semantic_preservation=0.0,
                            training_time=0.0,
                            memory_usage=0.0,
                            convergence_epochs=config.test_epochs,
                            parameter_count=0,
                            compression_ratio=0.0,
                            converged=False,
                            quality_passed=False
                        )
                        results.append(failed_result)
                        continue
        
        # Находим лучший результат для модели
        best_result = min([r for r in results if r.converged], 
                         key=lambda x: x.final_loss, 
                         default=None)
        
        if best_result:
            best_result.recommended = True
            self.logger.info(f"[TROPHY] Best config for {model_name}: {best_result.adapter_strategy}, lr={best_result.learning_rate}")
        
        return results
    
    def _run_single_test(self, 
                        model_name: str,
                        embedding_dim: int,
                        adapter_strategy: str,
                        learning_rate: float,
                        batch_size: int,
                        test_epochs: int,
                        test_data: Dict[str, torch.Tensor],
                        quality_threshold: float) -> ModelTestResult:
        """Запуск одного теста с конкретными параметрами"""
        
        start_time = time.time()
        
        # Создаем trainer конфигурацию
        trainer_config = AdapterIntegrationConfig(
            teacher_model=model_name,
            teacher_embedding_dim=embedding_dim,
            cube_dimensions=self.cube_dimensions,
            surface_strategy=self.surface_strategy,
            adapter_strategy=adapter_strategy,
            adapter_learning_rate=learning_rate,
            cube_learning_rate=learning_rate * 0.5,  # Cube LR меньше adapter LR
            joint_training=True
        )
        
        # Создаем trainer
        trainer = AdapterCubeTrainer(trainer_config, device=str(self.device))
        trainer.initialize_components()
        
        # Базовые метрики
        initial_memory = torch.cuda.memory_allocated() if self.device.type == 'cuda' else 0
        parameter_count = trainer.adapter.get_parameter_count()
        compression_ratio = trainer.adapter.get_compression_ratio()
        
        # Training loop (упрощенный для быстрого тестирования)  
        losses = []
        
        question_embeddings = test_data["questions"][:batch_size].to(self.device)
        answer_embeddings = test_data["answers"][:batch_size].to(self.device)
        
        for epoch in range(test_epochs):
            metrics = trainer.train_step(question_embeddings, answer_embeddings)
            losses.append(metrics["total_loss"])
            
            # Early stopping если loss растет
            if len(losses) > 2 and losses[-1] > losses[-2] > losses[-3]:
                break
        
        training_time = time.time() - start_time
        final_memory = torch.cuda.memory_allocated() if self.device.type == 'cuda' else 0
        memory_usage = final_memory - initial_memory
        
        # Финальная оценка качества
        final_metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        # Определяем convergence и quality
        converged = len(losses) >= 3 and all(l < 1.0 for l in losses[-3:])
        quality_passed = final_metrics.get("cosine_similarity", 0.0) > quality_threshold
        
        return ModelTestResult(
            model_name=model_name,
            adapter_strategy=adapter_strategy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            final_loss=final_metrics["total_loss"],
            reconstruction_loss=final_metrics.get("reconstruction_loss", 0.0),
            cosine_similarity=final_metrics.get("cosine_similarity", 0.0),
            semantic_preservation=final_metrics.get("semantic_preservation", 0.0),
            training_time=training_time,
            memory_usage=float(memory_usage) / 1024**2,  # В MB
            convergence_epochs=len(losses),
            parameter_count=parameter_count,
            compression_ratio=compression_ratio,
            converged=converged,
            quality_passed=quality_passed
        )
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Сравнение нескольких моделей и создание отчета
        
        Args:
            model_names: Список моделей для сравнения
            
        Returns:
            Dict: Comprehensive comparison report
        """
        self.logger.info(f"🏁 Starting comparison of {len(model_names)} models...")
        
        # Запускаем тесты для всех моделей
        all_results = []
        for model_name in model_names:
            if model_name not in self.test_embeddings:
                self.logger.warning(f"[WARNING] Skipping {model_name} - no test data")
                continue
            
            model_results = self.test_single_model(model_name)
            all_results.extend(model_results)
            self.test_results.extend(model_results)
        
        # Анализ результатов
        comparison_report = self._analyze_results(all_results)
        
        # Сохранение отчета
        self._save_comparison_report(comparison_report)
        
        return comparison_report
    
    def _analyze_results(self, results: List[ModelTestResult]) -> Dict[str, Any]:
        """Анализ результатов тестирования"""
        
        # Группировка по моделям
        by_model = defaultdict(list)
        for result in results:
            by_model[result.model_name].append(result)
        
        analysis = {
            "summary": {
                "total_tests": len(results),
                "models_tested": len(by_model),
                "successful_tests": len([r for r in results if r.converged]),
                "quality_passed": len([r for r in results if r.quality_passed])
            },
            "models": {},
            "best_overall": None,
            "recommendations": {}
        }
        
        # Анализ по моделям
        for model_name, model_results in by_model.items():
            converged_results = [r for r in model_results if r.converged]
            quality_results = [r for r in model_results if r.quality_passed]
            
            if converged_results:
                best_result = min(converged_results, key=lambda x: x.final_loss)
                
                analysis["models"][model_name] = {
                    "embedding_dim": best_result.compression_ratio * self.surface_size,  # Восстанавливаем original dim
                    "compression_ratio": best_result.compression_ratio,
                    "best_strategy": best_result.adapter_strategy,
                    "best_learning_rate": best_result.learning_rate,
                    "best_batch_size": best_result.batch_size,
                    "final_loss": best_result.final_loss,
                    "cosine_similarity": best_result.cosine_similarity,
                    "training_time": best_result.training_time,
                    "parameter_count": best_result.parameter_count,
                    "quality_passed": len(quality_results) > 0,
                    "convergence_rate": len(converged_results) / len(model_results)
                }
        
        # Лучшая модель overall
        best_models = [(name, data) for name, data in analysis["models"].items() 
                      if data["quality_passed"]]
        
        if best_models:
            best_model_name, best_model_data = min(best_models, 
                                                  key=lambda x: x[1]["final_loss"])
            analysis["best_overall"] = {
                "model": best_model_name,
                **best_model_data
            }
        
        # Рекомендации
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация рекомендаций на основе анализа"""
        recommendations = {
            "production_ready": [],
            "development_suitable": [],
            "high_compression": [],
            "fast_training": [],
            "best_quality": []
        }
        
        for model_name, data in analysis["models"].items():
            # Production ready: quality + stable convergence
            if data["quality_passed"] and data["convergence_rate"] > 0.7:
                recommendations["production_ready"].append(model_name)
            
            # Development suitable: at least converges
            if data["convergence_rate"] > 0.5:
                recommendations["development_suitable"].append(model_name)
            
            # High compression: ratio > 10x
            if data["compression_ratio"] > 10.0:
                recommendations["high_compression"].append(model_name)
            
            # Fast training: < 30 seconds per test
            if data["training_time"] < 30.0:
                recommendations["fast_training"].append(model_name)
            
            # Best quality: cosine similarity > 0.8
            if data["cosine_similarity"] > 0.8:
                recommendations["best_quality"].append(model_name)
        
        return recommendations
    
    def _save_comparison_report(self, report: Dict[str, Any]):
        """Сохранение отчета о сравнении"""
        
        # JSON отчет
        json_path = self.output_dir / "comparison_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Подробные результаты
        results_path = self.output_dir / "detailed_results.json"
        results_data = [
            {
                "model_name": r.model_name,
                "adapter_strategy": r.adapter_strategy,
                "learning_rate": r.learning_rate,
                "batch_size": r.batch_size,
                "final_loss": r.final_loss,
                "reconstruction_loss": r.reconstruction_loss,
                "cosine_similarity": r.cosine_similarity,
                "semantic_preservation": r.semantic_preservation,
                "training_time": r.training_time,
                "memory_usage": r.memory_usage,
                "convergence_epochs": r.convergence_epochs,
                "parameter_count": r.parameter_count,
                "compression_ratio": r.compression_ratio,
                "converged": r.converged,  
                "quality_passed": r.quality_passed,
                "recommended": r.recommended
            }
            for r in self.test_results
        ]
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[DATA] Comparison report saved:")
        self.logger.info(f"   Summary: {json_path}")
        self.logger.info(f"   Details: {results_path}")
    
    def get_recommended_config_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Получение рекомендуемой конфигурации для модели на основе тестирования"""
        model_results = [r for r in self.test_results if r.model_name == model_name and r.recommended]
        
        if not model_results:
            return None
        
        best_result = model_results[0]  # recommended уже выбран как лучший
        
        return {
            "model_name": model_name,
            "adapter_strategy": best_result.adapter_strategy,
            "learning_rate": best_result.learning_rate,
            "batch_size": best_result.batch_size,
            "expected_quality": {
                "final_loss": best_result.final_loss,
                "cosine_similarity": best_result.cosine_similarity,
                "training_time": best_result.training_time
            },
            "compression_info": {
                "compression_ratio": best_result.compression_ratio,
                "parameter_count": best_result.parameter_count
            }
        }


# Удобные функции для быстрого тестирования
def quick_model_comparison(models: List[str] = None, 
                          cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                          device: str = "cpu") -> Dict[str, Any]:
    """
    Быстрое сравнение моделей с базовыми настройками
    
    Args:
        models: Список моделей (или default set)
        cube_dimensions: Размеры куба
        device: Устройство
        
    Returns:
        Comparison report
    """
    if models is None:
        # Default set для Stage 3.1.3
        models = ["Meta-Llama-3-8B", "DistilBERT", "BERT-large", "GPT-3.5", "RoBERTa-large"]
    
    suite = ModelComparisonSuite(
        cube_dimensions=cube_dimensions,
        device=device
    )
    
    return suite.compare_models(models)


def test_single_model_quick(model_name: str, 
                           cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                           device: str = "cpu") -> List[ModelTestResult]:
    """Быстрое тестирование одной модели"""
    suite = ModelComparisonSuite(
        cube_dimensions=cube_dimensions,
        device=device
    )
    
    return suite.test_single_model(model_name) 