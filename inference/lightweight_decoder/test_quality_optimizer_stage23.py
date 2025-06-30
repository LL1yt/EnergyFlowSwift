"""
[TEST] ТЕСТЫ QUALITY OPTIMIZER - Stage 2.3 Production Integration
Комплексное тестирование системы оптимизации качества генерации

Tests:
1. AdvancedQualityAssessment functionality
2. Parameter optimization system
3. Production readiness evaluation
4. Integration с GenerativeDecoder
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import logging
from pathlib import Path
import time

# Импорты модулей
from quality_optimizer import (
    QualityMetrics, OptimizationConfig, AdvancedQualityAssessment,
    GenerationParameterOptimizer, create_quality_optimizer
)
from generative_decoder import GenerativeDecoder, GenerativeConfig

# Настройка логирования для тестов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityOptimizerStage23Test(unittest.TestCase):
    """
    [TARGET] STAGE 2.3 QUALITY OPTIMIZATION TESTS
    
    Проверяем полную функциональность системы оптимизации качества:
    - Продвинутые метрики качества (BLEU, ROUGE, coherence, fluency)
    - Оптимизация параметров генерации
    - Production readiness assessment
    - Интеграция с GenerativeDecoder RET v2.1
    """
    
    @classmethod
    def setUpClass(cls):
        """Инициализация для всех тестов"""
        logger.info("[START] Настройка тестов Quality Optimizer Stage 2.3...")
        
        # Конфигурация для тестирования
        cls.config = OptimizationConfig(
            target_bleu=0.45,
            target_rouge_l=0.35,
            target_coherence=0.75,
            target_fluency=0.80,
            max_optimization_iterations=10,  # Reduced for testing
            patience=3
        )
        
        # Инициализация quality assessor
        cls.quality_assessor = AdvancedQualityAssessment(cls.config)
        
        # Тестовые данные
        cls.test_cases = [
            {
                'generated': 'The quick brown fox jumps over the lazy dog.',
                'reference': 'A fast brown fox leaps over a sleeping dog.',
                'expected_quality': 0.5  # Approximate expected quality
            },
            {
                'generated': 'Hello world, this is a test sentence with proper grammar.',
                'reference': 'Hello world, this represents a test sentence using correct grammar.',
                'expected_quality': 0.6
            },
            {
                'generated': 'Machine learning models can generate text effectively.',
                'reference': 'AI systems are capable of producing text efficiently.',
                'expected_quality': 0.4
            }
        ]
        
        # Mock embeddings для тестирования
        cls.test_embeddings = [torch.randn(768) for _ in range(len(cls.test_cases))]
        cls.reference_texts = [case['reference'] for case in cls.test_cases]
        
        logger.info("[OK] Настройка завершена")
    
    def test_01_quality_metrics_basic(self):
        """Test 1: Базовая функциональность QualityMetrics"""
        logger.info("[TEST] Test 1: QualityMetrics basic functionality")
        
        # Создание метрик
        metrics = QualityMetrics(
            bleu_score=0.5,
            rouge_l=0.4,
            coherence_score=0.7,
            fluency_score=0.8,
            overall_quality=0.6
        )
        
        # Проверка значений
        self.assertEqual(metrics.bleu_score, 0.5)
        self.assertEqual(metrics.rouge_l, 0.4)
        self.assertEqual(metrics.coherence_score, 0.7)
        self.assertEqual(metrics.fluency_score, 0.8)
        self.assertEqual(metrics.overall_quality, 0.6)
        
        # Конвертация в dict
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('bleu_score', metrics_dict)
        self.assertIn('overall_quality', metrics_dict)
        
        logger.info("[OK] QualityMetrics basic functionality - PASSED")
    
    def test_02_advanced_quality_assessment(self):
        """Test 2: AdvancedQualityAssessment comprehensive evaluation"""
        logger.info("[TEST] Test 2: AdvancedQualityAssessment comprehensive evaluation")
        
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(case=i):
                # Оценка качества
                metrics = self.quality_assessor.assess_comprehensive_quality(
                    test_case['generated'],
                    test_case['reference'],
                    generation_time=0.1
                )
                
                # Проверка базовых метрик
                self.assertIsInstance(metrics.bleu_score, float)
                self.assertIsInstance(metrics.coherence_score, float)
                self.assertIsInstance(metrics.fluency_score, float)
                self.assertIsInstance(metrics.overall_quality, float)
                self.assertIsInstance(metrics.production_readiness, float)
                
                # Проверка диапазонов
                self.assertGreaterEqual(metrics.bleu_score, 0.0)
                self.assertLessEqual(metrics.bleu_score, 1.0)
                self.assertGreaterEqual(metrics.coherence_score, 0.0)
                self.assertLessEqual(metrics.coherence_score, 1.0)
                self.assertGreaterEqual(metrics.fluency_score, 0.0)
                self.assertLessEqual(metrics.fluency_score, 1.0)
                
                # Performance метрики
                self.assertEqual(metrics.generation_time, 0.1)
                self.assertGreater(metrics.tokens_per_second, 0)
                
                logger.info(f"   Case {i}: BLEU={metrics.bleu_score:.3f}, "
                          f"Coherence={metrics.coherence_score:.3f}, "
                          f"Fluency={metrics.fluency_score:.3f}, "
                          f"Overall={metrics.overall_quality:.3f}")
        
        logger.info("[OK] AdvancedQualityAssessment comprehensive evaluation - PASSED")
    
    def test_03_quality_assessment_edge_cases(self):
        """Test 3: Quality assessment edge cases"""
        logger.info("[TEST] Test 3: Quality assessment edge cases")
        
        # Пустые строки
        metrics_empty = self.quality_assessor.assess_comprehensive_quality("", "test")
        self.assertEqual(metrics_empty.overall_quality, 0.0)
        
        # Очень короткие тексты
        metrics_short = self.quality_assessor.assess_comprehensive_quality("Hi", "Hello")
        self.assertGreater(metrics_short.overall_quality, 0.0)
        
        # Идентичные тексты
        identical_text = "This is a test sentence."
        metrics_identical = self.quality_assessor.assess_comprehensive_quality(
            identical_text, identical_text
        )
        self.assertGreaterEqual(metrics_identical.overall_quality, 0.8)  # Should be very high
        
        # Очень длинные тексты
        long_text = " ".join(["word"] * 100)
        metrics_long = self.quality_assessor.assess_comprehensive_quality(
            long_text, "This is a reference text."
        )
        self.assertGreaterEqual(metrics_long.overall_quality, 0.0)
        
        logger.info("[OK] Quality assessment edge cases - PASSED")
    
    def test_04_optimization_config_validation(self):
        """Test 4: OptimizationConfig validation"""
        logger.info("[TEST] Test 4: OptimizationConfig validation")
        
        # Создание конфигурации
        config = OptimizationConfig(
            target_bleu=0.45,
            target_rouge_l=0.35,
            max_optimization_iterations=10
        )
        
        # Проверка значений
        self.assertEqual(config.target_bleu, 0.45)
        self.assertEqual(config.target_rouge_l, 0.35)
        self.assertEqual(config.max_optimization_iterations, 10)
        
        # Проверка диапазонов параметров
        self.assertIsInstance(config.temperature_range, tuple)
        self.assertIsInstance(config.top_k_range, tuple)
        self.assertIsInstance(config.top_p_range, tuple)
        
        # Проверка корректности диапазонов
        self.assertLess(config.temperature_range[0], config.temperature_range[1])
        self.assertLess(config.top_k_range[0], config.top_k_range[1])
        self.assertLess(config.top_p_range[0], config.top_p_range[1])
        
        logger.info("[OK] OptimizationConfig validation - PASSED")
    
    def test_05_parameter_optimizer_initialization(self):
        """Test 5: GenerationParameterOptimizer initialization"""
        logger.info("[TEST] Test 5: GenerationParameterOptimizer initialization")
        
        # Создание оптимизатора
        optimizer = GenerationParameterOptimizer(self.config)
        
        # Проверка инициализации
        self.assertEqual(optimizer.config, self.config)
        self.assertIsInstance(optimizer.quality_assessor, AdvancedQualityAssessment)
        self.assertEqual(len(optimizer.optimization_history), 0)
        self.assertIsNone(optimizer.best_params)
        self.assertEqual(optimizer.best_score, 0.0)
        
        # Проверка начальных параметров
        initial_params = optimizer._get_initial_parameters()
        self.assertIn('temperature', initial_params)
        self.assertIn('top_k', initial_params)
        self.assertIn('top_p', initial_params)
        self.assertIn('repetition_penalty', initial_params)
        
        logger.info("[OK] GenerationParameterOptimizer initialization - PASSED")
    
    def test_06_parameter_generation(self):
        """Test 6: Parameter generation и mutation"""
        logger.info("[TEST] Test 6: Parameter generation и mutation")
        
        optimizer = GenerationParameterOptimizer(self.config)
        base_params = optimizer._get_initial_parameters()
        
        # Генерация candidate параметров
        for iteration in [0, 10, 25]:
            candidate_params = optimizer._generate_candidate_parameters(base_params, iteration)
            
            # Проверка наличия всех параметров
            self.assertIn('temperature', candidate_params)
            self.assertIn('top_k', candidate_params)
            self.assertIn('top_p', candidate_params)
            self.assertIn('repetition_penalty', candidate_params)
            
            # Проверка диапазонов
            self.assertGreaterEqual(candidate_params['temperature'], self.config.temperature_range[0])
            self.assertLessEqual(candidate_params['temperature'], self.config.temperature_range[1])
            
            self.assertGreaterEqual(candidate_params['top_k'], self.config.top_k_range[0])
            self.assertLessEqual(candidate_params['top_k'], self.config.top_k_range[1])
            
            self.assertGreaterEqual(candidate_params['top_p'], self.config.top_p_range[0])
            self.assertLessEqual(candidate_params['top_p'], self.config.top_p_range[1])
            
            logger.info(f"   Iteration {iteration}: temp={candidate_params['temperature']:.2f}, "
                      f"top_k={candidate_params['top_k']}, top_p={candidate_params['top_p']:.2f}")
        
        logger.info("[OK] Parameter generation и mutation - PASSED")
    
    def test_07_generative_decoder_integration(self):
        """Test 7: Интеграция с GenerativeDecoder"""
        logger.info("[TEST] Test 7: Интеграция с GenerativeDecoder")
        
        # Создание GenerativeDecoder
        decoder_config = GenerativeConfig(
            architecture_type="resource_efficient_v21",
            embedding_dim=768,
            max_length=50,
            verbose_logging=False
        )
        decoder = GenerativeDecoder(decoder_config)
        
        # Тестовое embedding
        test_embedding = torch.randn(768)
        
        # Генерация текста
        try:
            result = decoder.generate(test_embedding)
            self.assertIn('text', result)
            self.assertIsInstance(result['text'], str)
            
            generated_text = result['text']
            
            # Оценка качества сгенерированного текста
            metrics = self.quality_assessor.assess_comprehensive_quality(
                generated_text,
                "This is a reference text for comparison.",
                result.get('generation_time', 0.0)
            )
            
            # Проверка метрик
            self.assertGreaterEqual(metrics.overall_quality, 0.0)
            self.assertLessEqual(metrics.overall_quality, 1.0)
            
            logger.info(f"   Generated: '{generated_text}'")
            logger.info(f"   Quality: {metrics.overall_quality:.3f}")
            logger.info(f"   Coherence: {metrics.coherence_score:.3f}")
            logger.info(f"   Fluency: {metrics.fluency_score:.3f}")
            
        except Exception as e:
            self.fail(f"GenerativeDecoder integration failed: {e}")
        
        logger.info("[OK] Интеграция с GenerativeDecoder - PASSED")
    
    def test_08_mock_parameter_optimization(self):
        """Test 8: Mock parameter optimization (abbreviated)"""
        logger.info("[TEST] Test 8: Mock parameter optimization")
        
        # Создание mock GenerativeDecoder для быстрого тестирования
        class MockGenerativeDecoder:
            def generate(self, embedding, **kwargs):
                # Простая генерация для тестирования
                temp = kwargs.get('temperature', 0.8)
                top_k = kwargs.get('top_k', 50)
                
                # Симуляция качества на основе параметров
                quality_factor = 1.0 - abs(temp - 0.7) - abs(top_k - 40) / 100.0
                quality_factor = max(0.3, min(0.9, quality_factor))
                
                # Возвращаем результат с разным качеством
                if quality_factor > 0.7:
                    text = "This is a high quality generated text with good coherence."
                elif quality_factor > 0.5:
                    text = "This is moderate quality text generation result."
                else:
                    text = "Low quality text result."
                
                return {
                    'text': text,
                    'generation_time': 0.05,
                    'quality_factor': quality_factor
                }
        
        mock_decoder = MockGenerativeDecoder()
        optimizer = GenerationParameterOptimizer(self.config)
        
        # Быстрая оптимизация (3 итерации)
        quick_config = OptimizationConfig(max_optimization_iterations=3, patience=2)
        quick_optimizer = GenerationParameterOptimizer(quick_config)
        
        # Оптимизация с mock моделью
        try:
            result = quick_optimizer.optimize_parameters(
                mock_decoder,
                self.test_embeddings[:2],  # Только 2 embedding для скорости
                self.reference_texts[:2],
                max_iterations=3
            )
            
            # Проверка результатов
            self.assertIn('best_params', result)
            self.assertIn('best_score', result)
            self.assertIn('optimization_history', result)
            
            self.assertIsInstance(result['best_params'], dict)
            self.assertIsInstance(result['best_score'], float)
            self.assertGreaterEqual(result['best_score'], 0.0)
            
            logger.info(f"   Best score: {result['best_score']:.3f}")
            logger.info(f"   Best params: {result['best_params']}")
            logger.info(f"   Iterations: {result['total_iterations']}")
            
        except Exception as e:
            self.fail(f"Parameter optimization failed: {e}")
        
        logger.info("[OK] Mock parameter optimization - PASSED")
    
    def test_09_production_readiness_evaluation(self):
        """Test 9: Production readiness evaluation"""
        logger.info("[TEST] Test 9: Production readiness evaluation")
        
        # Тестовые cases с разным уровнем качества
        test_cases = [
            {
                'metrics': QualityMetrics(
                    bleu_score=0.5, rouge_l=0.4, coherence_score=0.8,
                    fluency_score=0.85, overall_quality=0.7, generation_time=0.5
                ),
                'expected_readiness': 'high'
            },
            {
                'metrics': QualityMetrics(
                    bleu_score=0.3, rouge_l=0.25, coherence_score=0.6,
                    fluency_score=0.65, overall_quality=0.5, generation_time=1.5
                ),
                'expected_readiness': 'medium'
            },
            {
                'metrics': QualityMetrics(
                    bleu_score=0.1, rouge_l=0.15, coherence_score=0.4,
                    fluency_score=0.45, overall_quality=0.3, generation_time=2.0
                ),
                'expected_readiness': 'low'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            with self.subTest(case=i):
                metrics = test_case['metrics']
                
                # Пересчитаем production readiness
                readiness = self.quality_assessor._calculate_production_readiness(metrics)
                
                # Проверка диапазона
                self.assertGreaterEqual(readiness, 0.0)
                self.assertLessEqual(readiness, 1.0)
                
                # Логическая проверка с учетом float precision
                if test_case['expected_readiness'] == 'high':
                    self.assertGreaterEqual(readiness, 0.7)
                elif test_case['expected_readiness'] == 'medium':
                    self.assertGreaterEqual(readiness, 0.2)
                    self.assertLessEqual(readiness, 0.7)  # Используем LessEqual для float precision
                else:  # low
                    self.assertLess(readiness, 0.4)
                
                logger.info(f"   Case {i} ({test_case['expected_readiness']}): "
                          f"Production readiness = {readiness:.3f}")
        
        logger.info("[OK] Production readiness evaluation - PASSED")
    
    def test_10_factory_function(self):
        """Test 10: Factory function create_quality_optimizer"""
        logger.info("[TEST] Test 10: Factory function create_quality_optimizer")
        
        # Создание оптимизатора через factory
        optimizer = create_quality_optimizer(
            target_bleu=0.5,
            target_rouge_l=0.4,
            max_iterations=20
        )
        
        # Проверка конфигурации
        self.assertIsInstance(optimizer, GenerationParameterOptimizer)
        self.assertEqual(optimizer.config.target_bleu, 0.5)
        self.assertEqual(optimizer.config.target_rouge_l, 0.4)
        self.assertEqual(optimizer.config.max_optimization_iterations, 20)
        
        logger.info("[OK] Factory function create_quality_optimizer - PASSED")
    
    def test_11_optimization_results_serialization(self):
        """Test 11: Сохранение и загрузка результатов оптимизации"""
        logger.info("[TEST] Test 11: Optimization results serialization")
        
        optimizer = GenerationParameterOptimizer(self.config)
        
        # Симуляция результатов оптимизации
        optimizer.best_params = {'temperature': 0.75, 'top_k': 45, 'top_p': 0.9}
        optimizer.best_score = 0.85
        optimizer.optimization_history = [
            {'iteration': 0, 'score': 0.7, 'params': {'temperature': 0.8}},
            {'iteration': 1, 'score': 0.75, 'params': {'temperature': 0.75}}
        ]
        
        # Сохранение в временный файл
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            optimizer.save_optimization_results(temp_file)
            
            # Проверка что файл создан и не пустой
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
            
            # Проверка содержимого (базовая)
            import json
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('best_params', data)
            self.assertIn('best_score', data)
            self.assertIn('optimization_history', data)
            self.assertEqual(data['best_score'], 0.85)
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        logger.info("[OK] Optimization results serialization - PASSED")
    
    def test_12_stage23_integration_readiness(self):
        """Test 12: Полная готовность Stage 2.3"""
        logger.info("[TEST] Test 12: Stage 2.3 integration readiness")
        
        # Проверка всех компонентов Stage 2.3
        components_ready = {
            'AdvancedQualityAssessment': False,
            'GenerationParameterOptimizer': False,
            'Factory Functions': False,
            'Serialization': False,
            'GenerativeDecoder Integration': False
        }
        
        try:
            # 1. AdvancedQualityAssessment
            assessor = AdvancedQualityAssessment(self.config)
            test_metrics = assessor.assess_comprehensive_quality(
                "Test text", "Reference text", 0.1
            )
            self.assertGreater(test_metrics.overall_quality, 0.0)
            components_ready['AdvancedQualityAssessment'] = True
            
            # 2. GenerationParameterOptimizer
            optimizer = GenerationParameterOptimizer(self.config)
            initial_params = optimizer._get_initial_parameters()
            self.assertIn('temperature', initial_params)
            components_ready['GenerationParameterOptimizer'] = True
            
            # 3. Factory Functions
            factory_optimizer = create_quality_optimizer()
            self.assertIsInstance(factory_optimizer, GenerationParameterOptimizer)
            components_ready['Factory Functions'] = True
            
            # 4. Serialization
            optimizer.best_params = {'test': 'value'}
            optimizer.best_score = 0.5
            try:
                temp_path = tempfile.mktemp(suffix='.json')
                optimizer.save_optimization_results(temp_path)
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                components_ready['Serialization'] = True
            except Exception as e:
                logger.warning(f"Serialization test warning: {e}")
                components_ready['Serialization'] = False
            
            # 5. GenerativeDecoder Integration (basic check)
            try:
                from generative_decoder import GenerativeDecoder, GenerativeConfig
                config = GenerativeConfig(verbose_logging=False)
                decoder = GenerativeDecoder(config)
                components_ready['GenerativeDecoder Integration'] = True
            except Exception as e:
                logger.warning(f"GenerativeDecoder integration warning: {e}")
            
        except Exception as e:
            self.fail(f"Stage 2.3 component failure: {e}")
        
        # Финальная проверка готовности
        ready_components = sum(components_ready.values())
        total_components = len(components_ready)
        readiness_percentage = (ready_components / total_components) * 100
        
        logger.info(f"[DATA] Stage 2.3 Component Readiness:")
        for component, ready in components_ready.items():
            status = "[OK] READY" if ready else "[ERROR] NOT READY"
            logger.info(f"   {component}: {status}")
        
        logger.info(f"[TARGET] Overall Readiness: {readiness_percentage:.1f}% ({ready_components}/{total_components})")
        
        # Требуем минимум 80% готовности для Stage 2.3
        self.assertGreaterEqual(readiness_percentage, 80.0)
        
        logger.info("[OK] Stage 2.3 integration readiness - PASSED")


def run_stage23_quality_tests():
    """Запуск всех тестов Stage 2.3"""
    print("[START] Запуск тестов Quality Optimizer Stage 2.3...")
    
    # Создание test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(QualityOptimizerStage23Test)
    
    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Результаты
    if result.wasSuccessful():
        print("[SUCCESS] ВСЕ ТЕСТЫ STAGE 2.3 ПРОШЛИ УСПЕШНО!")
        print(f"[OK] Тестов выполнено: {result.testsRun}")
        print(f"[OK] Ошибок: {len(result.errors)}")
        print(f"[OK] Неудач: {len(result.failures)}")
        return True
    else:
        print("[ERROR] НЕКОТОРЫЕ ТЕСТЫ STAGE 2.3 НЕ ПРОШЛИ")
        print(f"[DATA] Тестов выполнено: {result.testsRun}")
        print(f"[ERROR] Ошибок: {len(result.errors)}")
        print(f"[ERROR] Неудач: {len(result.failures)}")
        return False


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении
    success = run_stage23_quality_tests()
    exit(0 if success else 1)