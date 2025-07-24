"""
🧪 Basic Tests - Model Comparison Suite (Stage 3.1.3)
Базовые тесты для инфраструктуры Model-Agnostic Training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
import tempfile
from pathlib import Path

from training.embedding_trainer.model_comparison import (
    ModelComparisonSuite,
    ModelDetectionSystem,
    ModelTestConfig,
    quick_model_comparison,
    test_single_model_quick
)

class TestModelDetectionSystem:
    """Тестирование системы определения моделей"""
    
    def test_detection_system_init(self):
        """Тест инициализации системы"""
        detector = ModelDetectionSystem()
        
        assert detector is not None
        assert len(detector.model_database) >= 12  # Основные модели + расширенные
        assert "Meta-Llama-3-8B" in detector.model_database
        assert "DistilBERT" in detector.model_database
        assert "Claude-3" in detector.model_database  # Новая модель
    
    def test_model_detection(self):
        """Тест определения моделей"""
        detector = ModelDetectionSystem()
        
        # Точное совпадение
        llama_info = detector.detect_model("Meta-Llama-3-8B")
        assert llama_info is not None
        assert llama_info["embedding_dim"] == 4096
        
        # Fuzzy matching
        distil_info = detector.detect_model("distilbert")
        assert distil_info is not None
        assert distil_info["embedding_dim"] == 768
        
        # Неизвестная модель
        unknown_info = detector.detect_model("UnknownModel")
        assert unknown_info is None
    
    def test_recommended_config(self):
        """Тест получения рекомендуемых конфигураций"""
        detector = ModelDetectionSystem()
        
        # High compression model (LLaMA)
        llama_config = detector.get_recommended_config("Meta-Llama-3-8B", target_surface_size=225)
        assert llama_config.model_name == "Meta-Llama-3-8B"
        assert llama_config.embedding_dim == 4096
        assert "hierarchical" in llama_config.adapter_strategies  # High compression
        
        # Medium compression model (BERT-large)
        bert_config = detector.get_recommended_config("BERT-large", target_surface_size=225)
        assert bert_config.model_name == "BERT-large"
        assert bert_config.embedding_dim == 1024
        assert "learned_linear" in bert_config.adapter_strategies  # Medium compression
        
        # Low compression model (DistilBERT)
        distil_config = detector.get_recommended_config("DistilBERT", target_surface_size=225)
        assert distil_config.model_name == "DistilBERT"
        assert distil_config.embedding_dim == 768
        assert distil_config.adapter_strategies == ["learned_linear"]  # Low compression
    
    def test_list_supported_models(self):
        """Тест получения списка поддерживаемых моделей"""
        detector = ModelDetectionSystem()
        models = detector.list_supported_models()
        
        assert len(models) >= 12
        assert "Meta-Llama-3-8B" in models
        assert "GPT-4" in models
        assert "Claude-3" in models


class TestModelComparisonSuite:
    """Тестирование ModelComparisonSuite"""
    
    def test_suite_initialization(self):
        """Тест инициализации suite"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ModelComparisonSuite(
                cube_dimensions=(15, 15, 11),
                output_dir=temp_dir,
                device="cpu"
            )
            
            assert suite is not None
            assert suite.surface_size == 225  # 15*15
            assert suite.cube_dimensions == (15, 15, 11)
            assert len(suite.test_embeddings) >= 12  # Для всех моделей
            assert Path(temp_dir).exists()
    
    def test_test_data_preparation(self):
        """Тест подготовки тестовых данных"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ModelComparisonSuite(output_dir=temp_dir, device="cpu")
            
            # Проверяем что данные созданы для всех моделей
            assert "Meta-Llama-3-8B" in suite.test_embeddings
            assert "DistilBERT" in suite.test_embeddings
            
            # Проверяем размеры данных
            llama_data = suite.test_embeddings["Meta-Llama-3-8B"]
            assert "questions" in llama_data
            assert "answers" in llama_data
            assert llama_data["questions"].shape == (20, 4096)  # 20 questions, 4096D
            assert llama_data["answers"].shape == (20, 4096)    # 20 answers, 4096D
            
            distil_data = suite.test_embeddings["DistilBERT"]
            assert distil_data["questions"].shape == (20, 768)   # 20 questions, 768D
            assert distil_data["answers"].shape == (20, 768)     # 20 answers, 768D
    
    def test_single_model_testing_structure(self):
        """Тест структуры тестирования одной модели (без полного запуска)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ModelComparisonSuite(output_dir=temp_dir, device="cpu")
            
            # Проверяем что метод существует и принимает правильные параметры
            assert hasattr(suite, 'test_single_model')
            assert hasattr(suite, '_run_single_test')
            assert hasattr(suite, 'compare_models')
            
            # Проверяем что model detector работает
            config = suite.model_detector.get_recommended_config("DistilBERT")
            assert config.model_name == "DistilBERT"
            assert config.embedding_dim == 768
    
    def test_result_analysis_structure(self):
        """Тест структуры анализа результатов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = ModelComparisonSuite(output_dir=temp_dir, device="cpu")
            
            # Проверяем методы анализа
            assert hasattr(suite, '_analyze_results')
            assert hasattr(suite, '_generate_recommendations') 
            assert hasattr(suite, '_save_comparison_report')
            assert hasattr(suite, 'get_recommended_config_for_model')


class TestQuickFunctions:
    """Тестирование quick functions"""
    
    def test_quick_functions_structure(self):
        """Тест структуры quick functions"""
        # Проверяем что функции импортируются
        assert quick_model_comparison is not None
        assert test_single_model_quick is not None
        
        # Проверяем что принимают правильные параметры
        # (Без реального запуска для экономии времени)
        pass


class TestIntegrationReadiness:
    """Тест готовности к интеграции с существующей системой"""
    
    def test_adapter_integration_imports(self):
        """Тест импорта компонентов AdapterCubeTrainer"""
        try:
            from training.embedding_trainer.adapter_integration import (
                AdapterCubeTrainer, 
                AdapterIntegrationConfig
            )
            from data.embedding_adapter.universal_adapter import (
                KNOWN_MODELS, 
                UniversalEmbeddingAdapter
            )
            
            # Проверяем что все импорты работают
            assert AdapterCubeTrainer is not None
            assert AdapterIntegrationConfig is not None
            assert KNOWN_MODELS is not None
            assert len(KNOWN_MODELS) >= 9  # Базовые модели
            
        except ImportError as e:
            pytest.fail(f"Integration import failed: {e}")
    
    def test_model_compatibility(self):
        """Тест совместимости с существующими моделями"""
        detector = ModelDetectionSystem()
        
        # Тестируем основные модели из KNOWN_MODELS
        essential_models = ["Meta-Llama-3-8B", "DistilBERT", "BERT-large", "GPT-3.5"]
        
        for model in essential_models:
            config = detector.get_recommended_config(model)
            assert config is not None
            assert config.embedding_dim > 0
            assert len(config.adapter_strategies) > 0
            assert len(config.learning_rates) > 0
            assert len(config.batch_sizes) > 0


def run_basic_tests():
    """Запуск базовых тестов Model Comparison Suite"""
    print("🧪 Running Model Comparison Suite Basic Tests...")
    
    test_classes = [
        TestModelDetectionSystem,
        TestModelComparisonSuite, 
        TestQuickFunctions,
        TestIntegrationReadiness
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n[INFO] Testing {test_class.__name__}:")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  [OK] {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")
    
    print(f"\n[TARGET] Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("[OK] All Model Comparison Suite basic tests passed!")
        return True
    else:
        print("[ERROR] Some tests failed!")
        return False


if __name__ == "__main__":
    run_basic_tests() 