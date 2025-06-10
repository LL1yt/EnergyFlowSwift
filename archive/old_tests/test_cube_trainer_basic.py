"""
Тесты для CubeTrainer - основной класс обучения 3D Cubic Core

Этот файл содержит тесты для проверки:
1. Инициализации CubeTrainer
2. Конфигурационной системы
3. Интеграции с компонентами
4. Базовой функциональности

Автор: 3D Cellular Neural Network Project  
Версия: Phase 3.1 - Stage 1.1
Дата: 6 июня 2025
"""

import sys
import torch
import numpy as np
import traceback
from pathlib import Path

def test_cube_trainer_import():
    """Тест 1: Проверка импорта CubeTrainer"""
    print("🧪 Тест 1: Импорт CubeTrainer")
    
    try:
        from training.embedding_trainer import CubeTrainer, TrainingConfig, EmbeddingMetrics
        print("[OK] CubeTrainer, TrainingConfig, EmbeddingMetrics успешно импортированы")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Неожиданная ошибка: {e}")
        return False

def test_training_config():
    """Тест 2: Проверка TrainingConfig"""
    print("\n🧪 Тест 2: TrainingConfig")
    
    try:
        from training.embedding_trainer import TrainingConfig
        
        # Создание конфигурации по умолчанию
        config = TrainingConfig()
        
        print(f"[OK] Конфигурация по умолчанию создана")
        print(f"   Mode: {config.mode}")
        print(f"   Device: {config.device}")
        print(f"   Lattice size: {config.lattice_size}")
        print(f"   Embedding dim: {config.embedding_dim}")
        
        # Проверка значений по умолчанию
        assert config.mode == "autoencoder"
        assert config.device == "cpu"
        assert config.lattice_size == [8, 8, 8]
        assert config.embedding_dim == 768
        
        # Создание кастомной конфигурации
        custom_config = TrainingConfig(
            mode="dialogue",
            device="cpu",
            lattice_size=[6, 6, 6],
            learning_rate=0.002
        )
        
        print(f"[OK] Кастомная конфигурация создана")
        print(f"   Mode: {custom_config.mode}")
        print(f"   Lattice size: {custom_config.lattice_size}")
        print(f"   Learning rate: {custom_config.learning_rate}")
        
        assert custom_config.mode == "dialogue"
        assert custom_config.lattice_size == [6, 6, 6]
        assert custom_config.learning_rate == 0.002
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка TrainingConfig: {e}")
        return False

def test_embedding_metrics():
    """Тест 3: Проверка EmbeddingMetrics"""
    print("\n🧪 Тест 3: EmbeddingMetrics")
    
    try:
        from training.embedding_trainer import EmbeddingMetrics
        
        metrics = EmbeddingMetrics(device="cpu")
        print("[OK] EmbeddingMetrics инициализированы")
        
        # Создание тестовых эмбедингов
        batch_size = 2
        embedding_dim = 768
        
        # Идентичные эмбединги (должны дать similarity = 1.0)
        identical_emb = torch.randn(batch_size, embedding_dim)
        
        similarity = metrics.calculate_cosine_similarity(identical_emb, identical_emb)
        mse_loss = metrics.calculate_mse_loss(identical_emb, identical_emb)
        
        print(f"[OK] Метрики для идентичных эмбедингов:")
        print(f"   Cosine similarity: {similarity:.4f} (ожидается ~1.0)")
        print(f"   MSE loss: {mse_loss:.6f} (ожидается ~0.0)")
        
        # Проверка результатов
        assert abs(similarity - 1.0) < 0.001, f"Similarity должен быть ~1.0, получен {similarity}"
        assert mse_loss < 0.001, f"MSE loss должен быть ~0.0, получен {mse_loss}"
        
        # Случайные разные эмбединги
        emb1 = torch.randn(batch_size, embedding_dim)
        emb2 = torch.randn(batch_size, embedding_dim)
        
        batch_metrics = metrics.compute_batch_metrics(emb1, emb2)
        
        print(f"[OK] Метрики для разных эмбедингов:")
        for metric_name, value in batch_metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # Проверка наличия всех метрик
        expected_metrics = ['cosine_similarity', 'mse_loss', 'semantic_preservation']
        for metric in expected_metrics:
            assert metric in batch_metrics, f"Метрика {metric} отсутствует"
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка EmbeddingMetrics: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_cube_trainer_initialization():
    """Тест 4: Инициализация CubeTrainer"""
    print("\n🧪 Тест 4: Инициализация CubeTrainer")
    
    try:
        from training.embedding_trainer import CubeTrainer, TrainingConfig
        
        # Инициализация с конфигурацией по умолчанию
        trainer = CubeTrainer()
        
        print("[OK] CubeTrainer инициализирован с настройками по умолчанию")
        print(f"   Mode: {trainer.config.mode}")
        print(f"   Device: {trainer.config.device}")
        print(f"   Lattice size: {trainer.config.lattice_size}")
        
        # Проверка базовых атрибутов
        assert trainer.config.mode == "autoencoder"
        assert trainer.config.device == "cpu"
        assert trainer.current_epoch == 0
        assert isinstance(trainer.training_history, list)
        assert len(trainer.training_history) == 0
        
        # Инициализация с кастомной конфигурацией
        custom_config = TrainingConfig(
            mode="dialogue",
            lattice_size=[4, 4, 4],
            learning_rate=0.002
        )
        
        custom_trainer = CubeTrainer(config=custom_config)
        
        print("[OK] CubeTrainer инициализирован с кастомной конфигурацией")
        print(f"   Mode: {custom_trainer.config.mode}")
        print(f"   Lattice size: {custom_trainer.config.lattice_size}")
        print(f"   Learning rate: {custom_trainer.config.learning_rate}")
        
        assert custom_trainer.config.mode == "dialogue"
        assert custom_trainer.config.lattice_size == [4, 4, 4]
        assert custom_trainer.config.learning_rate == 0.002
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации CubeTrainer: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_cube_trainer_config_loading():
    """Тест 5: Загрузка конфигурации из dict"""
    print("\n🧪 Тест 5: Загрузка конфигурации CubeTrainer")
    
    try:
        from training.embedding_trainer import CubeTrainer
        
        # Конфигурация из словаря
        config_dict = {
            'mode': 'dialogue',
            'device': 'cpu',
            'lattice_size': [6, 6, 6],
            'embedding_dim': 768,
            'learning_rate': 0.0015,
            'epochs': 30,
            'target_similarity': 0.92
        }
        
        trainer = CubeTrainer(config=config_dict)
        
        print("[OK] CubeTrainer инициализирован из словаря")
        print(f"   Mode: {trainer.config.mode}")
        print(f"   Lattice size: {trainer.config.lattice_size}")
        print(f"   Learning rate: {trainer.config.learning_rate}")
        print(f"   Target similarity: {trainer.config.target_similarity}")
        
        # Проверка загруженных значений
        assert trainer.config.mode == 'dialogue'
        assert trainer.config.lattice_size == [6, 6, 6]
        assert trainer.config.learning_rate == 0.0015
        assert trainer.config.epochs == 30
        assert trainer.config.target_similarity == 0.92
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки конфигурации: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_cube_trainer_info():
    """Тест 6: Получение информации о CubeTrainer"""
    print("\n🧪 Тест 6: Информация о CubeTrainer")
    
    try:
        from training.embedding_trainer import CubeTrainer
        
        trainer = CubeTrainer(mode="dialogue", device="cpu")
        info = trainer.get_info()
        
        print("[OK] Информация о CubeTrainer получена:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Проверка обязательных полей
        required_fields = [
            'mode', 'device', 'lattice_size', 'embedding_dim',
            'current_epoch', 'optimizer', 'loss_function',
            'target_similarity', 'components_initialized'
        ]
        
        for field in required_fields:
            assert field in info, f"Поле {field} отсутствует в info"
        
        # Проверка значений
        assert info['mode'] == 'dialogue'
        assert info['device'] == 'cpu'
        assert info['current_epoch'] == 0
        assert info['components_initialized'] == False  # Компоненты еще не инициализированы
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка получения информации: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_cube_trainer_mode_switching():
    """Тест 7: Переключение режимов CubeTrainer"""
    print("\n🧪 Тест 7: Переключение режимов")
    
    try:
        from training.embedding_trainer import CubeTrainer
        
        trainer = CubeTrainer(mode="autoencoder")
        
        print(f"[OK] Начальный режим: {trainer.config.mode}")
        assert trainer.config.mode == "autoencoder"
        
        # Переключение на dialogue
        trainer.set_mode("dialogue")
        print(f"[OK] Режим изменен на: {trainer.config.mode}")
        assert trainer.config.mode == "dialogue"
        
        # Переключение на mixed
        trainer.set_mode("mixed")
        print(f"[OK] Режим изменен на: {trainer.config.mode}")
        assert trainer.config.mode == "mixed"
        
        # Проверка неверного режима
        try:
            trainer.set_mode("invalid_mode")
            print("[ERROR] Должна была быть ошибка для неверного режима")
            return False
        except ValueError:
            print("[OK] Корректно обработан неверный режим")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка переключения режимов: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_cube_trainer_dependencies():
    """Тест 8: Проверка зависимостей CubeTrainer"""
    print("\n🧪 Тест 8: Проверка зависимостей")
    
    try:
        from training.embedding_trainer import CubeTrainer
        
        # Создание trainer без инициализации компонентов
        trainer = CubeTrainer()
        
        # Проверка, что компоненты еще не инициализированы
        assert trainer.embedding_processor is None
        assert trainer.embedding_reshaper is None
        assert trainer.embedding_loader is None
        
        print("[OK] Компоненты корректно не инициализированы")
        
        # Попытка forward pass без инициализации (должна дать ошибку)
        try:
            test_input = torch.randn(1, 768)
            output = trainer.forward(test_input)
            print("[ERROR] Forward pass должен был выдать ошибку")
            return False
        except ValueError as e:
            print("[OK] Forward pass корректно выдал ошибку без инициализации")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка проверки зависимостей: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("[START] ТЕСТИРОВАНИЕ CUBETRAINER")
    print("   Phase 3.1 - Stage 1.1 - Basic CubeTrainer Tests")
    print("=" * 60)
    
    tests = [
        test_cube_trainer_import,
        test_training_config,
        test_embedding_metrics,
        test_cube_trainer_initialization,
        test_cube_trainer_config_loading,
        test_cube_trainer_info,
        test_cube_trainer_mode_switching,
        test_cube_trainer_dependencies
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Критическая ошибка в {test_func.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("[DATA] РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ CUBETRAINER")
    print(f"[OK] Пройдено: {passed}")
    print(f"[ERROR] Провалено: {failed}")
    print(f"[CHART] Успешность: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("[SUCCESS] Все тесты пройдены! CubeTrainer базовая функциональность работает!")
        print("[START] Готов к Stage 1.2 - AutoencoderDataset")
    elif passed >= 6:
        print("[TARGET] Большинство тестов пройдено. Базовая функциональность работает")
        print("[WARNING]  Нужно исправить отказавшие тесты")
    else:
        print("[WARNING]  Критические проблемы. Требуется доработка")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 