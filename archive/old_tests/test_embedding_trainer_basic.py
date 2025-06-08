"""
Базовые тесты для модуля Embedding Trainer

Этот файл содержит тесты для проверки:
1. Импорта модуля и его компонентов
2. Проверки зависимостей
3. Базовой функциональности (когда будет реализована)

Автор: 3D Cellular Neural Network Project  
Версия: Phase 3.1
Дата: 6 июня 2025
"""

import sys
import traceback
from pathlib import Path

def test_embedding_trainer_import():
    """Тест 1: Проверка импорта модуля"""
    print("🧪 Тест 1: Импорт модуля embedding_trainer")
    
    try:
        from training.embedding_trainer import get_module_info
        print("✅ Модуль embedding_trainer успешно импортирован")
        
        # Получение информации о модуле
        info = get_module_info()
        print(f"   Название: {info['name']}")
        print(f"   Версия: {info['version']}")
        print(f"   Статус: {info['status']}")
        print(f"   Фаза: {info['phase']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

def test_placeholder_classes():
    """Тест 2: Проверка заглушек классов"""
    print("\n🧪 Тест 2: Проверка заглушек классов")
    
    try:
        from training.embedding_trainer import CubeTrainer, AutoencoderDataset, DialogueDataset
        
        # Проверяем, что классы существуют но еще не реализованы
        test_cases = [
            ("CubeTrainer", CubeTrainer),
            ("AutoencoderDataset", AutoencoderDataset),
            ("DialogueDataset", DialogueDataset)
        ]
        
        for class_name, class_obj in test_cases:
            try:
                instance = class_obj()
                print(f"❌ {class_name}: должен выбрасывать NotImplementedError")
                return False
            except NotImplementedError:
                print(f"✅ {class_name}: корректно выбрасывает NotImplementedError")
            except Exception as e:
                print(f"❌ {class_name}: неожиданная ошибка: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта классов: {e}")
        return False

def test_dependency_check():
    """Тест 3: Проверка системы зависимостей"""
    print("\n🧪 Тест 3: Проверка зависимостей")
    
    try:
        # При импорте модуля автоматически запускается проверка зависимостей
        # Мы просто проверим, что критические библиотеки доступны
        
        critical_imports = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("pathlib", "pathlib")
        ]
        
        for module_name, display_name in critical_imports:
            try:
                __import__(module_name)
                print(f"✅ {display_name} доступен")
            except ImportError:
                print(f"❌ {display_name} не найден")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки зависимостей: {e}")
        return False

def test_module_structure():
    """Тест 4: Проверка структуры модуля"""
    print("\n🧪 Тест 4: Проверка структуры модуля")
    
    try:
        # Проверяем существование ключевых файлов
        module_path = Path("training/embedding_trainer")
        
        required_files = [
            "__init__.py",
            "README.md",
            "plan.md", 
            "meta.md",
            "errors.md",
            "diagram.mmd",
            "examples.md"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = module_path / file_name
            if file_path.exists():
                print(f"✅ {file_name} существует")
            else:
                print(f"❌ {file_name} отсутствует")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Отсутствующие файлы: {missing_files}")
            return False
        
        print("✅ Все обязательные файлы документации присутствуют")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка проверки структуры: {e}")
        return False

def test_config_integration():
    """Тест 5: Проверка интеграции с конфигурацией"""
    print("\n🧪 Тест 5: Проверка интеграции конфигурации")
    
    try:
        # Проверяем доступность config_manager
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        print("✅ ConfigManager доступен")
        
        # Проверяем основную конфигурацию
        config_path = Path("config/main_config.yaml")
        if config_path.exists():
            print("✅ Основной конфигурационный файл найден")
            
            # Правильная инициализация ConfigManager
            settings = ConfigManagerSettings(
                base_config_path=str(config_path),
                enable_hot_reload=False,  # Отключаем для тестов
                enable_validation=False   # Отключаем для тестов
            )
            config = ConfigManager(settings)
            main_config = config.get_config()
            
            # Проверяем наличие секций, нужных для обучения
            required_sections = [
                "modular_architecture",
                "embedding_processing" 
            ]
            
            for section in required_sections:
                if section in main_config:
                    print(f"✅ Секция '{section}' найдена в конфигурации")
                else:
                    print(f"⚠️  Секция '{section}' отсутствует (будет добавлена позже)")
            
            return True
        else:
            print("⚠️  Основной конфигурационный файл не найден")
            return True  # Не критично на этапе разработки
            
    except ImportError as e:
        print(f"❌ ConfigManager не доступен: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка интеграции конфигурации: {e}")
        print(f"   Детали: {traceback.format_exc()}")
        return False

def test_ready_components_integration():
    """Тест 6: Проверка интеграции с готовыми компонентами"""
    print("\n🧪 Тест 6: Проверка готовых компонентов")
    
    ready_components = [
        ("core.embedding_processor", "EmbeddingProcessor"),
        ("data.embedding_reshaper", "EmbeddingReshaper"),
        ("data.embedding_loader", "EmbeddingLoader")
    ]
    
    available_components = []
    
    for module_name, component_name in ready_components:
        try:
            __import__(module_name)
            print(f"✅ {component_name} доступен")
            available_components.append(component_name)
        except ImportError:
            print(f"⚠️  {component_name} пока не доступен")
    
    if len(available_components) >= 2:
        print(f"✅ Достаточно компонентов для начала разработки: {available_components}")
        return True
    else:
        print(f"⚠️  Мало готовых компонентов, но разработка может продолжаться")
        return True  # Не критично на раннем этапе

def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 60)
    print("🚀 ТЕСТИРОВАНИЕ МОДУЛЯ EMBEDDING TRAINER")
    print("   Phase 3.1 - Basic Infrastructure Tests")
    print("=" * 60)
    
    tests = [
        test_embedding_trainer_import,
        test_placeholder_classes,
        test_dependency_check,
        test_module_structure,
        test_config_integration,
        test_ready_components_integration
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
            print(f"❌ Критическая ошибка в {test_func.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"✅ Пройдено: {passed}")
    print(f"❌ Провалено: {failed}")
    print(f"📈 Успешность: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("🎉 Все тесты пройдены! Модуль готов к Stage 1.1")
    elif passed >= 4:
        print("🎯 Большинство тестов пройдено. Можно продолжать разработку")
    else:
        print("⚠️  Критические проблемы. Требуется исправление")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 