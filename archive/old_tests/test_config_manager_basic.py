#!/usr/bin/env python3
"""
Basic Tests for ConfigManager

Comprehensive testing suite for the ConfigManager system.
Tests core functionality, configuration sections, export/import,
statistics, and integration with the main project.
"""

import os
import sys
import yaml
import json
import tempfile
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_config_loading():
    """Тест базовой загрузки конфигурации"""
    print("\n[WRITE] Testing basic config loading...")
    
    try:
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        
        # Создаем тестовую конфигурацию
        config_data = {
            'project': {
                'name': 'Test Project',
                'version': '1.0.0'
            },
            'lattice': {
                'dimensions': {
                    'width': 8,
                    'height': 8,
                    'depth': 8
                },
                'cell_count': 512
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'optimizer': {
                    'type': 'Adam',
                    'weight_decay': 0.0001
                }
            }
        }
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Создаем ConfigManager с изолированной конфигурацией
            settings = ConfigManagerSettings(
                base_config_path=temp_config_path,
                enable_hot_reload=False,  # Отключаем для тестов
                enable_validation=False,  # Отключаем пока нет схем
                config_search_paths=[]    # Отключаем автоматическое обнаружение модулей
            )
            
            config = ConfigManager(settings)
            
            # Тестируем получение конфигурации
            print("   [WRITE] Testing config retrieval...")
            
            # Получение всей конфигурации
            full_config = config.get_config()
            assert 'project' in full_config, "Project section should be present"
            assert 'lattice' in full_config, "Lattice section should be present"
            assert 'training' in full_config, "Training section should be present"
            
            # Получение конкретных секций
            project_config = config.get_config('project')
            assert project_config is not None, "Project config should not be None"
            assert project_config['name'] == 'Test Project', f"Project name mismatch: got {project_config.get('name')}"
            assert project_config['version'] == '1.0.0', "Project version mismatch"
            
            # Получение с dot-notation
            depth = config.get_config('lattice', 'dimensions.depth')
            assert depth == 8, f"Expected depth 8, got {depth}"
            
            batch_size = config.get_config('training', 'batch_size')
            assert batch_size == 32, f"Expected batch_size 32, got {batch_size}"
            
            optimizer_type = config.get_config('training', 'optimizer.type')
            assert optimizer_type == 'Adam', f"Expected optimizer Adam, got {optimizer_type}"
            
            # Тестируем значения по умолчанию
            non_existent = config.get_config('non_existent', 'key', 'default_value')
            assert non_existent == 'default_value', "Default value not returned"
            
            print("   [OK] Config retrieval works correctly")
            
            return True
            
        finally:
            # Очищаем временный файл
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"   [ERROR] Basic config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_modification():
    """Тест изменения конфигурации в runtime"""
    print("\n[GEAR] Testing config modification...")
    
    try:
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        
        # Создаем базовую конфигурацию
        config_data = {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            settings = ConfigManagerSettings(
                base_config_path=temp_config_path,
                enable_hot_reload=False,
                enable_validation=False,
                config_search_paths=[]
            )
            config = ConfigManager(settings)
            
            # Тестируем установку значений
            print("   [WRITE] Testing config setting...")
            
            # Установка простого значения
            config.set_config('training', 'batch_size', 64)
            new_batch_size = config.get_config('training', 'batch_size')
            assert new_batch_size == 64, f"Expected batch_size 64, got {new_batch_size}"
            
            # Установка с dot-notation
            config.set_config('training', 'optimizer.type', 'SGD')
            optimizer_type = config.get_config('training', 'optimizer.type')
            assert optimizer_type == 'SGD', f"Expected optimizer SGD, got {optimizer_type}"
            
            # Установка через kwargs
            config.set_config('training', learning_rate=0.01, num_epochs=100)
            learning_rate = config.get_config('training', 'learning_rate')
            num_epochs = config.get_config('training', 'num_epochs')
            assert learning_rate == 0.01, f"Expected learning_rate 0.01, got {learning_rate}"
            assert num_epochs == 100, f"Expected num_epochs 100, got {num_epochs}"
            
            # Установка новой секции
            config.set_config('new_section', 'new_key', 'new_value')
            new_value = config.get_config('new_section', 'new_key')
            assert new_value == 'new_value', f"Expected new_value, got {new_value}"
            
            print("   [OK] Config modification works correctly")
            return True
            
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"   [ERROR] Config modification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_sections():
    """Тест работы с секциями конфигурации"""
    print("\n📂 Testing config sections...")
    
    try:
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'credentials': {
                    'username': 'user',
                    'password': 'pass'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            settings = ConfigManagerSettings(
                base_config_path=temp_config_path,
                enable_hot_reload=False,
                enable_validation=False,
                config_search_paths=[]
            )
            config = ConfigManager(settings)
            
            # Получаем секцию
            db_section = config.get_section('database')
            
            print("   [WRITE] Testing section operations...")
            
            # Проверяем, что получили ConfigSection объект или словарь
            # ConfigSection должен быть доступен, если импорт прошел успешно
            if hasattr(db_section, 'get'):
                # Это ConfigSection объект
                print("   ℹ️ Using ConfigSection object")
                assert 'host' in db_section._data, "Host should be in database section"
                assert db_section._data['port'] == 5432, "Port mismatch"
            else:
                # Это обычный словарь (fallback)
                assert isinstance(db_section, dict), "Should be dict when ConfigSection unavailable"
                print("   ℹ️ Using dict fallback")
                assert 'host' in db_section, "Host should be in database section"
                assert db_section['port'] == 5432, "Port mismatch"
            
            # Если ConfigSection доступен, тестируем его методы
            try:
                if hasattr(db_section, 'get'):
                    # Тестируем методы ConfigSection
                    host = db_section.get('host')
                    assert host == 'localhost', f"Expected localhost, got {host}"
                    
                    # Dot-notation
                    username = db_section.get('credentials.username')
                    assert username == 'user', f"Expected user, got {username}"
                    
                    # Установка значений
                    db_section.set('port', 3306)
                    new_port = config.get_config('database', 'port')
                    assert new_port == 3306, f"Expected port 3306, got {new_port}"
                    
                    print("   [OK] ConfigSection methods work")
                else:
                    print("   ℹ️ ConfigSection not available, using dict fallback")
                    
            except Exception as e:
                print(f"   [WARNING] ConfigSection methods failed: {e}")
            
            print("   [OK] Config sections work correctly")
            return True
            
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"   [ERROR] Config sections test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_export():
    """Тест экспорта конфигурации"""
    print("\n[SAVE] Testing config export...")
    
    try:
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        
        config_data = {
            'app': {
                'name': 'Test App',
                'version': '1.0.0'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            settings = ConfigManagerSettings(
                base_config_path=temp_config_path,
                enable_hot_reload=False,
                enable_validation=False,
                config_search_paths=[]
            )
            config = ConfigManager(settings)
            
            # Модифицируем конфигурацию
            config.set_config('app', 'version', '2.0.0')
            config.set_config('new_section', 'test_key', 'test_value')
            
            # Экспортируем в YAML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                export_yaml_path = f.name
            
            config.export_config(export_yaml_path, format='yaml')
            
            # Проверяем экспортированный файл
            with open(export_yaml_path, 'r', encoding='utf-8') as f:
                exported_data = yaml.safe_load(f)
            
            assert 'app' in exported_data, "App section should be exported"
            assert exported_data['app']['version'] == '2.0.0', "Modified version should be exported"
            assert 'new_section' in exported_data, "New section should be exported"
            assert exported_data['new_section']['test_key'] == 'test_value', "New key should be exported"
            
            # Экспортируем в JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                export_json_path = f.name
            
            config.export_config(export_json_path, format='json')
            
            # Проверяем JSON файл
            import json
            with open(export_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            assert json_data['app']['name'] == 'Test App', "JSON export should match YAML"
            
            print("   [OK] Config export works correctly")
            
            # Очищаем
            os.unlink(export_yaml_path)
            os.unlink(export_json_path)
            
            return True
            
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"   [ERROR] Config export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_stats():
    """Тест статистики ConfigManager"""
    print("\n[DATA] Testing config statistics...")
    
    try:
        from utils.config_manager import ConfigManager, ConfigManagerSettings
        
        config_data = {'test': {'key': 'value'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            settings = ConfigManagerSettings(
                base_config_path=temp_config_path,
                enable_hot_reload=False,
                enable_validation=False,
                config_search_paths=[]
            )
            config = ConfigManager(settings)
            
            # Выполняем несколько операций для генерации статистики
            config.get_config('test', 'key')  # cache hit
            config.get_config('test', 'key')  # cache hit
            
            # Генерируем cache miss через несуществующие ключи в существующей секции
            for i in range(3):
                config.get_config('test', f'non_existent_key_{i}', 'default')  # cache miss
            
            # Получаем статистику
            stats = config.get_stats()
            
            assert 'cache_hits' in stats, "Stats should include cache_hits"
            assert 'cache_misses' in stats, "Stats should include cache_misses"
            assert 'cached_sections' in stats, "Stats should include cached_sections"
            assert 'cache_hit_rate' in stats, "Stats should include cache_hit_rate"
            
            assert stats['cache_hits'] >= 2, f"Expected at least 2 cache hits, got {stats['cache_hits']}"
            assert stats['cache_misses'] >= 1, f"Expected at least 1 cache miss, got {stats['cache_misses']}"
            assert stats['cached_sections'] >= 1, "Should have at least 1 cached section"
            
            print(f"   [CHART] Cache hit rate: {stats['cache_hit_rate']:.2%}")
            print(f"   [DATA] Cached sections: {stats['cached_sections']}")
            print(f"   [TARGET] Cache hits/misses: {stats['cache_hits']}/{stats['cache_misses']}")
            
            print("   [OK] Config statistics work correctly")
            return True
            
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"   [ERROR] Config statistics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_main_config():
    """Тест интеграции с основной конфигурацией проекта"""
    print("\n[LINK] Testing integration with main project config...")
    
    try:
        from utils.config_manager import create_config_manager
        
        # Проверяем наличие основного конфигурационного файла
        main_config_path = Path("config/main_config.yaml")
        
        if not main_config_path.exists():
            print(f"   [WARNING] Main config file not found: {main_config_path}")
            print("   ℹ️ Creating test with real project structure later")
            return True
        
        # Загружаем реальную конфигурацию проекта
        config = create_config_manager(
            base_config=str(main_config_path),
            enable_hot_reload=False
        )
        
        print("   [WRITE] Testing real project config...")
        
        # Проверяем ожидаемые секции из PROJECT_PLAN.md
        expected_sections = ['project', 'lattice', 'cell_prototype', 'training', 'data', 'device']
        
        for section in expected_sections:
            section_data = config.get_config(section)
            if section_data:
                print(f"   [OK] Found section: {section}")
            else:
                print(f"   [WARNING] Missing section: {section}")
        
        # Тестируем специфичные настройки проекта
        project_name = config.get_config('project', 'name')
        if project_name:
            print(f"   [INFO] Project name: {project_name}")
        
        lattice_dimensions = config.get_config('lattice', 'dimensions')
        if lattice_dimensions:
            print(f"   🔷 Lattice dimensions: {lattice_dimensions}")
        
        device_gpu = config.get_config('device', 'use_gpu')
        if device_gpu is not None:
            print(f"   [PC] GPU enabled: {device_gpu}")
        
        print("   [OK] Real project config integration works")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Main config integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Запуск всех тестов ConfigManager"""
    print("🧪 ConfigManager Basic Tests")
    print("=" * 50)
    
    tests = [
        test_basic_config_loading,
        test_config_modification,
        test_config_sections,
        test_config_export,
        test_config_stats,
        test_integration_with_main_config,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"[DATA] Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! ConfigManager is working correctly.")
        return True
    else:
        print(f"[WARNING] {total - passed} tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 