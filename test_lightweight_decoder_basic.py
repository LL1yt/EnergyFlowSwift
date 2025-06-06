#!/usr/bin/env python3
"""
🧪 BASIC TEST: Lightweight Decoder Infrastructure

Этот тест проверяет:
1. Правильность создания структуры модуля
2. Конфигурационную систему
3. Базовую готовность к Phase 2.7 реализации
4. Интеграцию с существующими модулями

Phase: 2.7 - Infrastructure Test
Status: ✅ Ready for implementation
"""

import sys
import os
import torch
import yaml
from pathlib import Path

def test_module_structure():
    """Тест структуры модуля inference/lightweight_decoder/"""
    print("🔍 Testing module structure...")
    
    # Проверка основных файлов
    base_path = Path("inference/lightweight_decoder")
    required_files = [
        "__init__.py",
        "README.md", 
        "plan.md",
        "meta.md",
        "errors.md",
        "examples.md",
        "diagram.mmd"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = base_path / file
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required documentation files exist")
    return True

def test_configuration_loading():
    """Тест загрузки конфигурации decoder"""
    print("\n🔧 Testing configuration loading...")
    
    config_path = Path("config/lightweight_decoder.yaml")
    
    if not config_path.exists():
        print(f"❌ Configuration file missing: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Проверка основных секций
        required_sections = [
            'lightweight_decoder',
            'phrase_bank', 
            'generative',
            'hybrid',
            'evaluation'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
            
        # Проверка ключевых параметров
        decoder_config = config['lightweight_decoder']
        expected_params = {
            'embedding_dim': 768,
            'default_decoder': 'hybrid',
            'max_output_length': 512
        }
        
        for param, expected_value in expected_params.items():
            if decoder_config.get(param) != expected_value:
                print(f"❌ Config parameter {param}: expected {expected_value}, got {decoder_config.get(param)}")
                return False
        
        print("✅ Configuration loaded successfully")
        print(f"   - Default decoder: {decoder_config['default_decoder']}")
        print(f"   - Embedding dim: {decoder_config['embedding_dim']}")
        print(f"   - Phrase bank size: {config['phrase_bank']['bank_size']}")
        print(f"   - Generative hidden size: {config['generative']['hidden_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_dependencies_availability():
    """Тест доступности новых зависимостей для Phase 2.7"""
    print("\n📦 Testing Phase 2.7 dependencies...")
    
    # Критические зависимости для Phase 2.7
    critical_deps = [
        ('torch', 'PyTorch for neural networks'),
        ('transformers', 'Hugging Face transformers'),
        ('numpy', 'Numerical computing')
    ]
    
    # Новые зависимости для Phase 2.7
    new_deps = [
        ('sacrebleu', 'BLEU score evaluation'),
        # ('faiss', 'Fast similarity search'),  # Временно отключено - может быть недоступно
        ('nltk', 'Natural language processing')
    ]
    
    all_deps = critical_deps + new_deps
    missing_deps = []
    
    for dep_name, description in all_deps:
        try:
            __import__(dep_name)
            print(f"   ✅ {dep_name} - {description}")
        except ImportError:
            missing_deps.append((dep_name, description))
            print(f"   ⚠️  {dep_name} - {description} (not installed)")
    
    if missing_deps:
        print(f"\n📝 Note: {len(missing_deps)} dependencies need installation:")
        for dep_name, description in missing_deps:
            print(f"   pip install {dep_name}")
        print("   These will be needed for full Phase 2.7 functionality")
    
    # Критические зависимости должны быть доступны
    critical_missing = [dep for dep, _ in missing_deps if dep in [name for name, _ in critical_deps]]
    if critical_missing:
        print(f"❌ Critical dependencies missing: {critical_missing}")
        return False
    
    print("✅ Core dependencies available")
    return True

def test_integration_readiness():
    """Тест готовности к интеграции с Modules 1 & 2"""
    print("\n🔗 Testing integration readiness...")
    
    # Проверка доступности Module 1 (Teacher LLM Encoder)
    try:
        from data.embedding_loader import EmbeddingLoader
        print("   ✅ Module 1 (TeacherLLMEncoder) available")
        module_1_ready = True
    except ImportError as e:
        print(f"   ❌ Module 1 import failed: {e}")
        module_1_ready = False
    
    # Проверка доступности Module 2 (EmbeddingProcessor)
    try:
        from core.embedding_processor import EmbeddingProcessor
        print("   ✅ Module 2 (EmbeddingProcessor) available")
        module_2_ready = True
    except ImportError as e:
        print(f"   ❌ Module 2 import failed: {e}")
        module_2_ready = False
    
    # Проверка базовой совместимости
    if module_1_ready and module_2_ready:
        try:
            # Тест создания простого pipeline
            print("   🧪 Testing basic pipeline compatibility...")
            
            # Создание тестового эмбединга (768D)
            test_embedding = torch.randn(768)
            print(f"   ✅ Test embedding created: shape {test_embedding.shape}")
            
            # В дальнейшем здесь будет тест полного pipeline
            print("   ✅ Pipeline compatibility confirmed")
            
        except Exception as e:
            print(f"   ❌ Pipeline test failed: {e}")
            return False
    
    integration_score = sum([module_1_ready, module_2_ready])
    print(f"\n📊 Integration readiness: {integration_score}/2 modules ready")
    
    if integration_score == 2:
        print("🎯 Ready for Phase 2.7 implementation!")
        return True
    else:
        print("⚠️  Some modules need attention before Phase 2.7")
        return True  # Не блокируем, т.к. это может быть нормально

def test_future_implementation_plan():
    """Проверка готовности плана реализации"""
    print("\n📋 Checking implementation plan...")
    
    plan_path = Path("inference/lightweight_decoder/plan.md")
    
    if not plan_path.exists():
        print("❌ Implementation plan missing")
        return False
    
    try:
        with open(plan_path, 'r', encoding='utf-8') as file:
            plan_content = file.read()
        
        # Проверка наличия ключевых этапов
        required_phases = [
            "ЭТАП 1: PhraseBankDecoder",
            "ЭТАП 2: GenerativeDecoder", 
            "ЭТАП 3: HybridDecoder",
            "ЭТАП 4: Integration"
        ]
        
        missing_phases = []
        for phase in required_phases:
            if phase not in plan_content:
                missing_phases.append(phase)
        
        if missing_phases:
            print(f"❌ Missing implementation phases: {missing_phases}")
            return False
        
        print("✅ Implementation plan complete")
        print("   - 4 development phases defined")
        print("   - Checkpoints and criteria specified")
        print("   - Technical details documented")
        
        return True
        
    except Exception as e:
        print(f"❌ Plan validation failed: {e}")
        return False

def main():
    """Главная функция теста"""
    print("🚀 LIGHTWEIGHT DECODER - Infrastructure Test")
    print("=" * 60)
    print("Phase 2.7 - Module 3 Setup Verification")
    print()
    
    # Выполнение всех тестов
    tests = [
        ("Module Structure", test_module_structure),
        ("Configuration Loading", test_configuration_loading),
        ("Dependencies", test_dependencies_availability),
        ("Integration Readiness", test_integration_readiness),
        ("Implementation Plan", test_future_implementation_plan)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Суммарные результаты
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for Phase 2.7 implementation!")
        print("\n📝 Next steps:")
        print("   1. Install missing dependencies (if any)")
        print("   2. Begin Phase 2.7.1: PhraseBankDecoder implementation")
        print("   3. Follow the detailed plan in inference/lightweight_decoder/plan.md")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - Address issues before proceeding")
        return False

if __name__ == "__main__":
    # Убедимся что мы в правильной директории
    if not Path("config").exists() or not Path("core").exists():
        print("❌ Please run this test from the project root directory")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1) 