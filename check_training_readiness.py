#!/usr/bin/env python3
"""
Скрипт проверки готовности системы к реальному обучению
Проверяет все компоненты перед запуском start_real_training.py
"""

import torch
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """Проверяем доступность GPU"""
    result = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_names": [],
        "memory_info": {}
    }
    
    if torch.cuda.is_available():
        result["gpu_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            result["gpu_names"].append(gpu_name)
            
            # Проверяем память
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            result["memory_info"][f"gpu_{i}"] = {
                "name": gpu_name,
                "total_memory_gb": round(memory_total, 2),
                "allocated_gb": round(memory_allocated, 2),
                "reserved_gb": round(memory_reserved, 2),
                "free_gb": round(memory_total - memory_reserved, 2)
            }
    
    return result


def check_dataset_availability() -> Dict[str, Any]:
    """Проверяем доступность датасетов"""
    result = {
        "dialogue_cache": {"available": False, "count": 0},
        "prepared_embeddings": {"available": False, "count": 0},
        "cache_embeddings": {"available": False, "count": 0},
        "snli_generator": {"available": False}
    }
    
    # Dialogue cache
    dialogue_dir = Path("cache/dialogue_dataset")
    if dialogue_dir.exists():
        dialogue_files = list(dialogue_dir.glob("*.pt"))
        result["dialogue_cache"] = {
            "available": len(dialogue_files) > 0,
            "count": len(dialogue_files)
        }
    
    # Prepared embeddings
    embeddings_dir = Path("data/embeddings")
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob("*.pt"))
        result["prepared_embeddings"] = {
            "available": len(embedding_files) > 0,
            "count": len(embedding_files)
        }
    
    # Cache embeddings
    cache_files = list(Path("cache").glob("llm_*.pt"))
    result["cache_embeddings"] = {
        "available": len(cache_files) > 0,
        "count": len(cache_files)
    }
    
    # SNLI generator
    snli_script = Path("generate_snli_embedding_dataset.py")
    result["snli_generator"]["available"] = snli_script.exists()
    
    return result


def check_dependencies() -> Dict[str, Any]:
    """Проверяем зависимости Python"""
    required_modules = [
        "torch",
        "numpy", 
        "datasets",
        "transformers",
        "pathlib",
        "json",
        "logging"
    ]
    
    result = {"missing_modules": [], "available_modules": []}
    
    for module in required_modules:
        try:
            __import__(module)
            result["available_modules"].append(module)
        except ImportError:
            result["missing_modules"].append(module)
    
    return result


def check_new_rebuild_components() -> Dict[str, Any]:
    """Проверяем компоненты new_rebuild"""
    result = {"available_components": [], "missing_components": []}
    
    components_to_check = [
        "new_rebuild.config",
        "new_rebuild.core.training",
        "new_rebuild.core.cells",
        "new_rebuild.core.lattice",
        "new_rebuild.core.moe",
        "new_rebuild.utils.device_manager",
        "new_rebuild.utils.logging"
    ]
    
    for component in components_to_check:
        try:
            __import__(component)
            result["available_components"].append(component)
        except ImportError as e:
            result["missing_components"].append(f"{component}: {str(e)}")
    
    return result


def test_unified_dataset_loader() -> Dict[str, Any]:
    """Тестируем unified dataset loader"""
    result = {"success": False, "error": None, "sample_count": 0}
    
    try:
        from unified_dataset_loader import DatasetConfig, UnifiedEmbeddingDataset
        
        # Создаем тестовую конфигурацию с ограниченными данными
        config = DatasetConfig(
            use_dialogue_cache=True,
            use_prepared_embeddings=True,
            use_cache_embeddings=True,
            use_snli_generator=False,
            max_samples_per_source=10  # Только 10 образцов для теста
        )
        
        dataset = UnifiedEmbeddingDataset(config)
        result["sample_count"] = len(dataset)
        result["success"] = len(dataset) > 0
        
        # Проверяем что можем загрузить один образец
        if len(dataset) > 0:
            sample = dataset[0]
            embedding = sample['embedding']
            if embedding.shape[0] == 768:  # Проверяем размерность
                result["embedding_dim_ok"] = True
            else:
                result["embedding_dim_ok"] = False
                result["actual_dim"] = embedding.shape[0]
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_embedding_trainer_creation() -> Dict[str, Any]:
    """Тестируем создание EmbeddingTrainer"""
    result = {"success": False, "error": None}
    
    try:
        from new_rebuild.config import SimpleProjectConfig
        from new_rebuild.core.training import EmbeddingTrainer
        
        # Создаем минимальную конфигурацию
        config = SimpleProjectConfig()
        config.lattice.dimensions = (4, 4, 4)  # Маленький куб для теста
        config.training_embedding.test_mode = True
        
        trainer = EmbeddingTrainer(config)
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    """Главная функция проверки готовности"""
    print("🔍 CHECKING TRAINING READINESS")
    print("=" * 50)
    
    all_checks = {}
    
    # 1. GPU проверка
    print("\n🖥️ Checking GPU availability...")
    gpu_check = check_gpu_availability()
    all_checks["gpu"] = gpu_check
    
    if gpu_check["cuda_available"]:
        print(f"✅ CUDA available: {gpu_check['gpu_count']} GPUs")
        for gpu_name in gpu_check["gpu_names"]:
            print(f"   - {gpu_name}")
    else:
        print("❌ CUDA not available - will use CPU")
    
    # 2. Датасеты
    print("\n📂 Checking dataset availability...")
    dataset_check = check_dataset_availability()
    all_checks["datasets"] = dataset_check
    
    total_samples = 0
    for dataset_type, info in dataset_check.items():
        if dataset_type != "snli_generator":
            status = "✅" if info["available"] else "❌"
            count = info.get("count", 0)
            total_samples += count
            print(f"   {status} {dataset_type}: {count} files")
        else:
            status = "✅" if info["available"] else "❌"
            print(f"   {status} SNLI generator available")
    
    print(f"📊 Total potential samples: ~{total_samples}")
    
    # 3. Зависимости
    print("\n📦 Checking Python dependencies...")
    deps_check = check_dependencies()
    all_checks["dependencies"] = deps_check
    
    if deps_check["missing_modules"]:
        print("❌ Missing modules:")
        for module in deps_check["missing_modules"]:
            print(f"   - {module}")
    else:
        print("✅ All required modules available")
    
    # 4. New rebuild компоненты
    print("\n🧩 Checking new_rebuild components...")
    components_check = check_new_rebuild_components()
    all_checks["components"] = components_check
    
    if components_check["missing_components"]:
        print("❌ Missing components:")
        for component in components_check["missing_components"]:
            print(f"   - {component}")
    else:
        print("✅ All new_rebuild components available")
    
    # 5. Тестируем dataset loader
    print("\n🔄 Testing unified dataset loader...")
    loader_test = test_unified_dataset_loader()
    all_checks["dataset_loader"] = loader_test
    
    if loader_test["success"]:
        print(f"✅ Dataset loader works: {loader_test['sample_count']} samples loaded")
        if loader_test.get("embedding_dim_ok"):
            print("✅ Embedding dimensions correct (768D)")
        else:
            print(f"❌ Wrong embedding dimension: {loader_test.get('actual_dim')}")
    else:
        print(f"❌ Dataset loader failed: {loader_test.get('error')}")
    
    # 6. Тестируем EmbeddingTrainer
    print("\n🧠 Testing EmbeddingTrainer creation...")
    trainer_test = test_embedding_trainer_creation()
    all_checks["trainer"] = trainer_test
    
    if trainer_test["success"]:
        print("✅ EmbeddingTrainer creates successfully")
    else:
        print(f"❌ EmbeddingTrainer failed: {trainer_test.get('error')}")
    
    # Сохраняем результаты
    results_file = Path("training_readiness_check.json")
    with open(results_file, 'w') as f:
        json.dump(all_checks, f, indent=2)
    
    # Финальная оценка
    print(f"\n📊 READINESS SUMMARY")
    print("=" * 30)
    
    critical_checks = [
        (gpu_check["cuda_available"], "GPU (CUDA)"),
        (dataset_check["dialogue_cache"]["available"] or 
         dataset_check["prepared_embeddings"]["available"] or 
         dataset_check["cache_embeddings"]["available"], "Datasets"),
        (len(deps_check["missing_modules"]) == 0, "Dependencies"),
        (len(components_check["missing_components"]) == 0, "Components"),
        (loader_test["success"], "Dataset Loader"),
        (trainer_test["success"], "EmbeddingTrainer")
    ]
    
    all_ready = True
    for is_ready, check_name in critical_checks:
        status = "✅" if is_ready else "❌"
        print(f"{status} {check_name}")
        if not is_ready:
            all_ready = False
    
    print(f"\n{'🚀 SYSTEM READY FOR TRAINING!' if all_ready else '⚠️ FIX ISSUES BEFORE TRAINING'}")
    
    if all_ready:
        print("\nNext steps:")
        print("1. Run: python start_real_training.py")
        print("2. Monitor: logs/real_training/")
        print("3. Check results: experiments/")
    else:
        print("\nFix the issues marked with ❌ before proceeding")
    
    print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()