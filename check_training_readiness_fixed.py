#!/usr/bin/env python3
"""
Исправленный скрипт проверки готовности системы к реальному обучению
================================================================

Проверяет обновленные компоненты:
- Центральную конфигурацию с настройками для реального обучения
- Новый unified dataset loader в new_rebuild структуре
- Динамический neighbor_count
"""

import torch
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Any

from new_rebuild.config import SimpleProjectConfig

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
        "cache_embeddings": {"available": False, "count": 0}
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
    
    # Cache embeddings (пропускаем файлы с неправильной размерностью)
    cache_files = list(Path("cache").glob("llm_*.pt"))
    valid_cache_files = 0
    
    for file in cache_files[:5]:  # Проверяем только первые 5 файлов
        try:
            data = torch.load(file, map_location='cpu')
            if isinstance(data, torch.Tensor) and data.shape[-1] == 768:
                valid_cache_files += 1
        except:
            continue
    
    result["cache_embeddings"] = {
        "available": valid_cache_files > 0,
        "count": valid_cache_files,
        "total_files": len(cache_files)
    }
    
    return result


def check_central_config() -> Dict[str, Any]:
    """Проверяем центральную конфигурацию"""
    result = {"success": False, "error": None, "config_values": {}}
    
    try:
        from new_rebuild.config import SimpleProjectConfig
        
        config = SimpleProjectConfig()
        
        # Проверяем ключевые значения для реального обучения
        result["config_values"] = {
            "lattice_dimensions": config.lattice.dimensions,
            "test_mode": config.training_embedding.test_mode,
            "num_epochs": config.training_embedding.num_epochs,
            "target_embedding_dim": config.training_embedding.target_embedding_dim,
            "batch_size": config.training_embedding.embedding_batch_size,
            "neighbor_count": config.model.neighbor_count,
            "state_size": config.model.state_size,
            "hidden_dim": config.model.hidden_dim
        }
        
        # Проверяем что конфиг настроен для реального обучения
        checks = []
        if config.training_embedding.test_mode == False:
            checks.append("✅ Real training mode enabled")
        else:
            checks.append("⚠️ Test mode still enabled")
            
        if config.model.neighbor_count == -1:
            checks.append("✅ Dynamic neighbor count enabled")
        else:
            checks.append(f"ℹ️ Static neighbor count: {config.model.neighbor_count}")
            
        if config.lattice.dimensions == (8, 8, 8):
            checks.append("✅ Correct lattice size for first training")
        else:
            checks.append(f"ℹ️ Lattice size: {config.lattice.dimensions}")
        
        result["config_checks"] = checks
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_new_dataset_loader() -> Dict[str, Any]:
    """Тестируем новый unified dataset loader"""
    result = {"success": False, "error": None, "sample_count": 0}
    
    try:
        from new_rebuild.config import SimpleProjectConfig
        from new_rebuild.core.training.utils import create_training_dataloader
        
        config = SimpleProjectConfig()
        
        # Создаем DataLoader с ограниченными данными для теста
        dataloader, stats = create_training_dataloader(
            config=config,
            max_samples_per_source=10,  # Только 10 образцов для теста
            shuffle=True
        )
        
        result["sample_count"] = stats.total_samples
        result["embedding_dim"] = stats.embedding_dim
        result["source_distribution"] = stats.source_distribution
        result["success"] = stats.total_samples > 0
        
        # Проверяем что можем загрузить один батч
        if stats.total_samples > 0:
            for batch in dataloader:
                embeddings = batch['embedding']
                if embeddings.shape[1] == config.embedding.input_dim:
                    result["embedding_dim_ok"] = True
                    result["batch_shape"] = list(embeddings.shape)
                else:
                    result["embedding_dim_ok"] = False
                    result["expected_dim"] = config.embedding.input_dim
                    result["actual_dim"] = embeddings.shape[1]
                break
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_embedding_trainer_creation() -> Dict[str, Any]:
    """Тестируем создание EmbeddingTrainer с новой конфигурацией"""
    result = {"success": False, "error": None}
    
    try:
        from new_rebuild.config import SimpleProjectConfig
        from new_rebuild.core.training import EmbeddingTrainer
        
        # Используем центральную конфигурацию
        config = SimpleProjectConfig()
        
        # Переключаем в тестовый режим для быстрой проверки
        config.training_embedding.test_mode = True
        # config.lattice.dimensions = (4, 4, 4)  # НЕ переопределяем размер
        # Используем размер из центрального конфига
        
        trainer = EmbeddingTrainer(config)
        result["success"] = True
        result["total_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_dynamic_neighbors_quick(config: SimpleProjectConfig) -> Dict[str, Any]:
    """Быстрая проверка динамических соседей"""
    result = {"success": False, "error": None}
    
    try:
        # Проверяем основные настройки
        neighbor_count = config.model.neighbor_count
        dynamic_enabled = getattr(config.neighbors, 'dynamic_count', False) if hasattr(config, 'neighbors') and config.neighbors else False
        
        # Вычисляем характеристики решетки
        dimensions = config.lattice.dimensions
        max_radius = config.lattice.max_radius
        local_threshold = config.lattice.local_distance_threshold
        functional_threshold = config.lattice.functional_distance_threshold
        
        result.update({
            "success": True,
            "neighbor_count_setting": neighbor_count,
            "dynamic_enabled": dynamic_enabled,
            "lattice_dimensions": dimensions,
            "max_radius": round(max_radius, 2),
            "thresholds": {
                "local": round(local_threshold, 2),
                "functional": round(functional_threshold, 2)
            },
            "is_dynamic": neighbor_count == -1,
            "legacy_detected": neighbor_count in [6, 26]
        })
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_dependencies() -> Dict[str, Any]:
    """Проверяем зависимости Python"""
    required_modules = [
        "torch",
        "numpy", 
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


def main():
    """Главная функция проверки готовности"""
    print("🔍 CHECKING TRAINING READINESS (FIXED VERSION)")
    print("=" * 55)
    
    all_checks = {}
    
    # 1. GPU проверка
    print("\n🖥️ Checking GPU availability...")
    gpu_check = check_gpu_availability()
    all_checks["gpu"] = gpu_check
    
    if gpu_check["cuda_available"]:
        print(f"✅ CUDA available: {gpu_check['gpu_count']} GPUs")
        for gpu_name in gpu_check["gpu_names"]:
            print(f"   - {gpu_name}")
            
        # Показываем память RTX 5090
        for gpu_id, info in gpu_check["memory_info"].items():
            if "RTX 5090" in info["name"]:
                print(f"   💾 {info['name']}: {info['free_gb']:.1f}GB free")
    else:
        print("❌ CUDA not available - will use CPU")
    
    # 2. Датасеты
    print("\n📂 Checking dataset availability...")
    dataset_check = check_dataset_availability()
    all_checks["datasets"] = dataset_check
    
    total_samples = 0
    for dataset_type, info in dataset_check.items():
        status = "✅" if info["available"] else "❌"
        count = info.get("count", 0)
        if dataset_type == "cache_embeddings":
            total_files = info.get("total_files", 0)
            print(f"   {status} {dataset_type}: {count} valid (of {total_files} total)")
            total_samples += count * 50  # Примерно
        else:
            total_samples += count * 4  # Примерно по 4 эмбеддинга на файл
            print(f"   {status} {dataset_type}: {count} files")
    
    print(f"📊 Estimated total samples: ~{total_samples}")
    
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
    
    # 4. Центральная конфигурация
    print("\n⚙️ Checking central configuration...")
    config_check = check_central_config()
    all_checks["central_config"] = config_check
    
    if config_check["success"]:
        print("✅ Central config loads successfully")
        for check in config_check["config_checks"]:
            print(f"   {check}")
        
        # Показываем ключевые значения
        values = config_check["config_values"]
        print(f"   📏 Lattice: {values['lattice_dimensions']}")
        print(f"   🧠 State size: {values['state_size']}, Hidden: {values['hidden_dim']}")
        print(f"   📊 Batch size: {values['batch_size']}, Epochs: {values['num_epochs']}")
        print(f"   🎯 Target dim: {values['target_embedding_dim']}")
    else:
        print(f"❌ Central config failed: {config_check.get('error')}")
    
    # 5. Проверяем динамические соседи
    print("\n🎯 Checking dynamic neighbors...")
    if config_check["success"]:
        config = SimpleProjectConfig()
        neighbors_check = check_dynamic_neighbors_quick(config)
        all_checks["dynamic_neighbors"] = neighbors_check
        
        if neighbors_check["success"]:
            print("✅ Dynamic neighbors analysis completed")
            print(f"   📏 Lattice: {neighbors_check['lattice_dimensions']}")
            print(f"   🎯 Max radius: {neighbors_check['max_radius']}")
            print(f"   🔵 Local threshold: {neighbors_check['thresholds']['local']}")
            print(f"   🟡 Functional threshold: {neighbors_check['thresholds']['functional']}")
            
            if neighbors_check["is_dynamic"]:
                print("   ✅ Dynamic neighbor count enabled (neighbor_count = -1)")
            elif neighbors_check["legacy_detected"]:
                print(f"   ❌ Legacy neighbor count detected: {neighbors_check['neighbor_count_setting']}")
            else:
                print(f"   ℹ️ Static neighbor count: {neighbors_check['neighbor_count_setting']}")
                
            if neighbors_check["dynamic_enabled"]:
                print("   ✅ Dynamic count enabled in NeighborSettings")
        else:
            print(f"❌ Dynamic neighbors check failed: {neighbors_check.get('error')}")
    else:
        print("⏭️ Skipping (config failed)")
    
    # 6. Тестируем новый dataset loader
    print("\n🔄 Testing new unified dataset loader...")
    loader_test = test_new_dataset_loader()
    all_checks["dataset_loader"] = loader_test
    
    if loader_test["success"]:
        print(f"✅ Dataset loader works: {loader_test['sample_count']} samples loaded")
        print(f"   📊 Source distribution: {loader_test['source_distribution']}")
        if loader_test.get("embedding_dim_ok"):
            print(f"✅ Embedding dimensions correct ({loader_test['embedding_dim']}D)")
        else:
            print(f"❌ Wrong embedding dimension: expected {loader_test.get('expected_dim')}, got {loader_test.get('actual_dim')}")
    else:
        print(f"❌ Dataset loader failed: {loader_test.get('error')}")
    
    # 7. Тестируем EmbeddingTrainer
    print("\n🧠 Testing EmbeddingTrainer creation...")
    trainer_test = test_embedding_trainer_creation()
    all_checks["trainer"] = trainer_test
    
    if trainer_test["success"]:
        params = trainer_test.get("total_parameters", 0)
        print(f"✅ EmbeddingTrainer creates successfully ({params:,} parameters)")
    else:
        print(f"❌ EmbeddingTrainer failed: {trainer_test.get('error')}")
    
    # Сохраняем результаты
    results_file = Path("training_readiness_check_fixed.json")
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
        (config_check["success"], "Central Config"),
        (all_checks.get("dynamic_neighbors", {}).get("success", False) and 
         not all_checks.get("dynamic_neighbors", {}).get("legacy_detected", False), "Dynamic Neighbors"),
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
        print("1. Run: python real_training_simple.py")
        print("2. Monitor: experiments/")
        print("3. Check logs for progress")
        
        # Проверяем режим обучения
        if config_check["success"] and config_check["config_values"]["test_mode"]:
            print("\n⚠️ NOTE: test_mode=True in config")
            print("For real training, edit new_rebuild/config/config_components.py:")
            print("  Change: test_mode: bool = False")
    else:
        print("\nFix the issues marked with ❌ before proceeding")
    
    print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()