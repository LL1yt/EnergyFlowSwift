#!/usr/bin/env python3
"""
Анализ готовых датасетов для реального обучения
Использует готовые эмбеддинги из data/embeddings и cache/dialogue_dataset
"""

import torch
from pathlib import Path
import json
from typing import Dict, List, Any

def analyze_dialogue_datasets() -> Dict[str, Any]:
    """Анализ dialogue datasets из cache/dialogue_dataset/"""
    cache_dir = Path("cache/dialogue_dataset")
    files = list(cache_dir.glob("*.pt"))
    
    print(f"=== DIALOGUE DATASETS ANALYSIS ===")
    print(f"Found {len(files)} dialogue files")
    
    if not files:
        print("No dialogue files found!")
        return {}
    
    # Анализируем первый файл
    sample_file = files[0]
    print(f"\nAnalyzing sample file: {sample_file.name}")
    
    try:
        sample = torch.load(sample_file, map_location='cpu')
        print(f"Keys: {list(sample.keys())}")
        
        results = {"dialogue_files": len(files), "sample_structure": {}}
        
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                info = f"{v.shape} ({v.dtype})"
                print(f"  {k}: {info}")
                results["sample_structure"][k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                info = f"{type(v)} (len: {len(v) if hasattr(v, '__len__') else 'N/A'})"
                print(f"  {k}: {info}")
                results["sample_structure"][k] = {"type": str(type(v)), "length": len(v) if hasattr(v, '__len__') else None}
        
        return results
        
    except Exception as e:
        print(f"Error loading sample file: {e}")
        return {}

def analyze_prepared_embeddings() -> Dict[str, Any]:
    """Анализ готовых эмбеддингов из data/embeddings/"""
    embeddings_dir = Path("data/embeddings")
    files = list(embeddings_dir.glob("*.pt"))
    
    print(f"\n=== PREPARED EMBEDDINGS ANALYSIS ===")
    print(f"Found {len(files)} embedding files")
    
    if not files:
        print("No embedding files found!")
        return {}
    
    # Анализируем первые несколько файлов
    results = {"embedding_files": len(files), "samples": []}
    
    for i, file in enumerate(files[:3]):  # Первые 3 файла
        print(f"\nAnalyzing embedding file {i+1}: {file.name}")
        
        try:
            embedding = torch.load(file, map_location='cpu')
            
            if isinstance(embedding, torch.Tensor):
                info = f"Tensor: {embedding.shape} ({embedding.dtype})"
                print(f"  {info}")
                results["samples"].append({
                    "file": file.name,
                    "type": "tensor",
                    "shape": list(embedding.shape),
                    "dtype": str(embedding.dtype)
                })
            elif isinstance(embedding, dict):
                print(f"  Dict with keys: {list(embedding.keys())}")
                sample_info = {"file": file.name, "type": "dict", "keys": list(embedding.keys())}
                
                for k, v in embedding.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: {v.shape} ({v.dtype})")
                        sample_info[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                
                results["samples"].append(sample_info)
            else:
                print(f"  Unexpected type: {type(embedding)}")
                results["samples"].append({
                    "file": file.name,
                    "type": str(type(embedding))
                })
                
        except Exception as e:
            print(f"  Error loading file: {e}")
            results["samples"].append({
                "file": file.name,
                "error": str(e)
            })
    
    return results

def check_snli_generator():
    """Проверяем generate_snli_embedding_dataset.py"""
    print(f"\n=== SNLI GENERATOR ANALYSIS ===")
    
    snli_script = Path("generate_snli_embedding_dataset.py")
    if snli_script.exists():
        print(f"✅ Found SNLI generator: {snli_script}")
        
        # Читаем первые строки для понимания
        with open(snli_script, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:20]  # Первые 20 строк
            
        print("Script preview:")
        for i, line in enumerate(lines, 1):
            print(f"  {i:2d}: {line.rstrip()}")
            
        return {"snli_script_exists": True, "path": str(snli_script)}
    else:
        print("❌ SNLI generator not found")
        return {"snli_script_exists": False}

def check_cache_embeddings():
    """Проверяем cache эмбеддинги"""
    print(f"\n=== CACHE EMBEDDINGS ANALYSIS ===")
    
    cache_files = list(Path("cache").glob("llm_*.pt"))
    print(f"Found {len(cache_files)} cached LLM embeddings")
    
    if cache_files:
        # Анализируем один файл
        sample = cache_files[0]
        print(f"Sample cache file: {sample.name}")
        
        try:
            data = torch.load(sample, map_location='cpu')
            if hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}, dtype: {data.dtype}")
                return {
                    "cache_files_count": len(cache_files),
                    "sample_shape": list(data.shape),
                    "sample_dtype": str(data.dtype)
                }
            else:
                print(f"  Type: {type(data)}")
                return {
                    "cache_files_count": len(cache_files),
                    "sample_type": str(type(data))
                }
        except Exception as e:
            print(f"  Error: {e}")
            return {"cache_files_count": len(cache_files), "error": str(e)}
    
    return {"cache_files_count": 0}

def main():
    """Главная функция анализа всех доступных датасетов"""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS FOR REAL TRAINING")
    print("=" * 60)
    
    # Собираем всю информацию
    analysis_results = {
        "dialogue_analysis": analyze_dialogue_datasets(),
        "embeddings_analysis": analyze_prepared_embeddings(), 
        "snli_generator": check_snli_generator(),
        "cache_embeddings": check_cache_embeddings()
    }
    
    # Сохраняем результаты
    results_file = Path("dataset_analysis_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"✅ Analysis results saved to: {results_file}")
    
    # Краткая сводка
    dialogue_count = analysis_results["dialogue_analysis"].get("dialogue_files", 0)
    embedding_count = analysis_results["embeddings_analysis"].get("embedding_files", 0)
    cache_count = analysis_results["cache_embeddings"].get("cache_files_count", 0)
    snli_available = analysis_results["snli_generator"].get("snli_script_exists", False)
    
    print(f"📂 Dialogue datasets: {dialogue_count} files")
    print(f"📂 Prepared embeddings: {embedding_count} files") 
    print(f"📂 Cache embeddings: {cache_count} files")
    print(f"🔧 SNLI generator: {'✅ Available' if snli_available else '❌ Missing'}")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Run this script to understand data structure")
    print("2. Create unified dataset loader")
    print("3. Configure training for 8x8x8 cube")
    print("4. Start real training!")

if __name__ == "__main__":
    main()