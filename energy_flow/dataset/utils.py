"""
Утилиты для dataset модуля
==========================

Вспомогательные функции для валидации, диагностики и работы с данными
"""

import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time

from ..utils.logging import get_logger

logger = get_logger(__name__)


def validate_dataset_directory_structure(base_path: str) -> Dict[str, Any]:
    """
    Проверка структуры директорий для датасетов
    
    Args:
        base_path: Базовый путь проекта
        
    Returns:
        Результаты проверки структуры
    """
    base = Path(base_path)
    
    results = {
        'base_path_exists': base.exists(),
        'directories': {},
        'missing_directories': [],
        'permissions_ok': True,
        'recommendations': []
    }
    
    # Проверяем ключевые директории
    key_directories = {
        'data': base / 'data',
        'data/embeddings': base / 'data' / 'embeddings', 
        'cache': base / 'cache',
        'models': base / 'models',
        'models/local_cache': base / 'models' / 'local_cache'
    }
    
    for name, path in key_directories.items():
        exists = path.exists()
        results['directories'][name] = {
            'exists': exists,
            'path': str(path),
            'is_writable': path.parent.exists() and path.parent.is_dir()
        }
        
        if not exists:
            results['missing_directories'].append(name)
    
    # Рекомендации
    if results['missing_directories']:
        results['recommendations'].append(
            f"Create missing directories: {', '.join(results['missing_directories'])}"
        )
    
    return results


def scan_available_datasets(data_dirs: List[str]) -> Dict[str, Any]:
    """
    Сканирование доступных датасетов
    
    Args:
        data_dirs: Список директорий для сканирования
        
    Returns:
        Информация о найденных датасетах
    """
    scan_results = {
        'total_files': 0,
        'directories_scanned': [],
        'files_by_type': {},
        'files_by_directory': {},
        'corrupted_files': [],
        'largest_files': []
    }
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            continue
            
        scan_results['directories_scanned'].append(str(dir_path))
        scan_results['files_by_directory'][str(dir_path)] = []
        
        # Сканируем .pt файлы
        for file_path in dir_path.glob("*.pt"):
            try:
                file_info = {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size_mb': file_path.stat().st_size / 1024 / 1024,
                    'modified': file_path.stat().st_mtime
                }
                
                # Пытаемся определить тип файла
                try:
                    data = torch.load(file_path, map_location='cpu')
                    file_type = _determine_file_type(data)
                    file_info['type'] = file_type
                    file_info['samples'] = _estimate_sample_count(data)
                    
                except Exception as load_error:
                    file_info['type'] = 'corrupted'
                    file_info['error'] = str(load_error)
                    scan_results['corrupted_files'].append(file_info)
                
                scan_results['files_by_directory'][str(dir_path)].append(file_info)
                scan_results['total_files'] += 1
                
                # Группировка по типам
                file_type = file_info.get('type', 'unknown')
                if file_type not in scan_results['files_by_type']:
                    scan_results['files_by_type'][file_type] = []
                scan_results['files_by_type'][file_type].append(file_info)
                
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")
    
    # Находим самые большие файлы
    all_files = []
    for files in scan_results['files_by_directory'].values():
        all_files.extend(files)
    
    scan_results['largest_files'] = sorted(
        all_files, 
        key=lambda x: x.get('size_mb', 0), 
        reverse=True
    )[:5]
    
    return scan_results


def _determine_file_type(data: Any) -> str:
    """Определение типа файла по содержимому"""
    if isinstance(data, dict):
        if 'question_embeddings' in data and 'answer_embeddings' in data:
            return 'snli_dataset'
        elif 'input_embeddings' in data and 'target_embeddings' in data:
            return 'unified_dataset'
        elif 'embeddings' in data:
            return 'embedding_cache'
        else:
            return 'dict_format'
    elif isinstance(data, torch.Tensor):
        return 'tensor'
    elif isinstance(data, list):
        return 'list_format'
    else:
        return 'unknown'


def _estimate_sample_count(data: Any) -> Optional[int]:
    """Оценка количества образцов в файле"""
    try:
        if isinstance(data, dict):
            for key in ['question_embeddings', 'input_embeddings', 'embeddings']:
                if key in data and isinstance(data[key], torch.Tensor):
                    return data[key].shape[0]
        elif isinstance(data, torch.Tensor):
            return data.shape[0] if data.dim() > 0 else 1
        elif isinstance(data, list):
            return len(data)
    except:
        pass
    return None


def diagnose_teacher_model_setup(config) -> Dict[str, Any]:
    """
    Диагностика настройки модели-учителя
    
    Args:
        config: DatasetConfig
        
    Returns:
        Результаты диагностики
    """
    diagnosis = {
        'model_name': config.teacher_model,
        'use_local_model': config.use_local_model,
        'local_path_configured': str(config.local_model_path),
        'local_model_available': False,
        'local_model_complete': False,
        'huggingface_accessible': False,
        'recommendations': []
    }
    
    # Проверка локальной модели
    if config.use_local_model:
        local_path = Path(config.local_model_path)
        diagnosis['local_model_available'] = local_path.exists()
        
        if diagnosis['local_model_available']:
            # Проверяем комплектность
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            missing_files = []
            
            for file in required_files:
                if not (local_path / file).exists():
                    missing_files.append(file)
            
            diagnosis['local_model_complete'] = len(missing_files) == 0
            diagnosis['missing_files'] = missing_files
            
            if missing_files:
                diagnosis['recommendations'].append(
                    f"Download missing model files: {', '.join(missing_files)}"
                )
        else:
            diagnosis['recommendations'].append(
                "Download local model using download_distilbert.py"
            )
    
    # Проверка доступности HuggingFace (упрощенная)
    try:
        from transformers import AutoConfig
        config_obj = AutoConfig.from_pretrained(config.teacher_model)
        diagnosis['huggingface_accessible'] = config_obj is not None
    except Exception as e:
        diagnosis['huggingface_accessible'] = False
        diagnosis['huggingface_error'] = str(e)
        diagnosis['recommendations'].append(
            "Check internet connection for HuggingFace access"
        )
    
    # Общие рекомендации
    if not diagnosis['local_model_complete'] and not diagnosis['huggingface_accessible']:
        diagnosis['recommendations'].append(
            "CRITICAL: No working model source available!"
        )
    
    return diagnosis


def benchmark_embedding_generation(teacher_provider, sample_texts: List[str], 
                                 iterations: int = 3) -> Dict[str, Any]:
    """
    Бенчмарк производительности генерации эмбеддингов
    
    Args:
        teacher_provider: TeacherModelProvider
        sample_texts: Тексты для тестирования
        iterations: Количество итераций для усреднения
        
    Returns:
        Результаты бенчмарка
    """
    if not teacher_provider.ensure_initialized():
        return {'error': 'Teacher provider not initialized'}
    
    results = {
        'sample_count': len(sample_texts),
        'iterations': iterations,
        'times': [],
        'throughput': [],
        'embedding_dimension': None,
        'device': str(teacher_provider.device)
    }
    
    try:
        for i in range(iterations):
            start_time = time.time()
            
            # Генерируем эмбеддинги
            embeddings = teacher_provider.encode_texts(sample_texts)
            
            end_time = time.time()
            iteration_time = end_time - start_time
            
            results['times'].append(iteration_time)
            results['throughput'].append(len(sample_texts) / iteration_time)
            
            if results['embedding_dimension'] is None:
                results['embedding_dimension'] = embeddings.shape[1]
        
        # Вычисляем статистику
        results['avg_time'] = sum(results['times']) / len(results['times'])
        results['avg_throughput'] = sum(results['throughput']) / len(results['throughput'])
        results['min_time'] = min(results['times'])
        results['max_time'] = max(results['times'])
        
        logger.info(f"Embedding benchmark: {results['avg_throughput']:.1f} texts/sec, "
                   f"avg_time={results['avg_time']:.3f}s")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Benchmark failed: {e}")
    
    return results


def create_dataset_summary_report(dataset_manager) -> str:
    """
    Создание текстового отчета о состоянии датасета
    
    Args:
        dataset_manager: DatasetManager
        
    Returns:
        Текстовый отчет
    """
    validation = dataset_manager.validate_setup()
    statistics = dataset_manager.get_statistics()
    
    report_lines = [
        "="*60,
        "ENERGY FLOW DATASET SUMMARY REPORT",
        "="*60,
        "",
        "🔍 VALIDATION STATUS:",
        f"  Teacher Model: {'✅ OK' if validation['teacher_model'] else '❌ FAILED'}",
        f"  Data Providers: {sum(validation['providers'].values())}/{len(validation['providers'])} available",
        f"  Dataset Preparation: {'✅ OK' if validation['dataset_preparation'] else '❌ FAILED'}",
        f"  Overall Status: {'🎉 READY' if validation['overall_status'] else '⚠️ ISSUES'}",
        ""
    ]
    
    if validation['errors']:
        report_lines.extend([
            "❌ ERRORS:",
            *[f"  - {error}" for error in validation['errors']],
            ""
        ])
    
    if validation['warnings']:
        report_lines.extend([
            "⚠️ WARNINGS:",
            *[f"  - {warning}" for warning in validation['warnings']], 
            ""
        ])
    
    if 'error' not in statistics:
        report_lines.extend([
            "📊 DATASET STATISTICS:",
            f"  Total Samples: {statistics.get('total_samples', 'N/A'):,}",
            f"  Embedding Dimension: {statistics.get('embedding_dimension', 'N/A')}",
            f"  Preparation Time: {statistics.get('preparation_time_seconds', 'N/A'):.2f}s",
            f"  Providers Used: {', '.join(statistics.get('providers_used', []))}",
            ""
        ])
        
        # Распределение по источникам
        if 'source_distribution' in statistics:
            report_lines.extend([
                "📈 SOURCE DISTRIBUTION:",
                *[f"  {source}: {count:,} samples" 
                  for source, count in statistics['source_distribution'].items()],
                ""
            ])
    
    report_lines.extend([
        "="*60,
        f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "="*60
    ])
    
    return "\n".join(report_lines)