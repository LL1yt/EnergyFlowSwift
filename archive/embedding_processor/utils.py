"""
EmbeddingProcessor Utils - Утилиты для тестирования и валидации
=============================================================

Набор утилит для тестирования, валидации и экспорта результатов EmbeddingProcessor.
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

from .config import EmbeddingConfig, ProcessingMode
from .metrics import calculate_processing_quality, evaluate_semantic_preservation

logger = logging.getLogger(__name__)


def create_test_embedding_batch(batch_size: int = 4, 
                               embedding_dim: int = 768,
                               embedding_type: str = "random") -> torch.Tensor:
    """
    Создать тестовый батч эмбедингов для проверки EmbeddingProcessor
    
    Args:
        batch_size: Размер батча
        embedding_dim: Размерность эмбедингов
        embedding_type: Тип эмбедингов ("random", "semantic", "normalized")
        
    Returns:
        torch.Tensor: Тестовый батч [batch_size, embedding_dim]
    """
    if embedding_type == "random":
        # Случайные эмбединги
        embeddings = torch.randn(batch_size, embedding_dim)
        
    elif embedding_type == "semantic":
        # Семантически значимые эмбединги (имитируют реальные)
        embeddings = []
        for i in range(batch_size):
            # Создаем "семантический" паттерн
            base = torch.randn(embedding_dim) * 0.5
            
            # Добавляем структурированные компоненты
            semantic_component = torch.sin(torch.arange(embedding_dim, dtype=torch.float) * (i + 1) * 0.01)
            context_component = torch.cos(torch.arange(embedding_dim, dtype=torch.float) * (i + 1) * 0.005)
            
            embedding = base + semantic_component * 0.3 + context_component * 0.2
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings)
        
    elif embedding_type == "normalized":
        # Нормализованные эмбединги (единичная длина)
        embeddings = torch.randn(batch_size, embedding_dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    else:
        raise ValueError(f"Неизвестный тип эмбедингов: {embedding_type}")
    
    logger.debug(f"🧪 Создан тестовый батч: {embeddings.shape} ({embedding_type})")
    return embeddings


def validate_processor_output(input_embeddings: torch.Tensor,
                             output_embeddings: torch.Tensor,
                             config: EmbeddingConfig) -> Dict[str, Any]:
    """
    Валидировать выход EmbeddingProcessor
    
    Args:
        input_embeddings: Входные эмбединги
        output_embeddings: Выходные эмбединги
        config: Конфигурация процессора
        
    Returns:
        Dict с результатами валидации
    """
    validation_results = {
        "shape_valid": False,
        "dtype_valid": False,
        "similarity_valid": False,
        "quality_metrics": {},
        "errors": []
    }
    
    try:
        # Проверка формы
        if input_embeddings.shape == output_embeddings.shape:
            validation_results["shape_valid"] = True
        else:
            validation_results["errors"].append(
                f"Shape mismatch: {input_embeddings.shape} != {output_embeddings.shape}"
            )
        
        # Проверка типа данных
        if input_embeddings.dtype == output_embeddings.dtype:
            validation_results["dtype_valid"] = True
        else:
            validation_results["errors"].append(
                f"Dtype mismatch: {input_embeddings.dtype} != {output_embeddings.dtype}"
            )
        
        # Проверка качества
        quality_metrics = calculate_processing_quality(input_embeddings, output_embeddings)
        validation_results["quality_metrics"] = quality_metrics
        
        mean_similarity = quality_metrics["mean_cosine_similarity"]
        if mean_similarity >= config.target_similarity:
            validation_results["similarity_valid"] = True
        else:
            validation_results["errors"].append(
                f"Similarity too low: {mean_similarity:.3f} < {config.target_similarity:.3f}"
            )
        
        # Общая валидация
        validation_results["all_valid"] = (
            validation_results["shape_valid"] and 
            validation_results["dtype_valid"] and 
            validation_results["similarity_valid"]
        )
        
    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
        validation_results["all_valid"] = False
    
    return validation_results


def benchmark_processing_speed(processor, 
                              test_sizes: List[Tuple[int, int]] = None,
                              num_runs: int = 5) -> Dict[str, Any]:
    """
    Бенчмарк скорости обработки EmbeddingProcessor
    
    Args:
        processor: EmbeddingProcessor instance
        test_sizes: Список (batch_size, embedding_dim) для тестирования
        num_runs: Количество прогонов для усреднения
        
    Returns:
        Dict с результатами бенчмарка
    """
    if test_sizes is None:
        test_sizes = [(1, 768), (4, 768), (8, 768), (16, 768)]
    
    benchmark_results = {
        "test_configurations": [],
        "summary": {}
    }
    
    all_throughputs = []
    
    for batch_size, embedding_dim in test_sizes:
        logger.info(f"🔬 Бенчмарк: batch_size={batch_size}, embedding_dim={embedding_dim}")
        
        times = []
        throughputs = []
        
        for run in range(num_runs):
            # Создаем тестовые данные
            test_batch = create_test_embedding_batch(batch_size, embedding_dim, "normalized")
            
            # Измеряем время
            start_time = time.time()
            
            with torch.no_grad():
                output = processor.forward(test_batch)
            
            processing_time = time.time() - start_time
            throughput = batch_size / processing_time
            
            times.append(processing_time)
            throughputs.append(throughput)
        
        # Статистики для данной конфигурации
        config_results = {
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "mean_throughput": np.mean(throughputs),
            "std_throughput": np.std(throughputs),
            "runs": num_runs
        }
        
        benchmark_results["test_configurations"].append(config_results)
        all_throughputs.extend(throughputs)
        
        logger.info(f"[FAST] Пропускная способность: {config_results['mean_throughput']:.1f} ± {config_results['std_throughput']:.1f} эмб/сек")
    
    # Общая статистика
    benchmark_results["summary"] = {
        "overall_mean_throughput": np.mean(all_throughputs),
        "overall_std_throughput": np.std(all_throughputs),
        "best_throughput": np.max(all_throughputs),
        "total_tests": len(test_sizes) * num_runs
    }
    
    return benchmark_results


def run_comprehensive_test(processor, 
                          test_modes: List[ProcessingMode] = None,
                          batch_size: int = 8) -> Dict[str, Any]:
    """
    Запустить комплексный тест EmbeddingProcessor
    
    Args:
        processor: EmbeddingProcessor instance
        test_modes: Режимы для тестирования
        batch_size: Размер тестового батча
        
    Returns:
        Dict с результатами всех тестов
    """
    if test_modes is None:
        test_modes = [ProcessingMode.AUTOENCODER, ProcessingMode.GENERATOR, ProcessingMode.DIALOGUE]
    
    test_results = {
        "mode_tests": {},
        "overall_summary": {}
    }
    
    all_similarities = []
    all_times = []
    
    for mode in test_modes:
        logger.info(f"🧪 Тестирование режима: {mode.value}")
        
        # Устанавливаем режим
        original_mode = processor.config.processing_mode
        processor.set_mode(mode)
        
        # Создаем тестовые данные
        test_input = create_test_embedding_batch(batch_size, 768, "semantic")
        
        # Обработка
        start_time = time.time()
        test_output = processor.forward(test_input)
        processing_time = time.time() - start_time
        
        # Валидация
        validation = validate_processor_output(test_input, test_output, processor.config)
        
        # Качество
        quality = calculate_processing_quality(test_input, test_output)
        semantic_eval = evaluate_semantic_preservation(test_input, test_output)
        
        mode_results = {
            "mode": mode.value,
            "processing_time": processing_time,
            "validation": validation,
            "quality_metrics": quality,
            "semantic_preservation": semantic_eval,
            "throughput": batch_size / processing_time
        }
        
        test_results["mode_tests"][mode.value] = mode_results
        
        all_similarities.append(quality["mean_cosine_similarity"])
        all_times.append(processing_time)
        
        logger.info(f"[OK] {mode.value}: similarity={quality['mean_cosine_similarity']:.3f}, time={processing_time:.3f}s")
    
    # Восстанавливаем исходный режим
    processor.set_mode(original_mode)
    
    # Общая сводка
    test_results["overall_summary"] = {
        "all_modes_tested": len(test_modes),
        "mean_similarity": np.mean(all_similarities),
        "min_similarity": np.min(all_similarities),
        "max_similarity": np.max(all_similarities),
        "mean_processing_time": np.mean(all_times),
        "total_test_time": np.sum(all_times),
        "all_passed": all(
            result["validation"]["all_valid"] 
            for result in test_results["mode_tests"].values()
        )
    }
    
    return test_results


def export_processing_results(results: Dict[str, Any], 
                             output_path: str = "outputs/embedding_processor_results.json"):
    """
    Экспортировать результаты обработки в файл
    
    Args:
        results: Результаты для экспорта
        output_path: Путь для сохранения
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Конвертируем в JSON-сериализуемый формат
    json_results = convert_to_json_serializable(results)
    
    # Добавляем метаданные
    json_results["export_metadata"] = {
        "export_time": time.time(),
        "export_path": str(output_path),
        "phase": "2.5",
        "module": "embedding_processor"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[FOLDER] Результаты экспортированы в: {output_path}")


def convert_to_json_serializable(obj: Any) -> Any:
    """Конвертировать объект в JSON-сериализуемый формат"""
    
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, ProcessingMode):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    else:
        return obj


def create_quality_report(processor,
                         num_samples: int = 100,
                         report_path: str = "outputs/phase_2_5_quality_report.json") -> Dict[str, Any]:
    """
    Создать подробный отчет о качестве работы EmbeddingProcessor
    
    Args:
        processor: EmbeddingProcessor instance
        num_samples: Количество тестовых образцов
        report_path: Путь для сохранения отчета
        
    Returns:
        Dict с полным отчетом о качестве
    """
    logger.info(f"[DATA] Создание отчета о качестве (samples={num_samples})")
    
    # Тестируем на разных типах данных
    test_types = ["random", "semantic", "normalized"]
    
    quality_report = {
        "test_summary": {},
        "detailed_results": {},
        "phase_2_5_assessment": {}
    }
    
    all_similarities = []
    
    for test_type in test_types:
        logger.info(f"🔬 Тестирование на {test_type} эмбедингах")
        
        # Создаем множественные тесты
        similarities = []
        processing_times = []
        
        for i in range(num_samples // 10):  # Батчи по 10
            test_batch = create_test_embedding_batch(10, 768, test_type)
            
            start_time = time.time()
            output_batch = processor.forward(test_batch)
            proc_time = time.time() - start_time
            
            quality = calculate_processing_quality(test_batch, output_batch)
            
            similarities.append(quality["mean_cosine_similarity"])
            processing_times.append(proc_time)
            all_similarities.append(quality["mean_cosine_similarity"])
        
        type_results = {
            "test_type": test_type,
            "samples_tested": len(similarities) * 10,
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "mean_processing_time": np.mean(processing_times),
            "target_achievement_rate": np.mean(np.array(similarities) >= processor.config.target_similarity)
        }
        
        quality_report["detailed_results"][test_type] = type_results
    
    # Общая оценка
    overall_similarity = np.mean(all_similarities)
    target_achievement = np.mean(np.array(all_similarities) >= processor.config.target_similarity)
    
    # Phase 2.5 assessment
    phase_2_5_ready = overall_similarity >= processor.config.target_similarity
    
    quality_report["test_summary"] = {
        "total_samples": num_samples,
        "overall_mean_similarity": overall_similarity,
        "overall_target_achievement": target_achievement,
        "test_types_count": len(test_types)
    }
    
    quality_report["phase_2_5_assessment"] = {
        "target_similarity": processor.config.target_similarity,
        "achieved_similarity": overall_similarity,
        "phase_2_5_ready": phase_2_5_ready,
        "quality_level": "excellent" if overall_similarity >= 0.95 else 
                        "good" if overall_similarity >= 0.90 else
                        "acceptable" if overall_similarity >= 0.80 else "poor",
        "recommendation": "Ready for Phase 3" if phase_2_5_ready else "Needs improvement"
    }
    
    # Экспорт отчета
    export_processing_results(quality_report, report_path)
    
    logger.info(f"[OK] Отчет готов: similarity={overall_similarity:.3f}, Phase 2.5 ready: {phase_2_5_ready}")
    
    return quality_report 