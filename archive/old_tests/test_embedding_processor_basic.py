#!/usr/bin/env python3
"""
Базовый тест EmbeddingProcessor - Phase 2.5
==========================================

Первичная проверка функциональности центрального процессора эмбедингов.

Цель: Убедиться что EmbeddingProcessor корректно инициализируется и работает.
"""

import sys
import os
import torch
import logging
import time

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embedding_processor import (
    EmbeddingProcessor,
    EmbeddingConfig,
    ProcessingMode,
    create_autoencoder_config,
    create_test_embedding_batch,
    validate_processor_output,
    run_comprehensive_test
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_embedding_processor_initialization():
    """Тест 1: Инициализация EmbeddingProcessor"""
    
    logger.info("🧪 ТЕСТ 1: Инициализация EmbeddingProcessor")
    
    try:
        # Создаем конфигурацию
        config = EmbeddingConfig(
            processing_mode=ProcessingMode.AUTOENCODER,
            target_similarity=0.90,
            debug_mode=True,
            verbose_logging=True
        )
        
        # Инициализируем процессор
        processor = EmbeddingProcessor(config)
        
        logger.info(f"[OK] Процессор создан: {processor}")
        logger.info(f"[DATA] Режим: {processor.config.processing_mode.value}")
        logger.info(f"[TARGET] Целевая схожесть: {processor.config.target_similarity}")
        
        return True, processor
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка инициализации: {e}")
        return False, None


def test_single_embedding_processing(processor):
    """Тест 2: Обработка одиночного эмбединга"""
    
    logger.info("🧪 ТЕСТ 2: Обработка одиночного эмбединга")
    
    try:
        # Создаем тестовый эмбединг
        input_embedding = torch.randn(768)
        logger.info(f"📥 Входной эмбединг: {input_embedding.shape}")
        
        # Обработка
        start_time = time.time()
        output_embedding = processor.forward(input_embedding)
        processing_time = time.time() - start_time
        
        logger.info(f"📤 Выходной эмбединг: {output_embedding.shape}")
        logger.info(f"⏱️ Время обработки: {processing_time:.3f}s")
        
        # Проверка схожести
        similarity = torch.nn.functional.cosine_similarity(
            input_embedding, output_embedding, dim=0
        ).item()
        
        logger.info(f"[DATA] Cosine similarity: {similarity:.3f}")
        
        # Проверка достижения цели Phase 2.5
        target_achieved = similarity >= processor.config.target_similarity
        logger.info(f"[TARGET] Phase 2.5 цель достигнута: {target_achieved} (>{processor.config.target_similarity:.2f})")
        
        return True, similarity
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка обработки эмбединга: {e}")
        return False, 0.0


def test_batch_processing(processor):
    """Тест 3: Батчевая обработка"""
    
    logger.info("🧪 ТЕСТ 3: Батчевая обработка")
    
    try:
        batch_size = 4
        
        # Создаем батч эмбедингов
        input_batch = create_test_embedding_batch(batch_size, 768, "semantic")
        logger.info(f"📥 Входной батч: {input_batch.shape}")
        
        # Обработка
        start_time = time.time()
        output_batch = processor.forward(input_batch)
        processing_time = time.time() - start_time
        
        logger.info(f"📤 Выходной батч: {output_batch.shape}")
        logger.info(f"⏱️ Время обработки: {processing_time:.3f}s")
        logger.info(f"[FAST] Пропускная способность: {batch_size/processing_time:.1f} эмб/сек")
        
        # Валидация
        validation = validate_processor_output(input_batch, output_batch, processor.config)
        
        if validation["all_valid"]:
            logger.info("[OK] Валидация батча пройдена")
            similarity = validation["quality_metrics"]["mean_cosine_similarity"]
            logger.info(f"[DATA] Средняя схожесть батча: {similarity:.3f}")
            return True, similarity
        else:
            logger.warning("[WARNING] Валидация батча не пройдена:")
            for error in validation["errors"]:
                logger.warning(f"  - {error}")
            return False, 0.0
            
    except Exception as e:
        logger.error(f"[ERROR] Ошибка батчевой обработки: {e}")
        return False, 0.0


def test_multiple_modes(processor):
    """Тест 4: Тестирование всех режимов"""
    
    logger.info("🧪 ТЕСТ 4: Тестирование всех режимов")
    
    modes_results = {}
    
    # Тестовый эмбединг
    test_embedding = torch.randn(768)
    
    for mode in [ProcessingMode.AUTOENCODER, ProcessingMode.GENERATOR, ProcessingMode.DIALOGUE]:
        try:
            logger.info(f"[REFRESH] Тестирование режима: {mode.value}")
            
            # Устанавливаем режим
            processor.set_mode(mode)
            
            # Обработка
            output = processor.forward(test_embedding)
            
            # Схожесть
            similarity = torch.nn.functional.cosine_similarity(
                test_embedding, output, dim=0
            ).item()
            
            modes_results[mode.value] = similarity
            logger.info(f"[DATA] {mode.value}: similarity = {similarity:.3f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка в режиме {mode.value}: {e}")
            modes_results[mode.value] = 0.0
    
    # Сводка по режимам
    logger.info("=== СВОДКА ПО РЕЖИМАМ ===")
    for mode, similarity in modes_results.items():
        status = "[OK]" if similarity >= 0.80 else "[WARNING]"
        logger.info(f"{status} {mode}: {similarity:.3f}")
    
    return len(modes_results) == 3, modes_results


def test_metrics_collection(processor):
    """Тест 5: Сбор метрик"""
    
    logger.info("🧪 ТЕСТ 5: Сбор метрик")
    
    try:
        # Сбрасываем метрики
        processor.reset_metrics()
        
        # Делаем несколько обработок
        for i in range(5):
            test_embedding = torch.randn(768)
            processor.forward(test_embedding)
        
        # Получаем метрики
        metrics = processor.get_metrics()
        
        logger.info("=== СОБРАННЫЕ МЕТРИКИ ===")
        logger.info(f"[DATA] Средняя схожесть: {metrics['similarity']['mean']:.3f}")
        logger.info(f"[TARGET] Достижение цели: {metrics['quality']['target_achievement_rate']:.1%}")
        logger.info(f"[STAR] Уровень качества: {metrics['quality']['quality_level']}")
        logger.info(f"[FAST] Пропускная способность: {metrics['performance']['throughput_embeddings_per_sec']:.1f} эмб/сек")
        logger.info(f"🔢 Обработано: {metrics['total_processed']} эмбедингов")
        
        # Детальное логирование
        processor.metrics.log_current_stats()
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка сбора метрик: {e}")
        return False, {}


def main():
    """Главная функция - запуск всех тестов"""
    
    logger.info("[START] ЗАПУСК БАЗОВОГО ТЕСТИРОВАНИЯ EMBEDDINGPROCESSOR (Phase 2.5)")
    logger.info("=" * 70)
    
    test_results = {
        "initialization": False,
        "single_processing": False,
        "batch_processing": False,
        "multiple_modes": False,
        "metrics_collection": False
    }
    
    similarities = []
    processor = None
    
    # Тест 1: Инициализация
    success, processor = test_embedding_processor_initialization()
    test_results["initialization"] = success
    
    if not success:
        logger.error("💥 Критическая ошибка: не удалось инициализировать процессор")
        return False
    
    # Тест 2: Одиночная обработка
    success, similarity = test_single_embedding_processing(processor)
    test_results["single_processing"] = success
    if success:
        similarities.append(similarity)
    
    # Тест 3: Батчевая обработка
    success, similarity = test_batch_processing(processor)
    test_results["batch_processing"] = success
    if success:
        similarities.append(similarity)
    
    # Тест 4: Множественные режимы
    success, modes_results = test_multiple_modes(processor)
    test_results["multiple_modes"] = success
    if success:
        similarities.extend(modes_results.values())
    
    # Тест 5: Метрики
    success, metrics = test_metrics_collection(processor)
    test_results["metrics_collection"] = success
    
    # === ФИНАЛЬНАЯ СВОДКА ===
    logger.info("=" * 70)
    logger.info("[INFO] ФИНАЛЬНАЯ СВОДКА ТЕСТОВ")
    logger.info("=" * 70)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "[OK] ПРОЙДЕН" if result else "[ERROR] ПРОВАЛЕН"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n[DATA] ОБЩИЙ РЕЗУЛЬТАТ: {passed_tests}/{total_tests} тестов пройдено")
    
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        logger.info(f"[CHART] Средняя схожесть по всем тестам: {avg_similarity:.3f}")
        
        # Оценка готовности Phase 2.5
        phase_2_5_ready = avg_similarity >= 0.90
        logger.info(f"[TARGET] Phase 2.5 готовность: {'[OK] ДА' if phase_2_5_ready else '[ERROR] НЕТ'} (цель: >0.90)")
        
        if phase_2_5_ready:
            logger.info("[SUCCESS] ПОЗДРАВЛЯЕМ! EmbeddingProcessor готов к Phase 3!")
        else:
            logger.info("[CONFIG] Требуется доработка для достижения целей Phase 2.5")
    
    all_passed = all(test_results.values())
    logger.info(f"\n[TROPHY] ИТОГОВЫЙ СТАТУС: {'[SUCCESS] ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_passed else '[WARNING] ЕСТЬ ПРОБЛЕМЫ'}")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1) 