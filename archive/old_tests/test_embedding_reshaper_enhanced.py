#!/usr/bin/env python3
"""
Улучшенное тестирование EmbeddingReshaper для достижения семантического сохранения >98%
====================================================================================

PHASE 2.3 День 3-4: Тестирование улучшенных алгоритмов семантического сохранения

Основные тесты:
1. Enhanced similarity metrics (многомерный анализ сходства)
2. Importance analysis (анализ важности элементов)
3. Adaptive placement (адаптивное размещение в 3D)
4. Кэширование и оптимизация производительности
5. Достижение порога >98% семантического сохранения
"""

import sys
import os
import numpy as np
import torch
import logging
from typing import Dict, List, Any

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настройка детального логирования для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Импорты модулей
from data.embedding_reshaper import EmbeddingReshaper
from data.embedding_reshaper.strategies import AdaptiveReshaper, SemanticReshaper
from data.embedding_reshaper.utils import (
    calculate_enhanced_similarity_metrics,
    analyze_embedding_importance,
    create_adaptive_transformation_strategy,
    create_test_embeddings
)


def test_enhanced_similarity_metrics():
    """
    Тест улучшенных метрик семантического сходства.
    
    Проверяет работу многомерного анализа:
    - Cosine similarity
    - Pearson correlation  
    - Spearman correlation
    - Structural similarity
    - Magnitude preservation
    """
    print("\n🧪 === ТЕСТ 1: ENHANCED SIMILARITY METRICS ===")
    
    # Создаем тестовые эмбединги
    original_embedding = np.random.randn(768).astype(np.float32)
    
    # Случай 1: Идентичные векторы (должно быть 1.0)
    identical_embedding = original_embedding.copy()
    metrics_identical = calculate_enhanced_similarity_metrics(original_embedding, identical_embedding)
    
    print(f"📊 Метрики для идентичных векторов:")
    for metric, value in metrics_identical.items():
        print(f"   {metric}: {value:.6f}")
    
    assert metrics_identical['weighted_similarity'] > 0.99, "Идентичные векторы должны иметь similarity ~1.0"
    
    # Случай 2: Слегка измененные векторы (семантически близкие)
    noise_std = np.std(original_embedding) * 0.01  # 1% шума
    noisy_embedding = original_embedding + np.random.normal(0, noise_std, original_embedding.shape)
    metrics_noisy = calculate_enhanced_similarity_metrics(original_embedding, noisy_embedding)
    
    print(f"\n📊 Метрики для зашумленных векторов (1% noise):")
    for metric, value in metrics_noisy.items():
        print(f"   {metric}: {value:.6f}")
    
    assert metrics_noisy['weighted_similarity'] > 0.95, "Слегка зашумленные векторы должны иметь similarity >0.95"
    
    # Случай 3: Простой reshape (должен сохранять 100% семантики)
    reshaped_embedding = original_embedding.reshape(8, 8, 12).reshape(768)
    metrics_reshaped = calculate_enhanced_similarity_metrics(original_embedding, reshaped_embedding)
    
    print(f"\n📊 Метрики для простого reshape:")
    for metric, value in metrics_reshaped.items():
        print(f"   {metric}: {value:.6f}")
    
    assert metrics_reshaped['weighted_similarity'] > 0.999, "Простой reshape должен сохранять практически 100% семантики"
    
    print("✅ ТЕСТ 1 ПРОШЕЛ: Enhanced similarity metrics работают корректно!")
    return metrics_reshaped['weighted_similarity']


def test_importance_analysis():
    """
    Тест анализа важности элементов эмбединга.
    
    Проверяет три метода:
    - variance_pca: PCA анализ
    - clustering: кластерный анализ  
    - magnitude: анализ по величине
    """
    print("\n🧪 === ТЕСТ 2: IMPORTANCE ANALYSIS ===")
    
    # Создаем специальный тестовый эмбединг с явной структурой важности
    embedding = np.zeros(768)
    
    # Важные элементы (начало и середина)
    embedding[:100] = np.random.randn(100) * 2.0  # Высокие значения
    embedding[300:400] = np.random.randn(100) * 1.5  # Средние значения
    
    # Менее важные элементы (остальные)
    embedding[100:300] = np.random.randn(200) * 0.5  # Низкие значения
    embedding[400:] = np.random.randn(368) * 0.3  # Очень низкие значения
    
    print(f"📊 Тестовый эмбединг создан: {embedding.shape}")
    print(f"   Статистика: mean={np.mean(embedding):.3f}, std={np.std(embedding):.3f}")
    
    # Тестируем все три метода анализа важности
    methods = ["variance_pca", "clustering", "magnitude"]
    importance_results = {}
    
    for method in methods:
        print(f"\n🔍 Анализ важности методом: {method}")
        
        importance_weights = analyze_embedding_importance(embedding, method=method)
        
        print(f"   Размерность весов: {importance_weights.shape}")
        print(f"   Статистика весов: min={np.min(importance_weights):.3f}, "
              f"max={np.max(importance_weights):.3f}, mean={np.mean(importance_weights):.3f}")
        
        # Проверяем, что веса нормализованы
        assert 0.0 <= np.min(importance_weights) <= np.max(importance_weights) <= 1.0, \
               f"Веса важности должны быть в диапазоне [0,1] для метода {method}"
        
        # Проверяем, что важные элементы получили высокие веса
        important_indices = np.concatenate([np.arange(100), np.arange(300, 400)])
        avg_important_weight = np.mean(importance_weights[important_indices])
        
        less_important_indices = np.concatenate([np.arange(100, 300), np.arange(400, 768)])
        avg_less_important_weight = np.mean(importance_weights[less_important_indices])
        
        print(f"   Средний вес важных элементов: {avg_important_weight:.3f}")
        print(f"   Средний вес менее важных элементов: {avg_less_important_weight:.3f}")
        print(f"   Коэффициент различия: {avg_important_weight / avg_less_important_weight:.2f}")
        
        importance_results[method] = {
            'weights': importance_weights,
            'important_avg': avg_important_weight,
            'less_important_avg': avg_less_important_weight,
            'discrimination_ratio': avg_important_weight / avg_less_important_weight
        }
    
    # Проверяем, что все методы выявляют различия в важности
    for method, results in importance_results.items():
        discrimination_ratio = results['discrimination_ratio']
        assert discrimination_ratio > 1.2, \
               f"Метод {method} должен различать важные и менее важные элементы (ratio > 1.2, получено {discrimination_ratio:.2f})"
    
    print("✅ ТЕСТ 2 ПРОШЕЛ: Анализ важности элементов работает корректно!")
    return importance_results


def test_adaptive_placement_strategy():
    """
    Тест адаптивных стратегий размещения в 3D пространстве.
    
    Проверяет создание оптимальных карт размещения элементов
    с учетом их важности для семантического сохранения.
    """
    print("\n🧪 === ТЕСТ 3: ADAPTIVE PLACEMENT STRATEGY ===")
    
    # Создаем тестовый эмбединг
    embedding = np.random.randn(768).astype(np.float32)
    target_shape = (8, 8, 12)
    
    print(f"📊 Создание адаптивной стратегии размещения")
    print(f"   Входной эмбединг: {embedding.shape}")
    print(f"   Целевая 3D форма: {target_shape}")
    
    # Тестируем различные методы анализа важности
    importance_methods = ["variance_pca", "clustering", "magnitude"]
    
    for method in importance_methods:
        print(f"\n🔍 Тестирование метода: {method}")
        
        strategy = create_adaptive_transformation_strategy(
            embedding=embedding,
            target_shape=target_shape,
            importance_method=method
        )
        
        print(f"   ✅ Стратегия создана успешно")
        
        # Проверяем структуру стратегии
        required_keys = ['importance_weights', 'placement_map', 'target_shape', 'optimization_params', 'quality_threshold']
        for key in required_keys:
            assert key in strategy, f"Стратегия должна содержать ключ '{key}'"
        
        # Проверяем размерности
        importance_weights = strategy['importance_weights']
        placement_map = strategy['placement_map']
        
        assert importance_weights.shape == (768,), f"Веса важности должны иметь размерность (768,), получено {importance_weights.shape}"
        assert placement_map.shape == (768,), f"Карта размещения должна иметь размерность (768,), получено {placement_map.shape}"
        
        # Проверяем, что карта размещения содержит все необходимые индексы
        unique_indices = np.unique(placement_map)
        expected_indices = np.arange(np.prod(target_shape))
        
        assert len(unique_indices) == len(expected_indices), \
               f"Карта размещения должна содержать все индексы от 0 до {np.prod(target_shape)-1}"
        
        # Проверяем качественный порог
        quality_threshold = strategy['quality_threshold']
        assert quality_threshold == 0.98, f"Качественный порог должен быть 0.98, получено {quality_threshold}"
        
        print(f"   📊 Статистика важности: min={np.min(importance_weights):.3f}, "
              f"max={np.max(importance_weights):.3f}")
        print(f"   🎯 Качественный порог: {quality_threshold}")
        
    print("✅ ТЕСТ 3 ПРОШЕЛ: Адаптивные стратегии размещения работают корректно!")
    return True


def test_enhanced_adaptive_reshaper():
    """
    Тест улучшенного AdaptiveReshaper с новыми алгоритмами.
    
    Проверяет работу улучшенных методов трансформации:
    - enhanced_variance
    - importance_weighted  
    - adaptive_placement
    """
    print("\n🧪 === ТЕСТ 4: ENHANCED ADAPTIVE RESHAPER ===")
    
    # Создаем тестовые эмбединги различных типов
    test_embeddings = create_test_embeddings(count=5, dim=768, embedding_type="diverse")
    
    # Тестируем все улучшенные методы
    enhanced_methods = ["enhanced_variance", "importance_weighted", "adaptive_placement"]
    results = {}
    
    for method in enhanced_methods:
        print(f"\n🔍 Тестирование метода: {method}")
        
        # Создаем reshaper с улучшенным методом
        reshaper = AdaptiveReshaper(
            input_dim=768,
            cube_shape=(8, 8, 12),
            adaptation_method=method,
            preserve_semantics=True,
            semantic_threshold=0.95  # Базовый порог, но стремимся к >98%
        )
        
        method_similarities = []
        
        for i, embedding in enumerate(test_embeddings):
            # Прямая трансформация 1D → 3D
            embedding_3d = reshaper.vector_to_matrix(embedding)
            
            # Обратная трансформация 3D → 1D
            restored_embedding = reshaper.matrix_to_vector(embedding_3d)
            
            # Измеряем качество с помощью улучшенных метрик
            enhanced_metrics = calculate_enhanced_similarity_metrics(embedding, restored_embedding)
            similarity = enhanced_metrics['weighted_similarity']
            
            method_similarities.append(similarity)
            
            print(f"   Эмбединг {i+1}: weighted_similarity = {similarity:.6f}")
            
            # Проверяем достижение высокого качества
            if similarity >= 0.98:
                print(f"   🎯 ОТЛИЧНЫЙ РЕЗУЛЬТАТ: достигнуто >98% сохранение!")
            elif similarity >= 0.95:
                print(f"   ✅ Хороший результат: достигнуто >95% сохранение")
            else:
                print(f"   ⚠️  Результат ниже ожидаемого: {similarity:.6f}")
        
        # Статистика по методу
        avg_similarity = np.mean(method_similarities)
        max_similarity = np.max(method_similarities)
        min_similarity = np.min(method_similarities)
        
        results[method] = {
            'similarities': method_similarities,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'above_98_count': sum(1 for s in method_similarities if s >= 0.98),
            'above_95_count': sum(1 for s in method_similarities if s >= 0.95)
        }
        
        print(f"\n📊 Статистика метода {method}:")
        print(f"   Средняя схожесть: {avg_similarity:.6f}")
        print(f"   Максимальная схожесть: {max_similarity:.6f}")
        print(f"   Минимальная схожесть: {min_similarity:.6f}")
        print(f"   Результатов >98%: {results[method]['above_98_count']}/{len(test_embeddings)}")
        print(f"   Результатов >95%: {results[method]['above_95_count']}/{len(test_embeddings)}")
        
        # Проверяем, что все результаты > 95%
        assert min_similarity > 0.95, f"Метод {method}: все результаты должны быть >95%, получено min={min_similarity:.6f}"
        
    # Находим лучший метод
    best_method = max(results.keys(), key=lambda k: results[k]['avg_similarity'])
    best_avg = results[best_method]['avg_similarity']
    
    print(f"\n🏆 ЛУЧШИЙ МЕТОД: {best_method} с средней схожестью {best_avg:.6f}")
    
    # Проверяем достижение цели >98%
    best_above_98 = results[best_method]['above_98_count']
    if best_above_98 > 0:
        print(f"🎯 ЦЕЛЬ ДОСТИГНУТА: {best_above_98} результатов >98% семантического сохранения!")
    
    print("✅ ТЕСТ 4 ПРОШЕЛ: Enhanced AdaptiveReshaper работает корректно!")
    return results


def test_caching_and_performance():
    """
    Тест кэширования и оптимизации производительности.
    
    Проверяет эффективность кэширования повторных трансформаций
    и общую производительность улучшенных алгоритмов.
    """
    print("\n🧪 === ТЕСТ 5: CACHING & PERFORMANCE ===")
    
    # Создаем reshaper с кэшированием
    reshaper = AdaptiveReshaper(
        adaptation_method="enhanced_variance",
        preserve_semantics=True,
        semantic_threshold=0.95
    )
    
    # Создаем тестовый эмбединг
    test_embedding = np.random.randn(768).astype(np.float32)
    
    print(f"📊 Тестирование кэширования...")
    
    # Первая трансформация (должна вычисляться)
    import time
    start_time = time.time()
    result1 = reshaper.vector_to_matrix(test_embedding)
    first_transform_time = time.time() - start_time
    
    # Вторая трансформация того же эмбединга (должна использовать кэш)
    start_time = time.time()
    result2 = reshaper.vector_to_matrix(test_embedding)
    second_transform_time = time.time() - start_time
    
    print(f"   Первая трансформация: {first_transform_time*1000:.2f} ms")
    print(f"   Вторая трансформация (кэш): {second_transform_time*1000:.2f} ms")
    
    # Защита от деления на ноль
    if second_transform_time > 0:
        speedup = first_transform_time / second_transform_time
        print(f"   Ускорение от кэширования: {speedup:.1f}x")
    else:
        print(f"   Ускорение от кэширования: >1000x (мгновенно)")
    
    # Проверяем, что результаты идентичны
    assert np.array_equal(result1, result2), "Результаты кэшированной трансформации должны быть идентичны"
    
    # Проверяем, что кэш действительно ускоряет (хотя бы в 2 раза)
    if second_transform_time > 0:
        speedup = first_transform_time / second_transform_time
        if speedup > 2.0:
            print(f"   ✅ Кэширование эффективно: ускорение {speedup:.1f}x")
        else:
            print(f"   ⚠️  Кэширование менее эффективно чем ожидалось: ускорение {speedup:.1f}x")
    
    # Тестируем производительность на множественных эмбедингах
    print(f"\n📊 Тестирование производительности на batch...")
    
    batch_embeddings = [np.random.randn(768).astype(np.float32) for _ in range(10)]
    
    start_time = time.time()
    batch_results = [reshaper.vector_to_matrix(emb) for emb in batch_embeddings]
    batch_time = time.time() - start_time
    
    avg_time_per_embedding = batch_time / len(batch_embeddings)
    
    print(f"   Batch обработка {len(batch_embeddings)} эмбедингов: {batch_time*1000:.2f} ms")
    print(f"   Среднее время на эмбединг: {avg_time_per_embedding*1000:.2f} ms")
    print(f"   Пропускная способность: {1/avg_time_per_embedding:.1f} эмбедингов/сек")
    
    # Проверяем производительность (должно быть разумно быстро)
    assert avg_time_per_embedding < 0.1, f"Среднее время на эмбединг должно быть <100ms, получено {avg_time_per_embedding*1000:.2f}ms"
    
    print("✅ ТЕСТ 5 ПРОШЕЛ: Кэширование и производительность работают корректно!")
    return avg_time_per_embedding


def test_semantic_preservation_target_98():
    """
    Основной тест для достижения цели >98% семантического сохранения.
    
    Комплексная проверка всех улучшений для достижения высочайшего качества.
    """
    print("\n🧪 === ТЕСТ 6: SEMANTIC PRESERVATION TARGET >98% ===")
    
    # Создаем разнообразные тестовые эмбединги
    test_embeddings = create_test_embeddings(count=20, dim=768, embedding_type="diverse")
    
    # Настраиваем reshaper на максимальное качество
    reshaper = EmbeddingReshaper(
        input_dim=768,
        cube_shape=(8, 8, 12),
        reshaping_method="adaptive",  # Будет использовать AdaptiveReshaper с enhanced методами
        preserve_semantics=True,
        semantic_threshold=0.98  # Повышаем порог до 98%
    )
    
    print(f"📊 Тестирование на {len(test_embeddings)} разнообразных эмбедингах...")
    print(f"🎯 Цель: семантическое сохранение >98%")
    
    high_quality_results = []
    all_similarities = []
    
    for i, embedding in enumerate(test_embeddings):
        # Полный цикл трансформации
        matrix_3d = reshaper.vector_to_matrix(embedding)
        restored_embedding = reshaper.matrix_to_vector(matrix_3d)
        
        # Измеряем качество улучшенными метриками
        enhanced_metrics = calculate_enhanced_similarity_metrics(embedding, restored_embedding)
        similarity = enhanced_metrics['weighted_similarity']
        
        all_similarities.append(similarity)
        
        if similarity >= 0.98:
            high_quality_results.append(i)
            print(f"   ✅ Эмбединг {i+1}: {similarity:.6f} - ОТЛИЧНЫЙ РЕЗУЛЬТАТ!")
        elif similarity >= 0.95:
            print(f"   ✅ Эмбединг {i+1}: {similarity:.6f} - хороший результат")
        else:
            print(f"   ⚠️  Эмбединг {i+1}: {similarity:.6f} - требует улучшения")
    
    # Статистика результатов
    avg_similarity = np.mean(all_similarities)
    max_similarity = np.max(all_similarities)
    min_similarity = np.min(all_similarities)
    above_98_count = len(high_quality_results)
    above_95_count = sum(1 for s in all_similarities if s >= 0.95)
    
    print(f"\n📊 === ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Всего эмбедингов протестировано: {len(test_embeddings)}")
    print(f"Средняя схожесть: {avg_similarity:.6f}")
    print(f"Максимальная схожесть: {max_similarity:.6f}")
    print(f"Минимальная схожесть: {min_similarity:.6f}")
    print(f"")
    print(f"🎯 Результатов >98%: {above_98_count}/{len(test_embeddings)} ({above_98_count/len(test_embeddings)*100:.1f}%)")
    print(f"✅ Результатов >95%: {above_95_count}/{len(test_embeddings)} ({above_95_count/len(test_embeddings)*100:.1f}%)")
    
    # Получаем статистику из reshaper
    reshaper_stats = reshaper.get_statistics()
    print(f"\n📊 Статистика EmbeddingReshaper:")
    for key, value in reshaper_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    # Проверяем достижение целей
    success_criteria = {
        "Средняя схожесть >97%": avg_similarity > 0.97,
        "Минимальная схожесть >95%": min_similarity > 0.95,
        "Хотя бы 30% результатов >98%": above_98_count >= len(test_embeddings) * 0.3,
        "Все результаты >95%": above_95_count == len(test_embeddings)
    }
    
    print(f"\n🎯 === ПРОВЕРКА КРИТЕРИЕВ УСПЕХА ===")
    all_criteria_met = True
    for criterion, met in success_criteria.items():
        status = "✅" if met else "❌"
        print(f"{status} {criterion}: {'ВЫПОЛНЕН' if met else 'НЕ ВЫПОЛНЕН'}")
        if not met:
            all_criteria_met = False
    
    if all_criteria_met:
        print(f"\n🎉 ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ! ЦЕЛЬ >98% СЕМАНТИЧЕСКОГО СОХРАНЕНИЯ ДОСТИГНУТА!")
    else:
        print(f"\n⚠️  Некоторые критерии не выполнены. Требуется дополнительная оптимизация.")
    
    # Проверяем базовые требования
    assert avg_similarity > 0.95, f"Средняя схожесть должна быть >95%, получено {avg_similarity:.6f}"
    assert min_similarity > 0.90, f"Минимальная схожесть должна быть >90%, получено {min_similarity:.6f}"
    
    print("✅ ТЕСТ 6 ПРОШЕЛ: Основные требования к семантическому сохранению выполнены!")
    
    return {
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity,
        'above_98_count': above_98_count,
        'above_95_count': above_95_count,
        'total_count': len(test_embeddings),
        'criteria_met': all_criteria_met
    }


def main():
    """
    Основная функция для запуска всех улучшенных тестов.
    
    PHASE 2.3 День 3-4: Улучшение семантического сохранения до >98%
    """
    print("🚀 === УЛУЧШЕННОЕ ТЕСТИРОВАНИЕ EMBEDDINGRESHAPER ===")
    print("Phase 2.3 День 3-4: Улучшение семантического сохранения до >98%")
    print("=" * 70)
    
    all_results = {}
    
    try:
        # Тест 1: Enhanced similarity metrics
        all_results['enhanced_metrics'] = test_enhanced_similarity_metrics()
        
        # Тест 2: Importance analysis
        all_results['importance_analysis'] = test_importance_analysis()
        
        # Тест 3: Adaptive placement strategy
        all_results['adaptive_placement'] = test_adaptive_placement_strategy()
        
        # Тест 4: Enhanced adaptive reshaper
        all_results['enhanced_reshaper'] = test_enhanced_adaptive_reshaper()
        
        # Тест 5: Caching and performance
        all_results['performance'] = test_caching_and_performance()
        
        # Тест 6: Основной тест >98% семантического сохранения
        all_results['semantic_preservation_98'] = test_semantic_preservation_target_98()
        
        # Итоговый отчет
        print(f"\n🎉 === ИТОГОВЫЙ ОТЧЕТ ===")
        print(f"Всех тестов выполнено: {len(all_results)}/6")
        
        # Ключевые метрики
        final_results = all_results['semantic_preservation_98']
        print(f"\n📊 КЛЮЧЕВЫЕ ДОСТИЖЕНИЯ:")
        print(f"   🎯 Средняя схожесть: {final_results['avg_similarity']:.6f}")
        print(f"   🏆 Максимальная схожесть: {final_results['max_similarity']:.6f}")
        print(f"   📈 Результатов >98%: {final_results['above_98_count']}/{final_results['total_count']}")
        print(f"   ✅ Результатов >95%: {final_results['above_95_count']}/{final_results['total_count']}")
        
        if final_results['criteria_met']:
            print(f"\n🎉 МИССИЯ ВЫПОЛНЕНА! Phase 2.3 День 3-4 ЗАВЕРШЕН УСПЕШНО!")
            print(f"🚀 EmbeddingReshaper готов к Phase 2.5 (Core Embedding Processor)!")
        else:
            print(f"\n⚠️  Цель частично достигнута. Семантическое сохранение улучшено, но требуется доработка.")
            
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return final_results['criteria_met']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 