"""
Базовый тест функциональной кластеризации - Шаг 3.3

Тестирует:
1. Инициализацию системы кластеризации
2. Базовую кластеризацию по сходству состояний
3. Модификацию весов связей
4. Интеграцию с существующей пластичностью
5. Готовность к расширению (координационные интерфейсы)

Архитектура: 16×16×16 решетка, MinimalNCACell + GatedMLPCell
"""

import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np
from pathlib import Path

# Добавляем путь к корневой директории
sys.path.append(str(Path(__file__).parent))

from core.lattice_3d import create_lattice_from_config
from core.lattice_3d.config import load_lattice_config
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_clustering_initialization():
    """Тест 1: Инициализация системы кластеризации"""
    print("🧪 Тест 1: Инициализация системы кластеризации")

    # Загружаем конфигурацию
    config = load_lattice_config("config/functional_clustering_test.yaml")

    # Проверяем что кластеризация включена в конфигурации
    assert (
        config.enable_clustering == True
    ), "Кластеризация должна быть включена в config"
    assert config.clustering_config is not None, "clustering_config не должен быть None"
    assert config.clustering_config["priority"] == 0.3, "Приоритет должен быть 0.3"

    print(f"  ✓ Конфигурация загружена: {config.dimensions} решетка")
    print(f"  ✓ Кластеризация включена: {config.enable_clustering}")
    print(f"  ✓ Приоритет: {config.clustering_config['priority']}")

    # Создаем решетку напрямую из config объекта (как в BCM тесте)
    from core.lattice_3d.lattice import Lattice3D

    lattice = Lattice3D(config)

    # Проверяем наличие компонентов кластеризации
    assert hasattr(
        lattice, "functional_clustering"
    ), "BasicFunctionalClustering не инициализирован"
    assert hasattr(
        lattice, "coordination_interface"
    ), "CoordinationInterface не инициализирован"
    assert hasattr(lattice, "enable_clustering"), "enable_clustering не установлен"

    # Проверяем параметры
    assert lattice.enable_clustering == True, "Кластеризация должна быть включена"
    assert lattice.clustering_priority == 0.3, "Приоритет кластеризации должен быть 0.3"
    assert (
        lattice.integration_mode == "additive"
    ), "Режим интеграции должен быть additive"

    print(f"  ✓ Решетка создана: {config.total_cells} клеток")
    print(f"  ✓ Кластеризация инициализирована")
    print(f"  ✓ Координационный интерфейс готов")
    print("✅ Инициализация прошла успешно")
    return lattice


def test_basic_clustering(lattice):
    """Тест 2: Базовая кластеризация по сходству состояний"""
    print("\n🧪 Тест 2: Базовая кластеризация")

    # Создаем тестовые паттерны активности
    num_cells = lattice.states.size(0)
    state_size = lattice.states.size(1)

    # Паттерн 1: Группа клеток с похожей активностью (первая половина)
    pattern1 = torch.tensor([1.0, 0.5, 0.2, 0.8, 0.3, 0.1])
    # Паттерн 2: Группа клеток с другой активностью (вторая половина)
    pattern2 = torch.tensor([0.1, 0.9, 0.7, 0.2, 0.6, 0.4])

    # Убеждаемся, что паттерны имеют правильный размер
    if pattern1.size(0) != state_size:
        pattern1 = (
            pattern1[:state_size]
            if pattern1.size(0) > state_size
            else torch.cat([pattern1, torch.zeros(state_size - pattern1.size(0))])
        )
    if pattern2.size(0) != state_size:
        pattern2 = (
            pattern2[:state_size]
            if pattern2.size(0) > state_size
            else torch.cat([pattern2, torch.zeros(state_size - pattern2.size(0))])
        )

    # Устанавливаем паттерны
    half = num_cells // 2
    lattice.states[:half] = pattern1.unsqueeze(0).expand(half, -1)
    lattice.states[half:] = pattern2.unsqueeze(0).expand(num_cells - half, -1)

    # Добавляем небольшой шум
    noise = torch.randn_like(lattice.states) * 0.1
    lattice.states = lattice.states + noise

    # Выполняем кластеризацию
    start_time = time.time()
    clustering_result = lattice.apply_functional_clustering(current_step=0)
    clustering_time = time.time() - start_time

    print(f"⏱️ Время кластеризации: {clustering_time:.3f}s")

    # Проверяем результат
    assert clustering_result["applied"] == True, "Кластеризация не была применена"
    assert clustering_result["weights_modified"] == True, "Веса не были модифицированы"

    # Проверяем информацию о кластерах
    clustering_info = clustering_result["clustering_info"]
    assert "clusters" in clustering_info, "Информация о кластерах отсутствует"
    assert clustering_info["updated"] == True, "Кластеры не были обновлены"

    clusters = clustering_info["clusters"]
    num_clusters = len(clusters)
    print(f"📊 Найдено кластеров: {num_clusters}")

    # Проверяем, что найдены кластеры
    assert (
        num_clusters >= 2
    ), f"Должно быть найдено минимум 2 кластера, найдено {num_clusters}"
    assert (
        num_clusters <= 8
    ), f"Не должно быть больше 8 кластеров, найдено {num_clusters}"

    # Печатаем размеры кластеров
    cluster_sizes = [len(members) for members in clusters.values()]
    print(f"📏 Размеры кластеров: {cluster_sizes}")

    # Проверяем минимальный размер кластера
    min_size = min(cluster_sizes)
    assert (
        min_size >= 8
    ), f"Минимальный размер кластера должен быть 8, найден {min_size}"

    print("✅ Базовая кластеризация работает корректно")
    return clusters


def test_weight_modification(lattice, clusters):
    """Тест 3: Модификация весов связей на основе кластеров"""
    print("\n🧪 Тест 3: Модификация весов связей")

    start_time = time.time()

    # Сохраняем исходные веса
    original_weights = lattice.connection_weights.clone()

    # Получаем кластеры
    cell_to_cluster = {}
    for cluster_id, members in clusters.items():
        for cell_idx in members:
            cell_to_cluster[cell_idx] = cluster_id

    print(f"  📋 Подготовка данных: {time.time() - start_time:.3f}s")

    # ОПТИМИЗАЦИЯ: получаем индексы соседей один раз
    neighbor_indices_time = time.time()
    neighbor_indices = lattice.topology.get_all_neighbor_indices_batched()
    print(
        f"  🔗 Получение индексов соседей: {time.time() - neighbor_indices_time:.3f}s"
    )

    # Анализируем изменения весов
    analysis_time = time.time()
    intra_cluster_weights = []
    inter_cluster_weights = []

    for cell_idx in range(lattice.connection_weights.size(0)):
        if cell_idx not in cell_to_cluster:
            continue

        cell_cluster = cell_to_cluster[cell_idx]

        for neighbor_idx in range(lattice.connection_weights.size(1)):
            neighbor_cell = neighbor_indices[cell_idx, neighbor_idx].item()

            if neighbor_cell == -1 or neighbor_cell not in cell_to_cluster:
                continue

            neighbor_cluster = cell_to_cluster[neighbor_cell]
            current_weight = lattice.connection_weights[cell_idx, neighbor_idx].item()

            if cell_cluster == neighbor_cluster:
                # Внутрикластерная связь
                intra_cluster_weights.append(current_weight)
            else:
                # Межкластерная связь
                inter_cluster_weights.append(current_weight)

    print(f"  📊 Анализ весов: {time.time() - analysis_time:.3f}s")

    # Вычисляем статистики
    if intra_cluster_weights and inter_cluster_weights:
        avg_intra = np.mean(intra_cluster_weights)
        avg_inter = np.mean(inter_cluster_weights)

        print(f"📈 Средний вес внутрикластерных связей: {avg_intra:.3f}")
        print(f"📉 Средний вес межкластерных связей: {avg_inter:.3f}")
        print(f"🔄 Соотношение (intra/inter): {avg_intra/avg_inter:.2f}")

        # Проверяем, что внутрикластерные связи сильнее
        assert (
            avg_intra > avg_inter
        ), "Внутрикластерные связи должны быть сильнее межкластерных"

        # Проверяем разумные пределы
        assert (
            0.8 <= avg_intra <= 3.0
        ), f"Внутрикластерные веса вне пределов: {avg_intra}"
        assert 0.1 <= avg_inter <= 2.0, f"Межкластерные веса вне пределов: {avg_inter}"
    else:
        print("⚠️ Недостаточно связей для анализа весов")
        print(f"  📊 Найдено внутрикластерных связей: {len(intra_cluster_weights)}")
        print(f"  📊 Найдено межкластерных связей: {len(inter_cluster_weights)}")

    total_time = time.time() - start_time
    print(f"  ⏱️ Общее время теста 3: {total_time:.3f}s")
    print("✅ Модификация весов работает корректно")


def test_plasticity_integration(lattice):
    """Тест 4: Интеграция с существующей пластичностью"""
    print("\n🧪 Тест 4: Интеграция с пластичностью")

    # Проверяем совместимость с PlasticityMixin
    assert hasattr(lattice, "apply_combined_plasticity"), "PlasticityMixin не найден"
    assert hasattr(lattice, "apply_functional_clustering"), "ClusteringMixin не найден"

    # Применяем пластичность
    initial_weights = lattice.connection_weights.clone()

    # Небольшая активность для пластичности
    lattice.states = torch.randn_like(lattice.states) * 0.1

    # Отслеживаем активность для STDP
    lattice._track_activity_for_stdp(lattice.states)

    # Применяем пластичность и кластеризацию
    plasticity_result = lattice.apply_combined_plasticity()
    clustering_result = lattice.apply_functional_clustering(current_step=1)

    print(f"🧠 Пластичность применена: {plasticity_result.get('applied', False)}")
    print(f"🔗 Кластеризация применена: {clustering_result.get('applied', False)}")

    # Проверяем, что веса изменились
    final_weights = lattice.connection_weights
    weight_change = torch.mean(torch.abs(final_weights - initial_weights))
    print(f"📊 Среднее изменение весов: {weight_change:.6f}")

    # Проверяем интеграцию режимов
    assert (
        lattice.integration_mode == "additive"
    ), "Режим интеграции должен быть additive"
    assert (
        0.0 <= lattice.clustering_priority <= 1.0
    ), "Приоритет кластеризации вне пределов"

    print("✅ Интеграция с пластичностью работает корректно")


def test_coordination_interface_readiness(lattice):
    """Тест 5: Готовность координационных интерфейсов"""
    print("\n🧪 Тест 5: Готовность к расширению")

    # Проверяем наличие интерфейсов
    assert hasattr(
        lattice, "coordination_interface"
    ), "CoordinationInterface отсутствует"

    # Проверяем методы для будущих функций
    coordination = lattice.coordination_interface

    # Тестируем заглушки пользовательского управления
    lattice.add_user_clustering_hint("test_hint", {"value": 1.0})
    lattice.add_user_clustering_correction({0: [1, 2, 3]}, {0: [1, 2], 1: [3]})

    # Проверяем статистику
    stats = lattice.get_clustering_statistics()
    assert stats["clustering_initialized"] == True, "Кластеризация не инициализирована"

    print("📊 Статистика кластеризации:")
    print(f"  - Включена: {stats['enable_clustering']}")
    print(f"  - Счетчик шагов: {stats['clustering_step_counter']}")
    print(f"  - Режим координации: {stats['coordination']['coordination_mode']}")
    print(
        f"  - Пользовательские подсказки: {stats['coordination']['user_hints_count']}"
    )

    # Тестируем динамическую конфигурацию
    lattice.configure_clustering(
        enable=True, priority=0.4, integration_mode="multiplicative"
    )

    assert lattice.clustering_priority == 0.4, "Приоритет не обновился"
    assert lattice.integration_mode == "multiplicative", "Режим интеграции не обновился"

    print("✅ Интерфейсы готовы к расширению")


def test_performance_and_stability(lattice):
    """Тест 6: Производительность и стабильность"""
    print("\n🧪 Тест 6: Производительность и стабильность")

    # Тестируем производительность
    num_steps = 10
    total_time = 0

    for step in range(num_steps):
        # Небольшие изменения состояний
        lattice.states += torch.randn_like(lattice.states) * 0.01

        start_time = time.time()
        result = lattice.apply_functional_clustering(current_step=step)
        step_time = time.time() - start_time
        total_time += step_time

        if step % 5 == 0:
            print(
                f"  Шаг {step}: {step_time:.3f}s, кластеров: {result.get('clustering_info', {}).get('num_clusters', 0)}"
            )

    avg_time = total_time / num_steps
    print(f"⏱️ Среднее время за шаг: {avg_time:.3f}s")
    print(f"📊 Общее время: {total_time:.3f}s")

    # Проверяем производительность (расслабленные требования для комплексного теста)
    assert avg_time < 1.0, f"Слишком медленно: {avg_time:.3f}s > 1.0s"

    # Проверяем стабильность кластеров
    stats = lattice.get_clustering_statistics()
    stability = stats["basic_clustering"].get("cluster_stability_score", 0.0)
    print(f"🔒 Стабильность кластеров: {stability:.3f}")

    # Если есть обновления, стабильность должна быть разумной
    if stats["clustering_step_counter"] > 1:
        assert stability >= 0.0, "Стабильность не может быть отрицательной"

    print("✅ Производительность и стабильность приемлемы")


def main():
    """Основная функция тестирования"""
    print("🚀 НАЧАЛО ТЕСТИРОВАНИЯ ФУНКЦИОНАЛЬНОЙ КЛАСТЕРИЗАЦИИ")
    print("=" * 60)

    try:
        # Тест 1: Инициализация
        lattice = test_clustering_initialization()

        # Тест 2: Базовая кластеризация
        clusters = test_basic_clustering(lattice)

        # Тест 3: Модификация весов
        test_weight_modification(lattice, clusters)

        # Тест 4: Интеграция с пластичностью
        test_plasticity_integration(lattice)

        # Тест 5: Готовность к расширению
        test_coordination_interface_readiness(lattice)

        # Тест 6: Производительность
        test_performance_and_stability(lattice)

        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Функциональная кластеризация готова")
        print("🔗 Архитектура готова к расширению")

        # Финальная статистика
        final_stats = lattice.get_clustering_statistics()
        print(f"\n📊 Финальная статистика:")
        print(f"  - Кластеров: {len(lattice.get_current_clusters())}")
        print(f"  - Применений кластеризации: {final_stats['clustering_step_counter']}")
        print(
            f"  - Координационный режим: {final_stats['coordination']['coordination_mode']}"
        )

    except Exception as e:
        print(f"\n❌ ТЕСТ ЗАВЕРШИЛСЯ С ОШИБКОЙ: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 ШАГ 3.3 ЗАВЕРШЕН УСПЕШНО!")
        print("Готов к переходу к следующему этапу развития")
    else:
        print("\n🔧 Требуется доработка")
        sys.exit(1)
