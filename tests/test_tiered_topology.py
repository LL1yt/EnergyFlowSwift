"""
Тест для новой трехуровневой топологии (Tiered Topology)
=========================================================

Этот скрипт проверяет, что рефакторинг модуля `core.lattice_3d`
прошел успешно и что новая трехуровневая стратегия соседства
инициализируется и работает корректно.

Что проверяется:
1. Корректная загрузка конфигурации с `tiered` стратегией.
2. Успешное создание `Lattice3D` с новой архитектурой.
3. Возможность выполнить один шаг симуляции (`forward` pass).
4. Базовая валидность состояний и весов после шага.
"""

import torch
import logging
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
# для корректного импорта модулей
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    from core.lattice_3d import create_lattice_from_config, Lattice3D
except ImportError as e:
    logging.error(f"Не удалось импортировать компоненты из core.lattice_3d: {e}")
    logging.error("Убедитесь, что __init__.py в core/lattice_3d настроен правильно.")
    sys.exit(1)


def run_test(config_path: str):
    """
    Основная функция для запуска теста.
    """
    logging.info(f"--- Запуск теста для конфигурации: {config_path} ---")

    if not Path(config_path).exists():
        logging.error(f"Файл конфигурации не найден: {config_path}")
        return

    try:
        # 1. Создание решетки из конфигурации
        logging.info("1. Создание Lattice3D из файла конфигурации...")
        lattice = create_lattice_from_config(config_path)
        logging.info("   ✅ Lattice3D успешно создан.")
        logging.info(f"   - Класс решетки: {type(lattice)}")
        logging.info(f"   - Устройство: {lattice.device}")
        logging.info(
            f"   - Стратегия соседства: {lattice.config.neighbor_finding_strategy}"
        )

        # 2. Проверка базовых параметров
        logging.info("2. Проверка инициализированных параметров...")
        assert isinstance(lattice, Lattice3D)
        assert lattice.config.neighbor_finding_strategy == "tiered"
        assert lattice.states.shape == (lattice.config.total_cells, lattice.state_size)
        assert lattice.connection_weights.shape == (
            lattice.config.total_cells,
            lattice.config.neighbors,
        )
        logging.info("   ✅ Параметры инициализированы корректно.")

        # 3. Выполнение одного шага симуляции
        logging.info("3. Выполнение одного шага forward pass...")
        initial_states_norm = torch.norm(lattice.get_states()).item()

        lattice.forward(external_inputs=None)

        final_states_norm = torch.norm(lattice.get_states()).item()
        logging.info("   ✅ Forward pass выполнен без ошибок.")

        # 4. Проверка результатов
        logging.info("4. Проверка состояния после шага симуляции...")
        assert not torch.isnan(
            lattice.get_states()
        ).any(), "Найдены NaN значения в состояниях"
        assert torch.isfinite(
            lattice.get_states()
        ).all(), "Найдены inf значения в состояниях"
        logging.info(f"   - Норма состояний (до): {initial_states_norm:.4f}")
        logging.info(f"   - Норма состояний (после): {final_states_norm:.4f}")
        assert (
            initial_states_norm != final_states_norm
        ), "Состояния не изменились после forward pass"
        logging.info("   ✅ Состояния изменились, NaN/inf отсутствуют.")

        # 5. Вывод статистики
        perf_stats = lattice.get_performance_stats()
        io_info = lattice.get_io_point_info()
        logging.info("5. Статистика выполнения:")
        logging.info(
            f"   - Время выполнения шага: {perf_stats['avg_time_per_step'] * 1000:.4f} ms"
        )
        logging.info(f"   - Количество входных точек: {io_info['input_points_count']}")
        logging.info(
            f"   - Количество выходных точек: {io_info['output_points_count']}"
        )

        logging.info(
            "\n--- [SUCCESS] Тест трехуровневой топологии успешно пройден! ---"
        )

    except Exception as e:
        logging.error(
            f"\n--- [FAILURE] Тест провалился на одном из этапов. ---", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    # Указываем путь к нашей новой тестовой конфигурации
    test_config_file = root_dir / "config" / "hybrid_neighbor_test.yaml"
    run_test(str(test_config_file))
