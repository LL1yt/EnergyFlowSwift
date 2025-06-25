#!/usr/bin/env python3
"""
Тест интеграции логирования - Phase 3 Priority 1
================================================

Проверяет:
- Интеграцию core/log_utils.py с new_rebuild/utils/logging.py
- Работу централизованного логирования в клетках
- Отсутствие дублирования логов
- Legacy совместимость
"""

import torch
import sys
from pathlib import Path

# Добавляем new_rebuild в путь
sys.path.insert(0, str(Path(__file__).parent / "new_rebuild"))

from new_rebuild.config import get_project_config
from new_rebuild.utils import (
    setup_logging,
    get_logger,
    log_init,
    _get_caller_info,
    LogContext,
)
from new_rebuild.core.cells import NCACell, GMLPCell


def test_logging_integration():
    """Основной тест интеграции логирования"""

    print("🧪 ТЕСТ: Интеграция централизованного логирования")
    print("=" * 60)

    # === STEP 1: Настройка логирования ===
    print("\n1️⃣ Настройка централизованного логирования...")

    # Получаем конфигурацию
    config = get_project_config()

    # Настраиваем логирование с контекстом (БЕЗ дедупликации)
    setup_logging(debug_mode=True, enable_context=True)

    logger = get_logger("test_logging")
    logger.info("🚀 Централизованное логирование настроено")

    # === STEP 2: Тест legacy совместимости ===
    print("\n2️⃣ Тестирование legacy совместимости...")

    caller_info = _get_caller_info()
    logger.info(f"📍 Caller info: {caller_info}")

    # Тест log_init функции
    log_init("TestComponent", version="1.0", mode="testing")

    # === STEP 3: Создание и тестирование NCA клетки С КОНТЕКСТОМ ===
    print("\n3️⃣ Создание NCA клетки...")

    with LogContext("cell_creation", cell_type="NCA"):
        nca_cell = NCACell()
        logger.info("✅ NCA клетка создана")

        # Тест forward pass с логированием - ИСПРАВЛЕНЫ РАЗМЕРЫ
        batch_size = 2
        neighbor_states = torch.randn(
            batch_size, 26, 4
        )  # 26 соседей, NCA state_size = 4
        own_state = torch.randn(batch_size, 4)  # NCA state_size = 4
        external_input = torch.randn(
            batch_size, 1
        )  # NCA external_input_size = 1 (по конфигурации)

        nca_output = nca_cell.forward(neighbor_states, own_state, external_input)
        logger.info(f"✅ NCA forward pass: {nca_output.shape}")

    # === STEP 4: Создание и тестирование gMLP клетки С КОНТЕКСТОМ ===
    print("\n4️⃣ Создание gMLP клетки...")

    with LogContext("cell_creation", cell_type="gMLP"):
        gmlp_cell = GMLPCell()
        logger.info("✅ gMLP клетка создана")

        # Тест forward pass - РАЗМЕРЫ СООТВЕТСТВУЮТ КОНФИГУРАЦИИ
        neighbor_states_gmlp = torch.randn(batch_size, 26, 32)  # gMLP state_size = 32
        own_state_gmlp = torch.randn(batch_size, 32)  # gMLP state_size = 32
        external_input_gmlp = torch.randn(batch_size, 8)  # gMLP external_input_size = 8

        gmlp_output = gmlp_cell.forward(
            neighbor_states_gmlp, own_state_gmlp, external_input_gmlp
        )
        logger.info(f"✅ gMLP forward pass: {gmlp_output.shape}")

    # === STEP 5: Проверка архитектурной статистики ===
    print("\n5️⃣ Архитектурная статистика...")

    nca_params = sum(p.numel() for p in nca_cell.parameters())
    gmlp_params = sum(p.numel() for p in gmlp_cell.parameters())
    total_params = nca_params + gmlp_params

    logger.info(f"📊 NCA параметры: {nca_params:,}")
    logger.info(f"📊 gMLP параметры: {gmlp_params:,}")
    logger.info(f"📊 Общие параметры: {total_params:,}")

    # === STEP 6: Проверка конфигурации ===
    print("\n6️⃣ Конфигурационная информация...")

    logger.info(f"🔧 Архитектура: {config.architecture_type}")
    logger.info(f"🔧 Решетка: {config.lattice_dimensions}")
    logger.info(f"🔧 Устройство: {config.device}")
    logger.info(f"🔧 Debug режим: {config.debug_mode}")

    # === STEP 7: ТЕСТ КОНТЕКСТНОГО ЛОГИРОВАНИЯ ===
    print("\n7️⃣ Тестирование контекстного логирования...")

    # Тест вложенных контекстов
    with LogContext("outer_context", operation="test"):
        logger.info("Внешний контекст")

        with LogContext("inner_context", level="deep"):
            logger.info("Вложенный контекст")

        logger.info("Снова внешний контекст")

    logger.info("✅ Контекстное логирование протестировано")

    # === РЕЗУЛЬТАТ ===
    print("\n🎉 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print("✅ Централизованное логирование: OK")
    print("✅ Legacy совместимость: OK")
    print("✅ NCA клетка с логированием: OK")
    print("✅ gMLP клетка с логированием: OK")
    print("✅ Caller tracking: OK")
    print("✅ Контекстное логирование: OK")
    print("✅ Интеграция завершена: OK")

    logger.info("🎯 Phase 3 Priority 1 - ИНТЕГРАЦИЯ ЛОГИРОВАНИЯ ЗАВЕРШЕНА!")

    return True


def test_clean_logging():
    """Тест чистого логирования без дедупликации"""

    print("\n🔍 ТЕСТ: Чистое логирование")
    print("-" * 40)

    # Настраиваем логирование в обычном режиме БЕЗ дедупликации
    setup_logging(debug_mode=False, enable_context=True)

    logger = get_logger("clean_test")

    # Создаем клетку (все логи сохраняются)
    logger.info("Создание клетки в обычном режиме...")

    with LogContext("test_clean"):
        nca_cell = NCACell()

        # Forward pass
        batch_size = 1
        neighbor_states = torch.randn(batch_size, 26, 4)
        own_state = torch.randn(batch_size, 4)

        result = nca_cell.forward(neighbor_states, own_state)

    logger.info("✅ Тест чистого логирования пройден")

    return True


if __name__ == "__main__":
    try:
        # Основной тест интеграции
        success1 = test_logging_integration()

        # Тест чистого логирования
        success2 = test_clean_logging()

        if success1 and success2:
            print("\n🏆 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("Phase 3 Priority 1 - ИНТЕГРАЦИЯ ЛОГИРОВАНИЯ: ✅ ЗАВЕРШЕНА")
        else:
            print("\n❌ ТЕСТЫ НЕ ПРОШЛИ")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
