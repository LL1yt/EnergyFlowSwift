#!/usr/bin/env python3
"""
Тест для проверки исправления DEBUG логирования
===============================================

Проверяем что когда в конфигурации level="DEBUG",
то logger.debug() сообщения действительно выводятся.
"""

import logging
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.utils.logging import get_logger

print("=== DEBUG LOGGING TEST ===")

# Создаем конфигурацию (автоматически настраивает логирование)
config = SimpleProjectConfig()

print(f"Config logging level: {config.logging.level}")
print(f"Config debug_mode: {config.logging.debug_mode}")

# Получаем логгер
logger = get_logger("test_debug")

print("\n=== ТЕСТИРУЕМ УРОВНИ ЛОГИРОВАНИЯ ===")

# Тестируем все уровни
logger.debug("🔍 DEBUG: Это debug сообщение")
logger.info("ℹ️ INFO: Это info сообщение")
logger.warning("⚠️ WARNING: Это warning сообщение")
logger.error("❌ ERROR: Это error сообщение")

print("\n=== ТЕСТИРУЕМ DEBUG С МАРКЕРАМИ ===")

# DEBUG с важными маркерами (должны показываться даже при фильтрации)
logger.debug("🚀 INIT Test component initialized")
logger.debug("✅ OK: Operation completed successfully")
logger.debug("❌ ERROR: Operation failed")

print("\n=== ТЕСТИРУЕМ ОБЫЧНЫЕ DEBUG ===")

# Обычные DEBUG сообщения (должны показываться только если level="DEBUG")
logger.debug("Обычное debug сообщение без маркеров")
logger.debug("Еще одно debug сообщение")
logger.debug("Третье debug сообщение")

print("\n=== ПРОВЕРЯЕМ ROOT LOGGER НАСТРОЙКИ ===")

root_logger = logging.getLogger()
print(
    f"Root logger level: {root_logger.level} ({logging.getLevelName(root_logger.level)})"
)
print(f"Root logger handlers: {len(root_logger.handlers)}")

for i, handler in enumerate(root_logger.handlers):
    print(f"  Handler {i}: {type(handler).__name__}")
    print(f"    Level: {handler.level} ({logging.getLevelName(handler.level)})")
    print(f"    Filters: {len(handler.filters)}")
    for j, filter_obj in enumerate(handler.filters):
        print(f"      Filter {j}: {type(filter_obj).__name__}")
        if hasattr(filter_obj, "debug_mode"):
            print(f"        debug_mode: {filter_obj.debug_mode}")

print("\n=== TEST COMPLETE ===")
