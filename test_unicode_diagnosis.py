#!/usr/bin/env python3
"""
Диагностический тест для проблемы Unicode в логировании
=========================================================

Цель: Выяснить почему UTF8StreamHandler не работает в test_minimal_forward.py
"""

import logging
import sys
from pathlib import Path

print("=== UNICODE LOGGING DIAGNOSIS ===")
print(f"Python version: {sys.version}")
print(f"Default encoding: {sys.getdefaultencoding()}")
print(f"File system encoding: {sys.getfilesystemencoding()}")
print(f"Console encoding: {getattr(sys.stdout, 'encoding', 'unknown')}")

# Тест 1: Проверяем состояние логирования ДО импорта new_rebuild
print("\n1. СОСТОЯНИЕ ДО ИМПОРТА:")
root_logger = logging.getLogger()
print(f"   Root logger handlers: {len(root_logger.handlers)}")
print(f"   Root logger level: {root_logger.level}")
for i, handler in enumerate(root_logger.handlers):
    print(f"   Handler {i}: {type(handler).__name__}")

# Тест 2: Импортируем SimpleProjectConfig и смотрим что изменилось
print("\n2. ИМПОРТ SimpleProjectConfig:")
from new_rebuild.config import SimpleProjectConfig

# Проверяем состояние после импорта модуля (но до создания экземпляра)
print(f"   Root logger handlers: {len(root_logger.handlers)}")
for i, handler in enumerate(root_logger.handlers):
    print(f"   Handler {i}: {type(handler).__name__}")

# Тест 3: Создаем конфигурацию и смотрим что происходит в __post_init__
print("\n3. СОЗДАНИЕ КОНФИГУРАЦИИ (вызов __post_init__):")
config = SimpleProjectConfig()

# Проверяем состояние после __post_init__
print(f"   Root logger handlers: {len(root_logger.handlers)}")
for i, handler in enumerate(root_logger.handlers):
    print(f"   Handler {i}: {type(handler).__name__}")
    if hasattr(handler, "stream"):
        print(f"      Stream: {type(handler.stream).__name__}")
        print(
            f"      Stream encoding: {getattr(handler.stream, 'encoding', 'unknown')}"
        )

# Тест 4: Проверяем get_logger
print("\n4. ТЕСТИРУЕМ get_logger:")
from new_rebuild.utils.logging import get_logger

test_logger = get_logger("test_unicode")
print(f"   Test logger handlers: {len(test_logger.handlers)}")
print(f"   Test logger parent: {test_logger.parent}")
print(f"   Test logger propagate: {test_logger.propagate}")

# Тест 5: Тестируем логирование с русскими символами
print("\n5. ТЕСТ РУССКИХ СИМВОЛОВ:")
try:
    test_logger.info("Тест русских символов")
    print("   [OK] Русские символы работают")
except Exception as e:
    print(f"   [ERROR] Ошибка русских символов: {e}")

# Тест 6: Тестируем логирование со специальными символами
print("\n6. ТЕСТ СПЕЦИАЛЬНЫХ СИМВОЛОВ:")
try:
    test_logger.info("Test arrow symbol: 768D ↔ 64D")
    print("   [OK] Специальные символы работают")
except Exception as e:
    print(f"   [ERROR] Ошибка специальных символов: {e}")

# Тест 7: Проверяем конкретный символ из ошибки
print("\n7. ТЕСТ ПРОБЛЕМНОГО СИМВОЛА:")
try:
    test_logger.info("[SYNC] EmbeddingTransformer initialized: 768D ↔ 64D")
    print("   [OK] Проблемный символ работает")
except Exception as e:
    print(f"   [ERROR] Ошибка проблемного символа: {e}")

# Тест 8: Создаем логгер в новом модуле (имитируем EmbeddingTrainer)
print("\n8. ТЕСТ ЛОГГЕРА В МОДУЛЕ:")
module_logger = get_logger("new_rebuild.core.training.embedding_trainer")
print(f"   Module logger handlers: {len(module_logger.handlers)}")
print(f"   Module logger parent: {module_logger.parent}")
print(f"   Module logger propagate: {module_logger.propagate}")

try:
    module_logger.info("Инициализация EmbeddingTrainer на устройстве: cuda:0")
    print("   [OK] Модульный логгер работает")
except Exception as e:
    print(f"   [ERROR] Ошибка модульного логгера: {e}")

# Тест 9: Проверяем всю иерархию логгеров
print("\n9. ИЕРАРХИЯ ЛОГГЕРОВ:")
logger_names = [
    "",  # root
    "new_rebuild",
    "new_rebuild.core",
    "new_rebuild.core.training",
    "new_rebuild.core.training.embedding_trainer",
]

for name in logger_names:
    logger = logging.getLogger(name)
    print(f"   Logger '{name}':")
    print(f"      Handlers: {len(logger.handlers)}")
    print(f"      Level: {logger.level}")
    print(f"      Propagate: {logger.propagate}")
    if logger.handlers:
        for i, handler in enumerate(logger.handlers):
            print(f"         Handler {i}: {type(handler).__name__}")

print("\n=== DIAGNOSIS COMPLETE ===")
