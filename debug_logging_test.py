#!/usr/bin/env python3
"""
Тест системы логирования
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("[SYNC] Testing logging system...")

# Тест 1: Проверим логирование до setup_logging
from new_rebuild.utils.logging import get_logger
logger = get_logger(__name__)

print("[SYNC] Test 1: Logging without setup")
logger.info("This should NOT appear (no setup yet)")
logger.warning("This WARNING might appear")
logger.error("This ERROR should appear")

# Тест 2: Настроим логирование и проверим снова
print("\n[SYNC] Test 2: Setting up logging...")
from new_rebuild.utils.logging import setup_logging
setup_logging(debug_mode=True)

print("[SYNC] Test 3: Logging after setup")
logger.info("This should appear now! (after setup)")
logger.warning("This warning should appear")
logger.error("This error should definitely appear")

# Тест 3: Проверим через конфигурацию
print("\n[SYNC] Test 4: Through config system...")
from new_rebuild.config import SimpleProjectConfig
config = SimpleProjectConfig()

logger2 = get_logger("test_config_logger")
logger2.info("Logging through config system")
logger2.debug("Debug message through config")

print("\n[OK] Logging test completed!")