#!/usr/bin/env python3
"""
Тест управления уровнями логирования через конфигурацию
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_logging_levels():
    """Тестируем разные уровни логирования"""
    
    print("🧪 TESTING LOGGING LEVELS CONFIGURATION")
    print("=" * 50)
    
    from new_rebuild.config import SimpleProjectConfig
    from new_rebuild.utils.logging import get_logger
    
    # Тест 1: DEBUG уровень
    print("\n🔬 Test 1: DEBUG level")
    config1 = SimpleProjectConfig()
    config1.logging.level = "DEBUG"
    config1.logging.debug_mode = False  # Отключаем debug_mode, чтобы level работал
    
    logger = get_logger("test_debug")
    logger.debug("This DEBUG message should appear")
    logger.info("This INFO message should appear")
    logger.warning("This WARNING message should appear")
    
    # Тест 2: WARNING уровень
    print("\n🔬 Test 2: WARNING level")
    config2 = SimpleProjectConfig()
    config2.logging.level = "WARNING"
    config2.logging.debug_mode = False
    
    logger2 = get_logger("test_warning")
    logger2.debug("This DEBUG message should NOT appear")
    logger2.info("This INFO message should NOT appear")
    logger2.warning("This WARNING message should appear")
    logger2.error("This ERROR message should appear")
    
    # Тест 3: debug_mode переопределяет level
    print("\n🔬 Test 3: debug_mode overrides level")
    config3 = SimpleProjectConfig()
    config3.logging.level = "ERROR"  # Высокий уровень
    config3.logging.debug_mode = True  # Но debug_mode включен
    
    logger3 = get_logger("test_override")
    logger3.debug("This DEBUG message should appear (debug_mode override)")
    logger3.info("This INFO message should appear (debug_mode override)")
    
    print("\n✅ Logging levels test completed!")

if __name__ == "__main__":
    test_logging_levels()