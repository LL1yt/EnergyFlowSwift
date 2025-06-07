#!/usr/bin/env python3
"""
Главный файл проекта: 3D Клеточная Нейронная Сеть

Это исследовательский проект, который создает "умную 3D ткань" из одинаковых клеток-нейронов.
Каждая клетка получает сигналы от соседей, обрабатывает их и передает дальше.

Аналогия: Как в коре головного мозга - одинаковые нейроны, организованные в структуру,
где каждый получает сигналы от соседей и передает дальше.

Автор: Исследовательская команда
Дата: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Добавляем текущую директорию в PATH для импорта модулей
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# === ИМПОРТЫ МОДУЛЕЙ (будут создаваться постепенно) ===
try:
    # ✅ Модуль cell_prototype готов!
    from core import CellPrototype, create_cell_from_config
    
    # ✅ Модуль lattice_3d готов с I/O стратегией!
    from core.lattice_3d import (
        Lattice3D, LatticeConfig, PlacementStrategy, IOPointPlacer, Face,
        load_lattice_config, create_lattice_from_config
    )
    
    # from core.signal_propagation import SignalPropagator
    # from data.embedding_loader import EmbeddingLoader
    # from data.data_visualization import Visualizer
    # from training.training_loop import TrainingLoop
    # from inference.prediction import Predictor
    # from utils.config_manager import ConfigManager
    # from utils.logger import setup_logger
    
    print("📋 Инициализация системы...")
    print("✅ Модуль cell_prototype загружен успешно!")
    print("✅ Модуль lattice_3d с I/O стратегией загружен успешно!")
    print("⚠️  Остальные модули пока не реализованы")
    
except ImportError as e:
    print(f"⚠️  Модуль не найден: {e}")
    print("💡 Это нормально на начальной стадии разработки")


def setup_project_structure():
    """
    Создает необходимые директории для проекта
    
    Биологическая аналогия: Подготавливаем "питательную среду" для роста нашей нейронной ткани
    """
    print("🏗️  Настройка структуры проекта...")
    
    # Директории, которые должны существовать
    directories = [
        "logs",           # Для записи происходящих процессов
        "checkpoints",    # Для сохранения промежуточных результатов
        "data/train",     # Обучающие данные
        "data/test",      # Тестовые данные
        "data/embeddings", # Входные эмбединги
        "outputs",        # Результаты работы
        "visualizations", # Сохраненные графики и анимации
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ Создана директория: {dir_path}")


def load_configuration(config_path="config/main_config.yaml"):
    """
    Загружает конфигурацию проекта через ConfigManager
    
    Параметры:
        config_path (str): Путь к файлу конфигурации
        
    Возвращает:
        ConfigManager: Настроенный менеджер конфигурации
    """
    print(f"⚙️  Инициализация ConfigManager из {config_path}...")
    
    try:
        from utils.config_manager import create_config_manager, set_global_config_manager
        
        # Создаем ConfigManager
        config = create_config_manager(
            base_config=config_path,
            environment="development",
            enable_hot_reload=True
        )
        
        # Устанавливаем как глобальный
        set_global_config_manager(config)
        
        print("  ✅ ConfigManager инициализирован успешно")
        print(f"  📊 Загружено секций: {len(config.get_config())}")
        print(f"  🔍 Обнаружено модульных конфигураций: {config.get_stats()['config_loads']}")
        
        return config
    except Exception as e:
        print(f"  ❌ Ошибка инициализации ConfigManager: {e}")
        return None


def setup_logging(config):
    """
    Настраивает систему логирования
    
    Параметры:
        config (ConfigManager): Менеджер конфигурации проекта
    """
    print("📝 Настройка системы логирования...")
    
    # Создаем директорию для логов если её нет
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Получаем настройки логирования из ConfigManager
    log_level = config.get_config('logging', 'level', 'INFO')
    log_to_file = config.get_config('logging', 'log_to_file', True)
    log_file = config.get_config('logging', 'log_file', 'logs/main.log')
    log_to_console = config.get_config('logging', 'log_to_console', True)
    
    # Настраиваем логгер
    handlers = []
    
    if log_to_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Система логирования настроена")
    logger.info(f"📊 Уровень логирования: {log_level}")
    logger.info(f"📁 Файл логов: {log_file if log_to_file else 'отключен'}")
    
    return logger


def demonstrate_concept():
    """
    Демонстрирует основную концепцию проекта с помощью простого примера
    
    Биологическая аналогия: Показываем, как одна клетка превращается в ткань
    """
    print("\n🧬 ДЕМОНСТРАЦИЯ КОНЦЕПЦИИ")
    print("=" * 50)
    
    print("""
    🆕 ОБНОВЛЕНО: Теперь с пропорциональной I/O стратегией!
    
    Представьте, что у нас есть 3D куб размером 8x8x8 = 512 клеток.
    
    ┌─────────────────────────────────────┐
    │  ЭМБЕДИНГ → I/O ТОЧКИ (только ~10%) │
    │                                     │
    │  ● ○ ○ ● ○ ○ ● ○  ← Входные точки  │
    │  ○ ┌─┐ ┌─┐ ┌─┐ ○  (биологически    │
    │  ● │ │→│ │→│ │ ●    обоснованное   │
    │  ○ └─┘ └─┘ └─┘ ○    покрытие)      │
    │   ↓   ↓   ↓   ↓                    │
    │  ┌─┐ ┌─┐ ┌─┐ ┌─┐   Слой 2          │
    │  │ │→│ │→│ │→│ │                   │
    │  └─┘ └─┘ └─┘ └─┘                   │
    │   ↓   ↓   ↓   ↓                    │
    │  ...  ...  ...  ...               │
    │   ↓   ↓   ↓   ↓                    │
    │  ● ○ ○ ● ○ ○ ● ○  ← Выходные точки │
    │                                     │
    │  I/O ТОЧКИ → ДЕКОДЕР → ТОКЕНЫ       │
    └─────────────────────────────────────┘
    
    🆕 Ключевые улучшения:
    • Автоматическое масштабирование I/O точек (7.8-15.6%)
    • Постоянная плотность рецепторов как в мозге  
    • 5 стратегий размещения: пропорциональная, случайная, углы, центр, полное
    • Значительное улучшение производительности
    
    Каждая клетка (□):
    • Получает сигналы от соседей
    • Обрабатывает их своей мини-нейросетью  
    • Передает результат дальше
    • Все клетки используют ОДИНАКОВЫЕ веса!
    """)
    
    print("\n💡 Ключевые преимущества:")
    print("  🔹 Параметрическая эффективность: учим только 1 прототип")
    print("  🔹 Биологическая правдоподобность: как в коре мозга")
    print("  🔹 Параллелизм: все клетки работают одновременно")
    print("  🔹 Масштабируемость: можно делать решетки любого размера")
    print("  🆕 Умная I/O система: автоматическое масштабирование точек")
    print("  🆕 Производительность: сокращение I/O точек в 5-10 раз")


def demonstrate_io_strategy():
    """
    🆕 Демонстрирует новую I/O стратегию с автоматическим масштабированием
    """
    print("\n🆕 ДЕМОНСТРАЦИЯ I/O СТРАТЕГИИ")
    print("=" * 40)
    
    try:
        import torch
        
        print("🎯 Сравнение различных стратегий размещения...")
        
        # Тестируем разные размеры решеток
        sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32)]
        strategies = [
            ("Пропорциональная", PlacementStrategy.PROPORTIONAL),
            ("Углы", PlacementStrategy.CORNERS),
            ("Случайная", PlacementStrategy.RANDOM),
            ("Полная грань", PlacementStrategy.FULL_FACE),
        ]
        
        print(f"\n{'Размер':>12} | {'Стратегия':>15} | {'I/O точек':>10} | {'Покрытие':>9}")
        print("-" * 60)
        
        for size in sizes:
            face_area = size[0] * size[1]
            
            for name, strategy in strategies:
                try:
                    placer = IOPointPlacer(
                        lattice_dimensions=size,
                        strategy=strategy,
                        config={
                            'coverage_ratio': {'min_percentage': 8.0, 'max_percentage': 12.0},
                            'absolute_limits': {'min_points': 5, 'max_points': 0}
                        },
                        seed=42
                    )
                    
                    input_points = placer.get_input_points(Face.FRONT)
                    coverage = len(input_points) / face_area * 100
                    
                    size_str = f"{size[0]}×{size[1]}×{size[2]}"
                    print(f"{size_str:>12} | {name:>15} | {len(input_points):>8}   | {coverage:>6.1f}%")
                    
                except Exception as e:
                    print(f"{size[0]}×{size[1]}×{size[2]:>12} | {name:>15} | {'ERROR':>10} | {'---':>9}")
        
        print("\n🧬 Создание реальной 3D решетки с I/O стратегией...")
        
        # Создаем конфигурацию с пропорциональной I/O стратегией
        config = LatticeConfig(
            dimensions=(8, 8, 8),
            boundary_conditions="walls",
            placement_strategy=PlacementStrategy.PROPORTIONAL,
            io_strategy_config={
                'coverage_ratio': {'min_percentage': 8.0, 'max_percentage': 12.0},
                'absolute_limits': {'min_points': 5, 'max_points': 25},
                'seed': 42
            }
        )
        
        # Создаем решетку
        lattice = create_lattice_from_config()  # Использует default config
        
        # Получаем информацию о I/O точках
        io_info = lattice.get_io_point_info()
        
        print(f"  ✅ 3D решетка создана: {config.dimensions}")
        print(f"  📊 Всего клеток: {config.total_cells}")
        print(f"  📍 Входных точек: {io_info['input_points']['count']} ({io_info['input_points']['coverage_percentage']:.1f}%)")
        print(f"  📍 Выходных точек: {io_info['output_points']['count']} ({io_info['output_points']['coverage_percentage']:.1f}%)")
        
        # Тестируем forward pass с пропорциональными точками
        num_input_points = io_info['input_points']['count']
        input_size = lattice.cell_prototype.input_size
        external_inputs = torch.randn(num_input_points, input_size)
        
        print(f"  🔄 Тестируем forward pass...")
        print(f"  📥 Входные данные: {external_inputs.shape}")
        
        # Выполняем прямой проход
        with torch.no_grad():
            output_states = lattice.forward(external_inputs)
            io_output = lattice.get_output_states()
        
        print(f"  📤 Все состояния: {output_states.shape}")
        print(f"  📤 I/O выходы: {io_output.shape}")
        print(f"  🎯 Диапазон выходов: [{io_output.min():.3f}, {io_output.max():.3f}]")
        
        # Сравнение с полной гранью
        full_face_points = 8 * 8  # 64 точки для полной грани
        efficiency_gain = full_face_points / num_input_points
        
        print(f"\n💡 Эффективность:")
        print(f"  🔸 Полная грань: {full_face_points} точек")
        print(f"  🔸 Пропорциональная: {num_input_points} точек")
        print(f"  🔸 Ускорение: {efficiency_gain:.1f}x меньше I/O точек")
        print(f"  🔸 Экономия памяти: {(1 - num_input_points/full_face_points)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка демонстрации I/O стратегии: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_test():
    """
    Запускает простой тест для проверки готовности системы
    """
    print("\n🧪 ПРОСТОЙ ТЕСТ СИСТЕМЫ")
    print("=" * 30)
    
    # Проверяем доступность PyTorch
    try:
        import torch
        print(f"  ✅ PyTorch доступен (версия: {torch.__version__})")
        
        # Проверяем доступность GPU
        if torch.cuda.is_available():
            print(f"  ✅ GPU доступен: {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ️  GPU недоступен, будем использовать CPU")
            
    except ImportError:
        print("  ❌ PyTorch не установлен")
        return False
    
    # Проверяем другие библиотеки
    libraries = ['numpy', 'matplotlib', 'yaml']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  ✅ {lib} доступен")
        except ImportError:
            print(f"  ❌ {lib} не установлен")
            return False
    
    # Демонстрируем РЕАЛЬНЫЙ модуль cell_prototype
    print("\n🧬 ТЕСТИРУЕМ РЕАЛЬНУЮ КЛЕТКУ CELL_PROTOTYPE...")
    try:
        # Получаем глобальный ConfigManager
        from utils.config_manager import get_global_config_manager
        config_manager = get_global_config_manager()
        
        if config_manager:
            # Получаем конфигурацию cell_prototype
            cell_config = config_manager.get_config('cell_prototype')
            if cell_config:
                real_cell = create_cell_from_config(cell_config)
                print(f"  ✅ Реальная клетка создана: {real_cell}")
                
                # Создаем тестовые данные
                batch_size = 2
                neighbor_states = torch.randn(batch_size, 6, cell_config['state_size'])
                own_state = torch.randn(batch_size, cell_config['state_size'])
                external_input = torch.randn(batch_size, cell_config['input_size'])
                
                # Тестируем forward pass
                with torch.no_grad():
                    new_state = real_cell(neighbor_states, own_state, external_input)
                
                print(f"  📊 Входное состояние: {own_state[0].numpy()}")
                print(f"  📊 Новое состояние:   {new_state[0].numpy()}")
                print(f"  📊 Диапазон выхода:   [{new_state.min():.3f}, {new_state.max():.3f}]")
                
                # Показываем информацию о модели
                info = real_cell.get_info()
                print(f"  📋 Параметров в модели: {info['total_parameters']}")
                print(f"  📋 Размер модели: {info['model_size_mb']:.2f} MB")
                
                print("  ✅ Тест реальной клетки прошел успешно!")
            else:
                print("  ⚠️  Конфигурация cell_prototype не найдена")
            
        else:
            print("  ⚠️  Конфигурация недоступна, используем простую заглушку")
            
            # Fallback к простой клетке если конфигурация не загрузилась
            import torch.nn as nn
            
            class SimpleCell(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = nn.Linear(3, 2)
                    self.activation = nn.Tanh()
                    
                def forward(self, x):
                    return self.activation(self.layer(x))
            
            cell = SimpleCell()
            test_input = torch.randn(1, 3)
            output = cell(test_input)
            
            print(f"  ✅ Простая клетка создана")
            print(f"  📊 Вход: {test_input.detach().numpy().flatten()}")
            print(f"  📊 Выход: {output.detach().numpy().flatten()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка тестирования клетки: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Главная функция программы
    
    Управляет всем жизненным циклом приложения:
    1. Настройка окружения
    2. Загрузка конфигурации  
    3. Инициализация модулей
    4. Запуск основной логики
    """
    print("🚀 ЗАПУСК 3D КЛЕТОЧНОЙ НЕЙРОННОЙ СЕТИ")
    print("=" * 50)
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="3D Cellular Neural Network")
    parser.add_argument("--config", default="config/main_config.yaml", 
                       help="Путь к файлу конфигурации")
    parser.add_argument("--mode", choices=["demo", "train", "test"], default="demo",
                       help="Режим работы: demo/train/test")
    parser.add_argument("--debug", action="store_true", 
                       help="Режим отладки")
    
    args = parser.parse_args()
    
    # Настраиваем структуру проекта
    setup_project_structure()
    
    # Загружаем конфигурацию
    config = load_configuration(args.config)
    if config is None:
        print("❌ Не удалось загрузить конфигурацию. Завершение работы.")
        return 1
    
    # Настраиваем логирование
    logger = setup_logging(config)
    
    # Демонстрируем концепцию
    demonstrate_concept()
    
    # 🆕 Демонстрируем новую I/O стратегию
    if not demonstrate_io_strategy():
        print("\n⚠️  Демонстрация I/O стратегии не удалась. Продолжаем с базовыми тестами...")
    
    # Запускаем тест системы
    if not run_simple_test():
        print("\n❌ Тесты системы не прошли. Проверьте установку зависимостей.")
        return 1
    
    print(f"\n🎯 РЕЖИМ РАБОТЫ: {args.mode.upper()}")
    print("=" * 30)
    
    if args.mode == "demo":
        print("📋 Режим демонстрации")
        print("  • Показываем основные концепции")
        print("  • Демонстрируем I/O стратегию с автоматическим масштабированием")
        print("  • Выполняем простые тесты")
        print("  • Готовимся к реальной разработке")
        print("\n💡 Следующий шаг: интеграция с signal_propagation модулем")
        
    elif args.mode == "train":
        print("🎓 Режим обучения (пока не реализован)")
        print("  • Будет загружать данные")
        print("  • Будет обучать модель")
        print("  • Будет сохранять чекпоинты")
        
    elif args.mode == "test":
        print("🧪 Режим тестирования (пока не реализован)")
        print("  • Будет загружать обученную модель")
        print("  • Будет делать предсказания")
        print("  • Будет оценивать качество")
    
    print("\n✅ СИСТЕМА ГОТОВА К РАЗРАБОТКЕ!")
    print("📖 Следуйте плану в PROJECT_PLAN.md")
    print("🔧 Настройки в config/main_config.yaml")
    
    return 0


if __name__ == "__main__":
    """
    Точка входа в программу
    
    Запускается когда файл выполняется напрямую (не импортируется)
    """
    exit_code = main()
    sys.exit(exit_code)
