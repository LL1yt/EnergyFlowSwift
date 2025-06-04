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
    # from core.lattice_3d import Lattice3D
    # from core.signal_propagation import SignalPropagator
    # from data.embedding_loader import EmbeddingLoader
    # from data.data_visualization import Visualizer
    # from training.training_loop import TrainingLoop
    # from inference.prediction import Predictor
    # from utils.config_manager import ConfigManager
    # from utils.logger import setup_logger
    
    print("📋 Инициализация системы...")
    print("✅ Модуль cell_prototype загружен успешно!")
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
    Загружает конфигурацию проекта
    
    Параметры:
        config_path (str): Путь к файлу конфигурации
        
    Возвращает:
        dict: Словарь с настройками
    """
    print(f"⚙️  Загрузка конфигурации из {config_path}...")
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print("  ✅ Конфигурация загружена успешно")
        return config
    except FileNotFoundError:
        print(f"  ❌ Файл конфигурации не найден: {config_path}")
        return None
    except Exception as e:
        print(f"  ❌ Ошибка загрузки конфигурации: {e}")
        return None


def setup_logging(config):
    """
    Настраивает систему логирования
    
    Параметры:
        config (dict): Конфигурация проекта
    """
    print("📝 Настройка системы логирования...")
    
    # Создаем директорию для логов если её нет
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Настраиваем логгер
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Система логирования настроена")
    return logger


def demonstrate_concept():
    """
    Демонстрирует основную концепцию проекта с помощью простого примера
    
    Биологическая аналогия: Показываем, как одна клетка превращается в ткань
    """
    print("\n🧬 ДЕМОНСТРАЦИЯ КОНЦЕПЦИИ")
    print("=" * 50)
    
    print("""
    Представьте, что у нас есть 3D куб размером 5x5x5 = 125 клеток.
    
    ┌─────────────────────────────────────┐
    │  ЭМБЕДИНГ (числа) → ГРАНЬ ВХОДА     │
    │                                     │
    │  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐              │
    │  │ │→│ │→│ │→│ │→│ │  Слой 1       │
    │  └─┘ └─┘ └─┘ └─┘ └─┘              │
    │   ↓   ↓   ↓   ↓   ↓               │
    │  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐              │
    │  │ │→│ │→│ │→│ │→│ │  Слой 2       │
    │  └─┘ └─┘ └─┘ └─┘ └─┘              │
    │   ↓   ↓   ↓   ↓   ↓               │
    │  ...  ...  ...  ...  ...          │
    │   ↓   ↓   ↓   ↓   ↓               │
    │  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐              │
    │  │ │→│ │→│ │→│ │→│ │  Слой 5       │
    │  └─┘ └─┘ └─┘ └─┘ └─┘              │
    │                                     │
    │  ГРАНЬ ВЫХОДА → ДЕКОДЕР → ТОКЕНЫ    │
    └─────────────────────────────────────┘
    
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
        # Создаем клетку из конфигурации  
        config = load_configuration()
        if config:
            real_cell = create_cell_from_config(config)
            print(f"  ✅ Реальная клетка создана: {real_cell}")
            
            # Создаем тестовые данные
            batch_size = 2
            neighbor_states = torch.randn(batch_size, 6, config['cell_prototype']['state_size'])
            own_state = torch.randn(batch_size, config['cell_prototype']['state_size'])
            external_input = torch.randn(batch_size, config['cell_prototype']['input_size'])
            
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
    
    # Запускаем тест системы
    if not run_simple_test():
        print("\n❌ Тесты системы не прошли. Проверьте установку зависимостей.")
        return 1
    
    print(f"\n🎯 РЕЖИМ РАБОТЫ: {args.mode.upper()}")
    print("=" * 30)
    
    if args.mode == "demo":
        print("📋 Режим демонстрации")
        print("  • Показываем основные концепции")
        print("  • Выполняем простые тесты")
        print("  • Готовимся к реальной разработке")
        print("\n💡 Следующий шаг: создание первого модуля cell_prototype")
        
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
