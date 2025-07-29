#!/usr/bin/env python3
"""
Energy Flow Dataset Generator CLI
=================================

Интерактивный инструмент для создания именованных файлов датасетов.
Создает готовые файлы которые можно напрямую использовать в обучении.

Использование:
    python generate_energy_dataset.py                    # Интерактивный режим
    python generate_energy_dataset.py --mode debug       # Быстрый режим  
    python generate_energy_dataset.py --list             # Список датасетов
    python generate_energy_dataset.py --archive          # Архивирование старых
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, create_experiment_config, create_optimized_config, set_energy_config
from energy_flow.dataset.generator import (
    DatasetGenerator, GeneratorConfig,
    create_debug_generator_config,
    create_experiment_generator_config, 
    create_production_generator_config,
    create_dataset_generator
)
from energy_flow.utils.logging import get_logger

logger = get_logger(__name__)


def print_header():
    """Печать заголовка программы"""
    print("🚀 Energy Flow Dataset Generator")
    print("=" * 50)
    print("")


def interactive_mode():
    """Интерактивный режим выбора параметров"""
    print("📋 Интерактивный режим создания датасета")
    print("-" * 40)
    
    # 1. Выбор режима обучения
    print("\n1️⃣ Выберите режим обучения:")
    print("   [1] DEBUG - быстрое тестирование (500 пар)")
    print("   [2] EXPERIMENT - исследования (5K пар)")
    print("   [3] PRODUCTION - полное обучение (50K пар)")
    print("   [4] CUSTOM - настраиваемые параметры")
    
    while True:
        choice = input("\nВыбор (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            break
        print("❌ Неверный выбор, попробуйте снова")
    
    # Создаем базовую конфигурацию
    if choice == "1":
        mode = "debug"
        generator_config = create_debug_generator_config()
        energy_config = create_debug_config() 
    elif choice == "2":
        mode = "experiment"
        generator_config = create_experiment_generator_config()
        energy_config = create_experiment_config()
    elif choice == "3":
        mode = "production"
        generator_config = create_production_generator_config()
        energy_config = create_optimized_config()
    else:  # custom
        mode = "custom"
        generator_config, energy_config = custom_configuration()
    
    # 2. Настройка источников данных
    if mode != "custom":
        print(f"\n2️⃣ Источники данных для режима {mode.upper()}:")
        sources_str = ", ".join(generator_config.sources)
        print(f"   Используются: {sources_str}")
        
        modify = input("   Изменить источники? (y/N): ").strip().lower()
        if modify == 'y':
            generator_config.sources = choose_data_sources()
    
    # 3. Дополнительные параметры для SNLI
    if "snli" in generator_config.sources:
        print(f"\n3️⃣ Параметры SNLI:")
        print(f"   Текущая фракция: {generator_config.snli_fraction:.1%}")
        
        modify_snli = input("   Изменить параметры SNLI? (y/N): ").strip().lower()
        if modify_snli == 'y':
            while True:
                try:
                    fraction = float(input("   Введите фракцию SNLI (0.1-1.0): "))
                    if 0.1 <= fraction <= 1.0:
                        generator_config.snli_fraction = fraction
                        break
                    else:
                        print("   ❌ Значение должно быть от 0.1 до 1.0")
                except ValueError:
                    print("   ❌ Введите число")
    
    # 4. Пользовательское имя файла
    print(f"\n4️⃣ Имя файла:")
    preview_name = generator_config.generate_filename(generator_config.target_pairs)
    print(f"   Автоматическое: {preview_name}")
    
    custom_name = input("   Пользовательское имя (Enter для автоматического): ").strip()
    if not custom_name:
        custom_name = None
    
    # 5. Подтверждение и генерация
    print(f"\n5️⃣ Подтверждение параметров:")
    print(f"   Режим: {mode.upper()}")
    print(f"   Целевое количество пар: {generator_config.target_pairs:,}")
    print(f"   Источники: {', '.join(generator_config.sources)}")
    if "snli" in generator_config.sources:
        print(f"   SNLI фракция: {generator_config.snli_fraction:.1%}")
    print(f"   Имя файла: {custom_name or preview_name}")
    
    confirm = input("\n   Начать генерацию? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("🛑 Генерация отменена")
        return
    
    # Генерация датасета
    return generate_dataset(generator_config, energy_config, custom_name)


def custom_configuration():
    """Настройка пользовательских параметров"""
    print("\n🔧 Пользовательская конфигурация:")
    
    # Количество пар
    while True:
        try:
            target_pairs = int(input("   Количество пар (100-100000): "))
            if 100 <= target_pairs <= 100000:
                break
            else:
                print("   ❌ Значение должно быть от 100 до 100000")
        except ValueError:
            print("   ❌ Введите число")
    
    # Источники данных
    sources = choose_data_sources()
    
    # SNLI параметры
    snli_fraction = 0.2
    if "snli" in sources:
        while True:
            try:
                snli_fraction = float(input("   Фракция SNLI (0.1-1.0): "))
                if 0.1 <= snli_fraction <= 1.0:
                    break
                else:
                    print("   ❌ Значение должно быть от 0.1 до 1.0")
            except ValueError:
                print("   ❌ Введите число")
    
    # Создаем конфигурации
    generator_config = GeneratorConfig(
        mode="custom",
        target_pairs=target_pairs,
        sources=sources,
        snli_fraction=snli_fraction
    )
    
    # Energy config на основе размера датасета
    if target_pairs <= 1000:
        energy_config = create_debug_config()
    elif target_pairs <= 10000:
        energy_config = create_experiment_config()
    else:
        energy_config = create_optimized_config()
    
    return generator_config, energy_config


def choose_data_sources():
    """Выбор источников данных"""
    print("   Доступные источники:")
    print("   [1] Только precomputed (готовые эмбеддинги)")
    print("   [2] Только SNLI (генерация из датасета)")
    print("   [3] Mixed (precomputed + SNLI)")
    
    while True:
        choice = input("   Выбор (1-3): ").strip()
        if choice == "1":
            return ["precomputed"] 
        elif choice == "2":
            return ["snli"]
        elif choice == "3":
            return ["precomputed", "snli"]
        else:
            print("   ❌ Неверный выбор")


def generate_dataset(generator_config: GeneratorConfig, energy_config, custom_name: str = None):
    """Генерация датасета"""
    print(f"\n🔄 Генерация датасета...")
    print("-" * 30)
    
    try:
        # Устанавливаем energy config
        set_energy_config(energy_config)
        
        # Создаем генератор
        generator = DatasetGenerator(generator_config, energy_config)
        
        # Генерируем датасет
        result = generator.generate_dataset(custom_name)
        
        # Показываем результат
        print(f"\n🎉 Датасет успешно создан!")
        print(f"   📁 Файл: {result['filename']}")
        print(f"   📊 Образцов: {result['sample_count']:,}")
        print(f"   💾 Размер: {result['file_size_mb']:.1f} MB")
        print(f"   ⏱️ Время: {result['generation_time']:.1f}s")
        print(f"   📂 Путь: {result['filepath']}")
        
        # Инструкции по использованию
        print(f"\n💡 Использование в обучении:")
        print(f"   import torch")
        print(f"   data = torch.load('{result['filename']}')")
        print(f"   input_embeddings = data['input_embeddings']")
        print(f"   target_embeddings = data['target_embeddings']")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Ошибка генерации: {e}")
        logger.error(f"Dataset generation failed: {e}")
        return None


def list_datasets():
    """Показать список доступных датасетов"""
    print("📋 Доступные датасеты:")
    print("-" * 50)
    
    try:
        # Создаем простой генератор для получения списка
        generator = create_dataset_generator("experiment")
        datasets = generator.list_available_datasets(include_archive=True)
        
        if not datasets:
            print("   📭 Датасеты не найдены")
            print("   Используйте интерактивный режим для создания первого датасета")
            return
        
        # Группируем по категориям
        active_datasets = [d for d in datasets if d.get('category') == 'active']
        archive_datasets = [d for d in datasets if d.get('category') == 'archive']
        
        # Активные датасеты
        if active_datasets:
            print("🟢 АКТИВНЫЕ ДАТАСЕТЫ:")
            for i, dataset in enumerate(active_datasets, 1):
                creation_date = datetime.fromtimestamp(dataset['creation_time']).strftime('%Y-%m-%d %H:%M')
                print(f"   {i}. {dataset['filename']}")
                print(f"      📊 {dataset.get('sample_count', 'N/A'):,} образцов")
                print(f"      💾 {dataset['file_size_mb']:.1f} MB")
                print(f"      🕒 {creation_date}")
                print(f"      🏷️ {dataset.get('mode', 'unknown')} / {', '.join(dataset.get('sources', []))}")
                print()
        
        # Архивные датасеты
        if archive_datasets:
            print("📦 АРХИВНЫЕ ДАТАСЕТЫ:")
            for dataset in archive_datasets[:5]:  # Показываем только первые 5
                creation_date = datetime.fromtimestamp(dataset['creation_time']).strftime('%Y-%m-%d %H:%M')
                print(f"   📁 {dataset['filename']} ({dataset['file_size_mb']:.1f} MB, {creation_date})")
            
            if len(archive_datasets) > 5:
                print(f"   ... и еще {len(archive_datasets) - 5} файлов в архиве")
        
        print(f"\n📈 Статистика:")
        print(f"   Активных: {len(active_datasets)}")
        print(f"   В архиве: {len(archive_datasets)}")
        total_size = sum(d['file_size_mb'] for d in datasets)
        print(f"   Общий размер: {total_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Ошибка получения списка: {e}")


def archive_old_datasets():
    """Архивирование старых датасетов"""
    print("🗂️ Архивирование старых датасетов:")
    print("-" * 40)
    
    # Спрашиваем количество дней
    while True:
        try:
            days = int(input("Архивировать датасеты старше скольки дней? (по умолчанию 7): ") or "7")
            if days > 0:
                break
            else:
                print("❌ Количество дней должно быть больше 0")
        except ValueError:
            print("❌ Введите число")
    
    try:
        generator = create_dataset_generator("experiment")
        result = generator.archive_old_datasets(days_old=days)
        
        if result['archived_count'] > 0:
            print(f"✅ Заархивировано {result['archived_count']} файлов:")
            for file_info in result['archived_files']:
                print(f"   📁 {file_info['filename']}")
            print(f"\n📂 Архив: {result['archive_directory']}")
        else:
            print("📭 Нет файлов для архивирования")
        
        if result['errors']:
            print(f"\n⚠️ Ошибки ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"   ❌ {error}")
        
    except Exception as e:
        print(f"❌ Ошибка архивирования: {e}")


def quick_generate(mode: str):
    """Быстрая генерация с предустановленными параметрами"""
    print(f"⚡ Быстрая генерация датасета: {mode.upper()}")
    print("-" * 40)
    
    try:
        # Создаем конфигурации
        if mode == "debug":
            generator_config = create_debug_generator_config()
            energy_config = create_debug_config()
        elif mode == "experiment":
            generator_config = create_experiment_generator_config()
            energy_config = create_experiment_config()
        elif mode == "production":
            generator_config = create_production_generator_config()
            energy_config = create_optimized_config()
        else:
            print(f"❌ Неизвестный режим: {mode}")
            return
        
        # Показываем параметры
        print(f"📋 Параметры:")
        print(f"   Целевое количество: {generator_config.target_pairs:,} пар")
        print(f"   Источники: {', '.join(generator_config.sources)}")
        print(f"   SNLI фракция: {generator_config.snli_fraction:.1%}")
        
        # Генерируем
        return generate_dataset(generator_config, energy_config) 
        
    except Exception as e:
        print(f"❌ Ошибка быстрой генерации: {e}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Energy Flow Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["debug", "experiment", "production"],
        help="Быстрая генерация с предустановленными параметрами"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="Показать список доступных датасетов"
    )
    parser.add_argument(
        "--archive", 
        action="store_true",
        help="Архивировать старые датасеты"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    try:
        if args.list:
            list_datasets()
        elif args.archive:
            archive_old_datasets()
        elif args.mode:
            quick_generate(args.mode)
        else:
            # Интерактивный режим по умолчанию
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n🛑 Прервано пользователем")
    except Exception as e:
        print(f"\n💥 Неожиданная ошибка: {e}")
        logger.error(f"Unexpected error in main: {e}")


if __name__ == "__main__":
    main()