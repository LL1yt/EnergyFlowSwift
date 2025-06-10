#!/usr/bin/env python3
"""
ЭКСПЕРИМЕНТЫ С МАСШТАБОМ РЕШЕТКИ
Позволяет легко менять scale factor и видеть как это влияет на:
- Размер решетки
- Количество нейронов
- Потребление памяти
- Название checkpoint'ов
"""

import logging
import torch
from pathlib import Path
import sys
from datetime import datetime

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from utils.config_manager.dynamic_config import DynamicConfigManager, ScaleSettings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def experiment_with_scale(custom_scale: float = None):
    """Эксперимент с custom scale factor"""
    print(f"🧪 ЭКСПЕРИМЕНТ С МАСШТАБОМ РЕШЕТКИ")
    print("=" * 60)

    if custom_scale:
        print(f"🎯 Используем custom scale: {custom_scale}")
    else:
        print(f"📊 Сравниваем стандартные режимы")

    try:
        config_manager = DynamicConfigManager()

        # Если задан custom scale, создаем временную модификацию
        if custom_scale:
            # Временно изменяем scale settings
            original_dev_scale = config_manager.generator.scale_settings.development
            config_manager.generator.scale_settings.development = custom_scale

            # Генерируем конфигурацию
            config = config_manager.create_config_for_mode("development")

            # Восстанавливаем оригинальный scale
            config_manager.generator.scale_settings.development = original_dev_scale

            print(f"\n📐 КОНФИГУРАЦИЯ С CUSTOM SCALE {custom_scale}:")
            _print_config_info(config, custom_scale)

            # Показываем как будет называться checkpoint
            _show_checkpoint_naming(
                custom_scale, len_dataset=1000, epochs=50, similarity=0.234
            )

        else:
            # Показываем все стандартные режимы
            modes = ["development", "research", "validation", "production"]

            for mode in modes:
                config = config_manager.create_config_for_mode(mode)
                scale = config.get("_metadata", {}).get("scale_factor") or config.get(
                    "lattice", {}
                ).get("scale_factor")

                print(f"\n📊 РЕЖИМ: {mode.upper()} (scale={scale})")
                _print_config_info(config, scale)

                # Показываем название checkpoint'а
                _show_checkpoint_naming(
                    scale, len_dataset=1000, epochs=50, similarity=0.234, mode=mode
                )

    except Exception as e:
        print(f"❌ Ошибка эксперимента: {e}")


def _print_config_info(config, scale_factor):
    """Печать информации о конфигурации"""
    lattice = config["lattice"]
    embeddings = config["embeddings"]
    training = config["training"]

    print(f"   📏 Размер решетки: {lattice['xs']} × {lattice['ys']} × {lattice['zs']}")
    print(f"   🧠 Всего нейронов: {lattice['total_neurons']:,}")
    print(f"   📊 Embedding dim: {embeddings['embedding_dim']:,}")
    print(f"   🎯 Batch size: {training['batch_size']}")

    # Оценка памяти
    lattice_memory_gb = (lattice["total_neurons"] * embeddings["embedding_dim"] * 4) / (
        1024**3
    )
    batch_memory_mb = (training["batch_size"] * embeddings["embedding_dim"] * 4) / (
        1024**2
    )

    print(f"   💾 Память решетки: ~{lattice_memory_gb:.3f} GB")
    print(f"   💾 Память батча: ~{batch_memory_mb:.1f} MB")

    # Время вычислений (примерная оценка)
    operations_per_epoch = (
        lattice["total_neurons"] * embeddings["embedding_dim"] * training["batch_size"]
    )
    relative_time = operations_per_epoch / 1e9  # Относительная оценка
    print(f"   ⏱️ Относительное время эпохи: ~{relative_time:.2f}")


def _show_checkpoint_naming(
    scale_factor, len_dataset, epochs, similarity, mode="development"
):
    """Показать как будет называться checkpoint"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Формируем название как в системе
    result_name = f"dynamic_{mode}_scale{scale_factor}_{len_dataset}pairs_{epochs}epochs_{similarity:.3f}sim_{timestamp}"

    print(f"   🏷️ Checkpoint: {result_name}")


def suggest_optimal_scales():
    """Предложить оптимальные scale factor'ы для разных целей"""
    print(f"\n💡 РЕКОМЕНДАЦИИ ПО SCALE FACTOR:")
    print("=" * 60)

    suggestions = [
        (0.005, "Очень быстрая разработка", "~900 нейронов, мгновенные тесты"),
        (0.01, "Быстрая разработка (стандарт)", "~3,600 нейронов, быстрые тесты"),
        (0.02, "Средняя разработка", "~14,400 нейронов, детальное тестирование"),
        (0.05, "Детальная разработка", "~90,000 нейронов, полноценные эксперименты"),
        (0.1, "Исследования (стандарт)", "~360,000 нейронов, серьезные модели"),
        (0.2, "Большие исследования", "~1.4M нейронов, требует много памяти"),
        (0.3, "Валидация (стандарт)", "~3.2M нейронов, почти продакшен"),
    ]

    for scale, purpose, description in suggestions:
        print(f"   {scale:5.3f} - {purpose:25s} ({description})")

    print(f"\n⚠️ ВАЖНО:")
    print(f"   • Для RTX 5090 (32GB): можно до 0.3-0.5")
    print(f"   • Для RTX 4070 Ti (12GB): рекомендуется до 0.1-0.2")
    print(f"   • Для меньших GPU: не больше 0.05-0.1")


def create_custom_scale_test(scale_factor: float):
    """Создать тест с custom scale factor"""
    print(f"\n🔧 СОЗДАНИЕ ТЕСТА С SCALE {scale_factor}")
    print("=" * 60)

    try:
        # Создаем модифицированный run_dynamic_training для этого scale
        script_content = f'''#!/usr/bin/env python3
"""
ТЕСТ ОБУЧЕНИЯ с custom scale factor = {scale_factor}
Автоматически сгенерированный скрипт для эксперимента
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from run_dynamic_training import DynamicTrainingManager
from utils.config_manager.dynamic_config import DynamicConfigManager

def main():
    """Запуск с custom scale {scale_factor}"""
    print(f"🧪 ТЕСТ ОБУЧЕНИЯ С SCALE {scale_factor}")
    print("=" * 50)
    
    # Создаем manager и временно изменяем scale
    manager = DynamicTrainingManager("development")
    
    # Модифицируем scale factor
    manager.config_manager.generator.scale_settings.development = {scale_factor}
    
    # Пересоздаем конфигурацию с новым scale
    manager.dynamic_config = manager.config_manager.create_config_for_mode("development")
    manager.config_manager.merge_dynamic_config(manager.dynamic_config)
    
    print(f"✅ Конфигурация обновлена для scale {scale_factor}")
    print(f"📏 Размер решетки: {{manager.dynamic_config['lattice']['xs']}}×{{manager.dynamic_config['lattice']['ys']}}×{{manager.dynamic_config['lattice']['zs']}}")
    print(f"🧠 Нейронов: {{manager.dynamic_config['lattice']['total_neurons']:,}}")
    
    # Запуск обучения
    try:
        results = manager.run_training(
            dataset_limit=1000,  # Ограничиваем для теста
            epochs=20,  # Меньше эпох для теста
            batch_size=None  # Используем из конфигурации
        )
        
        print(f"🎉 Тест завершен успешно!")
        print(f"   Лучшая similarity: {{results['best_similarity']:.4f}}")
        print(f"   Время: {{results['total_time']/60:.1f}} минут")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {{e}}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''

        # Сохраняем скрипт
        script_path = f"test_scale_{scale_factor:.3f}.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"✅ Создан тестовый скрипт: {script_path}")
        print(f"   Запуск: python {script_path}")

        return script_path

    except Exception as e:
        print(f"❌ Ошибка создания теста: {e}")
        return None


def main():
    """Главная функция"""
    print("🧪 ЭКСПЕРИМЕНТАЛЬНАЯ ЛАБОРАТОРИЯ МАСШТАБОВ")
    print("Позволяет легко экспериментировать с размерами решетки")
    print()

    while True:
        print("\n🎯 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
        print("1. Показать стандартные режимы")
        print("2. Эксперимент с custom scale")
        print("3. Рекомендации по scale factor")
        print("4. Создать тестовый скрипт")
        print("5. Выход")

        choice = input("\nВведите номер (1-5): ").strip()

        if choice == "1":
            experiment_with_scale()

        elif choice == "2":
            try:
                scale = float(input("Введите scale factor (например, 0.02): "))
                if 0.001 <= scale <= 1.0:
                    experiment_with_scale(scale)
                else:
                    print("⚠️ Scale должен быть от 0.001 до 1.0")
            except ValueError:
                print("❌ Неверный формат числа")

        elif choice == "3":
            suggest_optimal_scales()

        elif choice == "4":
            try:
                scale = float(
                    input("Введите scale factor для теста (например, 0.02): ")
                )
                if 0.001 <= scale <= 1.0:
                    script_path = create_custom_scale_test(scale)
                    if script_path:
                        print(f"\n💡 Можете теперь запустить: python {script_path}")
                else:
                    print("⚠️ Scale должен быть от 0.001 до 1.0")
            except ValueError:
                print("❌ Неверный формат числа")

        elif choice == "5":
            print("👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор, попробуйте снова")


if __name__ == "__main__":
    main()
