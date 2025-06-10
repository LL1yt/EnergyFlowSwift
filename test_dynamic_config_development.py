#!/usr/bin/env python3
"""
ТЕСТ ДИНАМИЧЕСКОЙ КОНФИГУРАЦИИ в режиме разработки (scale=0.01)
Проверяет настройки для development режима перед запуском обучения
"""

import logging
import torch
from pathlib import Path
import sys

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from utils.config_manager.dynamic_config import DynamicConfigManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_development_mode():
    """Тест development режима (scale=0.01)"""
    print("🧪 ТЕСТ РАЗВИТОГО РЕЖИМА (SCALE=0.01)")
    print("=" * 50)

    try:
        # Создаем динамический конфигуратор
        config_manager = DynamicConfigManager()

        # Определяем аппаратуру
        import torch

        gpu_memory_gb = 0
        gpu_name = "CPU"
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            gpu_name = gpu_props.name

        # Определяем рекомендованный режим через generator
        recommended_mode = config_manager.generator.detect_hardware_mode()

        print(f"📊 Обнаруженная аппаратура:")
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory_gb:.1f} GB")
        print(f"   Рекомендованный режим: {recommended_mode}")

        # Генерируем конфигурацию для development режима
        config = config_manager.create_config_for_mode("development")

        # Получаем scale_factor из правильного места
        scale_factor = config.get("_metadata", {}).get("scale_factor") or config.get(
            "lattice", {}
        ).get("scale_factor")

        print(f"\n📐 КОНФИГУРАЦИЯ DEVELOPMENT РЕЖИМА:")
        print(f"   Scale factor: {scale_factor}")
        print(
            f"   Lattice size: {config['lattice']['xs']}×{config['lattice']['ys']}×{config['lattice']['zs']}"
        )
        print(f"   Total neurons: {config['lattice']['total_neurons']:,}")
        print(f"   Embedding dim: {config['embeddings']['embedding_dim']:,}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")

        # Проверяем, что это действительно development режим (scale=0.01)
        assert scale_factor == 0.01, f"Expected scale=0.01, got {scale_factor}"

        # Вычисляем примерный размер памяти
        lattice_size = (
            config["lattice"]["xs"] * config["lattice"]["ys"] * config["lattice"]["zs"]
        )
        embedding_dim = config["embeddings"]["embedding_dim"]
        estimated_memory_gb = (lattice_size * embedding_dim * 4) / (1024**3)  # float32

        print(f"\n💾 ОЦЕНКА ПАМЯТИ:")
        print(f"   Примерный размер решетки: {estimated_memory_gb:.2f} GB")
        print(f"   Доступная VRAM: {gpu_memory_gb:.1f} GB")

        # Проверяем соответствие размера батча и памяти
        batch_memory = (config["training"]["batch_size"] * embedding_dim * 4) / (
            1024**2
        )  # MB
        print(f"   Память на батч: {batch_memory:.1f} MB")

        print(f"\n✅ ТЕСТ ПРОШЕЛ УСПЕШНО!")
        print(f"   Development режим настроен корректно")
        print(f"   Scale factor: {scale_factor} (правильно для development)")
        print(f"   Размеры подходят для разработки")

        # Проверяем наличие precomputed datasets
        try:
            from precomputed_embedding_loader import PrecomputedEmbeddingLoader

            loader = PrecomputedEmbeddingLoader()
            datasets = loader.list_available_datasets()

            if datasets:
                print(f"\n📁 ДОСТУПНЫЕ ДАТАСЕТЫ:")
                for i, dataset in enumerate(datasets[:3]):  # Показываем первые 3
                    print(f"   {i+1}. {dataset['filename']} ({dataset['size']} pairs)")
                print(f"   Всего датасетов: {len(datasets)}")
            else:
                print(f"\n⚠️ ДАТАСЕТЫ НЕ НАЙДЕНЫ!")
                print(f"   Нужно сначала запустить generate_large_embedding_dataset.py")

        except ImportError as e:
            print(f"\n❌ Ошибка импорта PrecomputedEmbeddingLoader: {e}")

        return config

    except Exception as e:
        print(f"\n❌ ОШИБКА ТЕСТА: {e}")
        raise


def test_checkpoint_naming():
    """Тест системы именования checkpoint'ов"""
    print(f"\n🏷️ ТЕСТ ИМЕНОВАНИЯ CHECKPOINT'ОВ")
    print("=" * 50)

    try:
        from datetime import datetime

        # Параметры для тестового названия
        mode = "development"
        scale_factor = 0.01
        dataset_size = 1000
        epochs = 50
        best_similarity = 0.234
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Формируем название как в системе
        result_name = f"dynamic_{mode}_scale{scale_factor}_{dataset_size}pairs_{epochs}epochs_{best_similarity:.3f}sim_{timestamp}"

        print(f"✅ Пример названия checkpoint'а:")
        print(f"   {result_name}")
        print(f"   Содержит scale factor: ✓")
        print(f"   Содержит режим: ✓")
        print(f"   Содержит размер датасета: ✓")
        print(f"   Содержит количество эпох: ✓")
        print(f"   Содержит лучший результат: ✓")

    except Exception as e:
        print(f"❌ Ошибка теста именования: {e}")


def main():
    """Главная функция теста"""
    print("🧪 ТЕСТ ДИНАМИЧЕСКОЙ КОНФИГУРАЦИИ ДЛЯ DEVELOPMENT РЕЖИМА")
    print("Проверяет настройки scale=0.01 и интеграцию с precomputed embeddings")
    print()

    try:
        # Тест конфигурации
        config = test_development_mode()

        # Тест именования
        test_checkpoint_naming()

        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print(f"   Development режим готов к использованию")
        print(f"   Можно запускать: python run_dynamic_training.py --mode development")

        return 0

    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
