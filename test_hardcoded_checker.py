#!/usr/bin/env python3
"""
Тест системы проверки hardcoded значений
========================================

Демонстрирует как работает система предотвращения hardcoded значений.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import create_experiment_config, set_project_config
from new_rebuild.utils import (
    check_hardcoded_value, 
    strict_no_hardcoded, 
    no_hardcoded,
    HardcodedValueError,
    allow_hardcoded,
)


def example_with_hardcoded_values():
    """Пример функции с hardcoded значениями - должна выбросить ошибку"""
    # Эти значения должны вызвать ошибку
    learning_rate = 1e-4  # ❌ Hardcoded!
    dropout = 0.1  # ❌ Hardcoded!
    batch_size = 64  # ❌ Hardcoded!
    
    print(f"LR: {learning_rate}, Dropout: {dropout}, Batch: {batch_size}")


@no_hardcoded
def example_with_decorator(lr=1e-4, dropout=0.1):
    """Функция с декоратором проверки - должна выбросить ошибку"""
    return lr * dropout


def example_correct_decorator_usage(lr=None, dropout=None):
    """Правильное использование с дефолтными значениями из конфига"""
    from new_rebuild.config import get_project_config
    config = get_project_config()
    
    # Используем значения из конфига если не переданы
    if lr is None:
        lr = config.training_optimizer.learning_rate
    if dropout is None:
        dropout = config.architecture.cnf_dropout_rate
        
    # Теперь можно безопасно использовать
    return lr * dropout


def example_with_strict_check():
    """Пример использования strict_no_hardcoded"""
    # Это автоматически заменит hardcoded значение на значение из конфига
    max_neighbors = strict_no_hardcoded(1000, "architecture.spatial_max_neighbors")
    print(f"Max neighbors из конфига: {max_neighbors}")
    
    # А это выбросит ошибку если параметра нет в конфиге
    try:
        some_value = strict_no_hardcoded(12345, "non.existent.param")
    except HardcodedValueError as e:
        print(f"Ожидаемая ошибка: {e}")


def example_correct_usage():
    """Пример правильного использования - через конфиг"""
    from new_rebuild.config import get_project_config
    config = get_project_config()
    
    # ✅ Правильно - используем значения из конфига
    learning_rate = config.training_optimizer.learning_rate
    dropout = config.embedding_mapping.dropout_rate
    max_neighbors = config.architecture.spatial_max_neighbors
    
    print(f"✅ Правильно: LR={learning_rate}, Dropout={dropout}, Neighbors={max_neighbors}")
    
    # Маленькие числа разрешены
    for i in range(5):  # ✅ OK - маленькое число
        x = i * 2  # ✅ OK
    
    # Базовые константы тоже OK
    if True:  # ✅ OK
        y = 0.0  # ✅ OK
        z = 1.0  # ✅ OK


def example_with_context_manager():
    """Пример временного отключения проверок (только для миграции!)"""
    
    # Обычно это вызовет ошибку
    try:
        check_hardcoded_value(1e-4, "test context")
    except HardcodedValueError:
        print("✅ Проверка работает - hardcoded значение обнаружено")
    
    # Но можно временно отключить (ТОЛЬКО для миграции!)
    with allow_hardcoded("демонстрация для теста"):
        learning_rate = 1e-4  # Временно разрешено
        check_hardcoded_value(1e-4, "test context")  # Не выбросит ошибку
        print("⚠️ Внутри контекста hardcoded разрешены")
    
    # После выхода из контекста снова запрещены
    try:
        check_hardcoded_value(1e-4, "test context")
    except HardcodedValueError:
        print("✅ После контекста проверка снова работает")


def demonstrate_all_cases():
    """Демонстрация всех случаев использования"""
    
    print("\n=== 1. Правильное использование (через конфиг) ===")
    example_correct_usage()
    
    print("\n=== 2. Использование strict_no_hardcoded ===")
    example_with_strict_check()
    
    print("\n=== 3. Проверка простых hardcoded значений ===")
    try:
        check_hardcoded_value(8000, "MoE functional params")
    except HardcodedValueError as e:
        print(f"Поймана ошибка:\n{e}")
    
    print("\n=== 4. Проверка с декоратором ===")
    try:
        # Декоратор проверяет переданные аргументы
        example_with_decorator(lr=1e-4, dropout=0.1)
    except HardcodedValueError as e:
        print(f"Поймана ошибка:\n{e}")
    
    # Проверим что без аргументов (используя дефолты) тоже работает
    try:
        result = example_with_decorator()
        print(f"⚠️ Дефолтные значения в определении функции не проверяются автоматически")
        print(f"   Используйте strict_no_hardcoded внутри функции для дефолтов")
    except HardcodedValueError as e:
        print(f"Поймана ошибка:\n{e}")
    
    print("\n=== 5. Контекстный менеджер для миграции ===")
    example_with_context_manager()
    
    print("\n=== 6. Проверка функции с hardcoded ===")
    try:
        example_with_hardcoded_values()
        print("⚠️ Функция выполнилась (проверка вручную не добавлена)")
        print("   Используйте @no_hardcoded декоратор или check_hardcoded_value внутри функции")
    except Exception as e:
        print(f"Ошибка: {e}")
        
    print("\n=== 7. Правильное использование с дефолтными значениями ===")
    result = example_correct_decorator_usage()
    print(f"✅ Функция использует значения из конфига по умолчанию")
    print(f"   Результат: {result}")


def main():
    """Основная функция"""
    print("🔍 Демонстрация системы проверки hardcoded значений")
    print("=" * 60)
    
    # Устанавливаем конфиг для тестов
    config = create_experiment_config()
    set_project_config(config)
    
    # Демонстрируем все случаи
    demonstrate_all_cases()
    
    print("\n" + "=" * 60)
    print("✅ Демонстрация завершена!")
    print("\n💡 Рекомендации:")
    print("1. Всегда используйте значения из config вместо hardcoded")
    print("2. Применяйте @no_hardcoded декоратор к новым функциям")
    print("3. Используйте strict_no_hardcoded() для автоматической замены")
    print("4. allow_hardcoded() только для временной миграции!")


if __name__ == "__main__":
    main()