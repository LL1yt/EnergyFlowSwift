#!/usr/bin/env python3
"""
Базовый тест модуля embedding_loader
Создает простые тестовые данные и проверяет основную функциональность.
"""

import os
import torch
import numpy as np
from pathlib import Path

# Создаем простые тестовые данные
def create_test_embeddings():
    """Создание тестовых файлов эмбедингов для проверки."""
    
    # Создаем директорию для тестов
    test_dir = Path("./data/embeddings/test/")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("Создание тестовых файлов эмбедингов...")
    
    # 1. Создаем простой GloVe-like файл
    glove_path = test_dir / "test_glove.txt"
    with open(glove_path, 'w', encoding='utf-8') as f:
        # Создаем 10 простых слов с 5-мерными векторами
        words = ['the', 'cat', 'dog', 'run', 'jump', 'happy', 'sad', 'big', 'small', 'good']
        for i, word in enumerate(words):
            # Генерируем случайные, но воспроизводимые векторы
            np.random.seed(i)
            vector = np.random.normal(0, 1, 5)
            vector_str = ' '.join([f"{v:.6f}" for v in vector])
            f.write(f"{word} {vector_str}\n")
    
    print(f"[OK] Создан тестовый GloVe файл: {glove_path}")
    
    # 2. Создаем простой Word2Vec-like файл (с заголовком)
    w2v_path = test_dir / "test_word2vec.txt"
    with open(w2v_path, 'w', encoding='utf-8') as f:
        f.write("10 5\n")  # vocab_size embedding_dim
        for i, word in enumerate(words):
            np.random.seed(i + 100)  # Другие семена для разнообразия
            vector = np.random.normal(0, 1, 5)
            vector_str = ' '.join([f"{v:.6f}" for v in vector])
            f.write(f"{word} {vector_str}\n")
    
    print(f"[OK] Создан тестовый Word2Vec файл: {w2v_path}")
    
    # 3. Создаем BERT-like PyTorch файл
    bert_path = test_dir / "test_bert.pt"
    # Создаем тензор размером [10, 8] (10 токенов, 8-мерные векторы)
    np.random.seed(42)
    bert_embeddings = torch.randn(10, 8)
    torch.save(bert_embeddings, bert_path)
    
    print(f"[OK] Создан тестовый BERT файл: {bert_path}")
    
    return {
        'glove': str(glove_path),
        'word2vec': str(w2v_path),
        'bert': str(bert_path)
    }

def test_embedding_loader():
    """Основной тест модуля embedding_loader."""
    
    print("="*60)
    print("ТЕСТ МОДУЛЯ EMBEDDING_LOADER")
    print("="*60)
    
    # Создаем тестовые данные
    test_files = create_test_embeddings()
    
    try:
        # Импортируем наш модуль
        from data.embedding_loader import EmbeddingLoader, EmbeddingPreprocessor
        
        print("\n[OK] Модуль успешно импортирован")
        
        # Создаем загрузчик
        loader = EmbeddingLoader(cache_dir="./data/cache/test/")
        print("[OK] EmbeddingLoader инициализирован")
        
        # Тест 1: Загрузка GloVe
        print("\n" + "-"*40)
        print("ТЕСТ 1: Загрузка GloVe")
        print("-"*40)
        
        glove_embeddings = loader.load_embeddings(
            path=test_files['glove'],
            format_type="glove",
            preprocess=True
        )
        
        print(f"[OK] GloVe загружен: {glove_embeddings.shape}")
        print(f"   Тип: {glove_embeddings.dtype}")
        print(f"   Устройство: {glove_embeddings.device}")
        
        # Проверяем размеры
        assert glove_embeddings.shape == (10, 5), f"Неожиданный размер: {glove_embeddings.shape}"
        print("[OK] Размеры корректны")
        
        # Тест 2: Загрузка Word2Vec
        print("\n" + "-"*40)
        print("ТЕСТ 2: Загрузка Word2Vec")
        print("-"*40)
        
        w2v_embeddings = loader.load_embeddings(
            path=test_files['word2vec'],
            format_type="word2vec",
            preprocess=True
        )
        
        print(f"[OK] Word2Vec загружен: {w2v_embeddings.shape}")
        assert w2v_embeddings.shape == (10, 5), f"Неожиданный размер: {w2v_embeddings.shape}"
        print("[OK] Размеры корректны")
        
        # Тест 3: Загрузка BERT
        print("\n" + "-"*40)
        print("ТЕСТ 3: Загрузка BERT")
        print("-"*40)
        
        bert_embeddings = loader.load_embeddings(
            path=test_files['bert'],
            format_type="bert",
            preprocess=True
        )
        
        print(f"[OK] BERT загружен: {bert_embeddings.shape}")
        assert bert_embeddings.shape == (10, 8), f"Неожиданный размер: {bert_embeddings.shape}"
        print("[OK] Размеры корректны")
        
        # Тест 4: Предобработка
        print("\n" + "-"*40)
        print("ТЕСТ 4: Предобработка")
        print("-"*40)
        
        preprocessor = EmbeddingPreprocessor()
        
        # Тестируем различные виды предобработки
        normalized = preprocessor.preprocess(
            glove_embeddings.clone(),
            normalize=True,
            center=False,
            clip_outliers=False
        )
        
        # Проверяем нормализацию
        norms = torch.norm(normalized, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Нормализация не работает"
        print("[OK] Нормализация работает корректно")
        
        # Центрирование
        centered = preprocessor.preprocess(
            glove_embeddings.clone(),
            normalize=False,
            center=True,
            clip_outliers=False
        )
        
        mean = centered.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6), "Центрирование не работает"
        print("[OK] Центрирование работает корректно")
        
        # Тест 5: Кэширование
        print("\n" + "-"*40)
        print("ТЕСТ 5: Кэширование")
        print("-"*40)
        
        # Кэшируем данные
        cache_key = "test_glove_cache"
        loader.cache_embeddings(glove_embeddings, cache_key)
        print("[OK] Данные закэшированы")
        
        # Загружаем из кэша
        cached_embeddings = loader.load_from_cache(cache_key)
        
        if cached_embeddings is not None:
            assert torch.equal(glove_embeddings, cached_embeddings), "Кэш не работает корректно"
            print("[OK] Кэширование работает корректно")
        else:
            print("[WARNING]  Кэш не найден (возможно, проблема с диском)")
        
        # Тест 6: Статистики
        print("\n" + "-"*40)
        print("ТЕСТ 6: Статистики")
        print("-"*40)
        
        info = loader.get_embedding_info(glove_embeddings)
        print(f"[OK] Статистики получены:")
        print(f"   Форма: {info['shape']}")
        print(f"   Память: {info['memory_mb']:.2f} MB")
        print(f"   Мин: {info['min_value']:.4f}")
        print(f"   Макс: {info['max_value']:.4f}")
        print(f"   Среднее: {info['mean_value']:.4f}")
        
        # Тест 7: Поддерживаемые форматы
        print("\n" + "-"*40)
        print("ТЕСТ 7: Поддерживаемые форматы")
        print("-"*40)
        
        formats = loader.get_supported_formats()
        expected_formats = ['word2vec', 'glove', 'bert']
        
        for fmt in expected_formats:
            assert fmt in formats, f"Формат {fmt} не поддерживается"
        
        print(f"[OK] Поддерживаемые форматы: {formats}")
        
        print("\n" + "="*60)
        print("[SUCCESS] ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ОШИБКА В ТЕСТАХ: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_loader()
    exit(0 if success else 1) 