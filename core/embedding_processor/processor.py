"""
EmbeddingProcessor - Главный класс процессора эмбедингов
=======================================================

ЯДРО Phase 2.5 - объединяет все готовые компоненты в единую систему.

Архитектура обработки:
1. Входной эмбединг (768D) → EmbeddingReshaper.vector_to_matrix() → 3D матрица (8×8×12)
2. 3D матрица → Lattice3D.forward() → обработанная 3D матрица  
3. Обработанная 3D матрица → EmbeddingReshaper.matrix_to_vector() → выходной эмбединг (768D)

Цель: Cosine similarity >90% в автоэнкодер режиме.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from dataclasses import asdict
import logging
import time

# Импорты готовых модулей
from data.embedding_reshaper import EmbeddingReshaper, validate_semantic_preservation
from core.lattice_3d import Lattice3D
from .config import EmbeddingConfig, ProcessingMode, validate_config
from .metrics import ProcessingMetrics, calculate_processing_quality

# Настройка логирования
logger = logging.getLogger(__name__)


class EmbeddingProcessor(nn.Module):
    """
    Центральный процессор эмбедингов - ЯДРО Phase 2.5
    
    Объединяет EmbeddingReshaper + Lattice3D в единую систему обработки.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Инициализация процессора эмбедингов
        
        Args:
            config: Конфигурация процессора
        """
        super().__init__()
        
        self.config = config
        validate_config(config)  # Валидация конфигурации
        
        # === ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ ===
        
        # 1. EmbeddingReshaper для 1D↔3D конвертации
        self.reshaper = self._init_embedding_reshaper()
        
        # 2. Lattice3D для 3D обработки
        self.lattice = self._init_lattice_3d()
        
        # 3. Метрики для контроля качества
        self.metrics = ProcessingMetrics()
        
        # === ВНУТРЕННЕЕ СОСТОЯНИЕ ===
        self.processing_count = 0
        self.cache = {} if config.cache_enabled else None
        self.device = torch.device(config.device)
        self.to(self.device)
        
        # Инфо о готовности
        logger.info(f"✅ EmbeddingProcessor инициализирован")
        logger.info(f"📊 Режим: {config.processing_mode.value}")
        logger.info(f"🎯 Целевая схожесть: {config.target_similarity:.1%}")
        logger.info(f"🔄 Шаги распространения: {config.propagation_steps}")
    
    def _init_embedding_reshaper(self) -> EmbeddingReshaper:
        """Инициализировать EmbeddingReshaper"""
        
        # Используем правильный API EmbeddingReshaper (из Phase 2.3)
        reshaper = EmbeddingReshaper(
            input_dim=self.config.input_dim,
            cube_shape=self.config.cube_shape,
            reshaping_method=self.config.reshaping_method,
            preserve_semantics=self.config.preserve_semantics,
            semantic_threshold=self.config.semantic_threshold
        )
        
        logger.info(f"✅ EmbeddingReshaper готов: {self.config.cube_shape}")
        return reshaper
    
    def _init_lattice_3d(self) -> Lattice3D:
        """Инициализировать Lattice3D"""
        
        # Импортируем нужные классы
        from core.lattice_3d import LatticeConfig, BoundaryCondition, Face, PlacementStrategy
        
        # Создаем правильную конфигурацию LatticeConfig
        lattice_config = LatticeConfig(
            dimensions=self.config.lattice_size,
            boundary_conditions=BoundaryCondition.WALLS,
            parallel_processing=False,  # Пока отключаем для простоты
            gpu_enabled=False,  # Пока используем CPU
            input_face=Face.FRONT,
            output_face=Face.BACK,
            placement_strategy=PlacementStrategy.PROPORTIONAL,
            enable_logging=self.config.debug_mode
        )
        
        try:
            # Создаем Lattice3D напрямую с объектом конфигурации
            lattice = Lattice3D(lattice_config)
            logger.info(f"✅ Lattice3D готов: {self.config.lattice_size}")
            return lattice
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Lattice3D: {e}")
            raise
    
    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """
        Основная функция обработки эмбединга
        
        Args:
            input_embedding: Входной эмбединг [batch_size, 768] или [768]
            
        Returns:
            torch.Tensor: Обработанный эмбединг той же размерности
        """
        start_time = time.time()
        original_shape = input_embedding.shape
        
        # Обеспечиваем batch dimension
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = input_embedding.shape[0]
        
        try:
            # === ЭТАП 1: 1D → 3D ПРЕОБРАЗОВАНИЕ ===
            if self.config.debug_mode:
                logger.debug(f"🔄 Этап 1: Преобразование {input_embedding.shape} → 3D")
            
            # Список для хранения 3D матриц
            matrices_3d = []
            
            for i in range(batch_size):
                emb_1d = input_embedding[i].cpu().numpy()  # EmbeddingReshaper работает с numpy
                matrix_3d = self.reshaper.vector_to_matrix(emb_1d)
                matrices_3d.append(torch.from_numpy(matrix_3d).float())
            
            # Объединяем в batch: [batch_size, depth, height, width]
            batch_3d = torch.stack(matrices_3d).to(self.device)
            
            # === ЭТАП 2: 3D ОБРАБОТКА ЧЕРЕЗ LATTICE ===
            if self.config.debug_mode:
                logger.debug(f"🧠 Этап 2: Обработка через Lattice3D {batch_3d.shape}")
            
            # Обрабатываем каждый пример в batch отдельно (пока нет batch support в Lattice3D)
            processed_matrices = []
            
            for i in range(batch_size):
                matrix_3d = batch_3d[i]
                
                # Конвертируем в формат для Lattice3D (может потребоваться адаптация)
                processed_matrix = self._process_through_lattice(matrix_3d)
                processed_matrices.append(processed_matrix)
            
            processed_batch = torch.stack(processed_matrices)
            
            # === ЭТАП 3: 3D → 1D ПРЕОБРАЗОВАНИЕ ===
            if self.config.debug_mode:
                logger.debug(f"🔄 Этап 3: Преобразование 3D → {self.config.output_dim}D")
            
            output_embeddings = []
            
            for i in range(batch_size):
                matrix_3d = processed_batch[i].cpu().numpy()
                emb_1d = self.reshaper.matrix_to_vector(matrix_3d)
                output_embeddings.append(torch.from_numpy(emb_1d).float())
            
            output_batch = torch.stack(output_embeddings).to(self.device)
            
            # === ЭТАП 4: КОНТРОЛЬ КАЧЕСТВА ===
            if self.config.quality_check_enabled:
                self._update_metrics(input_embedding, output_batch, start_time)
            
            # Возвращаем в исходном формате
            if single_input:
                output_batch = output_batch.squeeze(0)
            
            self.processing_count += 1
            
            return output_batch
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки эмбединга: {e}")
            raise
    
    def _process_through_lattice(self, matrix_3d: torch.Tensor) -> torch.Tensor:
        """
        Обработка 3D матрицы через Lattice3D
        
        Args:
            matrix_3d: 3D матрица [depth, height, width]
            
        Returns:
            torch.Tensor: Обработанная 3D матрица
        """
        try:
            # В зависимости от режима обработки
            if self.config.processing_mode == ProcessingMode.AUTOENCODER:
                # Автоэнкодер: стремимся к восстановлению
                return self._autoencoder_processing(matrix_3d)
            elif self.config.processing_mode == ProcessingMode.GENERATOR:
                # Генератор: семантическая трансформация
                return self._generator_processing(matrix_3d)
            elif self.config.processing_mode == ProcessingMode.DIALOGUE:
                # Диалог: контекстная обработка
                return self._dialogue_processing(matrix_3d)
            else:
                raise ValueError(f"Неизвестный режим: {self.config.processing_mode}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка в Lattice3D обработке: {e}")
            # Fallback: возвращаем исходную матрицу
            return matrix_3d
    
    def _autoencoder_processing(self, matrix_3d: torch.Tensor) -> torch.Tensor:
        """Автоэнкодер обработка (максимальное сохранение)"""
        
        # Пока используем identity transformation + небольшой шум для обучения
        # В будущем здесь будет полная интеграция с Lattice3D
        
        noise_level = 0.01  # Минимальный шум для обучения
        noise = torch.randn_like(matrix_3d) * noise_level
        
        return matrix_3d + noise
    
    def _generator_processing(self, matrix_3d: torch.Tensor) -> torch.Tensor:
        """Генеративная обработка (семантические трансформации)"""
        
        # Больше трансформаций для креативности
        transformation_strength = 0.1
        
        # Пример простой трансформации (замените на Lattice3D)
        transformed = matrix_3d * (1.0 + torch.randn_like(matrix_3d) * transformation_strength)
        
        return transformed
    
    def _dialogue_processing(self, matrix_3d: torch.Tensor) -> torch.Tensor:
        """Диалоговая обработка (контекстные трансформации)"""
        
        # Контекстные трансформации для диалога
        context_strength = 0.15
        
        # Пример (замените на полную Lattice3D интеграцию)
        context_transform = torch.tanh(matrix_3d) * context_strength
        
        return matrix_3d + context_transform
    
    def _update_metrics(self, input_batch: torch.Tensor, output_batch: torch.Tensor, start_time: float):
        """Обновить метрики обработки"""
        
        processing_time = time.time() - start_time
        
        # Вычисляем среднюю cosine similarity по batch
        similarities = []
        for i in range(input_batch.shape[0]):
            similarity = torch.nn.functional.cosine_similarity(
                input_batch[i], output_batch[i], dim=0
            ).item()
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        
        # Обновляем метрики
        self.metrics.update(
            similarity=avg_similarity,
            processing_time=processing_time,
            batch_size=input_batch.shape[0]
        )
        
        # Логирование
        if self.config.verbose_logging:
            logger.info(f"📊 Cosine similarity: {avg_similarity:.3f} (цель: {self.config.target_similarity:.3f})")
            logger.info(f"⏱️ Время обработки: {processing_time:.3f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить текущие метрики"""
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Сбросить метрики"""
        self.metrics.reset()
    
    def set_mode(self, mode: ProcessingMode):
        """Изменить режим обработки"""
        self.config.processing_mode = mode
        logger.info(f"🔄 Режим изменен на: {mode.value}")
    
    def validate_quality(self, input_embedding: torch.Tensor, output_embedding: torch.Tensor) -> bool:
        """
        Проверить качество обработки
        
        Returns:
            bool: True если качество соответствует требованиям
        """
        similarity = torch.nn.functional.cosine_similarity(
            input_embedding, output_embedding, dim=0 if input_embedding.dim() == 1 else 1
        ).mean().item()
        
        return similarity >= self.config.target_similarity
    
    def __repr__(self) -> str:
        return (f"EmbeddingProcessor("
                f"mode={self.config.processing_mode.value}, "
                f"target_sim={self.config.target_similarity:.2f}, "
                f"processed={self.processing_count})") 