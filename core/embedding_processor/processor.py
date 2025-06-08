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
        
        # 1. EmbeddingReshaper для 1D↔3D конвертации (только для non-surface режимов)
        if config.processing_mode != ProcessingMode.SURFACE_ONLY:
            self.reshaper = self._init_embedding_reshaper()
        else:
            self.reshaper = None  # Не нужен для surface-only режима
            logger.info("📄 EmbeddingReshaper пропущен для SURFACE_ONLY режима")
        
        # 2. Lattice3D для 3D обработки (только для non-surface режимов)
        if config.processing_mode != ProcessingMode.SURFACE_ONLY:
            self.lattice = self._init_lattice_3d()
        else:
            self.lattice = None  # Не используется в surface-only режиме
            logger.info("🎲 Lattice3D пропущен для SURFACE_ONLY режима")
        
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
            input_embedding: Входной эмбединг [batch_size, dim] или [dim]
                           - Для обычных режимов: [batch_size, 768] или [768]
                           - Для SURFACE_ONLY: [batch_size, surface_size] или [surface_size]
            
        Returns:
            torch.Tensor: Обработанный эмбединг той же размерности
        """
        start_time = time.time()
        
        # Обеспечиваем batch dimension
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = input_embedding.shape[0]
        
        try:
            # Проверяем режим обработки
            if self.config.processing_mode == ProcessingMode.SURFACE_ONLY:
                # SURFACE-ONLY РЕЖИМ: прямая обработка surface embeddings
                output_batch = self._surface_only_processing(input_embedding)
            else:
                # СТАНДАРТНЫЙ РЕЖИМ: через EmbeddingReshaper и полный куб
                output_batch = self._standard_processing(input_embedding)
            
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
            elif self.config.processing_mode == ProcessingMode.SURFACE_ONLY:
                # Surface-only: должен использоваться другой pipeline
                logger.warning("⚠️  _process_through_lattice вызван для SURFACE_ONLY режима. Используйте _surface_only_processing.")
                return matrix_3d  # Возвращаем без изменений
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
    
    def _surface_only_processing(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """
        Surface-only обработка для Universal Adapter интеграции
        
        Args:
            input_embedding: Surface embeddings [batch_size, surface_size]
            
        Returns:
            torch.Tensor: Обработанные surface embeddings той же размерности
        """
        if self.config.debug_mode:
            logger.debug(f"🔄 Surface-only processing: {input_embedding.shape}")
        
        batch_size = input_embedding.shape[0]
        surface_size = input_embedding.shape[1]
        
        # Проверяем соответствие размера
        expected_surface_size = self.config.surface_dimensions[0] * self.config.surface_dimensions[1]
        if surface_size != expected_surface_size:
            logger.warning(f"Surface size mismatch: got {surface_size}, expected {expected_surface_size}")
        
        # Reshape surface embeddings в 2D surface для lattice processing
        h, w = self.config.surface_dimensions
        
        # Проход по каждому примеру в batch
        processed_surfaces = []
        
        for i in range(batch_size):
            surface_emb = input_embedding[i]  # [surface_size]
            
            # Reshape в 2D surface [height, width] 
            surface_2d = surface_emb.view(h, w)
            
            # Emergent processing через surface-aware метод
            processed_surface_2d = self._surface_emergent_processing(surface_2d)
            
            # Flatten обратно в 1D
            processed_surface_1d = processed_surface_2d.view(-1)
            processed_surfaces.append(processed_surface_1d)
        
        # Объединяем обратно в batch
        output_batch = torch.stack(processed_surfaces).to(self.device)
        
        if self.config.debug_mode:
            logger.debug(f"🎯 Surface-only результат: {output_batch.shape}")
        
        return output_batch
    
    def _standard_processing(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """
        Стандартная обработка через EmbeddingReshaper и полный куб
        
        Args:
            input_embedding: Входной эмбединг [batch_size, 768]
            
        Returns:
            torch.Tensor: Обработанный эмбединг [batch_size, 768]
        """
        batch_size = input_embedding.shape[0]
        
        # === ЭТАП 1: 1D → 3D ПРЕОБРАЗОВАНИЕ ===
        if self.config.debug_mode:
            logger.debug(f"🔄 Этап 1: Преобразование {input_embedding.shape} → 3D")
        
        # Список для хранения 3D матриц
        matrices_3d = []
        
        for i in range(batch_size):
            emb_1d = input_embedding[i]  # Сохраняем torch тензор для градиентов
            matrix_3d = self.reshaper.vector_to_matrix(emb_1d)  # EmbeddingReshaper поддерживает torch
            matrices_3d.append(matrix_3d)
        
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
            matrix_3d = processed_batch[i]  # Сохраняем torch тензор для градиентов
            emb_1d = self.reshaper.matrix_to_vector(matrix_3d)  # EmbeddingReshaper поддерживает torch
            output_embeddings.append(emb_1d)
        
        output_batch = torch.stack(output_embeddings).to(self.device)
        
        return output_batch
    
    def _surface_emergent_processing(self, surface_2d: torch.Tensor) -> torch.Tensor:
        """
        Emergent processing для surface embeddings согласно архитектуре из EMERGENT_ARCHITECTURE_CLARIFICATION
        
        Реализует emergent internal processing:
        - Input только на surface
        - Emergent internal layers (11 layers depth)  
        - Self-organization patterns
        - Output только с surface
        
        Args:
            surface_2d: 2D surface [height, width]
            
        Returns:
            torch.Tensor: Обработанная 2D surface [height, width]
        """
        h, w = surface_2d.shape
        depth = self.config.surface_processing_depth  # 11 layers по умолчанию
        
        if self.config.debug_mode:
            logger.debug(f"🧠 Emergent processing: surface {h}×{w}, depth {depth}")
        
        # Создаем 3D representation для emergent processing
        # surface → volume → surface (emergent internal behavior)
        
        # 1. Расширение surface в 3D volume через learned patterns
        volume_3d = self._expand_surface_to_volume(surface_2d, depth)
        
        # 2. Emergent spatial propagation через internal layers
        processed_volume = self._emergent_spatial_propagation(volume_3d)
        
        # 3. Extraction результата обратно в surface
        output_surface = self._extract_surface_from_volume(processed_volume)
        
        return output_surface
    
    def _expand_surface_to_volume(self, surface_2d: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Расширение surface в 3D volume для internal processing
        
        Эмулирует "surface injection" → "internal propagation"
        """
        h, w = surface_2d.shape
        
        # Создаем 3D volume и инициализируем surface слои
        volume = torch.zeros(depth, h, w, device=surface_2d.device, dtype=surface_2d.dtype)
        
        # Front surface (input layer)
        volume[0] = surface_2d.clone()
        
        # Propagation в internal layers через learned patterns
        for layer in range(1, depth):
            # Простая emergent propagation (можно заменить на более сложную)
            prev_layer = volume[layer - 1]
            
            # Spatial diffusion + learnable transformation
            diffused = self._spatial_diffusion(prev_layer)
            
            # Learnable layer transformation (если needed)
            volume[layer] = diffused
        
        return volume
    
    def _spatial_diffusion(self, layer_2d: torch.Tensor) -> torch.Tensor:
        """Spatial diffusion для emergent propagation"""
        
        # Простой spatial filter для diffusion
        # В реальной реализации здесь будут learnable параметры
        
        h, w = layer_2d.shape
        result = layer_2d.clone()
        
        # Применяем spatial averaging с небольшими трансформациями
        if h > 2 and w > 2:
            # Простая convolution-like операция
            center = layer_2d[1:-1, 1:-1]
            neighbors = (
                layer_2d[:-2, 1:-1] + layer_2d[2:, 1:-1] +   # vertical neighbors
                layer_2d[1:-1, :-2] + layer_2d[1:-1, 2:]     # horizontal neighbors
            ) / 4.0
            
            # Mixing с центральными значениями
            alpha = 0.7  # Learnable parameter в реальной реализации
            result[1:-1, 1:-1] = alpha * center + (1 - alpha) * neighbors
        
        # Добавляем немного нелинейности для emergent behavior
        result = torch.tanh(result)
        
        return result
    
    def _emergent_spatial_propagation(self, volume_3d: torch.Tensor) -> torch.Tensor:
        """
        Emergent spatial propagation через все internal layers
        
        Эмулирует self-organization и emergent patterns
        """
        depth, h, w = volume_3d.shape
        result = volume_3d.clone()
        
        # Multiple propagation steps для emergent behavior
        for step in range(self.config.propagation_steps):
            
            # Cross-layer influence (depth propagation)
            for layer in range(1, depth - 1):
                prev_layer = result[layer - 1]
                curr_layer = result[layer]
                next_layer = result[layer + 1]
                
                # Emergent mixing между layers
                depth_mixing = (prev_layer + 2 * curr_layer + next_layer) / 4.0
                
                # Spatial mixing в текущем layer
                spatial_mixed = self._spatial_diffusion(depth_mixing)
                
                # Update с learnable mixing ratio
                beta = 0.8  # В реальной реализации - learnable parameter
                result[layer] = beta * curr_layer + (1 - beta) * spatial_mixed
        
        return result
    
    def _extract_surface_from_volume(self, volume_3d: torch.Tensor) -> torch.Tensor:
        """
        Extraction финального surface из обработанного volume
        
        Эмулирует "output surface extraction" 
        """
        depth, h, w = volume_3d.shape
        
        # Back surface (output layer) 
        back_surface = volume_3d[-1]
        
        # Можно также смешивать несколько layers для более богатого output
        # Weighted combination последних layers
        if depth >= 3:
            # Взвешенная комбинация последних 3 layers
            weights = torch.tensor([0.2, 0.3, 0.5], device=volume_3d.device)  # Больше вес для последнего layer
            weighted_output = (
                weights[0] * volume_3d[-3] + 
                weights[1] * volume_3d[-2] + 
                weights[2] * volume_3d[-1]
            )
            return weighted_output
        else:
            return back_surface 