"""
Training Stage Runner - Выполнитель тренировочных стадий

Этот модуль отвечает за выполнение отдельных стадий обучения в рамках
автоматизированного процесса. Управляет subprocess-ами, мониторит
выполнение и обрабатывает результаты.
"""

import sys
import time
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

from .types import StageConfig, StageResult
from .process_runner import run_process
from utils.config_manager.dynamic_config import DynamicConfigManager
from utils.config_manager import get_global_config_manager

logger = logging.getLogger(__name__)


class TrainingStageRunner:
    """Выполнитель тренировочных стадий"""

    def __init__(
        self,
        mode: str = "development",
        scale: Optional[float] = None,
        timeout_multiplier: float = 2.0,
        verbose: bool = False,
    ):
        """
        Args:
            mode: Режим конфигурации (development, research, etc.)
            scale: Custom scale factor
            timeout_multiplier: Multiplier for the timeout
            verbose: Enable verbose logging for subprocess operations
        """
        self.mode = mode
        self.scale = scale
        self.timeout_multiplier = timeout_multiplier
        self.verbose = verbose

        # Минимальное логирование инициализации
        if timeout_multiplier > 2.0:  # Логируем только нестандартные значения
            logger.warning(f"[RUNNER] High timeout multiplier: {timeout_multiplier}")
        if verbose:
            logger.info(f"[RUNNER] Verbose mode enabled for subprocess logging")

    def run_stage(
        self, stage_config: StageConfig, estimated_time: float
    ) -> Optional[StageResult]:
        """
        Запускает одну стадию обучения

        Args:
            stage_config: Конфигурация стадии
            estimated_time: Оценочное время выполнения в минутах

        Returns:
            StageResult: Результат выполнения или None в случае ошибки
        """
        output_json_path = None
        temp_config_path = None
        try:
            # Создаем временный файл для JSON-результатов
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", encoding="utf-8"
            ) as tmp_json_file:
                output_json_path = tmp_json_file.name

            # Генерируем временный файл конфигурации
            temp_config_path = self._generate_temp_config(stage_config)
            if not temp_config_path:
                logger.error(
                    f"❌ Stage {stage_config.stage} failed: Could not generate temp config."
                )
                return None

            # Строим команду с путями к файлам
            cmd = self._build_command(stage_config, output_json_path, temp_config_path)

            # Запускаем обучение с опциональным verbose логированием
            start_time = time.time()
            timeout_seconds = estimated_time * 60 * self.timeout_multiplier

            if self.verbose:
                logger.info(
                    f"🔄 Starting Stage {stage_config.stage}: {stage_config.description}"
                )
                logger.info(
                    f"   Dataset: {stage_config.dataset_limit:,} samples, {stage_config.epochs} epochs"
                )
                logger.info(f"   Timeout: {timeout_seconds/60:.1f} minutes")
                logger.info(f"   Temp config: {temp_config_path}")

            result = run_process(
                cmd,
                timeout_seconds,
                process_id=f"Stage {stage_config.stage}",
                verbose=self.verbose,
            )

            end_time = time.time()
            actual_time = (end_time - start_time) / 60  # в минутах

            if result is None:
                logger.error(
                    f"❌ Stage {stage_config.stage} failed after {actual_time:.1f}min"
                )
                return None

            return self._process_result(
                result, output_json_path, stage_config, actual_time, estimated_time
            )

        except Exception as e:
            logger.error(f"❌ Stage {stage_config.stage} exception: {e}")
            return None
        finally:
            # Очищаем временные файлы
            import os

            if output_json_path and os.path.exists(output_json_path):
                os.remove(output_json_path)
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def _generate_temp_config(
        self, stage_config: Optional[StageConfig] = None
    ) -> Optional[str]:
        """Генерирует временный YAML файл с динамической конфигурацией."""
        try:
            logger.info(
                f"Generating temporary config for subprocess (mode={self.mode}, scale={self.scale})..."
            )

            # 1. Создаем менеджер
            dynamic_manager = DynamicConfigManager()

            # 2. Если есть кастомный scale, применяем его
            if self.scale is not None:
                setattr(dynamic_manager.generator.scale_settings, self.mode, self.scale)
                logger.info(
                    f"Applied custom scale factor: {self.scale} for mode '{self.mode}'"
                )

            # 3. Генерируем конфигурацию для нужного режима
            config_data = dynamic_manager.create_config_for_mode(self.mode)

            # === PHASE 4 INTEGRATION: Plasticity & Optimization ===
            if stage_config is not None:
                config_data = self._prepare_config_with_optimizations(
                    config_data, stage_config
                )

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".yaml", encoding="utf-8"
            ) as tmp_file:
                import yaml

                yaml.dump(config_data, tmp_file, allow_unicode=True)
                logger.info(f"Temporary config generated at {tmp_file.name}")
                return tmp_file.name
        except Exception as e:
            logger.error(
                f"Failed to generate temporary config file: {e}", exc_info=True
            )
            return None

    def _prepare_config_with_optimizations(
        self, config_data: Dict[str, Any], stage_config: StageConfig
    ) -> Dict[str, Any]:
        """
        Применяет оптимизации Phase 4 к конфигурации стадии обучения

        Args:
            config_data: Базовая конфигурация
            stage_config: Параметры стадии обучения

        Returns:
            Оптимизированная конфигурация
        """
        # PHASE 4 CRITICAL FIX: Принудительно используем hybrid архитектуру
        self._apply_hybrid_architecture(config_data)

        # Применяем пластичность по профилю
        self._apply_plasticity_profile(config_data, stage_config)

        # Применяем progressive scaling
        if getattr(stage_config, "progressive_scaling", False):
            self._apply_progressive_scaling(config_data, stage_config)

        # Применяем memory optimizations
        if getattr(stage_config, "memory_optimizations", False):
            self._apply_memory_optimizations(config_data, stage_config)

        # Применяем emergence tracking
        if getattr(stage_config, "emergence_tracking", False):
            self._apply_emergence_tracking(config_data, stage_config)

        return config_data

    def _apply_hybrid_architecture(self, config_data: Dict[str, Any]):
        """
        PHASE 4 CRITICAL: Принудительно применяет hybrid NCA+gMLP архитектуру
        """
        logger.info("🔧 PHASE 4: Applying hybrid NCA+gMLP architecture...")

        # 1. Устанавливаем hybrid mode
        config_data["architecture"] = {
            "hybrid_mode": True,
            "neuron_architecture": "minimal_nca",
            "connection_architecture": "gated_mlp",
            "disable_nca_scaling": True,
        }

        # 2. Конфигурация lattice для hybrid режима
        if "lattice_3d" not in config_data:
            config_data["lattice_3d"] = {}

        # Обновляем lattice конфигурацию на правильные значения
        config_data["lattice_3d"].update(
            {
                "dimensions": [16, 16, 16],  # PHASE 4: 16×16×16 вместо старых размеров
                "total_cells": 4096,  # 16×16×16 = 4096
                "neighbors": 26,  # PHASE 4: 26 neighbors вместо 6
                "neighbor_finding_strategy": "tiered",
            }
        )

        # 3. Конфигурация NCA нейронов
        config_data["minimal_nca_cell"] = {
            "state_size": 4,
            "neighbor_count": 26,  # Синхронизировать с lattice
            "hidden_dim": 3,
            "external_input_size": 1,
            "activation": "tanh",
            "target_params": 362,
            "alpha": 0.1,
            "beta": 0.05,
            "enable_lattice_scaling": False,
        }

        # 4. Конфигурация gMLP связей
        config_data["gmlp_cell"] = {
            "state_size": 8,
            "neighbor_count": 26,  # Синхронизировать с lattice
            "hidden_dim": 16,
            "external_input_size": 4,
            "use_memory": False,  # PHASE 4 FIX: отключаем memory
            "target_params": 2000,
            "activation": "gelu",
        }

        # 5. Cell prototype configuration
        config_data["cell_prototype"] = {
            "prototype_name": "minimal_nca_cell",  # Принудительно устанавливаем NCA
            "minimal_nca_cell": config_data["minimal_nca_cell"],
            "gmlp_cell": config_data["gmlp_cell"],
        }

        # 6. Обновляем fallback значения
        if "lattice" not in config_data:
            config_data["lattice"] = {}

        config_data["lattice"].update(
            {
                "xs": 16,
                "ys": 16,
                "zs": 16,
                "total_neurons": 4096,
                "connectivity": "26-neighbors",
            }
        )

        logger.info("✅ PHASE 4: Hybrid architecture configuration applied!")
        logger.info(f"   - Architecture: hybrid NCA+gMLP")
        logger.info(f"   - Lattice: 16×16×16 = 4096 cells")
        logger.info(f"   - Neighbors: 26 (3D Moore)")
        logger.info(
            f"   - NCA params: {config_data['minimal_nca_cell']['target_params']}"
        )
        logger.info(f"   - gMLP params: {config_data['gmlp_cell']['target_params']}")

    def _apply_plasticity_profile(
        self, config_data: Dict[str, Any], stage_config: StageConfig
    ):
        """
        Применяет пластичность по профилю
        """
        # Получаем генератор для создания секций
        dynamic_manager = DynamicConfigManager()
        generator = dynamic_manager.generator

        # Создаем контекст стадии
        stage_context = {
            "plasticity_profile": stage_config.plasticity_profile,
            "clustering_enabled": stage_config.clustering_enabled,
            "activity_threshold": stage_config.activity_threshold,
            "memory_optimizations": stage_config.memory_optimizations,
            "emergence_tracking": stage_config.emergence_tracking,
            "sparse_connection_ratio": stage_config.sparse_connection_ratio,
            "progressive_scaling": stage_config.progressive_scaling,
            "decoder_monitoring": stage_config.decoder_monitoring,
            "stage_number": stage_config.stage,
        }

        # Генерируем секции пластичности и оптимизации
        plasticity_section = generator.generate_plasticity_section(stage_context)
        optimization_section = generator.generate_optimization_section(stage_context)

        # Интегрируем в основную конфигурацию
        if plasticity_section:
            config_data["plasticity"] = plasticity_section
            logger.info(
                f"🧠 Applied plasticity profile: {stage_config.plasticity_profile}"
            )

        if optimization_section:
            config_data["optimization"] = optimization_section
            logger.info(
                f"🔧 Applied memory optimizations: {stage_config.memory_optimizations}"
            )

    def _apply_progressive_scaling(
        self, config_data: Dict[str, Any], stage_config: StageConfig
    ):
        """
        Применяет progressive scaling
        """
        adaptive_dims = self._get_adaptive_dimensions(stage_config.stage)
        if adaptive_dims:
            config_data["lattice"]["lattice_width"] = adaptive_dims[0]
            config_data["lattice"]["lattice_height"] = adaptive_dims[1]
            config_data["lattice"]["lattice_depth"] = adaptive_dims[2]
            logger.info(
                f"📐 Progressive scaling: {adaptive_dims[0]}×{adaptive_dims[1]}×{adaptive_dims[2]}"
            )

    def _apply_memory_optimizations(
        self, config_data: Dict[str, Any], stage_config: StageConfig
    ):
        """
        Применяет memory optimizations
        """
        # === PHASE 4 FIX: Explicit GPU device configuration ===
        import torch

        if torch.cuda.is_available():
            # Убеждаемся что GPU включен в настройках
            if "lattice_3d" not in config_data:
                config_data["lattice_3d"] = {}
            config_data["lattice_3d"]["gpu_enabled"] = True
            config_data["lattice_3d"]["parallel_processing"] = True

            # Добавляем device в training секцию
            if "training" not in config_data:
                config_data["training"] = {}
            config_data["training"]["device"] = "cuda"
            config_data["training"]["pin_memory"] = True

            # GPU optimizations для memory efficiency
            if stage_config.memory_optimizations:
                config_data["training"]["mixed_precision"] = True
                config_data["training"]["gradient_checkpointing"] = True

            logger.info(
                f"🚀 GPU configuration enabled: {torch.cuda.get_device_name(0)}"
            )
        else:
            logger.warning("⚠️  CUDA not available - using CPU")
            if "training" not in config_data:
                config_data["training"] = {}
            config_data["training"]["device"] = "cpu"

    def _apply_emergence_tracking(
        self, config_data: Dict[str, Any], stage_config: StageConfig
    ):
        """
        Применяет emergence tracking
        """
        # === PHASE 4 FIX: Explicit GPU device configuration ===
        import torch

        if torch.cuda.is_available():
            # Убеждаемся что GPU включен в настройках
            if "lattice_3d" not in config_data:
                config_data["lattice_3d"] = {}
            config_data["lattice_3d"]["gpu_enabled"] = True
            config_data["lattice_3d"]["parallel_processing"] = True

            # Добавляем device в training секцию
            if "training" not in config_data:
                config_data["training"] = {}
            config_data["training"]["device"] = "cuda"
            config_data["training"]["pin_memory"] = True

            # GPU optimizations для memory efficiency
            if stage_config.memory_optimizations:
                config_data["training"]["mixed_precision"] = True
                config_data["training"]["gradient_checkpointing"] = True

            logger.info(
                f"🚀 GPU configuration enabled: {torch.cuda.get_device_name(0)}"
            )
        else:
            logger.warning("⚠️  CUDA not available - using CPU")
            if "training" not in config_data:
                config_data["training"] = {}
            config_data["training"]["device"] = "cpu"

    def _get_adaptive_dimensions(self, stage: int) -> Optional[tuple]:
        """
        Получает адаптивные размеры решетки для прогрессивного масштабирования
        """
        # Прогрессия размеров по стадиям (TIER 2 scaling)
        SCALING_PROGRESSION = {
            1: (16, 16, 16),  # Baseline testing
            2: (20, 20, 20),  # Small growth
            3: (24, 24, 24),  # Medium scale + clustering
            4: (32, 32, 24),  # Large scale + consolidation
            5: (40, 40, 30),  # Production scale
            6: (48, 48, 36),  # Advanced scale
            7: (64, 64, 48),  # Large production
            8: (80, 80, 60),  # Ultra scale
        }

        return SCALING_PROGRESSION.get(
            stage, SCALING_PROGRESSION[5]
        )  # Default to stage 5

    def _build_command(
        self, config: StageConfig, output_json_path: str, temp_config_path: str
    ) -> List[str]:
        """Строит команду для запуска обучения"""
        cmd = [
            sys.executable,  # Используем текущий Python интерпретатор
            "smart_resume_training.py",
            "--config-path",  # Ключевой аргумент для передачи конфигурации
            temp_config_path,
            "--mode",
            self.mode,
            "--dataset-limit",
            str(config.dataset_limit),
            "--additional-epochs",
            str(config.epochs),
            "--batch-size",
            str(config.batch_size),
            "--output-json-path",  # Новый аргумент для JSON-результата
            output_json_path,
        ]

        # PHASE 4: Убираем scale параметр - используем прогрессивное масштабирование
        # if self.scale:
        #     cmd.extend(["--scale", str(self.scale)])

        if self.verbose:
            cmd.append("--verbose")

        return cmd

    def _process_result(
        self,
        process_result: Dict[str, Any],
        output_json_path: str,
        config: StageConfig,
        actual_time: float,
        estimated_time: float,
    ) -> Optional[StageResult]:
        """Обрабатывает результат выполнения, читая JSON-файл."""

        if process_result["return_code"] != 0:
            logger.error(
                f"❌ Stage {config.stage} failed (exit code: {process_result['return_code']})"
            )
            stderr_lines = process_result["stderr"].split("\n")
            if stderr_lines:
                logger.error(f"   Last error: {stderr_lines[-1][:100]}...")

            return StageResult(
                stage=config.stage,
                config=config,
                success=False,
                actual_time_minutes=actual_time,
                estimated_time_minutes=estimated_time,
                error=(
                    process_result["stderr"][-500:]
                    if process_result["stderr"]
                    else "Unknown error"
                ),
                stdout=None,
            )

        # Успешное завершение - читаем JSON-результат
        try:
            with open(output_json_path, "r") as f:
                training_results = json.load(f)
            final_similarity = training_results.get("final_similarity")
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Failed to read or parse results from {output_json_path}: {e}"
            )
            final_similarity = None

        if final_similarity or actual_time > estimated_time * 1.5:
            logger.warning(f"✅ Stage {config.stage}: {actual_time:.1f}min")
            if final_similarity:
                logger.warning(f"   Similarity: {final_similarity:.3f}")

        return StageResult(
            stage=config.stage,
            config=config,
            success=True,
            actual_time_minutes=actual_time,
            estimated_time_minutes=estimated_time,
            final_similarity=final_similarity,
            stdout=(
                process_result["stdout"][-1000:] if process_result["stdout"] else None
            ),
        )

    def _extract_similarity_from_output(self, output: str) -> Optional[float]:
        """
        Извлекает последнее значение 'final_similarity' из вывода скрипта.
        Предполагается, что вывод содержит строки вида:
        'final_similarity=0.123'
        """
        similarity = None
        try:
            for line in reversed(output.splitlines()):
                if "final_similarity=" in line:
                    similarity_str = line.split("=")[1].strip()
                    similarity = float(similarity_str)
                    break
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse similarity from output: {e}")

        return similarity
