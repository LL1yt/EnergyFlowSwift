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
        self, config: Dict[str, Any], stage_config: StageConfig
    ) -> Dict[str, Any]:
        """
        === PHASE 4 INTEGRATION ===
        Подготавливает конфигурацию с оптимизациями пластичности и памяти
        """
        try:
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
            optimization_section = generator.generate_optimization_section(
                stage_context
            )

            # Интегрируем в основную конфигурацию
            if plasticity_section:
                config["plasticity"] = plasticity_section
                logger.info(
                    f"🧠 Applied plasticity profile: {stage_config.plasticity_profile}"
                )

            if optimization_section:
                config["optimization"] = optimization_section
                logger.info(
                    f"🔧 Applied memory optimizations: {stage_config.memory_optimizations}"
                )

            # Адаптивные размеры решетки для прогрессивного масштабирования
            if stage_config.progressive_scaling:
                adaptive_dims = self._get_adaptive_dimensions(stage_config.stage)
                if adaptive_dims:
                    config["lattice"]["lattice_width"] = adaptive_dims[0]
                    config["lattice"]["lattice_height"] = adaptive_dims[1]
                    config["lattice"]["lattice_depth"] = adaptive_dims[2]
                    logger.info(
                        f"📐 Progressive scaling: {adaptive_dims[0]}×{adaptive_dims[1]}×{adaptive_dims[2]}"
                    )

            # === PHASE 4 FIX: Explicit GPU device configuration ===
            import torch

            if torch.cuda.is_available():
                # Убеждаемся что GPU включен в настройках
                if "lattice_3d" not in config:
                    config["lattice_3d"] = {}
                config["lattice_3d"]["gpu_enabled"] = True
                config["lattice_3d"]["parallel_processing"] = True

                # Добавляем device в training секцию
                if "training" not in config:
                    config["training"] = {}
                config["training"]["device"] = "cuda"
                config["training"]["pin_memory"] = True

                # GPU optimizations для memory efficiency
                if stage_config.memory_optimizations:
                    config["training"]["mixed_precision"] = True
                    config["training"]["gradient_checkpointing"] = True

                logger.info(
                    f"🚀 GPU configuration enabled: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.warning("⚠️  CUDA not available - using CPU")
                if "training" not in config:
                    config["training"] = {}
                config["training"]["device"] = "cpu"

            return config

        except Exception as e:
            logger.error(f"Failed to prepare config with optimizations: {e}")
            return config  # Возвращаем оригинальную конфигурацию при ошибке

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

        if self.scale:
            cmd.extend(["--scale", str(self.scale)])

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
