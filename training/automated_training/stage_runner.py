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
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from .progressive_config import StageConfig

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Результат выполнения стадии обучения"""

    stage: int
    config: StageConfig
    success: bool
    actual_time_minutes: float
    estimated_time_minutes: float
    final_similarity: Optional[float] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TrainingStageRunner:
    """Выполнитель тренировочных стадий"""

    def __init__(
        self,
        mode: str = "development",
        scale: Optional[float] = None,
        timeout_multiplier: float = 2.0,
    ):
        """
        Args:
            mode: Режим конфигурации (development, research, etc.)
            scale: Custom scale factor
            timeout_multiplier: Multiplier for the timeout
        """
        self.mode = mode
        self.scale = scale
        self.timeout_multiplier = timeout_multiplier

        logger.info("[RUNNER] TrainingStageRunner initialized")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   Timeout multiplier: {timeout_multiplier}")

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
        logger.info(
            f"[START] Starting Stage {stage_config.stage}: {stage_config.description}"
        )
        logger.info(f"   Dataset: {stage_config.dataset_limit:,} examples")
        logger.info(f"   Epochs: {stage_config.epochs}")
        logger.info(f"   Batch size: {stage_config.batch_size}")
        logger.info(f"   Estimated time: {estimated_time:.1f} minutes")

        # Строим команду
        cmd = self._build_command(stage_config)
        logger.info(f"   Command: {' '.join(cmd)}")

        # Запускаем обучение с real-time выводом
        start_time = time.time()
        timeout_seconds = estimated_time * 60 * self.timeout_multiplier

        logger.info(
            f"   [PROGRESS] Starting subprocess with timeout: {timeout_seconds/60:.1f} minutes"
        )

        try:
            result = self._run_subprocess(cmd, timeout_seconds, stage_config.stage)

            end_time = time.time()
            actual_time = (end_time - start_time) / 60  # в минутах

            if result is None:
                # Таймаут или ошибка запуска
                logger.error(
                    f"[ERROR] Stage {stage_config.stage} failed after {actual_time:.1f} minutes"
                )
                return None

            return self._process_result(
                result, stage_config, actual_time, estimated_time
            )

        except Exception as e:
            logger.error(
                f"[ERROR] Stage {stage_config.stage} failed with exception: {e}"
            )
            return None

    def _build_command(self, config: StageConfig) -> List[str]:
        """Строит команду для запуска обучения"""
        cmd = [
            sys.executable,  # Используем текущий Python интерпретатор
            "smart_resume_training.py",
            "--mode",
            self.mode,
            "--dataset-limit",
            str(config.dataset_limit),
            "--additional-epochs",
            str(config.epochs),
            "--batch-size",
            str(config.batch_size),
        ]

        if self.scale:
            cmd.extend(["--scale", str(self.scale)])

        return cmd

    def _run_subprocess(
        self, cmd: List[str], timeout_seconds: float, stage: int
    ) -> Optional[Dict[str, Any]]:
        """
        Запускает subprocess с мониторингом в реальном времени

        Returns:
            Dict с результатами выполнения или None при ошибке
        """
        try:
            # Запускаем процесс с захватом вывода для real-time логов
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                bufsize=1,  # Построчная буферизация
            )

            logger.info(f"   [PROGRESS] Process started with PID: {process.pid}")

            # Собираем весь вывод для последующего анализа
            stdout_lines = []
            stderr_lines = []

            # Читаем вывод в реальном времени
            stdout_thread = threading.Thread(
                target=self._read_output,
                args=(process.stdout, stdout_lines, "[SUBPROCESS]"),
            )
            stderr_thread = threading.Thread(
                target=self._read_output,
                args=(process.stderr, stderr_lines, "[SUBPROCESS-ERR]"),
            )

            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Ожидаем завершения с таймаутом
            try:
                process.wait(timeout=timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                logger.error(
                    f"   [TIMEOUT] Process timed out after {timeout_seconds/60:.1f} minutes"
                )
                process.kill()
                process.wait()
                return_code = -1

            # Ждем завершения потоков чтения
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            logger.info(
                f"   [PROGRESS] Process completed with return code: {return_code}"
            )

            return {
                "returncode": return_code,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
            }

        except Exception as e:
            logger.error(f"   [ERROR] Subprocess execution failed: {e}")
            return None

    def _read_output(self, pipe, output_list: List[str], prefix: str):
        """Читает вывод из pipe и добавляет в список"""
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    line = line.rstrip()
                    output_list.append(line)
                    logger.info(f"   {prefix}: {line}")
        except Exception as e:
            logger.error(f"   [ERROR] Reading {prefix}: {e}")
        finally:
            pipe.close()

    def _process_result(
        self,
        result: Dict[str, Any],
        config: StageConfig,
        actual_time: float,
        estimated_time: float,
    ) -> Optional[StageResult]:
        """Обрабатывает результат выполнения стадии"""

        if result["returncode"] == -1:
            # Таймаут
            logger.error(
                f"[ERROR] Stage {config.stage} timed out after {actual_time:.1f} minutes"
            )
            return StageResult(
                stage=config.stage,
                config=config,
                success=False,
                actual_time_minutes=actual_time,
                estimated_time_minutes=estimated_time,
                error="Timeout",
            )

        elif result["returncode"] == 0:
            # Успешное выполнение
            logger.info(f"[OK] Stage {config.stage} completed successfully")
            logger.info(f"   Actual time: {actual_time:.1f} minutes")

            # Показываем последние строки вывода для контекста
            stdout_lines = result["stdout"].strip().split("\n")
            logger.info(f"   Last few lines of output:")
            for line in stdout_lines[-5:]:
                if line.strip():
                    logger.info(f"      {line}")

            # Извлекаем метрики из вывода
            similarity = self._extract_similarity_from_output(result["stdout"])

            return StageResult(
                stage=config.stage,
                config=config,
                success=True,
                actual_time_minutes=actual_time,
                estimated_time_minutes=estimated_time,
                final_similarity=similarity,
                stdout=result["stdout"],
            )

        else:
            # Ошибка выполнения
            logger.error(
                f"[ERROR] Stage {config.stage} failed with return code {result['returncode']}"
            )
            logger.error(f"   STDOUT output:")
            logger.error(result["stdout"])
            logger.error(f"   STDERR output:")
            logger.error(result["stderr"])

            return StageResult(
                stage=config.stage,
                config=config,
                success=False,
                actual_time_minutes=actual_time,
                estimated_time_minutes=estimated_time,
                error=result["stderr"],
                stdout=result["stdout"],
            )

    def _extract_similarity_from_output(self, output: str) -> Optional[float]:
        """Извлекает final similarity из вывода обучения"""
        try:
            # Ищем строку с final_similarity
            for line in output.split("\n"):
                if "final_similarity:" in line:
                    # Извлекаем число
                    parts = line.split("final_similarity:")
                    if len(parts) > 1:
                        similarity_str = parts[1].strip()
                        return float(similarity_str)
        except:
            pass
        return None
