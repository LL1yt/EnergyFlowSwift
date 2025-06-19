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

        # Минимальное логирование инициализации
        if timeout_multiplier > 2.0:  # Логируем только нестандартные значения
            logger.warning(f"[RUNNER] High timeout multiplier: {timeout_multiplier}")

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
        # Убрали детальное логирование старта - используется глобальная функция

        # Строим команду
        cmd = self._build_command(stage_config)

        # Запускаем обучение с минимальным логированием
        start_time = time.time()
        timeout_seconds = estimated_time * 60 * self.timeout_multiplier

        try:
            result = self._run_subprocess(cmd, timeout_seconds, stage_config.stage)

            end_time = time.time()
            actual_time = (end_time - start_time) / 60  # в минутах

            if result is None:
                logger.error(
                    f"❌ Stage {stage_config.stage} failed after {actual_time:.1f}min"
                )
                return None

            return self._process_result(
                result, stage_config, actual_time, estimated_time
            )

        except Exception as e:
            logger.error(f"❌ Stage {stage_config.stage} exception: {e}")
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
            # Запускаем процесс с захватом вывода (без логирования PID)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                bufsize=1,  # Построчная буферизация
            )

            # Собираем весь вывод для последующего анализа
            stdout_lines = []
            stderr_lines = []

            # Читаем вывод в реальном времени (без логирования каждой строки)
            stdout_thread = threading.Thread(
                target=self._read_output,
                args=(process.stdout, stdout_lines, None),  # Убрали prefix
            )
            stderr_thread = threading.Thread(
                target=self._read_output,
                args=(process.stderr, stderr_lines, None),  # Убрали prefix
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
                    f"⏰ Stage {stage} timeout after {timeout_seconds/60:.1f}min"
                )
                process.kill()
                process.wait()
                return_code = -1

            # Ждем завершения потоков чтения
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            # Возвращаем результат без логирования completion
            return {
                "return_code": return_code,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
            }

        except Exception as e:
            logger.error(f"❌ Subprocess execution failed: {e}")
            return None

    def _read_output(self, pipe, output_list: List[str], prefix: str):
        """Читает вывод из pipe без логирования каждой строки"""
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    output_list.append(line.strip())
        except Exception as e:
            if prefix:  # Логируем ошибку только если prefix не None
                logger.error(f"Reading error: {e}")

    def _process_result(
        self,
        result: Dict[str, Any],
        config: StageConfig,
        actual_time: float,
        estimated_time: float,
    ) -> Optional[StageResult]:
        """Обрабатывает результат выполнения subprocess (минимальное логирование)"""

        if result["return_code"] != 0:
            logger.error(
                f"❌ Stage {config.stage} failed (exit code: {result['return_code']})"
            )
            # Показываем только последние строки stderr при ошибке
            stderr_lines = result["stderr"].split("\n")
            if stderr_lines:
                logger.error(f"   Last error: {stderr_lines[-1][:100]}...")

            return StageResult(
                stage=config.stage,
                config=config,
                success=False,
                actual_time_minutes=actual_time,
                estimated_time_minutes=estimated_time,
                error=result["stderr"][-500:] if result["stderr"] else "Unknown error",
                stdout=None,
            )

        # Успешное завершение - минимальное логирование
        final_similarity = self._extract_similarity_from_output(result["stdout"])

        # Логируем только если есть similarity или время превысило оценку
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
                result["stdout"][-1000:] if result["stdout"] else None
            ),  # Только last 1000 chars
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
