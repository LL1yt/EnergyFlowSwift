#!/usr/bin/env python3
"""
[BOT] Automated Long Training Script
Автоматизированное долгое обучение с прогрессивным увеличением сложности
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Консоль
        logging.FileHandler("logs/automated_training.log", encoding="utf-8"),  # Файл
    ],
)
logger = logging.getLogger(__name__)


class AutomatedTrainer:
    """Автоматизированный тренер для долгого обучения"""

    def __init__(
        self,
        mode: str = "development",
        scale: Optional[float] = None,
        max_total_time_hours: float = 8.0,
        dataset_limit_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        timeout_multiplier: float = 2.0,
    ):
        """
        Args:
            mode: Режим конфигурации (development, research, etc.)
            scale: Custom scale factor
            max_total_time_hours: Максимальное время обучения в часах
            dataset_limit_override: Переопределить dataset_limit для всех стадий (для тестирования)
            batch_size_override: Переопределить batch_size для всех стадий (для ускорения)
            timeout_multiplier: Multiplier for the timeout
        """
        self.mode = mode
        self.scale = scale
        self.max_total_time_hours = max_total_time_hours
        self.dataset_limit_override = dataset_limit_override
        self.batch_size_override = batch_size_override
        self.timeout_multiplier = timeout_multiplier
        self.start_time = datetime.now()

        # История обучения
        self.training_history = []

        # Создаем директорию для логов
        self.log_dir = Path("logs/automated_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем директорию для основных логов
        Path("logs").mkdir(exist_ok=True)

        # Файл лога сессии
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.log_dir / f"automated_session_{timestamp}.json"

        logger.info(f"[BOT] Automated Trainer initialized")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   Max time: {max_total_time_hours} hours")
        if dataset_limit_override:
            logger.info(f"   Dataset limit override: {dataset_limit_override}")
        if batch_size_override:
            logger.info(f"   Batch size override: {batch_size_override}")
        logger.info(f"   Timeout multiplier: {timeout_multiplier}")
        logger.info(f"   Session log: {self.session_log}")

    def get_progressive_config(self, stage: int) -> Dict[str, Any]:
        """
        Возвращает конфигурацию для определенной стадии обучения

        Стратегия: Постепенно увеличиваем сложность
        - Stage 1: Маленький датасет, много эпох (изучение основ)
        - Stage 2: Средний датасет, средние эпохи (консолидация)
        - Stage 3: Большой датасет, мало эпох (финальная подстройка)
        """
        configs = {
            1: {
                "dataset_limit": 2000,
                "epochs": 20,
                "batch_size": 32,
                "description": "Foundation Learning (small data, many epochs)",
            },
            2: {
                "dataset_limit": 5000,
                "epochs": 15,
                "batch_size": 64,
                "description": "Consolidation (medium data, medium epochs)",
            },
            3: {
                "dataset_limit": 10000,
                "epochs": 12,
                "batch_size": 64,
                "description": "Refinement (large data, fewer epochs)",
            },
            4: {
                "dataset_limit": 20000,
                "epochs": 8,
                "batch_size": 128,
                "description": "Mastery (very large data, few epochs)",
            },
            5: {
                "dataset_limit": 50000,
                "epochs": 5,
                "batch_size": 128,
                "description": "Perfection (massive data, minimal epochs)",
            },
        }

        # Возвращаем конфигурацию или последнюю если stage слишком большой
        config = configs.get(stage, configs[5])

        # Переопределяем параметры если заданы override
        if self.dataset_limit_override or self.batch_size_override:
            config = config.copy()  # Не изменяем оригинальную конфигурацию

            if self.dataset_limit_override:
                config["dataset_limit"] = self.dataset_limit_override
                config["description"] += f" (dataset: {self.dataset_limit_override})"

            if self.batch_size_override:
                config["batch_size"] = self.batch_size_override
                config["description"] += f" (batch: {self.batch_size_override})"

        return config

    def estimate_stage_time(self, config: Dict[str, Any]) -> float:
        """Оценивает время выполнения стадии в минутах"""
        dataset_size = config["dataset_limit"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        # Примерная оценка на основе размера датасета и режима
        if self.mode == "development":
            time_per_1k_examples = 2  # минут
        elif self.mode == "research":
            time_per_1k_examples = 5
        else:
            time_per_1k_examples = 10

        estimated_minutes = (dataset_size / 1000) * time_per_1k_examples * epochs / 10

        # Минимальное время для инициализации и маленьких датасетов
        min_time_minutes = 8.0  # минимум 8 минут (увеличено)
        if dataset_size <= 100:  # Для очень маленьких датасетов
            min_time_minutes = 12.0  # минимум 12 минут
        elif dataset_size <= 1000:  # Для маленьких датасетов
            min_time_minutes = 10.0  # минимум 10 минут

        # Корректировка для больших batch size (ускоряют обучение)
        if batch_size >= 128:
            estimated_minutes *= 0.5  # 50% ускорение
        elif batch_size >= 64:
            estimated_minutes *= 0.7  # 30% ускорение

        return max(estimated_minutes, min_time_minutes)

    def run_training_stage(
        self, stage: int, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Запускает одну стадию обучения"""
        logger.info(f"[START] Starting Stage {stage}: {config['description']}")
        logger.info(f"   Dataset: {config['dataset_limit']:,} examples")
        logger.info(f"   Epochs: {config['epochs']}")
        logger.info(f"   Batch size: {config['batch_size']}")

        estimated_time = self.estimate_stage_time(config)
        logger.info(f"   Estimated time: {estimated_time:.1f} minutes")

        # Строим команду
        cmd = [
            sys.executable,  # Используем текущий Python интерпретатор
            "smart_resume_training.py",
            "--mode",
            self.mode,
            "--dataset-limit",
            str(config["dataset_limit"]),
            "--additional-epochs",
            str(config["epochs"]),
            "--batch-size",
            str(config["batch_size"]),
        ]

        if self.scale:
            cmd.extend(["--scale", str(self.scale)])

        logger.info(f"   Command: {' '.join(cmd)}")

        # Запускаем обучение с real-time выводом
        start_time = time.time()
        timeout_seconds = (
            estimated_time * 60 * self.timeout_multiplier
        )  # Таймаут = timeout_multiplier * от оценки
        logger.info(
            f"   [PROGRESS] Starting subprocess with timeout: {timeout_seconds/60:.1f} minutes"
        )

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
            import select
            import threading
            from queue import Queue, Empty

            def read_output(pipe, output_list, prefix):
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

            # Запускаем потоки для чтения stdout и stderr
            stdout_thread = threading.Thread(
                target=read_output, args=(process.stdout, stdout_lines, "[SUBPROCESS]")
            )
            stderr_thread = threading.Thread(
                target=read_output,
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

            # Создаем объект result для совместимости с остальным кодом
            class MockResult:
                def __init__(self, returncode, stdout_list, stderr_list):
                    self.returncode = returncode
                    self.stdout = "\n".join(stdout_list)
                    self.stderr = "\n".join(stderr_list)

            result = MockResult(return_code, stdout_lines, stderr_lines)

            end_time = time.time()
            actual_time = (end_time - start_time) / 60  # в минутах

            if result.returncode == -1:
                # Таймаут
                logger.error(
                    f"[ERROR] Stage {stage} timed out after {actual_time:.1f} minutes"
                )
                return None
            elif result.returncode == 0:
                logger.info(f"[OK] Stage {stage} completed successfully")
                logger.info(f"   Actual time: {actual_time:.1f} minutes")

                # Показываем последние строки вывода для контекста
                stdout_lines = result.stdout.strip().split("\n")
                logger.info(f"   Last few lines of output:")
                for line in stdout_lines[-5:]:
                    if line.strip():
                        logger.info(f"      {line}")

                # Извлекаем метрики из вывода
                similarity = self._extract_similarity_from_output(result.stdout)

                stage_result = {
                    "stage": stage,
                    "config": config,
                    "success": True,
                    "actual_time_minutes": actual_time,
                    "estimated_time_minutes": estimated_time,
                    "final_similarity": similarity,
                    "timestamp": datetime.now().isoformat(),
                }

                self.training_history.append(stage_result)
                self._save_session_log()

                return stage_result
            else:
                logger.error(
                    f"[ERROR] Stage {stage} failed with return code {result.returncode}"
                )
                logger.error(f"   STDOUT output:")
                logger.error(result.stdout)
                logger.error(f"   STDERR output:")
                logger.error(result.stderr)

                stage_result = {
                    "stage": stage,
                    "config": config,
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "timestamp": datetime.now().isoformat(),
                }

                self.training_history.append(stage_result)
                self._save_session_log()

                return None

        except Exception as e:
            logger.error(f"[ERROR] Stage {stage} failed with exception: {e}")
            return None

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

    def _save_session_log(self):
        """Сохраняет лог сессии"""
        session_data = {
            "mode": self.mode,
            "scale": self.scale,
            "max_total_time_hours": self.max_total_time_hours,
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "elapsed_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "training_history": self.training_history,
            "summary": self._generate_summary(),
        }

        with open(self.session_log, "w") as f:
            json.dump(session_data, f, indent=2)

    def _generate_summary(self) -> Dict[str, Any]:
        """Генерирует сводку по текущей сессии"""
        successful_stages = [
            h for h in self.training_history if h.get("success", False)
        ]

        if not successful_stages:
            return {"total_stages": 0, "best_similarity": None}

        total_time = sum(h["actual_time_minutes"] for h in successful_stages)
        similarities = [
            h["final_similarity"]
            for h in successful_stages
            if h["final_similarity"] is not None
        ]

        return {
            "total_stages": len(successful_stages),
            "total_time_minutes": total_time,
            "best_similarity": max(similarities) if similarities else None,
            "avg_similarity": (
                sum(similarities) / len(similarities) if similarities else None
            ),
            "similarity_trend": (
                similarities[-3:] if len(similarities) >= 3 else similarities
            ),
        }

    def should_continue(self) -> bool:
        """Проверяет, стоит ли продолжать обучение"""
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        if elapsed_hours >= self.max_total_time_hours:
            logger.info(
                f"[TIME] Time limit reached: {elapsed_hours:.1f}/{self.max_total_time_hours} hours"
            )
            return False

        return True

    def run_automated_training(self):
        """Запускает автоматизированное обучение"""
        logger.info(f"🎯 ======== AUTOMATED TRAINING SESSION STARTED ========")
        logger.info(f"[TARGET] Starting automated training session")
        logger.info(f"   Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Max duration: {self.max_total_time_hours} hours")
        logger.info(
            f"   Target end time: {(self.start_time + timedelta(hours=self.max_total_time_hours)).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"   Mode: {self.mode}")
        if self.scale:
            logger.info(f"   Scale factor: {self.scale}")
        if self.dataset_limit_override:
            logger.info(f"   Dataset limit override: {self.dataset_limit_override}")
        logger.info(f"   Session log: {self.session_log}")
        logger.info("=" * 60)

        stage = 1

        while self.should_continue():
            config = self.get_progressive_config(stage)

            # Проверяем, хватит ли времени для этой стадии
            estimated_time_hours = self.estimate_stage_time(config) / 60
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            remaining_hours = self.max_total_time_hours - elapsed_hours

            if estimated_time_hours > remaining_hours:
                logger.info(f"[TIME] Not enough time for stage {stage}")
                logger.info(
                    f"   Estimated: {estimated_time_hours:.1f}h, Remaining: {remaining_hours:.1f}h"
                )
                break

            # Запускаем стадию
            logger.info(f"🚀 [STAGE-{stage}] ======== STARTING STAGE {stage} ========")
            stage_start_time = time.time()

            result = self.run_training_stage(stage, config)

            stage_end_time = time.time()
            stage_duration = (stage_end_time - stage_start_time) / 60

            if result is None:
                logger.error(
                    f"[ERROR] Stage {stage} failed after {stage_duration:.1f} minutes, stopping automated training"
                )
                break

            # Показываем детальный прогресс
            summary = self._generate_summary()
            elapsed_total = (datetime.now() - self.start_time).total_seconds() / 3600
            remaining_time = self.max_total_time_hours - elapsed_total

            logger.info(f"✅ [STAGE-{stage}] ======== STAGE {stage} COMPLETED ========")
            logger.info(f"   Stage duration: {stage_duration:.1f} minutes")
            logger.info(f"[DATA] Overall Progress:")
            logger.info(f"   Stages completed: {summary['total_stages']}")
            logger.info(
                f"   Session time: {elapsed_total:.1f}h / {self.max_total_time_hours}h"
            )
            logger.info(f"   Remaining time: {remaining_time:.1f}h")
            logger.info(
                f"   Best similarity: {summary['best_similarity']:.4f}"
                if summary["best_similarity"]
                else "   Best similarity: N/A"
            )
            logger.info(
                f"   Total training time: {summary['total_time_minutes']:.1f} minutes"
            )

            # Показываем тренд похожести
            if summary.get("similarity_trend") and len(summary["similarity_trend"]) > 1:
                trend = summary["similarity_trend"]
                logger.info(f"   Similarity trend: {[f'{s:.3f}' for s in trend]}")
                if len(trend) >= 2:
                    improvement = trend[-1] - trend[-2]
                    logger.info(f"   Last improvement: {improvement:+.4f}")

            stage += 1

            logger.info(f"[NEXT] Preparing for stage {stage}...")

            # Небольшая пауза между стадиями
            logger.info("   [PAUSE] 10 second break between stages...")
            time.sleep(10)

        # Финальная сводка
        self._print_final_summary()

    def _print_final_summary(self):
        """Выводит финальную сводку"""
        summary = self._generate_summary()
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        logger.info(f"\n[SUCCESS] Automated training session completed!")
        logger.info(f"[DATA] Final Summary:")
        logger.info(f"   Total duration: {elapsed_hours:.1f} hours")
        logger.info(f"   Stages completed: {summary['total_stages']}")
        logger.info(
            f"   Total training time: {summary.get('total_time_minutes', 0):.1f} minutes"
        )
        logger.info(
            f"   Best similarity achieved: {summary['best_similarity']:.4f}"
            if summary["best_similarity"]
            else "   Best similarity: N/A"
        )
        logger.info(
            f"   Average similarity: {summary['avg_similarity']:.4f}"
            if summary.get("avg_similarity")
            else "   Average similarity: N/A"
        )
        logger.info(f"   Session log saved: {self.session_log}")

        if summary.get("similarity_trend"):
            logger.info(
                f"   Recent similarity trend: {[f'{s:.3f}' for s in summary['similarity_trend']]}"
            )


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Long Training Script")
    parser.add_argument(
        "--mode",
        choices=["development", "research", "validation", "production"],
        default="development",
        help="Configuration mode",
    )
    parser.add_argument("--scale", type=float, default=None, help="Custom scale factor")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Maximum training time in hours (default: 8.0)",
    )
    parser.add_argument(
        "--test-config",
        action="store_true",
        help="Show training stages configuration and exit",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Override dataset_limit for all stages (useful for quick testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size for all stages (useful for faster training)",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=2.0,
        help="Timeout multiplier for the training process",
    )

    args = parser.parse_args()

    try:
        trainer = AutomatedTrainer(
            mode=args.mode,
            scale=args.scale,
            max_total_time_hours=args.max_hours,
            dataset_limit_override=args.dataset_limit,
            batch_size_override=args.batch_size,
            timeout_multiplier=args.timeout_multiplier,
        )

        if args.test_config:
            # Показываем конфигурацию стадий
            logger.info(f"\n[INFO] Training stages configuration:")
            for stage in range(1, 6):
                config = trainer.get_progressive_config(stage)
                estimated_time = trainer.estimate_stage_time(config)
                logger.info(f"   Stage {stage}: {config['description']}")
                logger.info(
                    f"      Dataset: {config['dataset_limit']:,}, Epochs: {config['epochs']}, Batch: {config['batch_size']}"
                )
                logger.info(f"      Estimated time: {estimated_time:.1f} minutes")
                logger.info("")
            return

        # Запускаем автоматизированное обучение
        trainer.run_automated_training()

    except KeyboardInterrupt:
        logger.info("[STOP] Automated training interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Automated training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
