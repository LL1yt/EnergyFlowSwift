"""
External Process Runner
"""

import logging
import subprocess
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_process(
    cmd: List[str], timeout_seconds: float, process_id: Any = "", verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Запускает subprocess с мониторингом в реальном времени.

    Args:
        cmd: Команда для выполнения.
        timeout_seconds: Таймаут в секундах.
        process_id: Идентификатор процесса для логирования.
        verbose: Если True, выводит логи subprocess-ов в реальном времени.

    Returns:
        Словарь с результатами или None при ошибке.
    """
    try:
        if verbose:
            logger.info(
                f"🚀 Starting subprocess '{process_id}': {' '.join(cmd[:3])}..."
            )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            bufsize=1,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        stdout_thread = threading.Thread(
            target=_read_pipe,
            args=(process.stdout, stdout_lines, verbose, process_id, "OUT"),
        )
        stderr_thread = threading.Thread(
            target=_read_pipe,
            args=(process.stderr, stderr_lines, verbose, process_id, "ERR"),
        )

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        try:
            process.wait(timeout=timeout_seconds)
            return_code = process.returncode
            if verbose:
                if return_code == 0:
                    logger.info(f"✅ Subprocess '{process_id}' completed successfully")
                else:
                    logger.error(
                        f"❌ Subprocess '{process_id}' failed with exit code {return_code}"
                    )
        except subprocess.TimeoutExpired:
            logger.error(
                f"⏰ Process '{process_id}' timed out after {timeout_seconds/60:.1f} min."
            )
            process.kill()
            process.wait()
            return_code = -1

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        return {
            "return_code": return_code,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
        }

    except Exception as e:
        logger.error(f"❌ Subprocess execution failed for '{process_id}': {e}")
        return None


def _read_pipe(
    pipe,
    output_list: List[str],
    verbose: bool = False,
    process_id: Any = "",
    stream_type: str = "",
):
    """Читает вывод из pipe и добавляет в список."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                stripped_line = line.strip()
                output_list.append(stripped_line)

                # В verbose режиме выводим логи subprocess в реальном времени
                if verbose and stripped_line:
                    # Форматируем лог с префиксом subprocess
                    prefix = f"[{process_id}:{stream_type}]"
                    if stream_type == "ERR":
                        logger.error(f"{prefix} {stripped_line}")
                    else:
                        logger.info(f"{prefix} {stripped_line}")

    except Exception as e:
        logger.error(f"Error reading pipe: {e}")
