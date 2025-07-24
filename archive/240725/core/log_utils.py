#!/usr/bin/env python3
"""
Вспомогательные утилиты для модуля Lattice3D
"""

import inspect
import os
from pathlib import Path


def _get_caller_info():
    """Возвращает информацию о вызывающем коде (файл, строка, функция)."""
    try:
        # inspect.stack() is slow, but for debugging it's acceptable.
        stack = inspect.stack()
        # Ищем первый фрейм не из этого файла
        for frame_info in stack[2:]:  # Go up 2 frames to get out of this helper
            if frame_info.filename != str(Path(__file__).resolve()):
                # Return a concise string with relevant info, relative to project root 'AA'
                try:
                    # Assumes the project root is named 'AA' or is the CWD
                    project_root = Path.cwd()
                    if "AA" not in project_root.parts:
                        # Fallback if not run from a subfolder of AA
                        rel_path = frame_info.filename
                    else:
                        # Find the 'AA' part and form the path from there
                        aa_index = project_root.parts.index("AA")
                        aa_root = Path(*project_root.parts[: aa_index + 1])
                        rel_path = os.path.relpath(frame_info.filename, start=aa_root)

                except (ValueError, TypeError):
                    rel_path = frame_info.filename
                return f"{rel_path}:{frame_info.lineno} ({frame_info.function})"
        return "N/A"
    except Exception:
        # This could fail in some environments (e.g. optimized Python), so we have a fallback.
        return "N/A"
