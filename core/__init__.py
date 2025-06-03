"""
Основные компоненты 3D Клеточной Нейронной Сети

Этот пакет содержит базовые компоненты для построения и работы
с 3D клеточной нейронной сетью.

Модули:
    cell_prototype: Прототип "умной клетки"
    lattice_3d: 3D решетка клеток (планируется)
    signal_propagation: Распространение сигналов (планируется)
"""

# Импорт с использованием importlib для модулей с дефисами
import importlib.util
import sys
from pathlib import Path

# Получаем путь к модулю cell-prototype
module_path = Path(__file__).parent / "cell-prototype" / "main.py"

# Загружаем модуль динамически
spec = importlib.util.spec_from_file_location("cell_prototype_main", module_path)
cell_prototype_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell_prototype_main)

# Экспортируем классы и функции
CellPrototype = cell_prototype_main.CellPrototype
create_cell_from_config = cell_prototype_main.create_cell_from_config

__all__ = [
    'CellPrototype', 
    'create_cell_from_config'
]

__version__ = '0.1.0' 