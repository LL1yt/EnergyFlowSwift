"""
Модуль управления вводом/выводом (I/O) на гранях решетки
=========================================================

Содержит класс IOPointPlacer, который отвечает за размещение
точек ввода и вывода на гранях 3D-решетки в соответствии
с выбранной стратегией (пропорциональной, случайной и т.д.).
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import torch

from .enums import Face, PlacementStrategy
from .position import Coordinates3D, Dimensions3D


class IOPointPlacer:
    """
    Управление размещением точек ввода/вывода с автоматическим масштабированием.

    Реализует различные стратегии размещения I/O точек на гранях решетки,
    включая биологически обоснованное пропорциональное масштабирование.
    """

    def __init__(
        self,
        lattice_dimensions: Dimensions3D,
        strategy: PlacementStrategy,
        config: Dict[str, Any],
        seed: int = 42,
    ):
        """
        Инициализация размещения I/O точек.

        Args:
            lattice_dimensions: Размеры решетки (X, Y, Z)
            strategy: Стратегия размещения точек
            config: Конфигурация размещения
            seed: Сид для воспроизводимости
        """
        self.dimensions = lattice_dimensions
        self.strategy = strategy
        self.config = config
        self.seed = seed

        # Устанавливаем сид для воспроизводимости
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Кэш для размещенных точек
        self._input_points_cache: Dict[str, List[Coordinates3D]] = {}
        self._output_points_cache: Dict[str, List[Coordinates3D]] = {}

    def calculate_num_points(self, face_area: int) -> Tuple[int, int]:
        """
        Рассчитывает количество точек для пропорциональной стратегии.

        Args:
            face_area: Площадь грани (количество клеток)

        Returns:
            Tuple[int, int]: (минимальное_количество, максимальное_количество)
        """
        if self.strategy != PlacementStrategy.PROPORTIONAL:
            raise ValueError("calculate_num_points только для PROPORTIONAL стратегии")

        coverage_config = self.config.get("coverage_ratio", {})
        min_percentage = coverage_config.get("min_percentage", 7.8)
        max_percentage = coverage_config.get("max_percentage", 15.6)

        # Рассчитываем количество точек
        min_points_calc = max(1, int(face_area * min_percentage / 100))
        max_points_calc = max(min_points_calc, int(face_area * max_percentage / 100))

        # Применяем абсолютные ограничения
        limits_config = self.config.get("absolute_limits", {})
        min_points_abs = limits_config.get("min_points", 1)
        max_points_abs = limits_config.get("max_points", 0)  # 0 = без ограничений

        min_points = max(min_points_calc, min_points_abs)
        max_points = max_points_calc

        if max_points_abs > 0:
            max_points = min(max_points, max_points_abs)

        # ИСПРАВЛЕНИЕ: Убеждаемся, что min_points <= max_points
        if min_points > max_points:
            if max_points_abs > 0:
                min_points = max_points
            else:
                max_points = min_points

        return min_points, max_points

    def get_input_points(self, face: Face) -> List[Coordinates3D]:
        """
        Получает координаты точек ввода на указанной грани.
        """
        cache_key = f"input_{face.value}"
        if cache_key in self._input_points_cache:
            return self._input_points_cache[cache_key]

        if self.strategy == PlacementStrategy.PROPORTIONAL:
            points = self._generate_proportional_points(face)
        elif self.strategy == PlacementStrategy.FULL_FACE:
            points = self._generate_full_face_points(face)
        elif self.strategy == PlacementStrategy.RANDOM:
            points = self._generate_random_points(face)
        elif self.strategy == PlacementStrategy.CORNERS:
            points = self._generate_corner_points(face)
        elif self.strategy == PlacementStrategy.CORNERS_CENTER:
            points = self._generate_corners_center_points(face)
        else:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")

        self._input_points_cache[cache_key] = points
        return points

    def get_output_points(self, face: Face) -> List[Coordinates3D]:
        """
        Получает координаты точек вывода на указанной грани.
        """
        cache_key = f"output_{face.value}"
        if cache_key in self._output_points_cache:
            return self._output_points_cache[cache_key]

        # Для вывода используем те же стратегии, что и для ввода
        points = self.get_input_points(face)

        self._output_points_cache[cache_key] = points
        return points

    def _generate_proportional_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки для пропорциональной стратегии."""
        face_coords = self._get_face_coordinates(face)
        face_area = len(face_coords)

        min_points, max_points = self.calculate_num_points(face_area)

        num_points = np.random.randint(min_points, max_points + 1)

        selected_indices = np.random.choice(
            len(face_coords), size=num_points, replace=False
        )
        return [face_coords[i] for i in selected_indices]

    def _generate_full_face_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки для полного покрытия грани."""
        return self._get_face_coordinates(face)

    def _generate_random_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует случайно размещенные точки."""
        face_coords = self._get_face_coordinates(face)
        num_points = max(1, len(face_coords) // 4)
        selected_indices = np.random.choice(
            len(face_coords), size=num_points, replace=False
        )
        return [face_coords[i] for i in selected_indices]

    def _generate_corner_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки в углах грани."""
        x_dim, y_dim, z_dim = self.dimensions
        x_max, y_max, z_max = x_dim - 1, y_dim - 1, z_dim - 1

        if face == Face.FRONT:
            return [(0, 0, 0), (x_max, 0, 0), (0, y_max, 0), (x_max, y_max, 0)]
        if face == Face.BACK:
            return [
                (0, 0, z_max),
                (x_max, 0, z_max),
                (0, y_max, z_max),
                (x_max, y_max, z_max),
            ]
        if face == Face.LEFT:
            return [(0, 0, 0), (0, y_max, 0), (0, 0, z_max), (0, y_max, z_max)]
        if face == Face.RIGHT:
            return [
                (x_max, 0, 0),
                (x_max, y_max, 0),
                (x_max, 0, z_max),
                (x_max, y_max, z_max),
            ]
        if face == Face.TOP:
            return [
                (0, y_max, 0),
                (x_max, y_max, 0),
                (0, y_max, z_max),
                (x_max, y_max, z_max),
            ]
        if face == Face.BOTTOM:
            return [(0, 0, 0), (x_max, 0, 0), (0, 0, z_max), (x_max, 0, z_max)]

        return []

    def _generate_corners_center_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки в углах и в центре грани."""
        points = self._generate_corner_points(face)
        points.append(self._get_face_center(face))
        return list(set(points))  # Удаляем дубликаты, если центр совпал с углом

    def _get_face_coordinates(self, face: Face) -> List[Coordinates3D]:
        """Возвращает список всех координат, принадлежащих грани."""
        x_dim, y_dim, z_dim = self.dimensions
        coords = []

        if face == Face.FRONT:
            for x in range(x_dim):
                for y in range(y_dim):
                    coords.append((x, y, 0))
        elif face == Face.BACK:
            for x in range(x_dim):
                for y in range(y_dim):
                    coords.append((x, y, z_dim - 1))
        elif face == Face.LEFT:
            for y in range(y_dim):
                for z in range(z_dim):
                    coords.append((0, y, z))
        elif face == Face.RIGHT:
            for y in range(y_dim):
                for z in range(z_dim):
                    coords.append((x_dim - 1, y, z))
        elif face == Face.TOP:
            for x in range(x_dim):
                for z in range(z_dim):
                    coords.append((x, y_dim - 1, z))
        elif face == Face.BOTTOM:
            for x in range(x_dim):
                for z in range(z_dim):
                    coords.append((x, 0, z))
        return coords

    def _get_face_center(self, face: Face) -> Coordinates3D:
        """Находит центральную клетку грани."""
        x_dim, y_dim, z_dim = self.dimensions
        cx, cy, cz = (x_dim - 1) // 2, (y_dim - 1) // 2, (z_dim - 1) // 2

        if face == Face.FRONT:
            return (cx, cy, 0)
        if face == Face.BACK:
            return (cx, cy, z_dim - 1)
        if face == Face.LEFT:
            return (0, cy, cz)
        if face == Face.RIGHT:
            return (x_dim - 1, cy, cz)
        if face == Face.TOP:
            return (cx, y_dim - 1, cz)
        if face == Face.BOTTOM:
            return (cx, 0, cz)

        # Fallback
        return (cx, cy, cz)
