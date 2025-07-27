"""
Визуализаторы для 3D клеточной нейронной сети.

Этот модуль содержит основные классы визуализации:
- Lattice3DVisualizer: Основной визуализатор решетки
- IOPointVisualizer: Специализированная визуализация I/O точек
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging
import time

# Импорты из основного модуля
from .main import VisualizationConfig, VisualizationMode, Coordinates3D
from core.lattice_3d import Lattice3D, IOPointPlacer, Face


# =============================================================================
# ОСНОВНОЙ ВИЗУАЛИЗАТОР РЕШЕТКИ
# =============================================================================

class Lattice3DVisualizer:
    """
    Основной класс для визуализации 3D решетки клеток.
    
    Предоставляет методы для отображения состояний клеток, связей,
    I/O точек и создания интерактивных 3D сцен.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Инициализация визуализатора.
        
        Args:
            config: Конфигурация визуализации
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Кэш для оптимизации производительности
        self._mesh_cache = {}
        self._color_cache = {}
        self._layout_cache = {}
        
        # История для анимации
        self._state_history = []
        self._animation_frame = 0
        
        # Метрики производительности
        self._render_times = []
        self._last_render_time = 0
        
    def visualize_lattice(self, lattice: Lattice3D, 
                         mode: VisualizationMode = VisualizationMode.STATIC) -> go.Figure:
        """
        Создает визуализацию 3D решетки.
        
        Args:
            lattice: Объект решетки для визуализации
            mode: Режим визуализации
            
        Returns:
            go.Figure: Plotly фигура с 3D визуализацией
        """
        start_time = time.time()
        
        try:
            # Проверяем размер решетки для производительности
            if self._is_lattice_too_large(lattice):
                self.logger.warning("Large lattice detected, applying optimizations")
                
            # Получаем состояния клеток
            states = lattice.get_states()
            dimensions = lattice.config.dimensions
            
            # Создаем основную фигуру
            fig = self._create_base_figure()
            
            # Добавляем клетки решетки
            self._add_lattice_cells(fig, states, dimensions)
            
            # Добавляем I/O точки
            self._add_io_points(fig, lattice)
            
            # Добавляем связи (если включено)
            if self.config.show_connections:
                self._add_cell_connections(fig, lattice)
                
            # Настраиваем интерактивность
            if self.config.enable_interaction:
                self._configure_interaction(fig)
                
            # Обновляем метрики производительности
            render_time = time.time() - start_time
            self._update_render_metrics(render_time)
            
            self.logger.info(f"Lattice visualization created in {render_time:.3f}s")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating lattice visualization: {e}")
            raise
            
    def _create_base_figure(self) -> go.Figure:
        """Создает базовую 3D фигуру с настройками"""
        fig = go.Figure()
        
        fig.update_layout(
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.background_color,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                bgcolor=self.config.background_color,
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True
        )
        
        return fig
        
    def _add_lattice_cells(self, fig: go.Figure, states: torch.Tensor, 
                          dimensions: Tuple[int, int, int]):
        """Добавляет клетки решетки на график"""
        x_dim, y_dim, z_dim = dimensions
        
        # Создаем координаты всех клеток
        x_coords, y_coords, z_coords = [], [], []
        colors = []
        activations = []
        
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    # Координаты
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    
                    # Получаем активацию клетки
                    linear_idx = z * (x_dim * y_dim) + y * x_dim + x
                    activation = float(states[linear_idx].mean()) if len(states.shape) > 1 else float(states[linear_idx])
                    activations.append(activation)
                    
                    # Определяем цвет на основе активации
                    color = self._get_cell_color(activation)
                    colors.append(color)
        
        # Добавляем scatter plot для клеток
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=self.config.cell_size * 10,
                color=colors,
                opacity=self.config.cell_opacity,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=[f"Cell ({x},{y},{z})<br>Activation: {act:.3f}" 
                  for x, y, z, act in zip(x_coords, y_coords, z_coords, activations)],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Neural Cells'
        ))
        
    def _get_cell_color(self, activation: float) -> str:
        """Определяет цвет клетки на основе уровня активации"""
        thresholds = self.config.activation_thresholds
        colors = self.config.cell_colors
        
        if activation < thresholds["low"]:
            return colors["inactive"]
        elif activation < thresholds["medium"]:
            return colors["active_low"]
        elif activation < thresholds["high"]:
            return colors["active_medium"]
        else:
            return colors["active_high"]
            
    def _add_io_points(self, fig: go.Figure, lattice: Lattice3D):
        """Добавляет визуализацию I/O точек"""
        io_info = lattice.get_io_point_info()
        
        # Точки ввода
        if 'input_points' in io_info:
            input_points = io_info['input_points']
            if input_points:
                x_in, y_in, z_in = zip(*input_points)
                fig.add_trace(go.Scatter3d(
                    x=x_in, y=y_in, z=z_in,
                    mode='markers',
                    marker=dict(
                        size=self.config.input_point_size * 12,
                        color=self.config.input_point_color,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    name='Input Points',
                    text=[f"Input Point ({x},{y},{z})" for x, y, z in input_points],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))
        
        # Точки вывода
        if 'output_points' in io_info:
            output_points = io_info['output_points']
            if output_points:
                x_out, y_out, z_out = zip(*output_points)
                fig.add_trace(go.Scatter3d(
                    x=x_out, y=y_out, z=z_out,
                    mode='markers',
                    marker=dict(
                        size=self.config.output_point_size * 12,
                        color=self.config.output_point_color,
                        symbol='square',
                        line=dict(width=2, color='white')
                    ),
                    name='Output Points',
                    text=[f"Output Point ({x},{y},{z})" for x, y, z in output_points],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))
                
    def _add_cell_connections(self, fig: go.Figure, lattice: Lattice3D):
        """Добавляет визуализацию связей между клетками"""
        # Эта функция будет реализована для показа соединений между соседними клетками
        # Пока пропускаем для оптимизации производительности
        self.logger.debug("Cell connections visualization not yet implemented")
        
    def _configure_interaction(self, fig: go.Figure):
        """Настраивает интерактивность графика"""
        if self.config.enable_interaction:
            fig.update_layout(
                scene=dict(
                    dragmode='orbit'
                )
            )
            
    def _is_lattice_too_large(self, lattice: Lattice3D) -> bool:
        """Проверяет, не слишком ли большая решетка для визуализации"""
        dimensions = lattice.config.dimensions
        max_dims = self.config.max_lattice_size
        
        return any(d > max_d for d, max_d in zip(dimensions, max_dims))
        
    def _update_render_metrics(self, render_time: float):
        """Обновляет метрики производительности рендеринга"""
        self._render_times.append(render_time)
        self._last_render_time = render_time
        
        # Ограничиваем историю
        if len(self._render_times) > 100:
            self._render_times = self._render_times[-100:]
            
    def get_performance_stats(self) -> Dict[str, float]:
        """Возвращает статистику производительности"""
        if not self._render_times:
            return {}
            
        return {
            'avg_render_time': np.mean(self._render_times),
            'min_render_time': np.min(self._render_times),
            'max_render_time': np.max(self._render_times),
            'last_render_time': self._last_render_time,
            'total_renders': len(self._render_times)
        }


# =============================================================================
# СПЕЦИАЛИЗИРОВАННЫЙ ВИЗУАЛИЗАТОР I/O ТОЧЕК
# =============================================================================

class IOPointVisualizer:
    """
    Специализированный визуализатор для I/O точек и их стратегий размещения.
    
    Предоставляет детальную визуализацию различных стратегий размещения
    точек ввода/вывода и их распределение по граням решетки.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Инициализация визуализатора I/O точек.
        
        Args:
            config: Конфигурация визуализации
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def visualize_io_strategy(self, io_placer: IOPointPlacer, 
                             face: Face = Face.FRONT) -> go.Figure:
        """
        Визуализирует стратегию размещения I/O точек на конкретной грани.
        
        Args:
            io_placer: Объект размещения I/O точек
            face: Грань для визуализации
            
        Returns:
            go.Figure: Plotly фигура с визуализацией стратегии
        """
        fig = self._create_io_strategy_figure()
        
        # Получаем точки для визуализации
        input_points = io_placer.get_input_points(face)
        output_points = io_placer.get_output_points(face)
        
        # Добавляем грань решетки
        self._add_face_grid(fig, io_placer.dimensions, face)
        
        # Добавляем I/O точки
        self._add_io_points_to_face(fig, input_points, output_points, face)
        
        # Добавляем информационную панель
        self._add_strategy_info(fig, io_placer, face)
        
        return fig
        
    def _create_io_strategy_figure(self) -> go.Figure:
        """Создает базовую фигуру для визуализации I/O стратегии"""
        fig = go.Figure()
        
        fig.update_layout(
            title="I/O Points Placement Strategy",
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
        
    def _add_face_grid(self, fig: go.Figure, dimensions: Tuple[int, int, int], face: Face):
        """Добавляет сетку грани решетки"""
        # Определяем размеры грани в зависимости от её ориентации
        if face in [Face.FRONT, Face.BACK]:  # Z-плоскость
            width, height = dimensions[0], dimensions[1]
        elif face in [Face.LEFT, Face.RIGHT]:  # X-плоскость
            width, height = dimensions[1], dimensions[2]
        else:  # Y-плоскость (TOP, BOTTOM)
            width, height = dimensions[0], dimensions[2]
            
        # Создаем координаты сетки
        x_grid, y_grid = [], []
        for x in range(width):
            for y in range(height):
                x_grid.append(x)
                y_grid.append(y)
                
        # Добавляем сетку как scatter plot
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid,
            mode='markers',
            marker=dict(
                size=8,
                color='lightgray',
                opacity=0.5
            ),
            name='Grid Cells',
            showlegend=False
        ))
        
    def _add_io_points_to_face(self, fig: go.Figure, input_points: List[Tuple], 
                              output_points: List[Tuple], face: Face):
        """Добавляет I/O точки на грань"""
        # Преобразуем 3D координаты в 2D для грани
        input_2d = self._convert_3d_to_face_2d(input_points, face)
        output_2d = self._convert_3d_to_face_2d(output_points, face)
        
        # Добавляем точки ввода
        if input_2d:
            x_in, y_in = zip(*input_2d)
            fig.add_trace(go.Scatter(
                x=x_in, y=y_in,
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.config.input_point_color,
                    symbol='diamond'
                ),
                name='Input Points'
            ))
            
        # Добавляем точки вывода
        if output_2d:
            x_out, y_out = zip(*output_2d)
            fig.add_trace(go.Scatter(
                x=x_out, y=y_out,
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.config.output_point_color,
                    symbol='square'
                ),
                name='Output Points'
            ))
            
    def _convert_3d_to_face_2d(self, points_3d: List[Tuple], face: Face) -> List[Tuple]:
        """Преобразует 3D координаты в 2D координаты грани"""
        points_2d = []
        
        for point in points_3d:
            x, y, z = point
            if face in [Face.FRONT, Face.BACK]:
                points_2d.append((x, y))
            elif face in [Face.LEFT, Face.RIGHT]:
                points_2d.append((y, z))
            else:  # TOP, BOTTOM
                points_2d.append((x, z))
                
        return points_2d
        
    def _add_strategy_info(self, fig: go.Figure, io_placer: IOPointPlacer, face: Face):
        """Добавляет информационную панель о стратегии"""
        strategy_name = io_placer.strategy.value
        dimensions = io_placer.dimensions
        
        # Рассчитываем статистику
        input_points = io_placer.get_input_points(face)
        output_points = io_placer.get_output_points(face)
        
        # Рассчитываем площадь грани
        if face in [Face.FRONT, Face.BACK]:
            face_area = dimensions[0] * dimensions[1]
        elif face in [Face.LEFT, Face.RIGHT]:
            face_area = dimensions[1] * dimensions[2]
        else:
            face_area = dimensions[0] * dimensions[2]
            
        coverage_percent = ((len(input_points) + len(output_points)) / face_area) * 100
        
        info_text = (
            f"Strategy: {strategy_name.title()}<br>"
            f"Lattice Size: {dimensions[0]}×{dimensions[1]}×{dimensions[2]}<br>"
            f"Face: {face.value.title()}<br>"
            f"Face Area: {face_area} cells<br>"
            f"Input Points: {len(input_points)}<br>"
            f"Output Points: {len(output_points)}<br>"
            f"Coverage: {coverage_percent:.1f}%"
        )
        
        fig.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
    def compare_strategies(self, lattice_dimensions: Tuple[int, int, int], 
                          strategies: List[str]) -> go.Figure:
        """
        Сравнивает различные стратегии размещения I/O точек.
        
        Args:
            lattice_dimensions: Размеры решетки
            strategies: Список названий стратегий для сравнения
            
        Returns:
            go.Figure: Фигура со сравнением стратегий
        """
        n_strategies = len(strategies)
        cols = min(3, n_strategies)
        rows = (n_strategies + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[s.title() for s in strategies],
            specs=[[{'type': 'scatter'}] * cols for _ in range(rows)]
        )
        
        for i, strategy_name in enumerate(strategies):
            row = i // cols + 1
            col = i % cols + 1
            
            try:
                from core.lattice_3d import PlacementStrategy
                strategy = PlacementStrategy(strategy_name)
                
                # Создаем IOPointPlacer для стратегии
                config = {'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6}}
                io_placer = IOPointPlacer(lattice_dimensions, strategy, config)
                
                # Получаем точки для фронтальной грани
                input_points = io_placer.get_input_points(Face.FRONT)
                output_points = io_placer.get_output_points(Face.FRONT)
                
                # Преобразуем в 2D координаты
                input_2d = [(p[0], p[1]) for p in input_points]
                output_2d = [(p[0], p[1]) for p in output_points]
                
                # Добавляем сетку
                x_grid = []
                y_grid = []
                for x in range(lattice_dimensions[0]):
                    for y in range(lattice_dimensions[1]):
                        x_grid.append(x)
                        y_grid.append(y)
                        
                fig.add_trace(
                    go.Scatter(x=x_grid, y=y_grid, mode='markers', 
                              marker=dict(size=4, color='lightgray', opacity=0.5),
                              showlegend=False, name='Grid'),
                    row=row, col=col
                )
                
                # Добавляем I/O точки
                if input_2d:
                    x_in, y_in = zip(*input_2d)
                    fig.add_trace(
                        go.Scatter(x=x_in, y=y_in, mode='markers',
                                  marker=dict(size=8, color='blue', symbol='diamond'),
                                  showlegend=False, name='Input'),
                        row=row, col=col
                    )
                    
                if output_2d:
                    x_out, y_out = zip(*output_2d)
                    fig.add_trace(
                        go.Scatter(x=x_out, y=y_out, mode='markers',
                                  marker=dict(size=8, color='green', symbol='square'),
                                  showlegend=False, name='Output'),
                        row=row, col=col
                    )
                    
            except Exception as e:
                self.logger.error(f"Error visualizing strategy {strategy_name}: {e}")
                
        fig.update_layout(
            title="I/O Placement Strategies Comparison",
            height=200 * rows,
            showlegend=False
        )
        
        return fig 