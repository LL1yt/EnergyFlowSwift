#!/usr/bin/env python3
"""
Simple 2D Demo - Демонстрация концепции клеточной нейронной сети

Это наглядная демонстрация того, как работает наша 3D система, но в упрощенном 2D виде.
Показывает распространение сигналов по решетке из "умных клеток".

Биологическая аналогия: Как нервный импульс распространяется по нервной ткани.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn

# Добавляем путь к проекту для импорта модулей
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.cell_prototype.main import CellPrototype
    print("[OK] Модули загружены успешно!")
except ImportError as e:
    print(f"[ERROR] Ошибка импорта: {e}")
    print("[IDEA] Убедитесь, что запускаете из корневой директории проекта")
    sys.exit(1)


class Simple2DLattice:
    """
    Простая 2D решетка из клеток для демонстрации концепции
    
    Аналогия: Плоский кусочек нервной ткани из одинаковых нейронов
    """
    
    def __init__(self, width=10, height=10, cell_config=None):
        """
        Инициализация 2D решетки
        
        Args:
            width (int): Ширина решетки
            height (int): Высота решетки  
            cell_config (dict): Конфигурация клеток
        """
        self.width = width
        self.height = height
        self.size = (height, width)
        
        # Конфигурация клетки по умолчанию
        default_config = {
            'input_size': 1,      # Размер внешнего входа
            'state_size': 4,      # Размер состояния клетки
            'hidden_size': 16,    # Размер скрытого слоя
            'num_neighbors': 4,   # 4 соседа в 2D
            'activation': 'tanh',
            'use_bias': True
        }
        
        self.cell_config = cell_config or default_config
        
        # Создаем один прототип клетки (используется всеми!)
        self.cell_prototype = CellPrototype(**self.cell_config)
        
        # Состояния всех клеток решетки
        self.states = torch.zeros(height, width, self.cell_config['state_size'])
        
        # История состояний для анимации
        self.history = []
        
        print(f"[OK] Создана 2D решетка {width}x{height} с общим прототипом клетки")
    
    def get_neighbors(self, row, col):
        """
        Получает состояния соседей для клетки (row, col)
        
        В 2D у каждой клетки 4 соседа: сверху, снизу, слева, справа
        """
        neighbors = []
        
        # Соседи: вверх, вниз, влево, вправо
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            
            # Граничные условия - зеркальное отражение
            if nr < 0: nr = 0
            if nr >= self.height: nr = self.height - 1
            if nc < 0: nc = 0  
            if nc >= self.width: nc = self.width - 1
            
            neighbors.append(self.states[nr, nc])
        
        return torch.stack(neighbors, dim=0)  # [4, state_size]
    
    def step(self, external_input=None):
        """
        Один шаг времени - все клетки обновляются синхронно
        
        Args:
            external_input (torch.Tensor): Внешний вход для клеток [height, width, input_size]
        """
        new_states = torch.zeros_like(self.states)
        
        # Обновляем каждую клетку
        for row in range(self.height):
            for col in range(self.width):
                # Получаем входы для клетки
                neighbors = self.get_neighbors(row, col)  # [4, state_size]
                own_state = self.states[row, col]  # [state_size]
                
                # Внешний вход для этой клетки
                if external_input is not None:
                    ext_input = external_input[row, col]
                else:
                    ext_input = torch.zeros(1)  # Нет внешнего входа
                
                # Вызываем прототип клетки с правильными аргументами
                # CellPrototype.forward(neighbor_states, own_state, external_input)
                new_states[row, col] = self.cell_prototype(
                    neighbor_states=neighbors.unsqueeze(0),  # [1, 4, state_size]
                    own_state=own_state.unsqueeze(0),        # [1, state_size] 
                    external_input=ext_input.unsqueeze(0)    # [1, ext_input_size]
                ).squeeze(0)
        
        # Обновляем состояния
        self.states = new_states
        
        # Сохраняем в историю
        self.history.append(self.states.clone().detach())
    
    def reset(self):
        """Сбрасывает состояния решетки"""
        self.states = torch.zeros_like(self.states)
        self.history = []
    
    def get_activity_map(self):
        """
        Возвращает карту активности для визуализации
        
        Returns:
            numpy.ndarray: 2D массив активности [height, width]
        """
        # Берем норму вектора состояния как активность
        activity = torch.norm(self.states, dim=2)
        return activity.detach().numpy()


class PatternGenerator:
    """
    Генератор паттернов для демонстрации
    
    Создает различные входные сигналы для демонстрации возможностей системы
    """
    
    @staticmethod
    def point_source(width, height, x, y, intensity=1.0):
        """Точечный источник сигнала"""
        pattern = torch.zeros(height, width, 1)
        pattern[y, x, 0] = intensity
        return pattern
    
    @staticmethod  
    def wave_source(width, height, side='left', intensity=1.0):
        """Волна с одной стороны"""
        pattern = torch.zeros(height, width, 1)
        
        if side == 'left':
            pattern[:, 0, 0] = intensity
        elif side == 'right':
            pattern[:, -1, 0] = intensity
        elif side == 'top':
            pattern[0, :, 0] = intensity  
        elif side == 'bottom':
            pattern[-1, :, 0] = intensity
            
        return pattern
    
    @staticmethod
    def pulse_pattern(width, height, center_x, center_y, radius, intensity=1.0):
        """Круглый импульс"""
        pattern = torch.zeros(height, width, 1)
        
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    pattern[y, x, 0] = intensity * (1 - distance / radius)
        
        return pattern


class Demo2DVisualizer:
    """
    Визуализатор для 2D демонстрации
    
    Создает красивые анимации и статические изображения
    """
    
    def __init__(self, lattice):
        self.lattice = lattice
        
        # Создаем красивую цветовую схему
        colors = ['#000033', '#000080', '#0000FF', '#4080FF', '#80C0FF', '#FFFFFF']
        self.cmap = LinearSegmentedColormap.from_list('neural', colors)
        
        # Настройки графика
        plt.style.use('dark_background')
    
    def plot_current_state(self, title="Текущее состояние решетки", save_path=None):
        """Показывает текущее состояние решетки"""
        activity = self.lattice.get_activity_map()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(activity, cmap=self.cmap, interpolation='bilinear')
        plt.colorbar(label='Активность клеток')
        plt.title(title, fontsize=16, color='white')
        plt.xlabel('X координата', color='white')
        plt.ylabel('Y координата', color='white')
        
        # Добавляем сетку для лучшей видимости клеток
        plt.grid(True, alpha=0.3, color='white')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='black')
            print(f"[SAVE] Изображение сохранено: {save_path}")
        
        plt.show()
    
    def create_animation(self, save_path=None, fps=10):
        """
        Создает анимацию истории состояний
        
        Args:
            save_path (str): Путь для сохранения анимации
            fps (int): Кадры в секунду
        """
        if not self.lattice.history:
            print("[ERROR] Нет истории для анимации. Запустите симуляцию сначала.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        
        # Находим максимальную активность для нормализации
        max_activity = 0
        for state in self.lattice.history:
            activity = torch.norm(state, dim=2).detach().numpy()
            max_activity = max(max_activity, activity.max())
        
        im = ax.imshow(np.zeros((self.lattice.height, self.lattice.width)), 
                      cmap=self.cmap, vmin=0, vmax=max_activity,
                      interpolation='bilinear')
        
        ax.set_title('Распространение сигнала по клеточной решетке', 
                    fontsize=16, color='white')
        ax.set_xlabel('X координата', color='white')
        ax.set_ylabel('Y координата', color='white')
        
        # Добавляем colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Активность клеток', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        
        # Текст с номером кадра
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           color='white', fontsize=12, va='top')
        
        def animate(frame):
            """Функция анимации для каждого кадра"""
            if frame < len(self.lattice.history):
                state = self.lattice.history[frame]
                activity = torch.norm(state, dim=2).detach().numpy()
                im.set_array(activity)
                time_text.set_text(f'Шаг времени: {frame}')
            return [im, time_text]
        
        # Создаем анимацию
        anim = animation.FuncAnimation(fig, animate, frames=len(self.lattice.history),
                                     interval=1000//fps, blit=True, repeat=True)
        
        if save_path:
            print(f"[SAVE] Сохраняю анимацию в {save_path}...")
            anim.save(save_path, writer='pillow', fps=fps)
            print("[OK] Анимация сохранена!")
        
        plt.show()
        return anim


def run_wave_demo():
    """
    Демонстрация #1: Распространение волны
    
    Показывает как сигнал распространяется по решетке волнами
    """
    print("\n[DATA] ДЕМОНСТРАЦИЯ #1: Распространение волны")
    print("=" * 50)
    
    # Создаем решетку
    lattice = Simple2DLattice(width=15, height=15)
    visualizer = Demo2DVisualizer(lattice)
    
    # Создаем волновой источник слева
    wave_input = PatternGenerator.wave_source(15, 15, side='left', intensity=2.0)
    
    print("🌊 Запускаем волну с левой стороны...")
    
    # Запускаем симуляцию
    for step in range(20):
        if step < 5:  # Подаем вход только первые 5 шагов
            lattice.step(external_input=wave_input)
        else:
            lattice.step()  # Дальше система работает сама
        
        print(f"  Шаг {step + 1}: максимальная активность = {lattice.get_activity_map().max():.3f}")
    
    # Показываем результат
    visualizer.plot_current_state("Распространение волны (финальное состояние)")
    
    # Создаем анимацию
    print("[MOVIE] Создаю анимацию...")
    anim = visualizer.create_animation(save_path="visualizations/wave_demo.gif", fps=5)
    
    return lattice, anim


def run_pulse_demo():
    """
    Демонстрация #2: Распространение импульса
    
    Показывает как точечный импульс расходится кругами
    """
    print("\n[DATA] ДЕМОНСТРАЦИЯ #2: Распространение импульса")
    print("=" * 50)
    
    # Создаем решетку
    lattice = Simple2DLattice(width=12, height=12)
    visualizer = Demo2DVisualizer(lattice)
    
    # Создаем импульс в центре  
    center_x, center_y = 6, 6
    pulse_input = PatternGenerator.pulse_pattern(12, 12, center_x, center_y, 
                                               radius=2, intensity=3.0)
    
    print(f"💥 Запускаем импульс в центре ({center_x}, {center_y})...")
    
    # Запускаем симуляцию
    for step in range(15):
        if step < 3:  # Подаем импульс только первые 3 шага
            lattice.step(external_input=pulse_input)
        else:
            lattice.step()
        
        print(f"  Шаг {step + 1}: активных клеток = {(lattice.get_activity_map() > 0.1).sum()}")
    
    # Показываем результат
    visualizer.plot_current_state("Распространение импульса (финальное состояние)")
    
    # Создаем анимацию
    print("[MOVIE] Создаю анимацию...")
    anim = visualizer.create_animation(save_path="visualizations/pulse_demo.gif", fps=4)
    
    return lattice, anim


def run_interference_demo():
    """
    Демонстрация #3: Интерференция волн
    
    Показывает как две волны взаимодействуют друг с другом
    """
    print("\n[DATA] ДЕМОНСТРАЦИЯ #3: Интерференция волн")
    print("=" * 50)
    
    # Создаем решетку
    lattice = Simple2DLattice(width=20, height=15)
    visualizer = Demo2DVisualizer(lattice)
    
    # Создаем два источника волн
    wave1 = PatternGenerator.wave_source(20, 15, side='left', intensity=1.5)
    wave2 = PatternGenerator.wave_source(20, 15, side='right', intensity=1.5)
    
    print("🌊 Запускаем две волны навстречу друг другу...")
    
    # Запускаем симуляцию
    for step in range(25):
        if step < 8:  # Подаем волны первые 8 шагов
            combined_input = wave1 + wave2
            lattice.step(external_input=combined_input)
        else:
            lattice.step()
        
        activity_map = lattice.get_activity_map()
        print(f"  Шаг {step + 1}: средняя активность = {activity_map.mean():.3f}, "
              f"пиковая = {activity_map.max():.3f}")
    
    # Показываем результат
    visualizer.plot_current_state("Интерференция волн (финальное состояние)")
    
    # Создаем анимацию
    print("[MOVIE] Создаю анимацию...")
    anim = visualizer.create_animation(save_path="visualizations/interference_demo.gif", fps=6)
    
    return lattice, anim


def main():
    """
    Главная функция демонстрации
    """
    print("[MASK] SIMPLE 2D DEMO - Клеточная Нейронная Сеть")
    print("=" * 60)
    print("""
    Эта демонстрация показывает базовые принципы работы нашей 3D системы
    на простом 2D примере. Каждая клетка в решетке:
    
    • [BRAIN] Получает сигналы от 4 соседей
    • [FAST] Обрабатывает их одинаковой нейросетью  
    • 📡 Передает результат дальше
    • [REFRESH] Все клетки обновляются синхронно
    
    Биологическая аналогия: Как нервный импульс распространяется по ткани
    """)
    
    # Создаем папку для визуализаций
    os.makedirs("visualizations", exist_ok=True)
    
    # Запускаем демонстрации
    demos = []
    
    try:
        # Демонстрация 1: Волна
        lattice1, anim1 = run_wave_demo()
        demos.append(("Волна", lattice1, anim1))
        
        # Небольшая пауза между демонстрациями
        input("\n⏸️  Нажмите Enter для следующей демонстрации...")
        
        # Демонстрация 2: Импульс  
        lattice2, anim2 = run_pulse_demo()
        demos.append(("Импульс", lattice2, anim2))
        
        # Пауза
        input("\n⏸️  Нажмите Enter для последней демонстрации...")
        
        # Демонстрация 3: Интерференция
        lattice3, anim3 = run_interference_demo()
        demos.append(("Интерференция", lattice3, anim3))
        
    except KeyboardInterrupt:
        print("\n[STOP]  Демонстрация прервана пользователем")
    
    # Итоговая информация
    print("\n[SUCCESS] ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 30)
    print(f"[OK] Проведено демонстраций: {len(demos)}")
    print("[FOLDER] Анимации сохранены в папке visualizations/")
    print("""
    [IDEA] Что мы увидели:
    • Как одинаковые "умные клетки" создают сложное поведение
    • Как сигналы распространяются через решетку
    • Как возникают паттерны и взаимодействия
    
    [START] Следующий шаг: 3D версия с полным функционалом!
    """)


if __name__ == "__main__":
    main() 