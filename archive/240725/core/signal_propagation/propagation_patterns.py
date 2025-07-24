"""
Propagation Patterns - анализ паттернов распространения сигналов

Этот модуль анализирует:
- Волновые паттерны
- Циклические паттерны  
- Пространственную динамику
- Статистику активности
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

class PatternType(Enum):
    """Типы паттернов распространения"""
    WAVE = "wave"  # Волновой паттерн
    SPIRAL = "spiral"  # Спиральный паттерн
    UNIFORM = "uniform"  # Равномерное распространение
    CLUSTERED = "clustered"  # Кластерное распространение
    CHAOTIC = "chaotic"  # Хаотичное поведение
    STATIC = "static"  # Статичное состояние

@dataclass
class PatternAnalysisResult:
    """Результат анализа паттерна"""
    pattern_type: PatternType
    confidence: float  # Уверенность в классификации (0-1)
    characteristics: Dict[str, float]  # Характеристики паттерна
    spatial_features: Dict[str, Any]  # Пространственные особенности
    temporal_features: Dict[str, Any]  # Временные особенности

class PatternAnalyzer:
    """
    Анализатор паттернов распространения сигналов
    
    Основные функции:
    - Классификация типов паттернов
    - Извлечение характеристик
    - Анализ динамики
    """
    
    def __init__(self, window_size: int = 10):
        """
        Инициализация анализатора
        
        Args:
            window_size: Размер окна для анализа временных паттернов
        """
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # История для анализа
        self.analysis_history = []
        
        # Параметры анализа
        self.wave_threshold = 0.3  # Порог для детекции волн
        self.spiral_threshold = 0.2  # Порог для детекции спиралей
        self.uniformity_threshold = 0.1  # Порог для равномерности
        
    def analyze_pattern(self, signal_history: List[torch.Tensor]) -> PatternAnalysisResult:
        """
        Анализ паттерна по истории сигналов
        
        Args:
            signal_history: История состояний сигналов
            
        Returns:
            PatternAnalysisResult: Результат анализа
        """
        if len(signal_history) < 2:
            return PatternAnalysisResult(
                pattern_type=PatternType.STATIC,
                confidence=1.0,
                characteristics={},
                spatial_features={},
                temporal_features={}
            )
        
        # Анализируем пространственные характеристики
        spatial_features = self._analyze_spatial_features(signal_history)
        
        # Анализируем временные характеристики
        temporal_features = self._analyze_temporal_features(signal_history)
        
        # Классифицируем паттерн
        pattern_type, confidence = self._classify_pattern(spatial_features, temporal_features)
        
        # Извлекаем характеристики
        characteristics = self._extract_characteristics(signal_history, pattern_type)
        
        result = PatternAnalysisResult(
            pattern_type=pattern_type,
            confidence=confidence,
            characteristics=characteristics,
            spatial_features=spatial_features,
            temporal_features=temporal_features
        )
        
        self.analysis_history.append(result)
        
        return result
    
    def _analyze_spatial_features(self, signal_history: List[torch.Tensor]) -> Dict[str, Any]:
        """Анализ пространственных особенностей"""
        latest_signal = signal_history[-1]
        
        features = {}
        
        # Центр масс
        features['center_of_mass'] = self._calculate_center_of_mass(latest_signal)
        
        # Пространственная дисперсия
        features['spatial_variance'] = self._calculate_spatial_variance(latest_signal)
        
        # Активные области
        features['active_regions'] = self._find_active_regions(latest_signal)
        
        # Градиенты
        features['gradients'] = self._calculate_gradients(latest_signal)
        
        # Симметрия
        features['symmetry'] = self._calculate_symmetry(latest_signal)
        
        return features
    
    def _analyze_temporal_features(self, signal_history: List[torch.Tensor]) -> Dict[str, Any]:
        """Анализ временных особенностей"""
        features = {}
        
        # Изменения во времени
        if len(signal_history) > 1:
            changes = []
            for i in range(1, len(signal_history)):
                change = torch.abs(signal_history[i] - signal_history[i-1]).mean().item()
                changes.append(change)
            
            features['average_change'] = np.mean(changes)
            features['change_variance'] = np.var(changes)
            features['change_trend'] = self._calculate_trend(changes)
        
        # Периодичность
        features['periodicity'] = self._detect_periodicity(signal_history)
        
        # Скорость распространения
        features['propagation_speed'] = self._estimate_propagation_speed(signal_history)
        
        return features
    
    def _classify_pattern(self, spatial_features: Dict, temporal_features: Dict) -> Tuple[PatternType, float]:
        """Классификация типа паттерна"""
        scores = {pattern_type: 0.0 for pattern_type in PatternType}
        
        # Анализ на основе временных изменений
        avg_change = temporal_features.get('average_change', 0)
        
        if avg_change < 1e-6:
            scores[PatternType.STATIC] += 0.8
        
        # Анализ на основе градиентов
        gradients = spatial_features.get('gradients', {})
        gradient_magnitude = gradients.get('magnitude', 0)
        
        if gradient_magnitude > self.wave_threshold:
            scores[PatternType.WAVE] += 0.6
        
        # Анализ равномерности
        spatial_variance = spatial_features.get('spatial_variance', 0)
        if spatial_variance < self.uniformity_threshold:
            scores[PatternType.UNIFORM] += 0.5
        
        # Анализ периодичности
        periodicity = temporal_features.get('periodicity', 0)
        if periodicity > 0.5:
            scores[PatternType.WAVE] += 0.4
        
        # Анализ симметрии для спиралей
        symmetry = spatial_features.get('symmetry', {})
        if symmetry.get('rotational', 0) > 0.3:
            scores[PatternType.SPIRAL] += 0.5
        
        # Выбираем паттерн с максимальным счетом
        best_pattern = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_pattern], 1.0)
        
        # Если уверенность низкая, считаем хаотичным
        if confidence < 0.3:
            best_pattern = PatternType.CHAOTIC
            confidence = 1.0 - confidence
        
        return best_pattern, confidence
    
    def _extract_characteristics(self, signal_history: List[torch.Tensor], pattern_type: PatternType) -> Dict[str, float]:
        """Извлечение характеристик для конкретного типа паттерна"""
        characteristics = {}
        
        latest_signal = signal_history[-1]
        
        # Общие характеристики
        characteristics['amplitude'] = latest_signal.max().item()
        characteristics['energy'] = (latest_signal ** 2).sum().item()
        characteristics['active_fraction'] = (latest_signal.abs() > 1e-6).float().mean().item()
        
        # Специфичные для типа характеристики
        if pattern_type == PatternType.WAVE:
            characteristics['wavelength'] = self._estimate_wavelength(latest_signal)
            characteristics['wave_direction'] = self._estimate_wave_direction(signal_history)
        
        elif pattern_type == PatternType.SPIRAL:
            characteristics['spiral_arms'] = self._count_spiral_arms(latest_signal)
            characteristics['rotation_speed'] = self._estimate_rotation_speed(signal_history)
        
        elif pattern_type == PatternType.UNIFORM:
            characteristics['uniformity_index'] = self._calculate_uniformity_index(latest_signal)
        
        elif pattern_type == PatternType.CLUSTERED:
            characteristics['cluster_count'] = self._count_clusters(latest_signal)
            characteristics['cluster_size'] = self._average_cluster_size(latest_signal)
        
        return characteristics
    
    def _calculate_center_of_mass(self, signal: torch.Tensor) -> Tuple[float, float, float]:
        """Расчет центра масс сигнала"""
        x_size, y_size, z_size, _ = signal.shape
        
        # Суммарная интенсивность
        total_intensity = signal.abs().sum(dim=-1)
        total_mass = total_intensity.sum()
        
        if total_mass == 0:
            return (x_size/2, y_size/2, z_size/2)
        
        # Координаты центра масс
        x_coords, y_coords, z_coords = torch.meshgrid(
            torch.arange(x_size), torch.arange(y_size), torch.arange(z_size), indexing='ij'
        )
        
        center_x = (x_coords.float() * total_intensity).sum() / total_mass
        center_y = (y_coords.float() * total_intensity).sum() / total_mass
        center_z = (z_coords.float() * total_intensity).sum() / total_mass
        
        return (center_x.item(), center_y.item(), center_z.item())
    
    def _calculate_spatial_variance(self, signal: torch.Tensor) -> float:
        """Расчет пространственной дисперсии"""
        intensity = signal.abs().sum(dim=-1)
        return intensity.var().item()
    
    def _find_active_regions(self, signal: torch.Tensor, threshold: float = 1e-6) -> Dict[str, Any]:
        """Поиск активных областей"""
        active_mask = signal.abs().sum(dim=-1) > threshold
        active_count = active_mask.sum().item()
        total_count = active_mask.numel()
        
        return {
            'count': active_count,
            'fraction': active_count / total_count,
            'total_cells': total_count
        }
    
    def _calculate_gradients(self, signal: torch.Tensor) -> Dict[str, float]:
        """Расчет градиентов сигнала"""
        intensity = signal.abs().sum(dim=-1)
        
        # Градиенты по каждой оси
        grad_x = torch.diff(intensity, dim=0).abs().mean()
        grad_y = torch.diff(intensity, dim=1).abs().mean()
        grad_z = torch.diff(intensity, dim=2).abs().mean()
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return {
            'x': grad_x.item(),
            'y': grad_y.item(), 
            'z': grad_z.item(),
            'magnitude': magnitude.item()
        }
    
    def _calculate_symmetry(self, signal: torch.Tensor) -> Dict[str, float]:
        """Расчет симметрии сигнала"""
        intensity = signal.abs().sum(dim=-1)
        
        # Симметрия относительно центральных плоскостей
        x_sym = self._plane_symmetry(intensity, axis=0)
        y_sym = self._plane_symmetry(intensity, axis=1)
        z_sym = self._plane_symmetry(intensity, axis=2)
        
        # Примитивная оценка ротационной симметрии
        rotational = min(x_sym, y_sym, z_sym)
        
        return {
            'x_plane': x_sym,
            'y_plane': y_sym,
            'z_plane': z_sym,
            'rotational': rotational
        }
    
    def _plane_symmetry(self, tensor: torch.Tensor, axis: int) -> float:
        """Расчет симметрии относительно плоскости"""
        size = tensor.shape[axis]
        mid = size // 2
        
        if axis == 0:
            left = tensor[:mid]
            right = tensor[size-mid:].flip(dims=[0])
        elif axis == 1:
            left = tensor[:, :mid]
            right = tensor[:, size-mid:].flip(dims=[1])
        else:  # axis == 2
            left = tensor[:, :, :mid]
            right = tensor[:, :, size-mid:].flip(dims=[2])
        
        # Минимальный размер для сравнения
        min_size = min(left.shape[axis], right.shape[axis])
        if axis == 0:
            left = left[:min_size]
            right = right[:min_size]
        elif axis == 1:
            left = left[:, :min_size]
            right = right[:, :min_size]
        else:
            left = left[:, :, :min_size]
            right = right[:, :, :min_size]
        
        # Корреляция между левой и правой частями
        diff = torch.abs(left - right).mean()
        max_val = max(left.abs().max(), right.abs().max())
        
        if max_val == 0:
            return 1.0
        
        symmetry = 1.0 - (diff / max_val).item()
        return max(0.0, symmetry)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Расчет тренда в последовательности"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Простая линейная регрессия
        slope = np.corrcoef(x, y)[0, 1] if len(values) > 2 else 0.0
        return slope
    
    def _detect_periodicity(self, signal_history: List[torch.Tensor]) -> float:
        """Детекция периодичности в сигналах"""
        if len(signal_history) < 4:
            return 0.0
        
        # Используем автокорреляцию для детекции периодов
        intensities = [s.abs().sum().item() for s in signal_history]
        
        if len(intensities) < 4:
            return 0.0
        
        # Простая автокорреляция
        autocorr = np.correlate(intensities, intensities, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Нормализация
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0
        
        # Поиск пиков (упрощенный)
        peaks = []
        for i in range(1, min(len(autocorr) - 1, len(autocorr) // 2)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append(autocorr[i])
        
        return max(peaks) if peaks else 0.0
    
    def _estimate_propagation_speed(self, signal_history: List[torch.Tensor]) -> float:
        """Оценка скорости распространения"""
        if len(signal_history) < 2:
            return 0.0
        
        # Простая оценка на основе изменения центра масс
        centers = [self._calculate_center_of_mass(s) for s in signal_history[-3:]]
        
        if len(centers) < 2:
            return 0.0
        
        # Расстояние между последними центрами
        dist = np.sqrt(sum((a - b)**2 for a, b in zip(centers[-1], centers[-2])))
        
        return dist  # скорость = расстояние за 1 временной шаг
    
    def _estimate_wavelength(self, signal: torch.Tensor) -> float:
        """Оценка длины волны"""
        # Упрощенная оценка через автокорреляцию
        intensity = signal.abs().sum(dim=-1)
        
        # Берем срез по одной из осей
        slice_data = intensity[:, intensity.shape[1]//2, intensity.shape[2]//2]
        
        if slice_data.numel() < 4:
            return 0.0
        
        # Автокорреляция через NumPy
        slice_np = slice_data.detach().cpu().numpy()
        autocorr = np.correlate(slice_np, slice_np, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Поиск первого пика после 0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                return float(i)
        
        return 0.0
    
    def _estimate_wave_direction(self, signal_history: List[torch.Tensor]) -> Tuple[float, float, float]:
        """Оценка направления волны"""
        if len(signal_history) < 2:
            return (0.0, 0.0, 0.0)
        
        # Направление на основе смещения центра масс
        center1 = self._calculate_center_of_mass(signal_history[-2])
        center2 = self._calculate_center_of_mass(signal_history[-1])
        
        direction = tuple(b - a for a, b in zip(center1, center2))
        
        # Нормализация
        magnitude = np.sqrt(sum(d**2 for d in direction))
        if magnitude > 0:
            direction = tuple(d / magnitude for d in direction)
        
        return direction
    
    def _count_spiral_arms(self, signal: torch.Tensor) -> int:
        """Подсчет спиральных рукавов (упрощенная версия)"""
        # Заглушка для сложного анализа
        return 2  # По умолчанию предполагаем 2 рукава
    
    def _estimate_rotation_speed(self, signal_history: List[torch.Tensor]) -> float:
        """Оценка скорости вращения спирали"""
        # Заглушка для сложного анализа
        return 0.1  # Условная скорость
    
    def _calculate_uniformity_index(self, signal: torch.Tensor) -> float:
        """Расчет индекса равномерности"""
        intensity = signal.abs().sum(dim=-1)
        mean_intensity = intensity.mean()
        
        if mean_intensity == 0:
            return 1.0
        
        variance = intensity.var()
        uniformity = 1.0 / (1.0 + variance / (mean_intensity**2))
        
        return uniformity.item()
    
    def _count_clusters(self, signal: torch.Tensor) -> int:
        """Подсчет кластеров (упрощенная версия)"""
        # Заглушка для сложного анализа кластеризации
        intensity = signal.abs().sum(dim=-1)
        active_mask = intensity > intensity.mean()
        return active_mask.sum().item() // 10  # Примерная оценка
    
    def _average_cluster_size(self, signal: torch.Tensor) -> float:
        """Средний размер кластера"""
        # Заглушка
        return 5.0  # Условный размер

class PropagationPatterns:
    """
    Основной класс для анализа паттернов распространения
    
    Координирует анализ и предоставляет высокоуровневый интерфейс
    """
    
    def __init__(self):
        """Инициализация анализатора паттернов"""
        self.analyzer = PatternAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # История анализов
        self.pattern_history = []
        
    def analyze_propagation(self, signal_history: List[torch.Tensor]) -> PatternAnalysisResult:
        """
        Анализ паттерна распространения
        
        Args:
            signal_history: История состояний сигналов
            
        Returns:
            PatternAnalysisResult: Результат анализа
        """
        result = self.analyzer.analyze_pattern(signal_history)
        self.pattern_history.append(result)
        
        self.logger.info(f"Обнаружен паттерн: {result.pattern_type.value} "
                        f"(уверенность: {result.confidence:.2f})")
        
        return result
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Получение сводки по всем обнаруженным паттернам"""
        if not self.pattern_history:
            return {}
        
        # Статистика типов паттернов
        pattern_counts = {}
        for result in self.pattern_history:
            pattern_type = result.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Средняя уверенность
        avg_confidence = np.mean([r.confidence for r in self.pattern_history])
        
        # Последний паттерн
        latest_pattern = self.pattern_history[-1]
        
        return {
            'total_analyses': len(self.pattern_history),
            'pattern_distribution': pattern_counts,
            'average_confidence': avg_confidence,
            'latest_pattern': {
                'type': latest_pattern.pattern_type.value,
                'confidence': latest_pattern.confidence,
                'characteristics': latest_pattern.characteristics
            }
        }
    
    def reset(self):
        """Сброс истории анализов"""
        self.pattern_history.clear()
        self.analyzer.analysis_history.clear()
        self.logger.info("История паттернов сброшена") 