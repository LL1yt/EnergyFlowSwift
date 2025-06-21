#!/usr/bin/env python3
"""
Hybrid Cell V2 - биологически правдоподобная архитектура
=====================================================

Улучшенная гибридная клетка без потери информации:
1. NCA (4D) - внутриклеточная биохимическая динамика
2. GNN (32D) - межклеточная электрическая коммуникация
3. NCA модулирует GNN через дополнительные каналы

ПРИНЦИП: NCA и GNN работают в разных пространствах,
         но NCA влияет на эффективность GNN коммуникации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from .base_cell import BaseCell
from .nca_cell import NCACell
from .gnn_cell import GNNCell
from ...config import get_project_config
from ...utils.logging import (
    get_logger,
    log_cell_init,
    log_cell_forward,
    log_cell_component_params,
)

logger = get_logger(__name__)


class NCAModulator(nn.Module):
    """
    Модулятор влияния NCA на GNN без преобразования размерностей

    NCA (4D) генерирует модулирующие сигналы для GNN операций:
    - Attention weights modulation
    - Message passing efficiency
    - State update gates

    Биологический аналог: внутриклеточные процессы влияют на
    эффективность синаптической передачи
    """

    def __init__(self, nca_state_size: int, gnn_components: int = 3):
        super().__init__()

        self.nca_state_size = nca_state_size
        self.gnn_components = gnn_components  # attention, messages, update

        # NCA состояние → модулирующие коэффициенты для GNN
        self.modulation_network = nn.Sequential(
            nn.Linear(nca_state_size, nca_state_size * 2, bias=True),
            nn.Tanh(),  # ограничиваем модуляцию
            nn.Linear(nca_state_size * 2, gnn_components, bias=True),
            nn.Sigmoid(),  # модуляция в диапазоне [0, 1]
        )

        # Инициализация для нейтральной модуляции (0.5)
        with torch.no_grad():
            self.modulation_network[-2].bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, nca_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Генерация модулирующих коэффициентов из NCA состояния

        Args:
            nca_state: [batch, nca_state_size] - состояние NCA

        Returns:
            modulation: dict с коэффициентами для разных GNN компонентов
        """
        # Получаем модулирующие коэффициенты
        modulation_coeffs = self.modulation_network(
            nca_state
        )  # [batch, gnn_components]

        return {
            "attention_modulation": modulation_coeffs[:, 0:1],  # [batch, 1]
            "message_modulation": modulation_coeffs[:, 1:2],  # [batch, 1]
            "update_modulation": modulation_coeffs[:, 2:3],  # [batch, 1]
        }


class ModulatedGNNCell(GNNCell):
    """
    GNN клетка с модуляцией от NCA

    Расширяет базовую GNN клетку возможностью модуляции от NCA:
    - Модулирует attention weights
    - Модулирует эффективность message passing
    - Модулирует интенсивность state update
    """

    def __init__(self, *args, **kwargs):
        """Инициализация с вызовом родительского конструктора"""
        super().__init__(*args, **kwargs)

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        nca_modulation: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Модулированный GNN forward pass

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size]
            nca_modulation: dict с модулирующими коэффициентами от NCA

        Returns:
            new_state: [batch, state_size] - модулированное новое состояние
        """
        batch_size = own_state.shape[0]
        num_neighbors = neighbor_states.shape[1]

        # Обработка external input
        if external_input is not None:
            ext_input = external_input
        else:
            ext_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # === STEP 1: MESSAGE CREATION ===
        # Создаем сообщения от соседей к текущей клетке
        messages = []
        for i in range(self.neighbor_count):
            neighbor_state = neighbor_states[:, i, :]  # [batch, state_size]
            message = self.message_network(neighbor_state, own_state)
            messages.append(message)

        messages = torch.stack(messages, dim=1)  # [batch, neighbor_count, message_dim]

        # МОДУЛЯЦИЯ: влияние NCA на эффективность сообщений
        if nca_modulation and "message_modulation" in nca_modulation:
            message_mod = nca_modulation["message_modulation"]  # [batch, 1]
            message_mod = message_mod.unsqueeze(1)  # [batch, 1, 1]
            messages = messages * message_mod  # модулируем интенсивность сообщений

            # === STEP 2: ATTENTION AGGREGATION ===
        if self.use_attention:
            aggregated_message = self.aggregator(messages, own_state)

            # МОДУЛЯЦИЯ: влияние NCA на attention mechanism
            if nca_modulation and "attention_modulation" in nca_modulation:
                attention_mod = nca_modulation["attention_modulation"]  # [batch, 1]
                # Модулируем силу attention (ближе к 0.5 = меньше селективность)
                attention_strength = (attention_mod - 0.5) * 2  # [-1, 1]
                aggregated_message = aggregated_message * (
                    1.0 + attention_strength * 0.3
                )
        else:
            # Простая агрегация через среднее
            aggregated_message = torch.mean(messages, dim=1)

        # === STEP 3: STATE UPDATE ===
        new_state = self.state_updater(own_state, aggregated_message, ext_input)

        # МОДУЛЯЦИЯ: влияние NCA на интенсивность обновления состояния
        if nca_modulation and "update_modulation" in nca_modulation:
            update_mod = nca_modulation["update_modulation"]  # [batch, 1]
            # Интерполяция между старым и новым состоянием
            new_state = own_state * (1 - update_mod) + new_state * update_mod

        return new_state


class HybridCellV2(BaseCell):
    """
    Биологически правдоподобная гибридная клетка V2

    АРХИТЕКТУРА:
    1. NCA компонент (4D) - внутриклеточная динамика
    2. GNN компонент (32D) - межклеточная коммуникация
    3. NCAModulator - модуляция GNN через NCA сигналы
    4. Взвешенное объединение результатов

    БИОЛОГИЧЕСКОЕ ОБОСНОВАНИЕ:
    - NCA = биохимические процессы внутри нейрона
    - GNN = электрическая активность синапсов
    - Модуляция = влияние внутриклеточных процессов на синаптическую эффективность
    - Нет потери информации через преобразование размерностей
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        external_input_size: Optional[int] = None,
        activation: Optional[str] = None,
        nca_weight: Optional[float] = None,
        gnn_weight: Optional[float] = None,
        **kwargs,
    ):
        """
        HybridCell V2 с модуляцией вместо преобразования размерностей
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_project_config()
        nca_config = config.get_nca_config()
        gnn_config = config.get_gnn_config()

        # Основные параметры из GNN (доминирующая архитектура)
        self.state_size = state_size or gnn_config["state_size"]
        self.neighbor_count = neighbor_count or gnn_config["neighbor_count"]
        self.external_input_size = (
            external_input_size or gnn_config["external_input_size"]
        )

        # Веса влияния компонентов
        self.nca_weight = nca_weight or config.hybrid_nca_weight
        self.gnn_weight = gnn_weight or config.hybrid_gnn_weight

        # Нормализация весов
        total_weight = self.nca_weight + self.gnn_weight
        self.nca_weight = self.nca_weight / total_weight
        self.gnn_weight = self.gnn_weight / total_weight

        # Размеры состояний
        self.nca_state_size = nca_config["state_size"]
        self.gnn_state_size = gnn_config["state_size"]

        # === КОМПОНЕНТЫ АРХИТЕКТУРЫ ===

        # 1. NCA компонент - внутриклеточная динамика (4D)
        self.nca_cell = NCACell(
            state_size=self.nca_state_size,
            neighbor_count=self.neighbor_count,
            hidden_dim=nca_config["hidden_dim"],
            external_input_size=nca_config["external_input_size"],
            activation=nca_config["activation"],
            target_params=nca_config["target_params"],
        )

        # 2. NCA модулятор - влияние на GNN
        self.nca_modulator = NCAModulator(
            nca_state_size=self.nca_state_size,
            gnn_components=3,  # attention, messages, update
        )

        # 3. Модулированная GNN клетка - межклеточная коммуникация (32D)
        self.gnn_cell = ModulatedGNNCell(
            state_size=self.gnn_state_size,
            neighbor_count=self.neighbor_count,
            message_dim=gnn_config["message_dim"],
            hidden_dim=gnn_config["hidden_dim"],
            external_input_size=gnn_config["external_input_size"],
            activation=gnn_config["activation"],
            target_params=gnn_config["target_params"],
            use_attention=gnn_config["use_attention"],
        )

        # 4. Проекция NCA в GNN пространство для финального объединения
        # Это единственное место где нужно преобразование размерности
        self.nca_to_gnn_projection = nn.Linear(
            self.nca_state_size, self.gnn_state_size, bias=True
        )

        # Инициализация с малыми весами для стабильности
        nn.init.xavier_uniform_(self.nca_to_gnn_projection.weight, gain=0.1)
        nn.init.zeros_(self.nca_to_gnn_projection.bias)

        # 5. Целевые параметры
        self.target_params = nca_config["target_params"] + gnn_config["target_params"]

        # Логирование
        if config.debug_mode:
            self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование параметров улучшенной гибридной архитектуры"""
        total_params = sum(p.numel() for p in self.parameters())
        nca_params = sum(p.numel() for p in self.nca_cell.parameters())
        gnn_params = sum(p.numel() for p in self.gnn_cell.parameters())
        modulator_params = sum(p.numel() for p in self.nca_modulator.parameters())
        projection_params = sum(
            p.numel() for p in self.nca_to_gnn_projection.parameters()
        )

        log_cell_init(
            cell_type="HYBRID_V2",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            neighbor_count=self.neighbor_count,
            external_input_size=self.external_input_size,
            additional_info={
                "nca_params": nca_params,
                "gnn_params": gnn_params,
                "modulator_params": modulator_params,
                "projection_params": projection_params,
                "nca_weight": self.nca_weight,
                "gnn_weight": self.gnn_weight,
                "nca_state_size": self.nca_state_size,
                "gnn_state_size": self.gnn_state_size,
            },
        )

        # Детализация
        config = get_project_config()
        if config.debug_mode:
            component_params = {
                "nca_component": nca_params,
                "gnn_component": gnn_params,
                "nca_modulator": modulator_params,
                "nca_projection": projection_params,
            }
            log_cell_component_params(component_params, total_params)

    def _prepare_nca_inputs(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Подготовка входов для NCA (работает в своем 4D пространстве)

        NCA получает упрощенную проекцию GNN состояний для локальной динамики
        """
        # Простая проекция: берем первые 4 измерения из GNN состояний
        nca_own_state = own_state[:, : self.nca_state_size]

        # Аналогично для соседей
        batch_size, num_neighbors, _ = neighbor_states.shape
        nca_neighbor_states = neighbor_states[:, :, : self.nca_state_size]

        # External input для NCA
        nca_external_input = None
        if external_input is not None:
            nca_external_input = external_input[:, : self.nca_cell.external_input_size]

        return nca_neighbor_states, nca_own_state, nca_external_input

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Биологически правдоподобный hybrid forward pass

        1. NCA обновляет внутриклеточное состояние (4D)
        2. NCA модулирует эффективность GNN операций
        3. GNN выполняет модулированную межклеточную коммуникацию (32D)
        4. Результаты объединяются через взвешенную сумму
        """
        batch_size = own_state.shape[0]

        # Логирование
        config = get_project_config()
        if config.debug_mode:
            input_shapes = {
                "neighbor_states": neighbor_states.shape,
                "own_state": own_state.shape,
            }
            if external_input is not None:
                input_shapes["external_input"] = external_input.shape
            log_cell_forward("HYBRID_V2", input_shapes)

        # === STEP 1: NCA КОМПОНЕНТ (внутриклеточная динамика) ===

        # Подготовка входов для NCA
        nca_neighbor_states, nca_own_state, nca_external_input = (
            self._prepare_nca_inputs(neighbor_states, own_state, external_input)
        )

        # NCA forward pass (4D состояние)
        nca_new_state = self.nca_cell(
            neighbor_states=nca_neighbor_states,
            own_state=nca_own_state,
            external_input=nca_external_input,
        )

        # === STEP 2: NCA МОДУЛЯЦИЯ GNN ===

        # Генерируем модулирующие сигналы из NCA состояния
        nca_modulation = self.nca_modulator(nca_new_state)

        # === STEP 3: МОДУЛИРОВАННЫЙ GNN КОМПОНЕНТ ===

        # GNN forward pass с модуляцией от NCA
        gnn_new_state = self.gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=nca_modulation,  # КЛЮЧЕВОЕ ОТЛИЧИЕ
            **kwargs,
        )

        # === STEP 4: ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ ===

        # Проецируем NCA результат в GNN пространство для объединения
        nca_projected = self.nca_to_gnn_projection(nca_new_state)

        # Взвешенное объединение
        hybrid_new_state = (
            self.gnn_weight * gnn_new_state + self.nca_weight * nca_projected
        )

        return hybrid_new_state

    def get_component_analysis(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Детальный анализ работы компонентов (без потери информации)
        """
        # NCA компонент
        nca_neighbor_states, nca_own_state, nca_external_input = (
            self._prepare_nca_inputs(neighbor_states, own_state, external_input)
        )

        nca_new_state = self.nca_cell(
            neighbor_states=nca_neighbor_states,
            own_state=nca_own_state,
            external_input=nca_external_input,
        )

        # NCA модуляция
        nca_modulation = self.nca_modulator(nca_new_state)

        # GNN без модуляции (для сравнения)
        gnn_unmodulated = self.gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=None,
        )

        # GNN с модуляцией
        gnn_modulated = self.gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=nca_modulation,
        )

        # Финальное состояние
        nca_projected = self.nca_to_gnn_projection(nca_new_state)
        hybrid_final = self.gnn_weight * gnn_modulated + self.nca_weight * nca_projected

        return {
            "nca_state": nca_new_state,  # 4D внутриклеточное состояние
            "nca_modulation": nca_modulation,  # модулирующие сигналы
            "nca_projected": nca_projected,  # NCA проецированное в GNN пространство
            "gnn_unmodulated": gnn_unmodulated,  # GNN без модуляции
            "gnn_modulated": gnn_modulated,  # GNN с NCA модуляцией
            "hybrid_final": hybrid_final,  # итоговое состояние
            "modulation_effect": torch.mean(torch.abs(gnn_modulated - gnn_unmodulated)),
            "nca_weight": self.nca_weight,
            "gnn_weight": self.gnn_weight,
        }

    def reset_memory(self):
        """Сброс памяти в компонентах"""
        self.nca_cell.reset_memory()
        self.gnn_cell.reset_memory()

    def get_info(self) -> Dict[str, Any]:
        """Информация о HybridCell V2"""
        base_info = super().get_info()

        nca_params = sum(p.numel() for p in self.nca_cell.parameters())
        gnn_params = sum(p.numel() for p in self.gnn_cell.parameters())
        modulator_params = sum(p.numel() for p in self.nca_modulator.parameters())
        projection_params = sum(
            p.numel() for p in self.nca_to_gnn_projection.parameters()
        )

        base_info.update(
            {
                "architecture": "hybrid_v2",
                "nca_params": nca_params,
                "gnn_params": gnn_params,
                "modulator_params": modulator_params,
                "projection_params": projection_params,
                "nca_weight": self.nca_weight,
                "gnn_weight": self.gnn_weight,
                "nca_state_size": self.nca_state_size,
                "gnn_state_size": self.gnn_state_size,
                "biological_accuracy": "high",  # без потери информации
            }
        )

        return base_info
