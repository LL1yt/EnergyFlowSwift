#!/usr/bin/env python3
"""
MoE Processor - упрощенный Mixture of Experts процессор
=====================================================

Основной MoE процессор, использующий модульную архитектуру
для эффективной обработки связей в 3D нейронной решетке.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.checkpoint import checkpoint

from .gating_network import GatingNetwork
from .connection_classifier import UnifiedConnectionClassifier
from .connection_types import ConnectionCategory
from .simple_linear_expert import SimpleLinearExpert
from .hybrid_gnn_cnf_expert import HybridGNN_CNF_Expert
from ..cnf.gpu_enhanced_cnf import GPUEnhancedCNF, ConnectionType
from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_forward
from ...utils.device_manager import get_device_manager
from ..lattice.position import Position3D
from ..lattice.spatial_optimization.memory_manager import get_memory_pool_manager

logger = get_logger(__name__)


class MoEConnectionProcessor(nn.Module):
    """
    Упрощенный Mixture of Experts Connection Processor для 3D решетки 27×27×27

    ЭКСПЕРТЫ:
    - local_expert: SimpleLinear (2059 params) - 10% связей
    - functional_expert: HybridGNN_CNF (5500-12233 params) - 55% связей
    - distant_expert: LightweightCNF (1500-4000 params) - 35% связей

    УПРАВЛЕНИЕ:
    - gating_network: (808 params) - адаптивное взвешивание
    - connection_classifier: классификация связей по типам
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        lattice_dimensions: Optional[Tuple[int, int, int]] = None,
        neighbor_count: Optional[int] = None,
        enable_cnf: Optional[bool] = None,
        config: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = get_project_config()

        # === DEVICE MANAGEMENT ===
        self.device_manager = config.device_manager
        self.device = self.device_manager.get_device()
        self.memory_pool_manager = get_memory_pool_manager()

        # === ЦЕНТРАЛИЗОВАННАЯ КОНФИГУРАЦИЯ ===
        # Строгая проверка state_size
        if state_size is None:
            if not hasattr(config.model, 'state_size') or config.model.state_size is None:
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательный параметр config.model.state_size. "
                    "Проверьте конфигурацию в project_config.py"
                )
            self.state_size = config.model.state_size
        else:
            self.state_size = state_size

        # Строгая проверка lattice_dimensions
        if lattice_dimensions is None:
            if not hasattr(config.lattice, 'dimensions') or config.lattice.dimensions is None:
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательный параметр config.lattice.dimensions. "
                    "Проверьте конфигурацию в project_config.py"
                )
            self.lattice_dimensions = config.lattice.dimensions
        else:
            self.lattice_dimensions = lattice_dimensions

        self.adaptive_radius = config.calculate_adaptive_radius()
        
        # Получаем max_neighbors из NeighborSettings
        if not hasattr(config, "neighbors") or not hasattr(config.neighbors, 'max_neighbors'):
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательный параметр config.neighbors.max_neighbors. "
                "Проверьте конфигурацию в project_config.py"
            )
        self.max_neighbors = config.neighbors.max_neighbors
        
        # neighbor_count из ModelSettings используется для динамического определения
        if hasattr(config.model, 'neighbor_count'):
            self.dynamic_neighbors = (config.model.neighbor_count == -1)
        else:
            self.dynamic_neighbors = True
            
        logger.debug(f"[MoEConnectionProcessor] max_neighbors={self.max_neighbors}, dynamic_neighbors={self.dynamic_neighbors}")

        # Строгая проверка enable_cnf
        if enable_cnf is not None:
            self.enable_cnf = enable_cnf
        else:
            if not hasattr(config.cnf, 'enabled'):
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательный параметр config.cnf.enabled. "
                    "Проверьте конфигурацию в project_config.py"
                )
            self.enable_cnf = config.cnf.enabled

        # Конфигурация распределения связей: 10%/55%/35%
        # СТРОГАЯ ПРОВЕРКА - БЕЗ FALLBACK
        if not hasattr(config, "neighbors") or config.neighbors is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует конфигурация config.neighbors. "
                "Проверьте настройки в project_config.py. "
                "Необходимо определить neighbors с полями local_tier, functional_tier, distant_tier"
            )
        
        # Проверяем обязательные поля neighbors
        required_neighbor_fields = ['local_tier', 'functional_tier', 'distant_tier']
        for field in required_neighbor_fields:
            if not hasattr(config.neighbors, field) or getattr(config.neighbors, field) is None:
                raise RuntimeError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательное поле config.neighbors.{field}. "
                    f"Проверьте конфигурацию neighbors в project_config.py"
                )
        
        self.connection_ratios = {
            "local": config.neighbors.local_tier,
            "functional": config.neighbors.functional_tier,
            "distant": config.neighbors.distant_tier,
        }

        # === КЛАССИФИКАТОР СВЯЗЕЙ С КЭШИРОВАНИЕМ ===
        self.connection_classifier = UnifiedConnectionClassifier(
            lattice_dimensions=self.lattice_dimensions,
            enable_cache=True,  # Включаем кэш для исправления индексации
        )
        
        # Spatial optimizer будет установлен позже через setter
        # self.spatial_optimizer = None

        # === ЭКСПЕРТЫ ===
        self.local_expert = SimpleLinearExpert(state_size=self.state_size)

        # СТРОГАЯ ПРОВЕРКА параметров экспертов - БЕЗ FALLBACK
        if not hasattr(config, "expert") or config.expert is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует конфигурация config.expert. "
                "Проверьте настройки экспертов в project_config.py"
            )
        
        # Проверка functional expert
        if not hasattr(config.expert, "functional") or config.expert.functional is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.expert.functional. "
                "Настройте параметры functional expert в конфигурации"
            )
        
        if not hasattr(config.expert.functional, "params") or config.expert.functional.params is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.expert.functional.params. "
                "Укажите количество параметров для functional expert"
            )
        
        functional_params = config.expert.functional.params
        
        # Проверка distant expert
        if not hasattr(config.expert, "distant") or config.expert.distant is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.expert.distant. "
                "Настройте параметры distant expert в конфигурации"
            )
        
        if not hasattr(config.expert.distant, "params") or config.expert.distant.params is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.expert.distant.params. "
                "Укажите количество параметров для distant expert"
            )
        
        distant_params = config.expert.distant.params

        # Для функционального эксперта используем динамическое определение соседей
        self.functional_expert = HybridGNN_CNF_Expert(
            state_size=self.state_size,
            neighbor_count=-1 if self.dynamic_neighbors else self.max_neighbors,  # -1 означает динамическое определение
            target_params=functional_params,
            cnf_params=distant_params,
        )
        logger.info(f"[MoEConnectionProcessor] Functional expert создан с neighbor_count={-1 if self.dynamic_neighbors else self.max_neighbors}")

        # 3. Distant Expert - долгосрочная память (LightweightCNF)
        # СТРОГАЯ ПРОВЕРКА - БЕЗ FALLBACK для CNF
        if not self.enable_cnf:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: CNF отключен (config.cnf.enabled=False), "
                "но MoE архитектура требует CNF для distant_expert. "
                "Включите CNF в конфигурации: config.cnf.enabled = True"
            )
        
        # Проверяем необходимые параметры CNF
        cnf_required_fields = ['integration_steps', 'batch_processing_mode', 'max_batch_size', 'adaptive_method']
        for field in cnf_required_fields:
            if not hasattr(config.cnf, field) or getattr(config.cnf, field) is None:
                raise RuntimeError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует обязательный параметр config.cnf.{field}. "
                    f"Проверьте конфигурацию CNF в project_config.py"
                )
        
        self.distant_expert = GPUEnhancedCNF(
            state_size=self.state_size,
            connection_type=ConnectionType.DISTANT,
            integration_steps=config.cnf.integration_steps,
            batch_processing_mode=config.cnf.batch_processing_mode,
            max_batch_size=config.cnf.max_batch_size,
            adaptive_method=config.cnf.adaptive_method,
        )

        # === GATING NETWORK ===
        self.gating_network = GatingNetwork(state_size=self.state_size, num_experts=3)

        # === СТАТИСТИКА ===
        self.reset_stats()

        # Подсчет общих параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MoEConnectionProcessor: {total_params} параметров всего")

        # Перенос модели на правильное устройство
        self.device_manager.transfer_module(self)
        logger.info(f"MoEConnectionProcessor перенесен на устройство: {self.device}")
        
    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,  # DEPRECATED - будет получено из кэша
        cell_idx: int,
        neighbor_indices: List[int],  # DEPRECATED - будет получено из кэша
        external_input: Optional[torch.Tensor] = None,
        spatial_optimizer=None,  # DEPRECATED - больше не используется
        **kwargs,
    ) -> Dict[str, Any]:
        # DEBUG: Only log for extreme debug mode
        if logger.isEnabledFor(11):  # DEBUG_VERBOSE only
            logger.debug_verbose(f"🔍 MoE FORWARD called for cell {cell_idx}")
        """
        Основной forward pass с упрощенной логикой (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)

        Args:
            current_state: [state_size] - текущее состояние клетки
            neighbor_states: DEPRECATED - получается из кэша
            cell_idx: индекс текущей клетки
            neighbor_indices: DEPRECATED - получается из кэша
            external_input: внешний вход (опционально)
            spatial_optimizer: DEPRECATED - не используется
            **kwargs: должен содержать full_lattice_states

        Returns:
            Dict с результатами обработки
        """
        # === НОВАЯ АРХИТЕКТУРА: Используем только кэш ===
        if "full_lattice_states" not in kwargs:
            raise RuntimeError(
                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Для клетки {cell_idx} отсутствует full_lattice_states. "
                f"Этот параметр обязателен для новой архитектуры."
            )
            
        full_states = kwargs["full_lattice_states"]
        
        # Получаем соседей И классификацию одним вызовом из кэша
        neighbors_data = self.connection_classifier.get_cached_neighbors_and_classification(
            cell_idx=cell_idx,
            states=full_states
        )
        
        # Проверяем, что у нас есть хотя бы один сосед
        total_neighbors = (
            len(neighbors_data["local"]["indices"]) + 
            len(neighbors_data["functional"]["indices"]) + 
            len(neighbors_data["distant"]["indices"])
        )
        
        if total_neighbors == 0:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Клетка {cell_idx} не имеет соседей для обработки!")
            logger.error(f"   Это невозможно в 3D решетке - проверьте радиус поиска!")
            raise RuntimeError(
                f"Клетка {cell_idx} изолирована (0 соседей). "
                f"Проверьте конфигурацию адаптивного радиуса."
            )

        batch_size = 1
        device = current_state.device

        # Убеждаемся что все tensor'ы на правильном устройстве
        current_state = self.device_manager.ensure_device(current_state)
        if external_input is not None:
            external_input = self.device_manager.ensure_device(external_input)
            
        # Проверяем размерность current_state и приводим к правильной форме
        if current_state.dim() == 1:
            # Если одномерный [state_size], добавляем batch dimension
            current_state = current_state.unsqueeze(0)  # [1, state_size]
        elif current_state.dim() == 3:
            # Если трехмерный [batch, 1, state_size], убираем лишнее измерение
            current_state = current_state.squeeze(1)  # [batch, state_size]
            if current_state.shape[0] > 1:
                # Берем только первую клетку из батча
                current_state = current_state[0:1]  # [1, state_size]
        
        # Cache-based classification results (no logging for performance)

        # === 2. ОБРАБОТКА КАЖДЫМ ЭКСПЕРТОМ (ПАРАЛЛЕЛЬНАЯ АРХИТЕКТУРА) ===
        expert_outputs = []
        tensors_to_return = []

        # Подготавливаем данные для параллельной обработки
        local_data = neighbors_data["local"]
        functional_data = neighbors_data["functional"]
        distant_data = neighbors_data["distant"]

        # Для небольших решеток (≤32) используем параллельную обработку экспертов
        if self.device_manager.is_cuda() and max(current_state.shape) <= 1024:  # Оптимизация для GPU
            # Параллельная обработка с использованием CUDA streams
            with torch.cuda.device(self.device_manager.device):
                # Создаем отдельные stream'ы для каждого эксперта
                local_stream = torch.cuda.Stream()
                functional_stream = torch.cuda.Stream()
                distant_stream = torch.cuda.Stream() if self.enable_cnf else None

                # Async обработка экспертов
                with torch.cuda.stream(local_stream):
                    if local_data["indices"]:
                        local_neighbor_states = local_data["states"]
                        # Добавляем batch dimension к neighbor_states если нужно
                        if local_neighbor_states.dim() == 2:
                            local_neighbor_states = local_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                        def local_expert_wrapper(current, neighbors):
                            res = self.local_expert(current, neighbors)
                            if isinstance(res, dict):
                                return res.get("output", res.get("new_state", current))
                            return res
                        local_output = checkpoint(
                            local_expert_wrapper,
                            current_state,
                            local_neighbor_states,
                            use_reentrant=False,
                        )
                    else:
                        local_output = self.memory_pool_manager.get_tensor(
                            (1, self.state_size), dtype=current_state.dtype
                        )
                        tensors_to_return.append(local_output)

                with torch.cuda.stream(functional_stream):
                    if functional_data["indices"]:
                        functional_neighbor_states = functional_data["states"]
                        # Добавляем batch dimension к neighbor_states если нужно
                        if functional_neighbor_states.dim() == 2:
                            functional_neighbor_states = functional_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                        def functional_expert_wrapper(current, neighbors):
                            res = self.functional_expert(current, neighbors)
                            if isinstance(res, dict):
                                return res.get("output", res.get("new_state", current))
                            return res
                        # Отключаем checkpoint для functional expert из-за in-place operations
                        # functional_output = checkpoint(
                        #     functional_expert_wrapper,
                        #     current_state,
                        #     functional_neighbor_states,
                        #     use_reentrant=False,
                        # )
                        functional_output = functional_expert_wrapper(
                            current_state,
                            functional_neighbor_states
                        )
                    else:
                        functional_output = self.memory_pool_manager.get_tensor(
                            (1, self.state_size), dtype=current_state.dtype
                        )
                        tensors_to_return.append(functional_output)

                if self.enable_cnf and distant_stream:
                    with torch.cuda.stream(distant_stream):
                        if distant_data["indices"]:
                            distant_neighbor_states = distant_data["states"]
                            # Добавляем batch dimension к neighbor_states если нужно
                            if distant_neighbor_states.dim() == 2:
                                distant_neighbor_states = distant_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                            def distant_expert_wrapper(current, neighbors):
                                res = self.distant_expert(current, neighbors)
                                if isinstance(res, dict):
                                    return res.get("output", res.get("new_state", current))
                                return res
                            distant_output = checkpoint(
                                distant_expert_wrapper,
                                current_state,
                                distant_neighbor_states,
                                use_reentrant=False,
                            )
                        else:
                            distant_output = self.memory_pool_manager.get_tensor(
                                (1, self.state_size), dtype=current_state.dtype
                            )
                            tensors_to_return.append(distant_output)

                # Убираем блокирующие синхронизации - PyTorch автоматически управляет stream'ами
                # local_stream.synchronize()      # Отключено для GPU оптимизации
                # functional_stream.synchronize() # Отключено для GPU оптимизации
                # if distant_stream:
                #     distant_stream.synchronize() # Отключено для GPU оптимизации

        else:
            # Fallback к последовательной обработке для больших решеток
            # Local Expert
            if local_data["indices"]:
                local_neighbor_states = local_data["states"]
                # Добавляем batch dimension к neighbor_states если нужно
                if local_neighbor_states.dim() == 2:
                    local_neighbor_states = local_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                def local_expert_wrapper(current, neighbors):
                    res = self.local_expert(current, neighbors)
                    if isinstance(res, dict):
                        return res.get("output", res.get("new_state", current))
                    return res
                local_output = checkpoint(
                    local_expert_wrapper,
                    current_state,
                    local_neighbor_states,
                    use_reentrant=False,
                )
            else:
                local_output = self.memory_pool_manager.get_tensor(
                    (1, self.state_size), dtype=current_state.dtype
                )
                tensors_to_return.append(local_output)

            # Functional Expert
            if functional_data["indices"]:
                functional_neighbor_states = functional_data["states"]
                # Добавляем batch dimension к neighbor_states если нужно
                if functional_neighbor_states.dim() == 2:
                    functional_neighbor_states = functional_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                def functional_expert_wrapper(current, neighbors):
                    res = self.functional_expert(current, neighbors)
                    if isinstance(res, dict):
                        return res.get("output", res.get("new_state", current))
                    return res
                # Отключаем checkpoint для functional expert из-за in-place operations
                # functional_output = checkpoint(
                #     functional_expert_wrapper,
                #     current_state,
                #     functional_neighbor_states,
                #     use_reentrant=False,
                # )
                functional_output = functional_expert_wrapper(
                    current_state,
                    functional_neighbor_states
                )
            else:
                functional_output = self.memory_pool_manager.get_tensor(
                    (1, self.state_size), dtype=current_state.dtype
                )
                tensors_to_return.append(functional_output)

            # Distant Expert (только если CNF включен)
            if self.enable_cnf and distant_data["indices"]:
                distant_neighbor_states = distant_data["states"]
                # Добавляем batch dimension к neighbor_states если нужно
                if distant_neighbor_states.dim() == 2:
                    distant_neighbor_states = distant_neighbor_states.unsqueeze(0)  # [1, num_neighbors, state_size]
                def distant_expert_wrapper(current, neighbors):
                    res = self.distant_expert(current, neighbors)
                    if isinstance(res, dict):
                        return res.get("output", res.get("new_state", current))
                    return res
                distant_output = checkpoint(
                    distant_expert_wrapper,
                    current_state,
                    distant_neighbor_states,
                    use_reentrant=False,
                )
            else:
                distant_output = self.memory_pool_manager.get_tensor(
                    (1, self.state_size), dtype=current_state.dtype
                )
                tensors_to_return.append(distant_output)

        # Добавляем результаты в expert_outputs
        expert_outputs.append(local_output.squeeze(0))
        expert_outputs.append(functional_output.squeeze(0))
        if self.enable_cnf:
            expert_outputs.append(distant_output.squeeze(0))

        # === 3. КОМБИНИРОВАНИЕ РЕЗУЛЬТАТОВ ===
        try:
            # Предотвращение ошибки с пустыми expert_outputs
            if not expert_outputs:
                logger.warning(
                    f"⚠️ Нет выходов экспертов для клетки {cell_idx}, пропуск GatingNetwork."
                )
                final_state = current_state
                expert_weights = torch.zeros(
                    1, 3, device=device
                )  # Возвращаем нулевые веса
            else:
                # --- ИСПРАВЛЕНИЕ: Агрегация состояний соседей ---
                # Собираем все состояния соседей из всех категорий
                all_neighbor_states = []
                for category in ["local", "functional", "distant"]:
                    if neighbors_data[category]["states"].numel() > 0:
                        all_neighbor_states.append(neighbors_data[category]["states"])
                
                if all_neighbor_states:
                    # Объединяем все состояния соседей
                    combined_neighbor_states = torch.cat(all_neighbor_states, dim=0)
                    neighbor_activity = torch.mean(combined_neighbor_states, dim=0, keepdim=True)
                else:
                    # Если нет соседей, используем нулевой вектор
                    neighbor_activity = torch.zeros(
                        1, self.state_size, device=device, dtype=current_state.dtype
                    )
                # neighbor_activity computed

                # Вызов GatingNetwork с корректными по форме тензорами
                logger.debug_forward(f"[{cell_idx}] Вызов GatingNetwork...")
                combined_output, expert_weights = self.gating_network(
                    current_state=current_state,  # [1, state_size]
                    neighbor_activity=neighbor_activity,  # [1, state_size]
                    expert_outputs=expert_outputs,
                )
                logger.debug_forward(
                    f"[{cell_idx}] GatingNetwork завершен. combined_output: {combined_output.shape}, expert_weights: {expert_weights.shape}"
                )

                # Residual connection
                final_state = current_state + combined_output.squeeze(0)

        except Exception as e:
            logger.error(
                f"❌ MoE processor CRITICAL error on cell {cell_idx}: {e}",
                exc_info=True,
            )
            # В случае ошибки возвращаем исходное состояние, чтобы не прерывать процесс
            final_state = current_state
            expert_weights = torch.zeros(1, 3, device=device)

        # Освобождаем временные тензоры
        for t in tensors_to_return:
            self.memory_pool_manager.return_tensor(t)

        # === 4. ОБНОВЛЕНИЕ СТАТИСТИКИ ===
        # Преобразуем neighbors_data в формат classifications для совместимости
        classifications = {
            ConnectionCategory.LOCAL: [{"target_idx": idx} for idx in neighbors_data["local"]["indices"]],
            ConnectionCategory.FUNCTIONAL: [{"target_idx": idx} for idx in neighbors_data["functional"]["indices"]],
            ConnectionCategory.DISTANT: [{"target_idx": idx} for idx in neighbors_data["distant"]["indices"]]
        }
        self._update_stats(classifications, expert_weights)

        log_cell_forward(
            "MoEConnectionProcessor",
            input_shapes={
                "current_state": current_state.shape,
                "total_neighbors": total_neighbors,
            },
            output_shape=final_state.shape,
        )

        # Отдельное логирование expert_weights
        logger.debug_training(
            f"[{cell_idx}] Expert weights: {expert_weights.squeeze().tolist()}"
        )

        return {
            "new_state": final_state,
            "expert_weights": expert_weights,
            "classifications": classifications,
        }

    def _empty_forward_result(self, current_state: torch.Tensor) -> Dict[str, Any]:
        """Результат для случая без соседей"""
        return {
            "new_state": current_state,
            "expert_weights": torch.tensor(
                [1.0, 0.0, 0.0], device=current_state.device
            ),
            "classifications": {cat: [] for cat in ConnectionCategory},
            "expert_outputs": [
                current_state,
                torch.zeros_like(current_state),
                torch.zeros_like(current_state),
            ],
            "neighbor_count": 0,
        }

    def _update_stats(self, classifications: Dict, expert_weights: torch.Tensor):
        """Обновление статистики использования"""
        local_count = len(classifications[ConnectionCategory.LOCAL])
        functional_count = len(classifications[ConnectionCategory.FUNCTIONAL])
        distant_count = len(classifications[ConnectionCategory.DISTANT])

        self.usage_stats["local_connections"] += local_count
        self.usage_stats["functional_connections"] += functional_count
        self.usage_stats["distant_connections"] += distant_count
        self.usage_stats["total_forward_calls"] += 1

        # Статистика весов экспертов
        weights = expert_weights.detach()
        # Обрабатываем размерности weights - может быть [3] или [1, 3]
        if weights.dim() == 2:
            weights = weights.squeeze(0)  # [1, 3] -> [3]

        # Statistics collection disabled for performance in production
        # Debug mode check removed - stats always disabled for performance

    def get_usage_stats(self) -> Dict[str, Any]:
        """Получить статистику использования"""
        total_connections = max(
            1,
            self.usage_stats["local_connections"]
            + self.usage_stats["functional_connections"]
            + self.usage_stats["distant_connections"],
        )
        total_calls = max(1, self.usage_stats["total_forward_calls"])

        return {
            "connection_distribution": {
                "local_ratio": self.usage_stats["local_connections"]
                / total_connections,
                "functional_ratio": self.usage_stats["functional_connections"]
                / total_connections,
                "distant_ratio": self.usage_stats["distant_connections"]
                / total_connections,
            },
            "expert_usage": {
                "local_avg_weight": self.usage_stats["expert_weights"]["local"]
                / total_calls,
                "functional_avg_weight": self.usage_stats["expert_weights"][
                    "functional"
                ]
                / total_calls,
                "distant_avg_weight": self.usage_stats["expert_weights"]["distant"]
                / total_calls,
            },
            "total_forward_calls": total_calls,
            "total_connections": total_connections,
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.usage_stats = {
            "local_connections": 0,
            "functional_connections": 0,
            "distant_connections": 0,
            "total_forward_calls": 0,
            "expert_weights": {"local": 0.0, "functional": 0.0, "distant": 0.0},
        }

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Получить разбивку параметров по компонентам"""
        return {
            "local_expert": sum(p.numel() for p in self.local_expert.parameters()),
            "functional_expert": sum(
                p.numel() for p in self.functional_expert.parameters()
            ),
            "distant_expert": sum(p.numel() for p in self.distant_expert.parameters()),
            "gating_network": sum(p.numel() for p in self.gating_network.parameters()),
            "connection_classifier": sum(
                p.numel() for p in self.connection_classifier.parameters()
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Упрощенный forward для обработки эмбедингов без решетки

        Args:
            embeddings: [batch_size, embedding_dim] - входные эмбединги

        Returns:
            processed_embeddings: [batch_size, embedding_dim] - обработанные эмбединги
        """
        batch_size, embedding_dim = embeddings.shape
        device = embeddings.device

        # Для простоты используем только functional expert
        # который может работать с произвольными состояниями
        processed_batch = []

        for i in range(batch_size):
            current_embedding = embeddings[i : i + 1]  # [1, embedding_dim]

            # Создаем "соседей" из других эмбедингов в batch'е
            neighbor_indices = [j for j in range(batch_size) if j != i]
            if len(neighbor_indices) > 0:
                neighbor_embeddings = embeddings[
                    neighbor_indices
                ]  # [batch_size-1, embedding_dim]
            else:
                neighbor_embeddings = torch.zeros(0, embedding_dim, device=device)

            # Используем functional expert для обработки
            if neighbor_embeddings.shape[0] > 0:
                try:
                    result = self.functional_expert(
                        current_state=current_embedding,
                        neighbor_states=neighbor_embeddings,
                        external_input=None,
                    )
                    if isinstance(result, dict):
                        processed_embedding = result.get("new_state", current_embedding)
                    else:
                        processed_embedding = result
                except Exception as e:
                    logger.warning(f"Functional expert failed for embedding {i}: {e}")
                    processed_embedding = current_embedding
            else:
                processed_embedding = current_embedding

            processed_batch.append(processed_embedding)

        return torch.cat(processed_batch, dim=0)
