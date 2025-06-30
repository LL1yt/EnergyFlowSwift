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
        self.state_size = state_size or config.model.state_size
        self.lattice_dimensions = lattice_dimensions or config.lattice.dimensions
        self.adaptive_radius = config.calculate_adaptive_radius()
        # Используем neighbor_count из model settings как max_neighbors
        self.max_neighbors = getattr(
            config, "max_neighbors", config.model.neighbor_count
        )
        self.enable_cnf = enable_cnf if enable_cnf is not None else config.cnf.enabled

        # Конфигурация распределения связей: 10%/55%/35%
        # Используем fallback значения если neighbors не определен
        if hasattr(config, "neighbors") and config.neighbors:
            self.connection_ratios = {
                "local": config.neighbors.local_tier,
                "functional": config.neighbors.functional_tier,
                "distant": config.neighbors.distant_tier,
            }
        else:
            # Fallback значения
            self.connection_ratios = {
                "local": 0.1,
                "functional": 0.55,
                "distant": 0.35,
            }

        # === КЛАССИФИКАТОР СВЯЗЕЙ С КЭШИРОВАНИЕМ ===
        self.connection_classifier = UnifiedConnectionClassifier(
            lattice_dimensions=self.lattice_dimensions,
            enable_cache=True,  # Включаем кэширование для оптимизации
        )

        # === ЭКСПЕРТЫ ===
        self.local_expert = SimpleLinearExpert(state_size=self.state_size)

        # Используем fallback значения для параметров экспертов
        functional_params = 8000  # Default value
        distant_params = 4000  # Default value

        if hasattr(config, "expert") and config.expert:
            if hasattr(config.expert, "functional") and config.expert.functional:
                functional_params = getattr(
                    config.expert.functional, "params", functional_params
                )
            if hasattr(config.expert, "distant") and config.expert.distant:
                distant_params = getattr(
                    config.expert.distant, "params", distant_params
                )

        self.functional_expert = HybridGNN_CNF_Expert(
            state_size=self.state_size,
            neighbor_count=self.max_neighbors,
            target_params=functional_params,
            cnf_params=distant_params,
        )

        # 3. Distant Expert - долгосрочная память (LightweightCNF)
        if self.enable_cnf:
            self.distant_expert = GPUEnhancedCNF(
                state_size=self.state_size,
                connection_type=ConnectionType.DISTANT,
                integration_steps=config.cnf.integration_steps,
                batch_processing_mode=config.cnf.batch_processing_mode,
                max_batch_size=config.cnf.max_batch_size,
                adaptive_method=config.cnf.adaptive_method,
            )
        else:
            # Fallback к простому linear если CNF отключен
            self.distant_expert = SimpleLinearExpert(state_size=self.state_size)

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
        neighbor_states: torch.Tensor,
        cell_idx: int,
        neighbor_indices: List[int],
        external_input: Optional[torch.Tensor] = None,
        spatial_optimizer=None,
        **kwargs,
    ) -> Dict[str, Any]:
        # DEBUG: Reduced logging - only log for specific problematic cells
        if cell_idx in [223, 256, 260, 320] or logger.isEnabledFor(10):
            logger.debug(f"[SEARCH] MoE FORWARD called for cell {cell_idx}")
            logger.debug(f"[SEARCH] current_state.shape={current_state.shape}")
            logger.debug(f"[SEARCH] neighbor_states.shape={neighbor_states.shape if neighbor_states is not None else 'None'}")
            # Safe logging for neighbor_indices (could be list or tensor)
            if neighbor_indices is not None:
                if isinstance(neighbor_indices, torch.Tensor):
                    logger.debug(f"[SEARCH] neighbor_indices=tensor({neighbor_indices.tolist()}), len={neighbor_indices.numel()}")
                else:
                    logger.debug(f"[SEARCH] neighbor_indices={neighbor_indices}, len={len(neighbor_indices)}")
            else:
                logger.debug("[SEARCH] neighbor_indices=None, len=0")
            logger.debug(f"[SEARCH] spatial_optimizer={spatial_optimizer is not None}")
            logger.debug(f"[SEARCH] kwargs keys={list(kwargs.keys())}")
            if 'full_lattice_states' in kwargs:
                logger.debug(f"[SEARCH] full_lattice_states.shape={kwargs['full_lattice_states'].shape}")
        """
        Основной forward pass с упрощенной логикой

        Args:
            current_state: [state_size] - текущее состояние клетки
            neighbor_states: [num_neighbors, state_size] - состояния соседей
            cell_idx: индекс текущей клетки
            neighbor_indices: индексы соседних клеток
            external_input: внешний вход (опционально)
            spatial_optimizer: опциональный spatial optimizer для adaptive radius поиска

        Returns:
            Dict с результатами обработки
        """
        # === ОСНОВНОЙ МЕТОД: Adaptive Radius Neighbor Search ===
        if spatial_optimizer is not None and hasattr(
            spatial_optimizer, "find_neighbors_by_radius_safe"
        ):
            # ВСЕГДА используем spatial optimizer для безопасного поиска соседей
            adaptive_neighbors = spatial_optimizer.find_neighbors_by_radius_safe(
                cell_idx
            )

            if adaptive_neighbors and "full_lattice_states" in kwargs:
                # Получаем состояния соседей из полной решетки
                full_states = kwargs["full_lattice_states"]

                # ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ индексов перед использованием
                max_idx = full_states.shape[0] - 1
                valid_neighbors = [
                    idx for idx in adaptive_neighbors if 0 <= idx <= max_idx
                ]

                if len(valid_neighbors) != len(adaptive_neighbors):
                    logger.warning(
                        f"[WARN] Отфильтровано {len(adaptive_neighbors) - len(valid_neighbors)} невалидных индексов для клетки {cell_idx}"
                    )

                if valid_neighbors:
                    neighbor_indices = valid_neighbors
                    neighbor_states = full_states[neighbor_indices]

                    logger.debug(
                        f"[SEARCH] ОСНОВНОЙ РЕЖИМ: spatial_optimizer для клетки {cell_idx}: найдено {len(neighbor_indices)} валидных соседей"
                    )
                else:
                    neighbor_indices = []
                    neighbor_states = torch.empty(
                        0, full_states.shape[1], device=full_states.device
                    )
            else:
                raise RuntimeError(
                    f"[ERROR] КРИТИЧЕСКАЯ ОШИБКА: spatial_optimizer передан для клетки {cell_idx}, "
                    f"но full_lattice_states отсутствует. Согласно CLAUDE.md fallback'и запрещены."
                )
        else:
            # По правилам CLAУDE.md - никаких fallback'ов, proper error handling
            # Проверяем длину neighbor_indices (может быть list или tensor)
            neighbor_count = neighbor_indices.numel() if isinstance(neighbor_indices, torch.Tensor) else len(neighbor_indices)
            if neighbor_count == 0:
                raise RuntimeError(
                    f"[ERROR] КРИТИЧЕСКАЯ ОШИБКА: Для клетки {cell_idx} не переданы ни spatial_optimizer, ни neighbor_indices. "
                    f"Согласно CLAUDE.md fallback'и запрещены - исправьте интеграцию."
                )
            
            # Если neighbor_indices переданы, требуем full_lattice_states
            if "full_lattice_states" not in kwargs:
                raise RuntimeError(
                    f"[ERROR] КРИТИЧЕСКАЯ ОШИБКА: Для клетки {cell_idx} переданы neighbor_indices={neighbor_indices}, "
                    f"но отсутствует full_lattice_states. Требуется полная интеграция без fallback'ов."
                )
            
            full_states = kwargs["full_lattice_states"]
            logger.debug(f"[SEARCH] BEFORE extraction: full_states.shape={full_states.shape}")
            logger.debug(f"[SEARCH] neighbor_indices for cell {cell_idx}: {neighbor_indices}")
            
            # Правильное извлечение состояний соседей с учетом batch dimension
            # Преобразуем neighbor_indices в list если это tensor
            if isinstance(neighbor_indices, torch.Tensor):
                neighbor_indices_list = neighbor_indices.tolist()
            else:
                neighbor_indices_list = neighbor_indices
            
            if full_states.dim() == 3:  # [batch, num_cells, state_size]
                # Извлекаем состояния соседей для каждого элемента в batch
                # Но поскольку connection classifier ожидает [num_neighbors, state_size],
                # мы берем только первый элемент batch'а
                neighbor_states = full_states[0, neighbor_indices_list, :]  # [num_neighbors, state_size]
            elif full_states.dim() == 2:  # [num_cells, state_size]
                neighbor_states = full_states[neighbor_indices_list]  # [num_neighbors, state_size]
            else:
                raise RuntimeError(f"Неожиданная размерность full_lattice_states: {full_states.shape}")
            
            logger.debug(
                f"[OK] Извлечены состояния соседей из full_lattice_states для клетки {cell_idx}, shape={neighbor_states.shape}"
            )

        # Проверяем количество соседей (может быть list или tensor)
        neighbor_count = neighbor_indices.numel() if isinstance(neighbor_indices, torch.Tensor) else len(neighbor_indices)
        if neighbor_count == 0:
            return self._empty_forward_result(current_state)

        batch_size = 1
        device = current_state.device

        # Убеждаемся что все tensor'ы на правильном устройстве
        current_state = self.device_manager.ensure_device(current_state)
        neighbor_states = self.device_manager.ensure_device(neighbor_states)
        if external_input is not None:
            external_input = self.device_manager.ensure_device(external_input)

        # === 1. КЛАССИФИКАЦИЯ СВЯЗЕЙ ===
        logger.debug(f"[{cell_idx}] Шаг 1: Классификация связей...")
        classifications = self.connection_classifier.classify_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            cell_state=current_state,
            neighbor_states=neighbor_states,
        )
        logger.debug(f"[{cell_idx}] Классификация завершена.")

        # === 2. ОБРАБОТКА КАЖДЫМ ЭКСПЕРТОМ ===
        logger.debug(f"[{cell_idx}] Шаг 2: Обработка экспертами...")
        expert_outputs = []
        tensors_to_return = []

        # Local Expert
        local_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.LOCAL]
        ]
        logger.debug(f"[{cell_idx}] Local expert, {len(local_neighbors)} соседей.")
        if local_neighbors:
            # Создаем маску для местных соседей
            local_mask = torch.isin(
                neighbor_indices,
                torch.tensor(local_neighbors, device=neighbor_indices.device),
            )
            # Flatten маску для правильной индексации
            local_mask_flat = local_mask.flatten()
            local_neighbor_states = neighbor_states[local_mask_flat]
            logger.debug(
                f"[{cell_idx}] Local neighbor states shape: {local_neighbor_states.shape}"
            )

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
            logger.debug(
                f"[{cell_idx}] Local expert output shape: {local_output.shape}"
            )

        else:
            local_output = self.memory_pool_manager.get_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
            tensors_to_return.append(local_output)
        expert_outputs.append(local_output.squeeze(0))

        # Functional Expert
        functional_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.FUNCTIONAL]
        ]
        logger.debug(
            f"[{cell_idx}] Functional expert, {len(functional_neighbors)} соседей."
        )
        if functional_neighbors:
            # Создаем маску для функциональных соседей
            functional_mask = torch.isin(
                neighbor_indices,
                torch.tensor(functional_neighbors, device=neighbor_indices.device),
            )
            # Flatten маску для правильной индексации
            functional_mask_flat = functional_mask.flatten()
            functional_neighbor_states = neighbor_states[functional_mask_flat]
            logger.debug(
                f"[{cell_idx}] Functional neighbor states shape: {functional_neighbor_states.shape}"
            )

            def functional_expert_wrapper(current, neighbors):
                res = self.functional_expert(current, neighbors)
                if isinstance(res, dict):
                    return res.get("output", res.get("new_state", current))
                return res

            functional_output = checkpoint(
                functional_expert_wrapper,
                current_state,
                functional_neighbor_states,
                use_reentrant=False,
            )
            logger.debug(
                f"[{cell_idx}] Functional expert output shape: {functional_output.shape}"
            )
        else:
            functional_output = self.memory_pool_manager.get_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
            tensors_to_return.append(functional_output)
        expert_outputs.append(functional_output.squeeze(0))

        # Distant Expert (только если CNF включен)
        distant_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.DISTANT]
        ]
        logger.debug(f"[{cell_idx}] Distant expert, {len(distant_neighbors)} соседей.")
        if self.enable_cnf and distant_neighbors:
            # Создаем маску для дальних соседей
            distant_mask = torch.isin(
                neighbor_indices,
                torch.tensor(distant_neighbors, device=neighbor_indices.device),
            )
            # Flatten маску для правильной индексации
            distant_mask_flat = distant_mask.flatten()
            distant_neighbor_states = neighbor_states[distant_mask_flat]
            logger.debug(
                f"[{cell_idx}] Distant neighbor states shape: {distant_neighbor_states.shape}"
            )

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
            logger.debug(
                f"[{cell_idx}] Distant expert output shape: {distant_output.shape}"
            )
        else:
            distant_output = self.memory_pool_manager.get_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
            tensors_to_return.append(distant_output)
        expert_outputs.append(distant_output.squeeze(0))

        # === 3. КОМБИНИРОВАНИЕ РЕЗУЛЬТАТОВ ===
        logger.debug(
            f"[{cell_idx}] Шаг 3: Комбинирование результатов. expert_outputs: {[t.shape for t in expert_outputs]}"
        )
        try:
            # Предотвращение ошибки с пустыми expert_outputs
            if not expert_outputs:
                logger.warning(
                    f"[WARN] Нет выходов экспертов для клетки {cell_idx}, пропуск GatingNetwork."
                )
                final_state = current_state
                expert_weights = torch.zeros(
                    1, 3, device=device
                )  # Возвращаем нулевые веса
            else:
                # --- ИСПРАВЛЕНИЕ: Агрегация состояний соседей ---
                # Усредняем состояния всех соседей для получения единого вектора контекста
                logger.debug(
                    f"[{cell_idx}] Агрегация neighbor_states... Shape: {neighbor_states.shape}"
                )
                if neighbor_states.numel() > 0:
                    neighbor_activity = torch.mean(neighbor_states, dim=0, keepdim=True)
                else:
                    # Если нет соседей, используем нулевой вектор
                    neighbor_activity = torch.zeros(
                        1, self.state_size, device=device, dtype=current_state.dtype
                    )
                logger.debug(
                    f"[{cell_idx}] neighbor_activity shape: {neighbor_activity.shape}"
                )

                # Вызов GatingNetwork с корректными по форме тензорами
                logger.debug(f"[{cell_idx}] Вызов GatingNetwork...")
                combined_output, expert_weights = self.gating_network(
                    current_state=current_state,  # [1, state_size]
                    neighbor_activity=neighbor_activity,  # [1, state_size]
                    expert_outputs=expert_outputs,
                )
                logger.debug(
                    f"[{cell_idx}] GatingNetwork завершен. combined_output: {combined_output.shape}, expert_weights: {expert_weights.shape}"
                )

                # Residual connection
                final_state = current_state + combined_output.squeeze(0)

        except Exception as e:
            logger.error(
                f"[ERROR] MoE processor CRITICAL error on cell {cell_idx}: {e}",
                exc_info=True,
            )
            # В случае ошибки возвращаем исходное состояние, чтобы не прерывать процесс
            final_state = current_state
            expert_weights = torch.zeros(1, 3, device=device)

        # Освобождаем временные тензоры
        for t in tensors_to_return:
            self.memory_pool_manager.return_tensor(t)

        # === 4. ОБНОВЛЕНИЕ СТАТИСТИКИ ===
        self._update_stats(classifications, expert_weights)

        log_cell_forward(
            "MoEConnectionProcessor",
            input_shapes={
                "current_state": current_state.shape,
                "neighbor_states": neighbor_states.shape,
            },
            output_shape=final_state.shape,
        )

        # Отдельное логирование expert_weights
        logger.debug(
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

        self.usage_stats["expert_weights"]["local"] += weights[0].item()
        self.usage_stats["expert_weights"]["functional"] += weights[1].item()
        self.usage_stats["expert_weights"]["distant"] += weights[2].item()

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
