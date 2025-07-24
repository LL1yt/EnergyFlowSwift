"""
Mixin для интеграции функциональной кластеризации с Lattice3D.

Объединяет:
- BasicFunctionalClustering: базовая кластеризация
- CoordinationInterface: координация кластеров
- Интеграция с существующей пластичностью (STDP, конкурентное обучение)

Архитектура:
- Модульность: каждый компонент независим
- Расширяемость: готов к добавлению новых методов
- Совместимость: интегрируется с существующей PlasticityMixin
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

from .basic_clustering import BasicFunctionalClustering
from .coordination_interface import CoordinationInterface

logger = logging.getLogger(__name__)


class ClusteringMixin:
    """
    Mixin для добавления функциональной кластеризации к Lattice3D.

    Предоставляет:
    - Автоматическую кластеризацию по сходству состояний
    - Координацию кластеров (расширяемую в будущем)
    - Модификацию весов связей на основе кластеров
    - Интеграцию с существующей пластичностью
    """

    def _init_clustering(self, config: Dict):
        """
        Инициализация системы кластеризации.

        Args:
            config: конфигурация с параметрами кластеризации
        """
        clustering_config = config.get("clustering_config", {})

        # Базовая кластеризация
        self.functional_clustering = BasicFunctionalClustering(clustering_config)

        # Координационный интерфейс
        coordination_config = clustering_config.get("coordination", {})
        self.coordination_interface = CoordinationInterface(coordination_config)

        # Параметры интеграции
        self.enable_clustering = clustering_config.get("enable_clustering", True)
        self.clustering_priority = clustering_config.get(
            "priority", 0.5
        )  # vs plasticity
        self.integration_mode = clustering_config.get("integration_mode", "additive")

        # Состояние кластеризации
        self.current_clustering_info = {}
        self.clustering_step_counter = 0

        logger.info(
            f"ClusteringMixin initialized: enable={self.enable_clustering}, "
            f"priority={self.clustering_priority}, mode={self.integration_mode}"
        )

    def apply_functional_clustering(
        self, current_step: int, external_input: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Применяет функциональную кластеризацию к текущему состоянию решетки.

        Args:
            current_step: текущий шаг симуляции
            external_input: внешние входы (для контекста)

        Returns:
            clustering_result: информация о применении кластеризации
        """
        if not self.enable_clustering:
            return {"applied": False, "reason": "clustering_disabled"}

        try:
            # Подготавливаем контекст для координации
            context = {
                "step": current_step,
                "external_input": external_input,
                "lattice_dimensions": self.config.dimensions,
                "total_cells": self.states.size(0),
            }

            # Получаем индексы соседей
            neighbor_indices = self.topology.get_all_neighbor_indices_batched()

            # Выполняем базовую кластеризацию
            modified_weights, clustering_info = (
                self.functional_clustering.update_clustering(
                    cell_states=self.states,
                    connection_weights=self.connection_weights,
                    neighbor_indices=neighbor_indices,
                    current_step=current_step,
                )
            )

            # Применяем координацию
            if clustering_info.get("updated", False):
                coordinated_clusters = self.coordination_interface.coordinate_clusters(
                    base_clusters=clustering_info["clusters"], context=context
                )

                # Пересчитываем веса с координированными кластерами
                if coordinated_clusters != clustering_info["clusters"]:
                    modified_weights = self.functional_clustering.apply_cluster_weights(
                        connection_weights=self.connection_weights,
                        neighbor_indices=neighbor_indices,
                        clusters=coordinated_clusters,
                    )
                    clustering_info["clusters"] = coordinated_clusters
                    clustering_info["coordinated"] = True
                else:
                    clustering_info["coordinated"] = False

            # Интегрируем с существующими весами
            final_weights = self._integrate_clustering_weights(
                original_weights=self.connection_weights,
                clustering_weights=modified_weights,
                integration_mode=self.integration_mode,
            )

            # Обновляем веса связей
            self.connection_weights = final_weights

            # Сохраняем информацию о кластеризации
            self.current_clustering_info = clustering_info
            self.clustering_step_counter += 1

            result = {
                "applied": True,
                "clustering_info": clustering_info,
                "weights_modified": True,
                "integration_mode": self.integration_mode,
            }

            if clustering_info.get("updated", False):
                logger.debug(
                    f"Applied clustering at step {current_step}: "
                    f"{clustering_info.get('num_clusters', 0)} clusters, "
                    f"coordinated={clustering_info.get('coordinated', False)}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in functional clustering: {e}")
            return {"applied": False, "reason": f"error: {e}"}

    def _integrate_clustering_weights(
        self,
        original_weights: torch.Tensor,
        clustering_weights: torch.Tensor,
        integration_mode: str,
    ) -> torch.Tensor:
        """
        Интегрирует веса кластеризации с существующими весами.

        Args:
            original_weights: исходные веса связей
            clustering_weights: веса с учетом кластеризации
            integration_mode: режим интеграции

        Returns:
            integrated_weights: итоговые веса
        """
        if integration_mode == "replace":
            # Полная замена весов
            return clustering_weights

        elif integration_mode == "additive":
            # Аддитивная интеграция с нормализацией
            priority = self.clustering_priority
            integrated = (
                1 - priority
            ) * original_weights + priority * clustering_weights
            return torch.clamp(integrated, 0.1, 3.0)

        elif integration_mode == "multiplicative":
            # Мультипликативная интеграция
            ratio = clustering_weights / (original_weights + 1e-8)
            integrated = original_weights * torch.lerp(
                torch.ones_like(ratio), ratio, self.clustering_priority
            )
            return torch.clamp(integrated, 0.1, 3.0)

        elif integration_mode == "selective":
            # Селективная интеграция (только значимые изменения)
            diff = torch.abs(clustering_weights - original_weights)
            threshold = 0.1  # Порог значимости изменений
            mask = diff > threshold

            integrated = original_weights.clone()
            integrated[mask] = torch.lerp(
                original_weights[mask],
                clustering_weights[mask],
                self.clustering_priority,
            )
            return integrated

        else:
            logger.warning(f"Unknown integration mode: {integration_mode}")
            return original_weights

    def get_clustering_statistics(self) -> Dict:
        """Возвращает статистику кластеризации."""
        if not hasattr(self, "functional_clustering"):
            return {"clustering_initialized": False}

        stats = {
            "clustering_initialized": True,
            "enable_clustering": self.enable_clustering,
            "clustering_step_counter": self.clustering_step_counter,
            "current_clustering_info": self.current_clustering_info,
        }

        # Статистика базовой кластеризации
        clustering_stats = self.functional_clustering.get_statistics()
        stats["basic_clustering"] = clustering_stats

        # Статистика координации
        coordination_stats = self.coordination_interface.get_coordination_statistics()
        stats["coordination"] = coordination_stats

        return stats

    def configure_clustering(
        self,
        enable: Optional[bool] = None,
        priority: Optional[float] = None,
        integration_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Динамическая настройка кластеризации.

        Args:
            enable: включить/выключить кластеризацию
            priority: приоритет кластеризации vs пластичности
            integration_mode: режим интеграции весов
            **kwargs: дополнительные параметры
        """
        if enable is not None:
            self.enable_clustering = enable
            logger.info(f"Clustering enabled: {enable}")

        if priority is not None:
            self.clustering_priority = max(0.0, min(1.0, priority))
            logger.info(f"Clustering priority: {self.clustering_priority}")

        if integration_mode is not None:
            if integration_mode in [
                "replace",
                "additive",
                "multiplicative",
                "selective",
            ]:
                self.integration_mode = integration_mode
                logger.info(f"Integration mode: {integration_mode}")
            else:
                logger.warning(f"Invalid integration mode: {integration_mode}")

        # Передаем дополнительные параметры компонентам
        if hasattr(self, "functional_clustering") and kwargs:
            logger.debug(f"Updating clustering parameters: {kwargs}")
            # TODO: Добавить метод update_config в BasicFunctionalClustering

    def add_user_clustering_hint(self, hint_type: str, hint_data: Dict):
        """
        Добавляет пользовательскую подсказку для кластеризации.

        Интерфейс для будущего пользовательского управления.
        """
        if hasattr(self, "coordination_interface"):
            self.coordination_interface.add_user_hint(hint_type, hint_data)
            logger.info(f"Added clustering hint: {hint_type}")
        else:
            logger.warning("Coordination interface not initialized")

    def add_user_clustering_correction(
        self,
        wrong_clustering: Dict[int, List[int]],
        correct_clustering: Dict[int, List[int]],
    ):
        """
        Добавляет пользовательскую коррекцию кластеризации.

        Интерфейс для будущего обучения координации.
        """
        if hasattr(self, "coordination_interface"):
            self.coordination_interface.add_user_correction(
                wrong_clustering, correct_clustering
            )
            logger.info("Added clustering correction")
        else:
            logger.warning("Coordination interface not initialized")

    def get_current_clusters(self) -> Dict[int, List[int]]:
        """Возвращает текущие кластеры."""
        if hasattr(self, "functional_clustering"):
            return self.functional_clustering.current_clusters
        else:
            return {}

    def get_cluster_centroids(self) -> Optional[torch.Tensor]:
        """Возвращает центроиды кластеров."""
        if hasattr(self, "functional_clustering"):
            return self.functional_clustering.cluster_centroids
        else:
            return None
