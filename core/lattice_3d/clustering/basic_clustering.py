"""
Базовая функциональная кластеризация для клеток решетки.

Алгоритм:
1. Вычисление cosine similarity между состояниями клеток
2. K-means кластеризация для группировки похожих клеток
3. Усиление связей внутри кластеров
4. Ослабление связей между кластерами
5. Динамическое обновление структуры

Биологическое обоснование:
- Клетки с похожей активностью формируют функциональные группы
- Внутрикластерные связи усиливаются (локальная специализация)
- Межкластерные связи ослабляются (конкуренция между группами)
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging

logger = logging.getLogger(__name__)


class BasicFunctionalClustering:
    """
    Базовая функциональная кластеризация клеток по сходству состояний.

    Параметры:
    - similarity_threshold: порог сходства для считания клеток похожими
    - max_clusters: максимальное количество кластеров
    - update_frequency: частота обновления кластеризации (в шагах)
    - intra_cluster_boost: коэффициент усиления связей внутри кластера
    - inter_cluster_dampening: коэффициент ослабления связей между кластерами
    """

    def __init__(self, config: Dict):
        # Основные параметры кластеризации
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_clusters = config.get("max_clusters", 8)
        self.update_frequency = config.get("update_frequency", 100)

        # Параметры модификации связей
        self.intra_cluster_boost = config.get("intra_cluster_boost", 1.2)
        self.inter_cluster_dampening = config.get("inter_cluster_dampening", 0.8)

        # Параметры стабилизации
        self.min_cluster_size = config.get("min_cluster_size", 5)
        self.stability_threshold = config.get("stability_threshold", 0.8)

        # Состояние кластеризации
        self.current_clusters = {}
        self.cluster_centroids = None
        self.cluster_stability = {}
        self.last_update_step = (
            -1
        )  # Инициализируем как -1, чтобы первый шаг (0) сработал

        # Статистика
        self.clustering_history = []
        self.performance_stats = {
            "total_clusterings": 0,
            "avg_clustering_time": 0.0,
            "cluster_stability_score": 0.0,
        }

        logger.info(
            f"BasicFunctionalClustering initialized: "
            f"similarity_threshold={self.similarity_threshold}, "
            f"max_clusters={self.max_clusters}, "
            f"update_frequency={self.update_frequency}"
        )

    def should_update_clusters(self, current_step: int) -> bool:
        """Определяет, нужно ли обновлять кластеризацию на текущем шаге."""
        should_update = (current_step - self.last_update_step) >= self.update_frequency
        logger.debug(
            f"Clustering update check: step={current_step}, last_update={self.last_update_step}, "
            f"frequency={self.update_frequency}, should_update={should_update}"
        )
        return should_update

    def compute_similarities(self, cell_states: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет cosine similarity между всеми парами клеток.

        Args:
            cell_states: [num_cells, state_size] - состояния клеток

        Returns:
            similarities: [num_cells, num_cells] - матрица сходства
        """
        # Нормализуем состояния для cosine similarity
        normalized_states = F.normalize(cell_states, p=2, dim=1)

        # Вычисляем cosine similarity: cos(θ) = (a·b) / (|a||b|)
        similarities = torch.mm(normalized_states, normalized_states.t())

        return similarities

    def cluster_cells(self, cell_states: torch.Tensor) -> Dict[int, List[int]]:
        """
        Выполняет кластеризацию клеток по сходству состояний.

        Args:
            cell_states: [num_cells, state_size] - состояния клеток

        Returns:
            clusters: {cluster_id: [cell_indices]} - словарь кластеров
        """
        start_time = time.time()
        num_cells = cell_states.size(0)

        # Определяем оптимальное количество кластеров
        n_clusters = min(self.max_clusters, max(2, num_cells // 50))

        # Пробуем GPU K-means, fallback на CPU
        try:
            cluster_labels = self._gpu_kmeans(cell_states, n_clusters)
            logger.debug(f"Used GPU K-means clustering")
        except Exception as e:
            logger.warning(f"GPU K-means failed: {e}. Trying CPU sklearn.")
            try:
                # Fallback на sklearn CPU
                states_cpu = cell_states.detach().cpu().numpy()
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=5, max_iter=50
                )
                cluster_labels = kmeans.fit_predict(states_cpu)
                self.cluster_centroids = torch.from_numpy(kmeans.cluster_centers_).to(
                    cell_states.device
                )
                logger.debug(f"Used CPU sklearn K-means clustering")
            except Exception as e2:
                logger.warning(
                    f"CPU K-means also failed: {e2}. Using similarity-based clustering."
                )
                cluster_labels = self._fallback_similarity_clustering(cell_states)

        # Преобразуем labels в словарь кластеров
        clusters = {}
        for cell_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(cell_idx)

        # Фильтруем маленькие кластеры
        clusters = {
            cid: members
            for cid, members in clusters.items()
            if len(members) >= self.min_cluster_size
        }

        # Обновляем статистику
        clustering_time = time.time() - start_time
        self.performance_stats["total_clusterings"] += 1
        self.performance_stats["avg_clustering_time"] = (
            self.performance_stats["avg_clustering_time"]
            * (self.performance_stats["total_clusterings"] - 1)
            + clustering_time
        ) / self.performance_stats["total_clusterings"]

        logger.debug(
            f"Clustered {num_cells} cells into {len(clusters)} clusters "
            f"(time: {clustering_time:.3f}s)"
        )

        return clusters

    def _gpu_kmeans(
        self, data: torch.Tensor, n_clusters: int, max_iter: int = 50
    ) -> np.ndarray:
        """
        GPU-оптимизированная K-means кластеризация.

        Args:
            data: [num_points, features] - данные для кластеризации
            n_clusters: количество кластеров
            max_iter: максимальное количество итераций

        Returns:
            cluster_labels: массив меток кластеров
        """
        device = data.device
        num_points, features = data.shape

        # Инициализация центроидов (K-means++)
        centroids = self._init_centroids_plus_plus(data, n_clusters)

        for iteration in range(max_iter):
            # Вычисляем расстояния до всех центроидов
            distances = torch.cdist(data, centroids)  # [num_points, n_clusters]

            # Назначаем точки к ближайшим центроидам
            cluster_assignments = torch.argmin(distances, dim=1)  # [num_points]

            # Обновляем центроиды
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = cluster_assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    # Если кластер пустой, оставляем старый центроид
                    new_centroids[k] = centroids[k]

            # Проверяем сходимость
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                logger.debug(f"GPU K-means converged after {iteration+1} iterations")
                break

            centroids = new_centroids

        # Сохраняем центроиды
        self.cluster_centroids = centroids

        return cluster_assignments.cpu().numpy()

    def _init_centroids_plus_plus(
        self, data: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:
        """
        K-means++ инициализация центроидов для лучшей сходимости.
        """
        device = data.device
        num_points, features = data.shape

        centroids = torch.zeros(n_clusters, features, device=device)

        # Выбираем первый центроид случайно
        first_idx = torch.randint(0, num_points, (1,), device=device)
        centroids[0] = data[first_idx]

        # Выбираем остальные центроиды с вероятностью пропорциональной квадрату расстояния
        for k in range(1, n_clusters):
            # Вычисляем расстояния до ближайших центроидов
            distances = torch.cdist(data, centroids[:k])  # [num_points, k]
            min_distances = torch.min(distances, dim=1)[0]  # [num_points]

            # Квадрат расстояний как веса для выбора
            weights = min_distances**2
            weights = weights / weights.sum()

            # Выбираем следующий центроид
            next_idx = torch.multinomial(weights, 1)
            centroids[k] = data[next_idx]

        return centroids

    def _fallback_similarity_clustering(self, cell_states: torch.Tensor) -> np.ndarray:
        """
        Резервная кластеризация на основе similarity, если KMeans не работает.
        """
        similarities = self.compute_similarities(cell_states)
        num_cells = cell_states.size(0)

        # Простая агломеративная кластеризация
        cluster_labels = np.arange(num_cells)  # Каждая клетка - свой кластер

        # Объединяем похожие клетки
        for i in range(num_cells):
            for j in range(i + 1, num_cells):
                if similarities[i, j] > self.similarity_threshold:
                    # Объединяем кластеры
                    cluster_labels[cluster_labels == cluster_labels[j]] = (
                        cluster_labels[i]
                    )

        # Переиндексируем кластеры
        unique_labels = np.unique(cluster_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([label_mapping[label] for label in cluster_labels])

        return cluster_labels

    def apply_cluster_weights(
        self,
        connection_weights: torch.Tensor,
        neighbor_indices: torch.Tensor,
        clusters: Dict[int, List[int]],
    ) -> torch.Tensor:
        """
        Применяет модификацию весов связей на основе кластеризации.

        Args:
            connection_weights: [num_cells, max_neighbors] - веса связей
            neighbor_indices: [num_cells, max_neighbors] - индексы соседей
            clusters: {cluster_id: [cell_indices]} - кластеры

        Returns:
            modified_weights: [num_cells, max_neighbors] - модифицированные веса
        """
        modified_weights = connection_weights.clone()

        # Создаем маппинг клетка -> кластер
        cell_to_cluster = {}
        for cluster_id, members in clusters.items():
            for cell_idx in members:
                cell_to_cluster[cell_idx] = cluster_id

        # Модифицируем веса связей
        for cell_idx in range(connection_weights.size(0)):
            if cell_idx not in cell_to_cluster:
                continue  # Клетка не в кластере

            cell_cluster = cell_to_cluster[cell_idx]

            for neighbor_idx in range(connection_weights.size(1)):
                neighbor_cell = neighbor_indices[cell_idx, neighbor_idx].item()

                if neighbor_cell == -1:  # Нет соседа
                    continue

                if neighbor_cell in cell_to_cluster:
                    neighbor_cluster = cell_to_cluster[neighbor_cell]

                    if cell_cluster == neighbor_cluster:
                        # Внутрикластерная связь - усиливаем
                        modified_weights[
                            cell_idx, neighbor_idx
                        ] *= self.intra_cluster_boost
                    else:
                        # Межкластерная связь - ослабляем
                        modified_weights[
                            cell_idx, neighbor_idx
                        ] *= self.inter_cluster_dampening

        # Ограничиваем веса в разумных пределах
        modified_weights = torch.clamp(modified_weights, 0.1, 3.0)

        return modified_weights

    def update_clustering(
        self,
        cell_states: torch.Tensor,
        connection_weights: torch.Tensor,
        neighbor_indices: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Основной метод обновления кластеризации и весов.

        Args:
            cell_states: [num_cells, state_size] - состояния клеток
            connection_weights: [num_cells, max_neighbors] - веса связей
            neighbor_indices: [num_cells, max_neighbors] - индексы соседей
            current_step: текущий шаг симуляции

        Returns:
            modified_weights: модифицированные веса связей
            clustering_info: информация о кластеризации
        """
        if not self.should_update_clusters(current_step):
            # Используем существующую кластеризацию
            logger.debug(f"Skipping clustering update at step {current_step}")
            if self.current_clusters:
                modified_weights = self.apply_cluster_weights(
                    connection_weights, neighbor_indices, self.current_clusters
                )
                logger.debug(f"Using existing {len(self.current_clusters)} clusters")
                return modified_weights, {
                    "clusters": self.current_clusters,
                    "updated": False,
                }
            else:
                logger.debug("No existing clusters, returning original weights")
                return connection_weights, {"clusters": {}, "updated": False}

        # Выполняем новую кластеризацию
        logger.info(f"Performing new clustering at step {current_step}")
        new_clusters = self.cluster_cells(cell_states)
        logger.info(
            f"Found {len(new_clusters)} clusters with sizes: {[len(members) for members in new_clusters.values()]}"
        )

        # Вычисляем стабильность кластеров
        stability_score = self._compute_cluster_stability(new_clusters)
        logger.debug(f"Cluster stability score: {stability_score:.3f}")

        # Обновляем состояние
        self.current_clusters = new_clusters
        self.last_update_step = current_step
        self.performance_stats["cluster_stability_score"] = stability_score

        # Применяем модификацию весов
        modified_weights = self.apply_cluster_weights(
            connection_weights, neighbor_indices, new_clusters
        )

        # Сохраняем в историю
        clustering_record = {
            "step": current_step,
            "num_clusters": len(new_clusters),
            "cluster_sizes": [len(members) for members in new_clusters.values()],
            "stability_score": stability_score,
            "clustering_time": self.performance_stats["avg_clustering_time"],
        }
        self.clustering_history.append(clustering_record)

        # Ограничиваем размер истории
        if len(self.clustering_history) > 100:
            self.clustering_history = self.clustering_history[-80:]

        logger.info(
            f"Updated clustering at step {current_step}: "
            f"{len(new_clusters)} clusters, "
            f"stability={stability_score:.3f}"
        )

        return modified_weights, {
            "clusters": new_clusters,
            "updated": True,
            "stability_score": stability_score,
            "num_clusters": len(new_clusters),
        }

    def _compute_cluster_stability(self, new_clusters: Dict[int, List[int]]) -> float:
        """Вычисляет стабильность кластеров по сравнению с предыдущими."""
        if not self.current_clusters:
            return 0.0

        # Простая метрика: процент клеток, оставшихся в том же кластере
        total_cells = sum(len(members) for members in new_clusters.values())
        stable_cells = 0

        # Создаем маппинги
        old_cell_to_cluster = {}
        for cid, members in self.current_clusters.items():
            for cell in members:
                old_cell_to_cluster[cell] = cid

        new_cell_to_cluster = {}
        for cid, members in new_clusters.items():
            for cell in members:
                new_cell_to_cluster[cell] = cid

        # Считаем стабильные клетки (находим наибольшее пересечение кластеров)
        for new_cid, new_members in new_clusters.items():
            old_cluster_counts = {}
            for cell in new_members:
                if cell in old_cell_to_cluster:
                    old_cid = old_cell_to_cluster[cell]
                    old_cluster_counts[old_cid] = old_cluster_counts.get(old_cid, 0) + 1

            if old_cluster_counts:
                # Максимальное пересечение с предыдущими кластерами
                max_overlap = max(old_cluster_counts.values())
                stable_cells += max_overlap

        stability = stable_cells / total_cells if total_cells > 0 else 0.0
        return stability

    def get_statistics(self) -> Dict:
        """Возвращает статистику кластеризации."""
        current_stats = self.performance_stats.copy()
        current_stats.update(
            {
                "current_num_clusters": len(self.current_clusters),
                "current_cluster_sizes": [
                    len(members) for members in self.current_clusters.values()
                ],
                "history_length": len(self.clustering_history),
                "last_update_step": self.last_update_step,
            }
        )
        return current_stats
