"""
Основная реализация Cell Prototype

Этот файл содержит класс CellPrototype - "умную клетку", которая является
базовым строительным блоком для 3D клеточной нейронной сети.

Биологическая аналогия:
Представьте нейрон в коре головного мозга:
- Получает сигналы от соседних нейронов (дендриты)
- Обрабатывает их (сома)
- Передает результат дальше (аксон)

Все нейроны в одном слое коры имеют похожую структуру,
но каждый обрабатывает свои уникальные входные сигналы.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Any

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)


class CellPrototype(nn.Module):
    """
    Прототип "умной клетки" для 3D клеточной нейронной сети

    Биологическая аналогия:
    Это как один нейрон в коре головного мозга:
    - Получает сигналы от соседей (neighbor_states)
    - Учитывает свое текущее состояние (own_state)
    - Может получать внешние сигналы (external_input)
    - Обрабатывает все это и выдает новое состояние

    Параметры архитектуры:
        input_size (int): Размер входного вектора
        state_size (int): Размер состояния клетки
        hidden_size (int): Размер скрытых слоев
        num_neighbors (int): Ожидаемое количество соседей (по умолчанию 6 для 3D)
        activation (str): Функция активации ('tanh', 'sigmoid', 'relu')
        use_bias (bool): Использовать ли bias в слоях
    """

    def __init__(
        self,
        input_size: int = 12,
        state_size: int = 8,
        hidden_size: int = 16,
        num_neighbors: int = 6,
        activation: str = "tanh",
        use_bias: bool = True,
    ):
        super(CellPrototype, self).__init__()

        # Сохраняем параметры конфигурации
        self.input_size = input_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_neighbors = num_neighbors
        self.activation_name = activation
        self.use_bias = use_bias

        # Вычисляем размер входа для нейросети
        # Входы от соседей + собственное состояние + внешний вход
        neighbor_input_size = num_neighbors * state_size  # Сигналы от соседних клеток
        own_state_size = state_size  # Собственное состояние
        external_input_size = input_size  # Внешний вход (для граничных клеток)

        total_input_size = neighbor_input_size + own_state_size + external_input_size

        logger.info(f"Создается CellPrototype:")
        logger.info(f"  - Входы от соседей: {neighbor_input_size}")
        logger.info(f"  - Собственное состояние: {own_state_size}")
        logger.info(f"  - Внешний вход: {external_input_size}")
        logger.info(f"  - Общий размер входа: {total_input_size}")
        logger.info(f"  - Размер выхода: {state_size}")

        # Создаем простую нейросеть: вход -> скрытый слой -> выход
        self.network = nn.Sequential(
            nn.Linear(total_input_size, hidden_size, bias=use_bias),
            self._get_activation_function(activation),
            nn.Linear(hidden_size, state_size, bias=use_bias),
            nn.Tanh(),  # Выходное состояние должно быть в диапазоне [-1, 1]
        )

        # Инициализируем веса для стабильности
        self._initialize_weights()

        logger.info(f"[OK] CellPrototype создан успешно")

    def _get_activation_function(self, activation: str) -> nn.Module:
        """
        Возвращает функцию активации по названию

        Параметры:
            activation (str): Название функции ('tanh', 'sigmoid', 'relu')

        Возвращает:
            nn.Module: Соответствующая функция активации
        """
        activation_map = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
        }

        if activation.lower() not in activation_map:
            logger.warning(f"Неизвестная активация '{activation}', используем 'tanh'")
            return nn.Tanh()

        return activation_map[activation.lower()]

    def _initialize_weights(self):
        """
        Инициализирует веса для стабильного обучения

        Биологическая аналогия:
        Как "настройка чувствительности" нейрона при его развитии
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Используем Xavier инициализацию для стабильности
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        logger.debug("Веса инициализированы")

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Основная функция обработки - "мышление" клетки

        Биологическая аналогия:
        Нейрон получает сигналы по дендритам, обрабатывает их в соме,
        и отправляет результат по аксону.

        Параметры:
            neighbor_states (torch.Tensor): Состояния соседних клеток
                Форма: (batch_size, num_neighbors, state_size)
            own_state (torch.Tensor): Текущее состояние этой клетки
                Форма: (batch_size, state_size)
            external_input (torch.Tensor, optional): Внешний вход
                Форма: (batch_size, input_size)

        Возвращает:
            torch.Tensor: Новое состояние клетки
                Форма: (batch_size, state_size)
        """
        batch_size = own_state.shape[0]

        # Шаг 1: Обрабатываем входы от соседей
        if neighbor_states.numel() > 0:  # Проверяем, что есть соседи
            # Преобразуем состояния соседей в плоский вектор
            neighbor_input = neighbor_states.view(batch_size, -1)
        else:
            # Если соседей нет, создаем нулевой вектор
            neighbor_input = torch.zeros(
                batch_size,
                self.num_neighbors * self.state_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Шаг 2: Подготавливаем внешний вход
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Шаг 3: Объединяем все входы
        # Биологическая аналогия: нейрон "суммирует" все сигналы на дендритах
        combined_input = torch.cat(
            [
                neighbor_input,  # Сигналы от соседей
                own_state,  # Собственное состояние
                external_input,  # Внешний сигнал
            ],
            dim=1,
        )

        # Шаг 4: Обрабатываем объединенный вход через нейросеть
        # Биологическая аналогия: обработка в соме нейрона
        new_state = self.network(combined_input)

        return new_state

    def get_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели для отладки

        Возвращает:
            Dict[str, Any]: Словарь с информацией о модели
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_size": self.input_size,
            "state_size": self.state_size,
            "hidden_size": self.hidden_size,
            "num_neighbors": self.num_neighbors,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Примерно в MB
        }

    def __repr__(self) -> str:
        """Строковое представление модели"""
        info = self.get_info()
        return (
            f"CellPrototype("
            f"input_size={info['input_size']}, "
            f"state_size={info['state_size']}, "
            f"hidden_size={info['hidden_size']}, "
            f"params={info['total_parameters']})"
        )


def create_cell_from_config(config: Dict[str, Any]):
    """
    Создает экземпляр клетки из конфигурации

    Поддерживает различные архитектуры:
    - gmlp_cell: GatedMLPCell
    - minimal_nca_cell: MinimalNCACell
    - cell_prototype: Простой CellPrototype (fallback)

    Параметры:
        config (Dict[str, Any]): Словарь с параметрами конфигурации

    Возвращает:
        nn.Module: Настроенный экземпляр клетки
    """
    prototype_name = config.get("prototype_name", "cell_prototype")

    logger.info(f"Создание клетки архитектуры: {prototype_name}")

    if prototype_name == "gmlp_cell":
        # Импорт GatedMLPCell
        from .architectures.gmlp_cell import GatedMLPCell

        # Извлекаем конфигурацию из правильного места
        gmlp_config = config.get("gmlp_cell", {})

        # Поддержка новой структуры с cell_prototype
        if not gmlp_config and "cell_prototype" in config:
            cell_proto_config = config.get("cell_prototype", {})
            gmlp_config = cell_proto_config.get("gmlp_cell", {})

        params = {
            "state_size": gmlp_config.get("state_size", 32),
            "neighbor_count": gmlp_config.get("neighbor_count", 6),
            "hidden_dim": gmlp_config.get("hidden_dim", 128),
            "external_input_size": gmlp_config.get("external_input_size", 12),
            "use_memory": gmlp_config.get("use_memory", True),
            "target_params": gmlp_config.get("target_params", 25000),
        }

        logger.info(f"Создаем GatedMLPCell с параметрами: {params}")
        return GatedMLPCell(**params)

    elif prototype_name == "minimal_nca_cell":
        # Импорт MinimalNCACell
        from .architectures.minimal_nca_cell import MinimalNCACell

        # Извлекаем конфигурацию из правильного места
        nca_config = config.get("minimal_nca_cell", {})

        # Поддержка новой структуры с cell_prototype
        if not nca_config and "cell_prototype" in config:
            cell_proto_config = config.get("cell_prototype", {})
            nca_config = cell_proto_config.get("minimal_nca_cell", {})

        params = {
            "state_size": nca_config.get("state_size", 32),
            "neighbor_count": nca_config.get("neighbor_count", 6),
            "hidden_dim": nca_config.get("hidden_dim", 64),
            "external_input_size": nca_config.get("external_input_size", 2),
            "activation": nca_config.get("activation", "tanh"),
            "dropout": nca_config.get("dropout", 0.0),
            "use_memory": nca_config.get("use_memory", False),
            "enable_lattice_scaling": nca_config.get("enable_lattice_scaling", False),
            "target_params": nca_config.get("target_params", 69),
            # alpha и beta не передаются в конструктор - они создаются как nn.Parameter внутри
        }

        logger.info(f"Создаем MinimalNCACell с параметрами: {params}")
        return MinimalNCACell(**params)

    else:
        # Fallback к простому CellPrototype
        cell_config = config.get("cell_prototype", {})

        params = {
            "input_size": cell_config.get("input_size", 12),
            "state_size": cell_config.get("state_size", 8),
            "hidden_size": cell_config.get("architecture", {}).get("hidden_size", 16),
            "num_neighbors": cell_config.get("num_neighbors", 6),
            "activation": cell_config.get("architecture", {}).get("activation", "tanh"),
            "use_bias": cell_config.get("architecture", {}).get("use_bias", True),
        }

        logger.info(f"Создаем CellPrototype из конфигурации: {params}")
        return CellPrototype(**params)


# Простая функция для тестирования
def test_cell_basic():
    """
    Базовый тест функциональности CellPrototype
    """
    print("🧪 Запуск базового теста CellPrototype...")

    # Создаем простую конфигурацию
    config = {
        "cell_prototype": {
            "input_size": 8,
            "state_size": 4,
            "architecture": {"hidden_size": 12, "activation": "tanh", "use_bias": True},
        }
    }

    # Создаем клетку
    cell = create_cell_from_config(config)

    # Подготавливаем тестовые данные
    batch_size = 2
    num_neighbors = 6

    # Создаем случайные входные данные
    neighbor_states = torch.randn(
        batch_size, num_neighbors, config["cell_prototype"]["state_size"]
    )
    own_state = torch.randn(batch_size, config["cell_prototype"]["state_size"])
    external_input = torch.randn(batch_size, config["cell_prototype"]["input_size"])

    print(f"  Входные данные:")
    print(f"    neighbor_states: {neighbor_states.shape}")
    print(f"    own_state: {own_state.shape}")
    print(f"    external_input: {external_input.shape}")

    # Запускаем forward pass
    with torch.no_grad():
        new_state = cell(neighbor_states, own_state, external_input)

    print(f"  Выходные данные:")
    print(f"    new_state: {new_state.shape}")
    print(
        f"    Диапазон значений: [{new_state.min().item():.3f}, {new_state.max().item():.3f}]"
    )

    # Проверяем корректность
    assert (
        new_state.shape == own_state.shape
    ), f"Неверная форма выхода: {new_state.shape} vs {own_state.shape}"
    assert torch.all(
        torch.abs(new_state) <= 1.0
    ), "Выходные значения должны быть в диапазоне [-1, 1]"

    # Выводим информацию о модели
    info = cell.get_info()
    print(f"  Информация о модели:")
    for key, value in info.items():
        print(f"    {key}: {value}")

    print("  [OK] Базовый тест пройден успешно!")
    return True


if __name__ == "__main__":
    # Настраиваем логирование для тестирования
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("[START] Тестирование модуля CellPrototype")
    print("=" * 40)

    try:
        test_cell_basic()
        print("\n[OK] Все тесты пройдены успешно!")
    except Exception as e:
        print(f"\n[ERROR] Ошибка в тестах: {e}")
        raise
