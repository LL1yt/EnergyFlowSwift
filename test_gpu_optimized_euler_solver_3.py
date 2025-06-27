import unittest
from unittest.mock import patch, MagicMock
import torch

from new_rebuild.config import (
    ProjectConfig,
    EulerSettings,
    get_project_config,
    set_project_config,
)
from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
    GPUOptimizedEulerSolver,
    IntegrationResult,
)
from new_rebuild.utils.device_manager import get_device_manager


# Простая функция производной для тестов
def simple_derivative(t, states):
    return -0.1 * states


class TestGPUOptimizedEulerSolver(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем конфигурацию для тестов
        config = get_project_config()
        config.euler = EulerSettings(max_batch_size=100)
        config.device.prefer_cuda = False  # Используем CPU для тестов
        set_project_config(config)

        self.solver = GPUOptimizedEulerSolver(config=config.euler)
        self.device_manager = get_device_manager()
        self.solver.device_manager = self.device_manager

    def test_placeholder(self):
        self.assertTrue(True)

    @patch("new_rebuild.utils.device_manager.DeviceManager.get_memory_stats")
    def test_calculate_adaptive_batch_size(self, mock_get_memory_stats):
        """Тестирует расчет адаптивного размера батча"""
        # Сценарий 1: Достаточно памяти для всех траекторий
        mock_get_memory_stats.return_value = {"available_mb": 2000}
        batch_size = self.solver._calculate_adaptive_batch_size(total_trajectories=500)
        # Ожидаем, что размер батча будет ограничен конфигурацией (100)
        self.assertEqual(batch_size, 100)

        # Сценарий 2: Памяти хватает только на часть траекторий
        # 200MB / (0.1 MB/traj * 0.8 safety) = 250, но ограничено конфигом в 100
        mock_get_memory_stats.return_value = {
            "available_mb": 20
        }  # ~20 / 0.1 * 0.8 = 160 -> 100
        self.solver.config.max_batch_size = 200
        batch_size = self.solver._calculate_adaptive_batch_size(total_trajectories=500)
        self.assertLessEqual(batch_size, 160)

        # Сценарий 3: Памяти очень мало
        mock_get_memory_stats.return_value = {"available_mb": 1}
        self.solver.config.max_batch_size = 100
        batch_size = self.solver._calculate_adaptive_batch_size(total_trajectories=500)
        # Ожидаем, что размер батча будет маленьким, но > 0
        self.assertLess(batch_size, 10)
        self.assertGreater(batch_size, 0)

        # Сценарий 4: Общее количество траекторий меньше, чем max_batch_size
        mock_get_memory_stats.return_value = {"available_mb": 2000}
        self.solver.config.max_batch_size = 100
        batch_size = self.solver._calculate_adaptive_batch_size(total_trajectories=50)
        self.assertEqual(batch_size, 50)

    @patch(
        "new_rebuild.core.cnf.gpu_optimized_euler_solver.GPUOptimizedEulerSolver.batch_integrate"
    )
    def test_batch_integrate_chunked(self, mock_batch_integrate):
        """Тестирует чанковую обработку в batch_integrate_chunked"""
        # Настраиваем мок для возврата корректного результата
        def batch_integrate_side_effect(derivative_fn, states, *args, **kwargs):
            batch_size = states.shape[0]
            return IntegrationResult(
                final_state=torch.rand(batch_size, 32), trajectory=None
            )
        mock_batch_integrate.side_effect = batch_integrate_side_effect

        total_trajectories = 250
        self.solver.config.max_batch_size = 100
        initial_states = torch.rand(total_trajectories, 32)

        # Вызываем тестируемую функцию
        result = self.solver.batch_integrate_chunked(
            derivative_fn=simple_derivative,
            initial_states=initial_states,
            adaptive_batch_size=False,  # Используем фиксированный размер батча для предсказуемости
        )

        # Проверяем, что batch_integrate был вызван 3 раза (250 / 100)
        self.assertEqual(mock_batch_integrate.call_count, 3)

        # Проверяем, что размеры батчей были правильными
        call_args_list = mock_batch_integrate.call_args_list
        self.assertEqual(call_args_list[0][0][1].shape[0], 100)  # Первый батч
        self.assertEqual(call_args_list[1][0][1].shape[0], 100)  # Второй батч
        self.assertEqual(call_args_list[2][0][1].shape[0], 50)  # Третий (остаток)

        # Проверяем, что итоговый результат имеет правильный размер
        self.assertEqual(result.final_state.shape[0], total_trajectories)

    @patch(
        "new_rebuild.core.cnf.gpu_optimized_euler_solver.GPUOptimizedEulerSolver._calculate_adaptive_batch_size"
    )
    def test_batch_integrate_chunked_with_adaptive_batch(
        self, mock_calculate_batch_size
    ):
        """Тестирует чанковую обработку с адаптивным размером батча"""
        # Настраиваем мок для возврата предсказуемого размера батча
        mock_calculate_batch_size.return_value = 75

        # Настраиваем мок для `batch_integrate`
        with patch.object(
            self.solver,
            "batch_integrate",
            MagicMock(return_value=IntegrationResult(final_state=torch.rand(10, 32))),
        ) as mock_integrate:
            total_trajectories = 350
            initial_states = torch.rand(total_trajectories, 32)

            # Вызываем функцию
            self.solver.batch_integrate_chunked(
                derivative_fn=simple_derivative,
                initial_states=initial_states,
                adaptive_batch_size=True,
            )

            # Проверяем, что _calculate_adaptive_batch_size был вызван
            mock_calculate_batch_size.assert_called_once_with(total_trajectories)

            # Проверяем, что batch_integrate вызывался 5 раз (350 / 75)
            self.assertEqual(mock_integrate.call_count, 5)

            # Проверяем размеры чанков
            call_args_list = mock_integrate.call_args_list
            self.assertEqual(call_args_list[0][0][1].shape[0], 75)
            self.assertEqual(call_args_list[1][0][1].shape[0], 75)
            self.assertEqual(call_args_list[2][0][1].shape[0], 75)
            self.assertEqual(call_args_list[3][0][1].shape[0], 75)
            self.assertEqual(call_args_list[4][0][1].shape[0], 50)


if __name__ == "__main__":
    unittest.main()
