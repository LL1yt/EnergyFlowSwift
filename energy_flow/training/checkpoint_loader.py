"""
Простой загрузчик чекпоинтов для energy_flow
===========================================

Обеспечивает загрузку чекпоинтов из активной папки без сложных проверок совместимости.
Полагается на ручное управление пользователем для размещения актуальных чекпоинтов.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.checkpoint_utils import (
    find_latest_checkpoint, find_best_checkpoint, 
    list_checkpoints, parse_checkpoint_name,
    create_checkpoint_summary
)
from ..config import EnergyConfig

logger = get_logger(__name__)


class SimpleCheckpointLoader:
    """
    Простой загрузчик чекпоинтов для energy_flow
    
    Особенности:
    - Загрузка из конкретной папки без автоматического поиска совместимости
    - Поддержка поиска последнего или лучшего чекпоинта
    - Простое логирование и отчетность
    - Ручное управление активными чекпоинтами
    """
    
    def __init__(self, active_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            active_dir: Директория с активными чекпоинтами
                       По умолчанию: checkpoints/energy_flow/active/
        """
        if active_dir is None:
            active_dir = Path("checkpoints/energy_flow/active")
        
        self.active_dir = Path(active_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"SimpleCheckpointLoader initialized: {self.active_dir}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path], 
                       current_config: Optional[EnergyConfig] = None,
                       strict_validation: bool = False) -> Optional[Dict[str, Any]]:
        """
        Загружает конкретный чекпоинт с валидацией конфигурации
        
        Args:
            checkpoint_path: Путь к чекпоинту
            current_config: Текущая конфигурация для валидации (опционально)
            strict_validation: Если True, несовместимость вызывает ошибку, иначе предупреждение
            
        Returns:
            Данные чекпоинта или None при ошибке
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path.name}")
            
            # Загружаем чекпоинт
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Валидация конфигурации если предоставлена текущая конфигурация
            if current_config is not None:
                validation_passed = self._validate_checkpoint_config(checkpoint, current_config, strict_validation)
                if not validation_passed and strict_validation:
                    logger.error(f"Checkpoint validation failed in strict mode. Aborting load.")
                    return None
            
            # Логируем информацию о чекпоинте
            self._log_checkpoint_info(checkpoint_path, checkpoint)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def load_latest_checkpoint(self, current_config: Optional[EnergyConfig] = None,
                              strict_validation: bool = False) -> Optional[Dict[str, Any]]:
        """
        Загружает самый свежий чекпоинт из активной директории
        
        Args:
            current_config: Текущая конфигурация для валидации (опционально)
            strict_validation: Если True, несовместимость вызывает ошибку, иначе предупреждение
        
        Returns:
            Данные чекпоинта или None, если не найден
        """
        latest_path = find_latest_checkpoint(self.active_dir)
        
        if latest_path is None:
            logger.warning(f"No checkpoints found in {self.active_dir}")
            return None
        
        logger.info(f"Found latest checkpoint: {latest_path.name}")
        return self.load_checkpoint(latest_path, current_config, strict_validation)
    
    def load_best_checkpoint(self, current_config: Optional[EnergyConfig] = None,
                            strict_validation: bool = False) -> Optional[Dict[str, Any]]:
        """
        Загружает лучший чекпоинт (с префиксом best_) из активной директории
        
        Args:
            current_config: Текущая конфигурация для валидации (опционально)
            strict_validation: Если True, несовместимость вызывает ошибку, иначе предупреждение
        
        Returns:
            Данные чекпоинта или None, если не найден
        """
        best_path = find_best_checkpoint(self.active_dir)
        
        if best_path is None:
            logger.warning(f"No best checkpoints found in {self.active_dir}")
            return None
        
        logger.info(f"Found best checkpoint: {best_path.name}")
        return self.load_checkpoint(best_path, current_config, strict_validation)
    
    def load_checkpoint_by_pattern(self, pattern: str, 
                                   current_config: Optional[EnergyConfig] = None,
                                   strict_validation: bool = False) -> Optional[Dict[str, Any]]:
        """
        Загружает чекпоинт по паттерну в имени файла
        
        Args:
            pattern: Паттерн для поиска (например, "epoch_050")
            current_config: Текущая конфигурация для валидации (опционально)
            strict_validation: Если True, несовместимость вызывает ошибку, иначе предупреждение
            
        Returns:
            Данные чекпоинта или None, если не найден
        """
        if not self.active_dir.exists():
            logger.warning(f"Active directory not found: {self.active_dir}")
            return None
        
        # Ищем файлы, содержащие паттерн
        matching_files = []
        for file_path in self.active_dir.glob("*.pt"):
            if pattern in file_path.name:
                matching_files.append(file_path)
        
        if not matching_files:
            logger.warning(f"No checkpoints found matching pattern '{pattern}' in {self.active_dir}")
            return None
        
        # Если несколько файлов, берем самый свежий
        if len(matching_files) > 1:
            logger.info(f"Found {len(matching_files)} checkpoints matching '{pattern}', using latest")
            checkpoints_with_time = []
            for file_path in matching_files:
                metadata = parse_checkpoint_name(file_path.name)
                if metadata:
                    checkpoints_with_time.append((file_path, metadata['timestamp']))
            
            if checkpoints_with_time:
                checkpoints_with_time.sort(key=lambda x: x[1], reverse=True)
                selected_path = checkpoints_with_time[0][0]
            else:
                selected_path = matching_files[0]  # Fallback
        else:
            selected_path = matching_files[0]
        
        logger.info(f"Loading checkpoint by pattern '{pattern}': {selected_path.name}")
        return self.load_checkpoint(selected_path, current_config, strict_validation)
    
    def list_available_checkpoints(self) -> list:
        """
        Возвращает список всех доступных чекпоинтов с метаданными
        
        Returns:
            Список кортежей (Path, metadata)
        """
        checkpoints = list_checkpoints(self.active_dir, sort_by="timestamp")
        
        if checkpoints:
            logger.info(f"Found {len(checkpoints)} checkpoints in {self.active_dir}:")
            for i, (path, metadata) in enumerate(checkpoints[:10]):  # Показываем первые 10
                status = "BEST" if metadata['is_best'] else "REG"
                logger.info(f"  {i+1:2d}. [{status}] {path.name} (epoch {metadata['epoch']}, loss {metadata['loss']:.4f})")
        else:
            logger.info(f"No checkpoints found in {self.active_dir}")
        
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Получает информацию о чекпоинте без его загрузки
        
        Args:
            checkpoint_path: Путь к чекпоинту
            
        Returns:
            Информация о чекпоинте или None
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return None
        
        # Получаем метаданные из имени файла
        metadata = parse_checkpoint_name(checkpoint_path.name)
        if not metadata:
            return None
        
        # Создаем summary
        summary = create_checkpoint_summary(checkpoint_path)
        
        return summary
    
    def ensure_active_directory(self) -> bool:
        """
        Создает активную директорию, если она не существует
        
        Returns:
            True если директория существует или была создана
        """
        try:
            self.active_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Active directory ensured: {self.active_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create active directory {self.active_dir}: {e}")
            return False
    
    def _log_checkpoint_info(self, checkpoint_path: Path, checkpoint: Dict[str, Any]):
        """
        Логирует информацию о загруженном чекпоинте
        
        Args:
            checkpoint_path: Путь к чекпоинту
            checkpoint: Данные чекпоинта
        """
        # Базовая информация
        epoch = checkpoint.get('epoch', 'N/A')
        step = checkpoint.get('global_step', 'N/A')
        loss = checkpoint.get('best_loss', 'N/A')
        
        logger.info(f"✅ Checkpoint loaded successfully:")
        logger.info(f"   File: {checkpoint_path.name}")
        logger.info(f"   Epoch: {epoch}, Step: {step}")
        logger.info(f"   Best loss: {loss}")
        
        # Информация о конфигурации
        config = checkpoint.get('config', {})
        if config:
            lattice_info = f"{config.get('lattice_width', '?')}x{config.get('lattice_height', '?')}x{config.get('lattice_depth', '?')}"
            batch_size = config.get('batch_size', '?')
            lr = config.get('learning_rate', '?')
            
            logger.info(f"   Config: lattice={lattice_info}, batch_size={batch_size}, lr={lr}")
        
        # Размер файла
        try:
            file_size = checkpoint_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            logger.info(f"   File size: {size_mb:.1f} MB")
        except:
            pass
        
        # Проверяем наличие text_bridge
        has_text_encoder = 'text_encoder_state_dict' in checkpoint
        has_text_decoder = 'text_decoder_state_dict' in checkpoint
        if has_text_encoder or has_text_decoder:
            logger.info(f"   Text bridge: encoder={has_text_encoder}, decoder={has_text_decoder}")
    
    def _validate_checkpoint_config(self, checkpoint: Dict[str, Any], 
                                   current_config: EnergyConfig, 
                                   strict: bool = False) -> bool:
        """
        Валидирует совместимость конфигурации чекпоинта с текущей конфигурацией
        
        Args:
            checkpoint: Данные чекпоинта
            current_config: Текущая конфигурация
            strict: Если True, любое несоответствие считается критическим
            
        Returns:
            True если валидация прошла успешно, False при критических несоответствиях
        """
        saved_config = checkpoint.get('config', {})
        if not saved_config:
            logger.warning("⚠️ Checkpoint has no config data. Skipping validation.")
            return True
        
        validation_passed = True
        warnings = []
        errors = []
        
        # Критические параметры (размеры решетки) - должны совпадать
        critical_params = [
            ('lattice_width', 'Lattice width'),
            ('lattice_height', 'Lattice height'), 
            ('lattice_depth', 'Lattice depth'),
            ('carrier_hidden_size', 'Carrier hidden size'),
            ('carrier_num_layers', 'Carrier layers'),
            ('neuron_hidden_dim', 'Neuron hidden dim'),
            ('neuron_output_dim', 'Neuron output dim')
        ]
        
        for param_name, display_name in critical_params:
            saved_value = saved_config.get(param_name)
            current_value = getattr(current_config, param_name, None)
            
            if saved_value is not None and current_value is not None:
                if saved_value != current_value:
                    msg = f"{display_name}: saved={saved_value}, current={current_value}"
                    errors.append(msg)
                    validation_passed = False
        
        # Важные параметры (влияют на поведение) - предупреждения
        important_params = [
            ('text_bridge_enabled', 'Text bridge'),
            ('relative_coordinates', 'Relative coordinates'),
            ('center_start_enabled', 'Center start'),
            ('dual_output_planes', 'Dual output planes'),
            ('boundary_reflection_enabled', 'Boundary reflection'),
            ('movement_based_spawn', 'Movement-based spawn'),
            ('enable_displacement_filtering', 'Displacement filtering'),
            ('convergence_enabled', 'Convergence detection')
        ]
        
        for param_name, display_name in important_params:
            saved_value = saved_config.get(param_name)
            current_value = getattr(current_config, param_name, None)
            
            if saved_value is not None and current_value is not None:
                if saved_value != current_value:
                    msg = f"{display_name}: saved={saved_value}, current={current_value}"
                    warnings.append(msg)
                    if strict:
                        validation_passed = False
        
        # Числовые параметры с допустимыми отклонениями
        numeric_params = [
            ('learning_rate', 'Learning rate', 0.0),
            ('batch_size', 'Batch size', 0),
            ('max_active_flows', 'Max active flows', 0),
            ('text_loss_weight', 'Text loss weight', 0.2),  # Допустимое отклонение 20%
            ('exploration_noise', 'Exploration noise', 0.1),
            ('displacement_scale', 'Displacement scale', 1.0)
        ]
        
        for param_name, display_name, tolerance in numeric_params:
            saved_value = saved_config.get(param_name)
            current_value = getattr(current_config, param_name, None)
            
            if saved_value is not None and current_value is not None:
                if tolerance > 0:
                    # Относительная разница для параметров с допуском
                    if saved_value != 0:
                        rel_diff = abs(saved_value - current_value) / abs(saved_value)
                        if rel_diff > tolerance:
                            msg = f"{display_name}: saved={saved_value:.4f}, current={current_value:.4f} (diff={rel_diff:.1%})"
                            warnings.append(msg)
                elif saved_value != current_value:
                    msg = f"{display_name}: saved={saved_value}, current={current_value}"
                    warnings.append(msg)
        
        # Логирование результатов валидации
        if errors:
            logger.error("❌ Checkpoint validation errors:")
            for error in errors:
                logger.error(f"   - {error}")
        
        if warnings:
            logger.warning("⚠️ Checkpoint validation warnings:")
            for warning in warnings:
                logger.warning(f"   - {warning}")
        
        if validation_passed and not errors and not warnings:
            logger.info("✅ Checkpoint config validation passed")
        elif validation_passed:
            logger.info("✅ Checkpoint loaded with warnings")
        
        return validation_passed


def create_checkpoint_loader(active_dir: Optional[Union[str, Path]] = None) -> SimpleCheckpointLoader:
    """
    Фабричная функция для создания SimpleCheckpointLoader
    
    Args:
        active_dir: Директория с активными чекпоинтами
        
    Returns:
        Экземпляр SimpleCheckpointLoader
    """
    return SimpleCheckpointLoader(active_dir)


def quick_load_latest() -> Optional[Dict[str, Any]]:
    """
    Быстрая загрузка последнего чекпоинта из стандартной директории
    
    Returns:
        Данные чекпоинта или None
    """
    loader = create_checkpoint_loader()
    return loader.load_latest_checkpoint()


def quick_load_best() -> Optional[Dict[str, Any]]:
    """
    Быстрая загрузка лучшего чекпоинта из стандартной директории
    
    Returns:
        Данные чекпоинта или None
    """
    loader = create_checkpoint_loader()
    return loader.load_best_checkpoint()