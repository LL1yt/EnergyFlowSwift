"""
Утилиты для умного именования и управления чекпоинтами energy_flow
================================================================

Обеспечивает консистентное именование чекпоинтов с информативными именами
для легкой навигации и понимания состояния модели.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import re

from ..config import EnergyConfig


def generate_checkpoint_name(
    config: EnergyConfig,
    epoch: int,
    loss: float,
    is_best: bool = False,
    custom_suffix: Optional[str] = None
) -> str:
    """
    Генерирует умное имя для чекпоинта на основе конфигурации и метрик
    
    Args:
        config: EnergyConfig с параметрами модели
        epoch: Номер эпохи
        loss: Значение loss
        is_best: Является ли чекпоинт лучшим
        custom_suffix: Дополнительный суффикс для имени
        
    Returns:
        Имя файла чекпоинта в формате:
        [best_]experiment_config_YYYYMMDD_HHMMSS_epoch_N_loss_X.XX[_suffix].pt
    """
    # Определяем тип конфигурации
    config_type = _get_config_type(config)
    
    # Формируем timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Формируем базовое имя
    base_parts = []
    
    if is_best:
        base_parts.append("best")
    
    base_parts.extend([
        config_type,
        "config",
        timestamp,
        f"epoch_{epoch:03d}",
        f"loss_{loss:.4f}".replace(".", "_")
    ])
    
    if custom_suffix:
        base_parts.append(custom_suffix)
    
    filename = "_".join(base_parts) + ".pt"
    
    return filename


def generate_checkpoint_path(
    config: EnergyConfig,
    epoch: int,
    loss: float,
    base_dir: Optional[Path] = None,
    is_best: bool = False,
    custom_suffix: Optional[str] = None
) -> Path:
    """
    Генерирует полный путь для чекпоинта
    
    Args:
        config: EnergyConfig с параметрами модели
        epoch: Номер эпохи
        loss: Значение loss
        base_dir: Базовая директория (по умолчанию checkpoints/energy_flow)
        is_best: Является ли чекпоинт лучшим
        custom_suffix: Дополнительный суффикс для имени
        
    Returns:
        Полный путь к файлу чекпоинта
    """
    if base_dir is None:
        base_dir = Path("checkpoints/energy_flow")
    
    filename = generate_checkpoint_name(
        config=config,
        epoch=epoch,
        loss=loss,
        is_best=is_best,
        custom_suffix=custom_suffix
    )
    
    return base_dir / filename


def parse_checkpoint_name(filename: str) -> Optional[Dict[str, Any]]:
    """
    Парсит имя чекпоинта и извлекает метаданные
    
    Args:
        filename: Имя файла чекпоинта
        
    Returns:
        Словарь с метаданными или None, если формат не распознан
    """
    # Убираем расширение
    name = Path(filename).stem
    
    # Паттерн для парсинга: [best_]config_type_config_timestamp_epoch_N_loss_X_XX[_suffix]
    pattern = r"^(?:(best)_)?(\w+)_config_(\d{8}_\d{6})_epoch_(\d+)_loss_(\d+_\d+)(?:_(.+))?$"
    
    match = re.match(pattern, name)
    if not match:
        return None
    
    is_best, config_type, timestamp_str, epoch_str, loss_str, suffix = match.groups()
    
    # Парсим timestamp
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    
    # Парсим loss (заменяем последний _ на .)
    loss_parts = loss_str.split('_')
    if len(loss_parts) >= 2:
        loss = float(f"{loss_parts[0]}.{loss_parts[1]}")
    else:
        return None
    
    return {
        'is_best': bool(is_best),
        'config_type': config_type,
        'timestamp': timestamp,
        'epoch': int(epoch_str),
        'loss': loss,
        'suffix': suffix,
        'original_filename': filename
    }


def find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """
    Находит самый свежий чекпоинт в директории по timestamp в имени
    
    Args:
        directory: Директория для поиска
        
    Returns:
        Путь к самому свежему чекпоинту или None
    """
    if not directory.exists():
        return None
    
    checkpoints = []
    
    for file_path in directory.glob("*.pt"):
        metadata = parse_checkpoint_name(file_path.name)
        if metadata:
            checkpoints.append((file_path, metadata['timestamp']))
    
    if not checkpoints:
        return None
    
    # Сортируем по timestamp (самый свежий первый)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    return checkpoints[0][0]


def find_best_checkpoint(directory: Path) -> Optional[Path]:
    """
    Находит лучший чекпоинт (с префиксом best_) в директории
    
    Args:
        directory: Директория для поиска
        
    Returns:
        Путь к лучшему чекпоинту или None
    """
    if not directory.exists():
        return None
    
    best_checkpoints = []
    
    for file_path in directory.glob("best_*.pt"):
        metadata = parse_checkpoint_name(file_path.name)
        if metadata and metadata['is_best']:
            best_checkpoints.append((file_path, metadata['loss'], metadata['timestamp']))
    
    if not best_checkpoints:
        return None
    
    # Сортируем по loss (лучший первый), потом по timestamp
    best_checkpoints.sort(key=lambda x: (x[1], -x[2].timestamp()))
    
    return best_checkpoints[0][0]


def list_checkpoints(directory: Path, sort_by: str = "timestamp") -> list:
    """
    Возвращает список всех чекпоинтов с метаданными
    
    Args:
        directory: Директория для поиска
        sort_by: Поле для сортировки ("timestamp", "loss", "epoch")
        
    Returns:
        Список кортежей (Path, metadata)
    """
    if not directory.exists():
        return []
    
    checkpoints = []
    
    for file_path in directory.glob("*.pt"):
        metadata = parse_checkpoint_name(file_path.name)
        if metadata:
            checkpoints.append((file_path, metadata))
    
    # Сортировка
    if sort_by == "timestamp":
        checkpoints.sort(key=lambda x: x[1]['timestamp'], reverse=True)
    elif sort_by == "loss":
        checkpoints.sort(key=lambda x: x[1]['loss'])
    elif sort_by == "epoch":
        checkpoints.sort(key=lambda x: x[1]['epoch'], reverse=True)
    
    return checkpoints


def cleanup_old_checkpoints(
    directory: Path,
    keep_best: int = 3,
    keep_latest: int = 5
) -> list:
    """
    Очищает старые чекпоинты, оставляя только лучшие и самые свежие
    
    Args:
        directory: Директория для очистки
        keep_best: Количество лучших чекпоинтов для сохранения
        keep_latest: Количество последних чекпоинтов для сохранения
        
    Returns:
        Список удаленных файлов
    """
    if not directory.exists():
        return []
    
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        return []
    
    # Собираем чекпоинты, которые нужно сохранить
    to_keep = set()
    
    # Лучшие чекпоинты (по loss)
    best_checkpoints = [cp for cp in checkpoints if cp[1]['is_best']]
    best_checkpoints.sort(key=lambda x: x[1]['loss'])
    for checkpoint, _ in best_checkpoints[:keep_best]:
        to_keep.add(checkpoint)
    
    # Последние чекпоинты (по timestamp)
    latest_checkpoints = list_checkpoints(directory, sort_by="timestamp")
    for checkpoint, _ in latest_checkpoints[:keep_latest]:
        to_keep.add(checkpoint)
    
    # Удаляем остальные
    deleted = []
    for checkpoint, metadata in checkpoints:
        if checkpoint not in to_keep:
            try:
                checkpoint.unlink()
                deleted.append(str(checkpoint))
            except Exception as e:
                print(f"Не удалось удалить {checkpoint}: {e}")
    
    return deleted


def _get_config_type(config: EnergyConfig) -> str:
    """
    Определяет тип конфигурации по её параметрам
    
    Args:
        config: EnergyConfig для анализа
        
    Returns:
        Строка с типом конфигурации
    """
    # Анализируем размеры решетки для определения типа
    total_cells = config.total_cells
    
    if total_cells <= 20 * 20 * 10:  # До 4000 клеток
        return "debug"
    elif total_cells <= 50 * 50 * 20:  # До 50000 клеток
        return "experiment"
    else:
        return "optimized"


def create_checkpoint_summary(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Создает summary чекпоинта для логирования
    
    Args:
        checkpoint_path: Путь к чекпоинту
        
    Returns:
        Словарь с информацией о чекпоинте
    """
    metadata = parse_checkpoint_name(checkpoint_path.name)
    if not metadata:
        return {'error': 'Invalid checkpoint name format'}
    
    # Получаем размер файла
    try:
        file_size = checkpoint_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
    except:
        size_mb = 0
    
    return {
        'name': checkpoint_path.name,
        'config_type': metadata['config_type'],
        'epoch': metadata['epoch'],
        'loss': metadata['loss'],
        'is_best': metadata['is_best'],
        'timestamp': metadata['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
        'size_mb': round(size_mb, 2),
        'suffix': metadata.get('suffix', None)
    }