"""
Dataset Generator для создания именованных файлов датасетов
===========================================================

Генератор создает готовые файлы датасетов для использования в обучении:
- Интерактивный выбор параметров
- Именованные файлы с метаданными
- Автоматическое управление архивом
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import time
from datetime import datetime
import json

from .config import DatasetConfig, create_dataset_config_from_energy
from .manager import DatasetManager, create_dataset_manager
from ..config import EnergyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GeneratorConfig:
    """Конфигурация для генератора датасетов"""
    
    # Основные параметры
    mode: str = "experiment"  # debug, experiment, production, custom
    target_pairs: int = 5000
    sources: List[str] = field(default_factory=lambda: ["precomputed", "snli"])
    
    # Параметры SNLI
    snli_fraction: float = 0.2
    snli_min_text_length: int = 10
    
    # Обработка данных
    normalize_embeddings: bool = True
    shuffle_data: bool = True
    validate_data: bool = True
    
    # Пути и именование
    output_dir: str = "data/energy_flow/active"
    archive_dir: str = "data/energy_flow/archive"
    name_template: str = "{mode}_{sources}_{count}pairs_{timestamp}"
    
    # Дополнительные опции
    save_text_pairs: bool = True  # Сохранять ли текстовые пары для анализа
    save_metadata: bool = True
    max_file_size_mb: int = 1024  # Максимальный размер файла
    
    def __post_init__(self):
        """Валидация параметров"""
        assert self.mode in ["debug", "experiment", "production", "custom"], \
            f"Invalid mode: {self.mode}"
        assert self.target_pairs > 0, "target_pairs must be > 0"
        assert 0 < self.snli_fraction <= 1.0, "snli_fraction must be in (0, 1]"
        
        # Создаем директории
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.archive_dir).mkdir(parents=True, exist_ok=True)
    
    def get_sources_string(self) -> str:
        """Получить короткое название источников для имени файла"""
        if set(self.sources) == {"precomputed", "snli"}:
            return "mixed"
        elif self.sources == ["snli"]:
            return "snli"
        elif self.sources == ["precomputed"]:
            return "precomputed"
        else:
            return "custom"
    
    def generate_filename(self, actual_count: int) -> str:
        """Генерация имени файла"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sources_str = self.get_sources_string()
        
        filename = self.name_template.format(
            mode=self.mode,
            sources=sources_str,
            count=actual_count,
            timestamp=timestamp
        )
        
        return f"{filename}.pt"


# Предустановленные конфигурации для разных режимов
def create_debug_generator_config() -> GeneratorConfig:
    """Конфигурация для отладки"""
    return GeneratorConfig(
        mode="debug",
        target_pairs=500,
        sources=["precomputed"],  # Только готовые данные для скорости
        snli_fraction=0.05,
        max_file_size_mb=64
    )


def create_experiment_generator_config() -> GeneratorConfig:
    """Конфигурация для экспериментов"""
    return GeneratorConfig(
        mode="experiment", 
        target_pairs=5000,
        sources=["precomputed", "snli"],
        snli_fraction=0.2,
        max_file_size_mb=256
    )


def create_production_generator_config() -> GeneratorConfig:
    """Конфигурация для продакшн обучения"""
    return GeneratorConfig(
        mode="production",
        target_pairs=50000,
        sources=["precomputed", "snli"],
        snli_fraction=0.3,
        max_file_size_mb=1024
    )


class DatasetGenerator:
    """
    Генератор именованных файлов датасетов для energy_flow
    
    Создает готовые файлы датасетов которые можно напрямую загружать
    в training скриптах без сложной интеграции
    """
    
    def __init__(self, config: GeneratorConfig, energy_config: Optional[EnergyConfig] = None):
        """
        Args:
            config: Конфигурация генератора
            energy_config: Основная конфигурация energy_flow
        """
        self.config = config
        self.energy_config = energy_config
        
        # Создаем DatasetManager для генерации
        dataset_config = self._create_dataset_config()
        self.dataset_manager = create_dataset_manager(dataset_config, energy_config)
        
        logger.info(f"🔧 DatasetGenerator initialized: mode={config.mode}, "
                   f"target={config.target_pairs} pairs")
    
    def _create_dataset_config(self) -> DatasetConfig:
        """Создание DatasetConfig на основе GeneratorConfig"""
        dataset_config_params = {
            'dataset_sources': self.config.sources,
            'max_samples_per_source': self.config.target_pairs // len(self.config.sources) if len(self.config.sources) > 1 else self.config.target_pairs,
            'snli_fraction': self.config.snli_fraction,
            'snli_min_text_length': self.config.snli_min_text_length,
            'normalize_embeddings': self.config.normalize_embeddings,
            'shuffle_data': self.config.shuffle_data,
            'validate_embeddings': self.config.validate_data
        }
        
        if self.energy_config:
            return create_dataset_config_from_energy(self.energy_config, **dataset_config_params)
        else:
            return DatasetConfig(**dataset_config_params)
    
    def generate_dataset(self, custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация датасета и сохранение в файл
        
        Args:
            custom_name: Пользовательское имя файла (без расширения)
            
        Returns:
            Информация о созданном датасете
        """
        logger.info(f"🚀 Starting dataset generation: {self.config.mode} mode")
        start_time = time.time()
        
        # Проверяем готовность системы
        validation = self.dataset_manager.validate_setup()
        if not validation['overall_status']:
            raise RuntimeError(f"Dataset system not ready: {validation['errors']}")
        
        # Подготавливаем датасет
        logger.info("📊 Preparing dataset...")
        dataset = self.dataset_manager.prepare_dataset()
        if not dataset:
            raise RuntimeError("Failed to prepare dataset")
        
        actual_count = len(dataset)
        logger.info(f"✅ Dataset prepared: {actual_count} samples")
        
        # Проверяем размер
        estimated_size_mb = self._estimate_file_size(dataset)
        if estimated_size_mb > self.config.max_file_size_mb:
            logger.warning(f"⚠️ Estimated file size {estimated_size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB")
        
        # Генерируем имя файла
        if custom_name:
            filename = f"{custom_name}.pt"
        else:
            filename = self.config.generate_filename(actual_count)
        
        output_path = Path(self.config.output_dir) / filename
        
        # Подготавливаем данные для сохранения
        save_data = self._prepare_save_data(dataset, actual_count, start_time)
        
        # Сохраняем файл
        logger.info(f"💾 Saving dataset: {filename}")
        torch.save(save_data, output_path)
        
        generation_time = time.time() - start_time
        actual_size_mb = output_path.stat().st_size / 1024 / 1024
        
        # Результат генерации
        result = {
            'filename': filename,
            'filepath': str(output_path),
            'sample_count': actual_count,
            'file_size_mb': actual_size_mb,
            'generation_time': generation_time,
            'mode': self.config.mode,
            'sources': self.config.sources,
            'config': self.config.__dict__.copy()
        }
        
        logger.info(f"🎉 Dataset generated successfully:")
        logger.info(f"   File: {filename}")
        logger.info(f"   Samples: {actual_count:,}")
        logger.info(f"   Size: {actual_size_mb:.1f} MB")
        logger.info(f"   Time: {generation_time:.1f}s")
        
        return result
    
    def _prepare_save_data(self, dataset, actual_count: int, start_time: float) -> Dict[str, Any]:
        """Подготовка данных для сохранения"""
        save_data = {
            # Основные данные для обучения
            'input_embeddings': dataset.input_embeddings,
            'target_embeddings': dataset.target_embeddings,
            
            # Метаданные генерации
            'generation_info': {
                'mode': self.config.mode,
                'sources': self.config.sources,
                'target_pairs': self.config.target_pairs,
                'actual_pairs': actual_count,
                'generation_timestamp': datetime.now().isoformat(),
                'generation_time': time.time() - start_time,
                'config': self.config.__dict__.copy()
            },
            
            # Статистика датасета
            'dataset_stats': self.dataset_manager.get_statistics()
        }
        
        # Опционально сохраняем текстовые пары
        if self.config.save_text_pairs:
            save_data['text_pairs'] = dataset.text_pairs
            save_data['metadata'] = dataset.metadata
        
        # Информация о системе
        if self.config.save_metadata:
            save_data['system_info'] = {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device': str(dataset.input_embeddings.device),
                'embedding_dimension': dataset.input_embeddings.shape[1]
            }
        
        return save_data
    
    def _estimate_file_size(self, dataset) -> float:
        """Оценка размера файла в MB"""
        # Размер тензоров эмбеддингов
        embeddings_size = dataset.input_embeddings.numel() + dataset.target_embeddings.numel()
        embeddings_mb = embeddings_size * 4 / 1024 / 1024  # float32 = 4 bytes
        
        # Дополнительные данные (примерно 20% от эмбеддингов)
        total_mb = embeddings_mb * 1.2
        
        return total_mb
    
    def list_available_datasets(self, include_archive: bool = False) -> List[Dict[str, Any]]:
        """Список доступных датасетов"""
        datasets = []
        
        # Активные датасеты
        active_dir = Path(self.config.output_dir)
        if active_dir.exists():
            for file_path in active_dir.glob("*.pt"):
                info = self._get_dataset_info(file_path, "active")
                if info:
                    datasets.append(info)
        
        # Архивные датасеты
        if include_archive:
            archive_dir = Path(self.config.archive_dir)
            if archive_dir.exists():
                for file_path in archive_dir.rglob("*.pt"):  # Рекурсивный поиск
                    info = self._get_dataset_info(file_path, "archive")
                    if info:
                        datasets.append(info)
        
        # Сортируем по времени создания (новые первыми)
        datasets.sort(key=lambda x: x.get('creation_time', 0), reverse=True)
        
        return datasets
    
    def _get_dataset_info(self, file_path: Path, category: str) -> Optional[Dict[str, Any]]:
        """Получение информации о датасете из файла"""
        try:
            # Быстрая загрузка метаданных
            data = torch.load(file_path, map_location='cpu')
            
            # Базовая информация о файле
            info = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'category': category,
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'creation_time': file_path.stat().st_mtime
            }
            
            # Информация из содержимого файла
            if isinstance(data, dict):
                # Размер датасета
                if 'input_embeddings' in data:
                    info['sample_count'] = data['input_embeddings'].shape[0]
                    info['embedding_dimension'] = data['input_embeddings'].shape[1]
                
                # Метаданные генерации
                if 'generation_info' in data:
                    gen_info = data['generation_info']
                    info.update({
                        'mode': gen_info.get('mode', 'unknown'),
                        'sources': gen_info.get('sources', []),
                        'generation_time': gen_info.get('generation_time', 0),
                        'target_pairs': gen_info.get('target_pairs', 0)
                    })
            
            return info
            
        except Exception as e:
            logger.warning(f"❌ Failed to read dataset info from {file_path.name}: {e}")
            return None
    
    def archive_old_datasets(self, days_old: int = 7) -> Dict[str, Any]:
        """Архивирование старых датасетов"""
        logger.info(f"🗂️ Archiving datasets older than {days_old} days...")
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        active_dir = Path(self.config.output_dir)
        archive_dir = Path(self.config.archive_dir)
        
        # Создаем папку для архива по дате
        archive_subdir = archive_dir / datetime.now().strftime("%Y-%m")
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        archived_files = []
        errors = []
        
        if active_dir.exists():
            for file_path in active_dir.glob("*.pt"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        # Перемещаем в архив
                        archive_path = archive_subdir / file_path.name
                        file_path.rename(archive_path)
                        
                        archived_files.append({
                            'filename': file_path.name,
                            'archive_path': str(archive_path)
                        })
                        
                        logger.info(f"📁 Archived: {file_path.name}")
                        
                except Exception as e:
                    errors.append(f"Failed to archive {file_path.name}: {e}")
                    logger.error(f"❌ {errors[-1]}")
        
        result = {
            'archived_count': len(archived_files),
            'archived_files': archived_files,
            'errors': errors,
            'archive_directory': str(archive_subdir)
        }
        
        logger.info(f"✅ Archiving completed: {len(archived_files)} files moved")
        
        return result


def create_dataset_generator(mode: str = "experiment", 
                           energy_config: Optional[EnergyConfig] = None) -> DatasetGenerator:
    """
    Фабричная функция для создания DatasetGenerator
    
    Args:
        mode: Режим генерации (debug, experiment, production, custom)
        energy_config: Конфигурация energy_flow
        
    Returns:
        Настроенный DatasetGenerator
    """
    if mode == "debug":
        config = create_debug_generator_config()
    elif mode == "experiment":
        config = create_experiment_generator_config()
    elif mode == "production":
        config = create_production_generator_config()
    else:
        config = GeneratorConfig(mode="custom")
    
    return DatasetGenerator(config, energy_config)