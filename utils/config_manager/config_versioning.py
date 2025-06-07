"""
Config Versioning - система версионирования и миграции конфигураций

Основные возможности:
- Отслеживание изменений конфигурации
- Автоматическое версионирование
- Миграция между версиями
- История изменений
- Rollback к предыдущим версиям
"""

import os
import json
import yaml
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging


class ChangeType(Enum):
    """Типы изменений в конфигурации"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class ConfigChange:
    """Представляет одно изменение в конфигурации"""
    path: str                           # Путь к измененному значению (dot notation)
    change_type: ChangeType            # Тип изменения
    old_value: Any = None              # Старое значение
    new_value: Any = None              # Новое значение
    timestamp: datetime = field(default_factory=datetime.now)
    user: Optional[str] = None         # Пользователь, сделавший изменение
    description: Optional[str] = None  # Описание изменения


@dataclass
class ConfigVersion:
    """Версия конфигурации"""
    version: str                       # Семантическая версия (e.g., "1.2.3")
    config_hash: str                   # Hash конфигурации для быстрого сравнения
    timestamp: datetime = field(default_factory=datetime.now)
    changes: List[ConfigChange] = field(default_factory=list)
    description: Optional[str] = None  # Описание версии
    is_stable: bool = True            # Стабильная версия или dev
    migration_required: bool = False   # Требуется ли миграция
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        return {
            'version': self.version,
            'config_hash': self.config_hash,
            'timestamp': self.timestamp.isoformat(),
            'changes': [
                {
                    'path': change.path,
                    'change_type': change.change_type.value,
                    'old_value': change.old_value,
                    'new_value': change.new_value,
                    'timestamp': change.timestamp.isoformat(),
                    'user': change.user,
                    'description': change.description
                }
                for change in self.changes
            ],
            'description': self.description,
            'is_stable': self.is_stable,
            'migration_required': self.migration_required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Создание из словаря"""
        changes = [
            ConfigChange(
                path=change_data['path'],
                change_type=ChangeType(change_data['change_type']),
                old_value=change_data.get('old_value'),
                new_value=change_data.get('new_value'),
                timestamp=datetime.fromisoformat(change_data['timestamp']),
                user=change_data.get('user'),
                description=change_data.get('description')
            )
            for change_data in data.get('changes', [])
        ]
        
        return cls(
            version=data['version'],
            config_hash=data['config_hash'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            changes=changes,
            description=data.get('description'),
            is_stable=data.get('is_stable', True),
            migration_required=data.get('migration_required', False)
        )


class ConfigMigration:
    """Базовый класс для миграции конфигурации"""
    
    def __init__(self, from_version: str, to_version: str, description: str = ""):
        self.from_version = from_version
        self.to_version = to_version
        self.description = description
    
    def can_migrate(self, from_version: str, to_version: str) -> bool:
        """Проверка возможности миграции"""
        return (self.from_version == from_version and 
                self.to_version == to_version)
    
    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение миграции"""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Описание миграции"""
        return f"Migration from {self.from_version} to {self.to_version}: {self.description}"


class ConfigVersionManager:
    """
    Менеджер версий конфигурации.
    
    Основные возможности:
    - Отслеживание изменений
    - Создание новых версий
    - Миграция между версиями
    - История изменений
    - Rollback к предыдущим версиям
    """
    
    def __init__(self, 
                 versions_dir: str = "config/versions",
                 current_version: str = "1.0.0",
                 auto_save: bool = True):
        """
        Инициализация менеджера версий.
        
        Args:
            versions_dir: Директория для хранения версий
            current_version: Текущая версия
            auto_save: Автоматическое сохранение изменений
        """
        self.versions_dir = Path(versions_dir)
        self.current_version = current_version
        self.auto_save = auto_save
        self.logger = logging.getLogger(__name__)
        
        # Создаем директорию если не существует
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Внутренние данные
        self._versions: Dict[str, ConfigVersion] = {}
        self._current_config: Optional[Dict[str, Any]] = None
        self._migrations: List[ConfigMigration] = []
        
        # Загружаем существующие версии
        self._load_versions()
        
        self.logger.info(f"📚 ConfigVersionManager initialized with {len(self._versions)} versions")
    
    def track_changes(self, 
                     old_config: Dict[str, Any], 
                     new_config: Dict[str, Any],
                     user: Optional[str] = None,
                     description: Optional[str] = None) -> List[ConfigChange]:
        """
        Отслеживание изменений между конфигурациями.
        
        Args:
            old_config: Старая конфигурация
            new_config: Новая конфигурация
            user: Пользователь, внесший изменения
            description: Описание изменений
            
        Returns:
            Список изменений
        """
        changes = []
        
        # Находим все изменения
        all_paths = set()
        self._collect_paths(old_config, "", all_paths)
        self._collect_paths(new_config, "", all_paths)
        
        for path in all_paths:
            old_value = self._get_nested_value(old_config, path)
            new_value = self._get_nested_value(new_config, path)
            
            if old_value is None and new_value is not None:
                # Добавлено
                changes.append(ConfigChange(
                    path=path,
                    change_type=ChangeType.ADDED,
                    new_value=new_value,
                    user=user,
                    description=description
                ))
            elif old_value is not None and new_value is None:
                # Удалено
                changes.append(ConfigChange(
                    path=path,
                    change_type=ChangeType.DELETED,
                    old_value=old_value,
                    user=user,
                    description=description
                ))
            elif old_value != new_value:
                # Изменено
                changes.append(ConfigChange(
                    path=path,
                    change_type=ChangeType.MODIFIED,
                    old_value=old_value,
                    new_value=new_value,
                    user=user,
                    description=description
                ))
        
        return changes
    
    def create_version(self, 
                      config_data: Dict[str, Any],
                      version: Optional[str] = None,
                      description: Optional[str] = None,
                      is_stable: bool = True,
                      user: Optional[str] = None) -> ConfigVersion:
        """
        Создание новой версии конфигурации.
        
        Args:
            config_data: Данные конфигурации
            version: Версия (если None, будет сгенерирована автоматически)
            description: Описание версии
            is_stable: Стабильная версия
            user: Пользователь
            
        Returns:
            Созданная версия
        """
        # Генерируем версию если не указана
        if version is None:
            version = self._generate_next_version()
        
        # Вычисляем hash конфигурации
        config_hash = self._calculate_config_hash(config_data)
        
        # Находим изменения относительно текущей версии
        changes = []
        if self._current_config is not None:
            changes = self.track_changes(
                self._current_config, 
                config_data, 
                user=user, 
                description=description
            )
        else:
            # Если это первая версия, все поля считаются добавленными
            self._collect_paths(config_data, "", set())
            all_paths = set()
            self._collect_paths(config_data, "", all_paths)
            
            for path in all_paths:
                value = self._get_nested_value(config_data, path)
                if value is not None:
                    changes.append(ConfigChange(
                        path=path,
                        change_type=ChangeType.ADDED,
                        new_value=value,
                        user=user,
                        description=description or "Initial version"
                    ))
        
        # Создаем версию
        config_version = ConfigVersion(
            version=version,
            config_hash=config_hash,
            changes=changes,
            description=description,
            is_stable=is_stable
        )
        
        # Сохраняем версию
        self._versions[version] = config_version
        self._current_config = config_data.copy()
        self.current_version = version
        
        if self.auto_save:
            self._save_version(config_version, config_data)
        
        self.logger.info(f"📌 Created config version {version} with {len(changes)} changes")
        return config_version
    
    def get_version(self, version: str) -> Optional[ConfigVersion]:
        """Получение информации о версии"""
        return self._versions.get(version)
    
    def list_versions(self, stable_only: bool = False) -> List[ConfigVersion]:
        """
        Список всех версий.
        
        Args:
            stable_only: Только стабильные версии
            
        Returns:
            Список версий отсортированный по времени
        """
        versions = list(self._versions.values())
        
        if stable_only:
            versions = [v for v in versions if v.is_stable]
        
        return sorted(versions, key=lambda v: v.timestamp, reverse=True)
    
    def get_config_for_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Загрузка конфигурации для указанной версии.
        
        Args:
            version: Версия конфигурации
            
        Returns:
            Данные конфигурации или None если не найдена
        """
        version_file = self.versions_dir / f"{version}.yaml"
        
        if not version_file.exists():
            self.logger.warning(f"Configuration file for version {version} not found")
            return None
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config for version {version}: {e}")
            return None
    
    def rollback_to_version(self, target_version: str) -> Optional[Dict[str, Any]]:
        """
        Откат к указанной версии.
        
        Args:
            target_version: Целевая версия
            
        Returns:
            Конфигурация целевой версии или None
        """
        if target_version not in self._versions:
            self.logger.error(f"Version {target_version} not found")
            return None
        
        config_data = self.get_config_for_version(target_version)
        if config_data is None:
            return None
        
        # Создаем новую версию как rollback
        rollback_version = self._generate_next_version()
        self.create_version(
            config_data=config_data,
            version=rollback_version,
            description=f"Rollback to version {target_version}",
            is_stable=True
        )
        
        self.logger.info(f"🔄 Rolled back to version {target_version} as {rollback_version}")
        return config_data
    
    def add_migration(self, migration: ConfigMigration):
        """Добавление миграции"""
        self._migrations.append(migration)
        self.logger.info(f"📋 Added migration: {migration.get_description()}")
    
    def migrate_to_version(self, 
                          from_version: str, 
                          to_version: str, 
                          config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Миграция конфигурации между версиями.
        
        Args:
            from_version: Исходная версия
            to_version: Целевая версия
            config_data: Данные для миграции
            
        Returns:
            Мигрированные данные или None если миграция невозможна
        """
        # Ищем подходящую миграцию
        migration = None
        for m in self._migrations:
            if m.can_migrate(from_version, to_version):
                migration = m
                break
        
        if migration is None:
            self.logger.error(f"No migration found from {from_version} to {to_version}")
            return None
        
        try:
            migrated_data = migration.migrate(config_data)
            self.logger.info(f"🔄 Migrated config from {from_version} to {to_version}")
            return migrated_data
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return None
    
    def get_changes_since_version(self, since_version: str) -> List[ConfigChange]:
        """
        Получение всех изменений с указанной версии.
        
        Args:
            since_version: Версия с которой смотреть изменения
            
        Returns:
            Список всех изменений
        """
        all_changes = []
        
        for version in sorted(self._versions.keys()):
            if version > since_version:
                version_obj = self._versions[version]
                all_changes.extend(version_obj.changes)
        
        return all_changes
    
    def export_version_history(self, output_file: str, format: str = "json"):
        """
        Экспорт истории версий.
        
        Args:
            output_file: Файл для сохранения
            format: Формат (json/yaml)
        """
        history = {
            'current_version': self.current_version,
            'versions': [version.to_dict() for version in self._versions.values()],
            'exported_at': datetime.now().isoformat()
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(history, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"📊 Exported version history to {output_file}")
    
    # ========================================
    # PRIVATE METHODS
    # ========================================
    
    def _load_versions(self):
        """Загрузка существующих версий"""
        metadata_file = self.versions_dir / "versions.json"
        
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for version_data in metadata.get('versions', []):
                version = ConfigVersion.from_dict(version_data)
                self._versions[version.version] = version
            
            self.current_version = metadata.get('current_version', '1.0.0')
            
        except Exception as e:
            self.logger.error(f"Error loading versions metadata: {e}")
    
    def _save_version(self, version: ConfigVersion, config_data: Dict[str, Any]):
        """Сохранение версии на диск"""
        # Сохраняем конфигурацию
        config_file = self.versions_dir / f"{version.version}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        # Обновляем метаданные
        self._save_metadata()
    
    def _save_metadata(self):
        """Сохранение метаданных версий"""
        metadata = {
            'current_version': self.current_version,
            'versions': [version.to_dict() for version in self._versions.values()],
            'updated_at': datetime.now().isoformat()
        }
        
        metadata_file = self.versions_dir / "versions.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_next_version(self) -> str:
        """Генерация следующего номера версии"""
        if not self._versions:
            return "1.0.0"
        
        # Находим максимальную версию
        max_version = max(self._versions.keys(), key=lambda v: tuple(map(int, v.split('.'))))
        major, minor, patch = map(int, max_version.split('.'))
        
        # Увеличиваем patch версию
        return f"{major}.{minor}.{patch + 1}"
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Вычисление hash конфигурации"""
        config_str = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def _collect_paths(self, data: Dict[str, Any], prefix: str, paths: set):
        """Сбор всех путей в конфигурации"""
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            
            if isinstance(value, dict):
                self._collect_paths(value, path, paths)
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Получение вложенного значения по пути"""
        if not path:
            return data
        
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


# =============================================================================
# ПРЕДУСТАНОВЛЕННЫЕ МИГРАЦИИ
# =============================================================================

class LatticeV1ToV2Migration(ConfigMigration):
    """Миграция lattice конфигурации с v1.0 на v2.0"""
    
    def __init__(self):
        super().__init__(
            from_version="1.0.0",
            to_version="2.0.0",
            description="Add new connectivity options and boundary conditions"
        )
    
    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение миграции"""
        migrated = config_data.copy()
        
        # Добавляем новые поля если их нет
        if 'lattice_3d' in migrated:
            lattice_config = migrated['lattice_3d']
            
            # Добавляем connectivity если не существует
            if 'connectivity' not in lattice_config:
                lattice_config['connectivity'] = {
                    'type': '6-neighbors',
                    'weight_sharing': True
                }
            
            # Добавляем boundary_conditions если не существует
            if 'boundary_conditions' not in lattice_config:
                lattice_config['boundary_conditions'] = 'walls'
        
        return migrated


# Предустановленные миграции
DEFAULT_MIGRATIONS = [
    LatticeV1ToV2Migration(),
]