#!/usr/bin/env python3
"""
Dataset Management Utilities
=============================

Утилиты для управления датасетами energy_flow:
- Анализ и статистика датасетов
- Очистка дублей и поврежденных файлов
- Управление архивом
- Валидация данных
"""

import sys
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

# Добавляем корень проекта в path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from energy_flow.dataset.generator import create_dataset_generator
from energy_flow.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """Менеджер для управления датасетами energy_flow"""
    
    def __init__(self, active_dir: str = "data/energy_flow/active", 
                 archive_dir: str = "data/energy_flow/archive"):
        self.active_dir = Path(active_dir)
        self.archive_dir = Path(archive_dir)
        
        # Создаем директории если не существуют
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self, include_archive: bool = False, detailed: bool = False) -> List[Dict[str, Any]]:
        """Получить список всех датасетов с метаданными"""
        datasets = []
        
        # Активные датасеты
        for file_path in self.active_dir.glob("*.pt"):
            info = self._analyze_dataset(file_path, "active", detailed)
            if info:
                datasets.append(info)
        
        # Архивные датасеты
        if include_archive:
            for file_path in self.archive_dir.rglob("*.pt"):
                info = self._analyze_dataset(file_path, "archive", detailed)
                if info:
                    datasets.append(info)
        
        # Сортируем по времени создания
        datasets.sort(key=lambda x: x.get('creation_time', 0), reverse=True)
        
        return datasets
    
    def _analyze_dataset(self, file_path: Path, category: str, detailed: bool = False) -> Optional[Dict[str, Any]]:
        """Анализ одного файла датасета"""
        try:
            # Базовая информация о файле
            stat = file_path.stat()
            info = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'category': category,
                'file_size_mb': stat.st_size / 1024 / 1024,
                'creation_time': stat.st_mtime,
                'creation_date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if not detailed:
                return info
            
            # Детальный анализ содержимого
            try:
                data = torch.load(file_path, map_location='cpu')
                
                if isinstance(data, dict):
                    # Информация о размере данных  
                    if 'input_embeddings' in data and 'target_embeddings' in data:
                        input_emb = data['input_embeddings']
                        target_emb = data['target_embeddings']
                        
                        info.update({
                            'sample_count': input_emb.shape[0],
                            'embedding_dimension': input_emb.shape[1],
                            'input_shape': list(input_emb.shape),
                            'target_shape': list(target_emb.shape),
                        })
                        
                        # Статистика эмбеддингов
                        info.update({
                            'input_norm_mean': input_emb.norm(dim=1).mean().item(),
                            'input_norm_std': input_emb.norm(dim=1).std().item(),
                            'target_norm_mean': target_emb.norm(dim=1).mean().item(),
                            'target_norm_std': target_emb.norm(dim=1).std().item(),
                        })
                    
                    # Метаданные генерации
                    if 'generation_info' in data:
                        gen_info = data['generation_info']
                        info.update({
                            'mode': gen_info.get('mode', 'unknown'),
                            'sources': gen_info.get('sources', []),
                            'generation_time': gen_info.get('generation_time', 0),
                            'target_pairs': gen_info.get('target_pairs', 0),
                            'generation_timestamp': gen_info.get('generation_timestamp', 'unknown')
                        })
                    
                    # Системная информация
                    if 'system_info' in data:
                        sys_info = data['system_info']
                        info.update({
                            'torch_version': sys_info.get('torch_version', 'unknown'),
                            'device_used': sys_info.get('device', 'unknown'),
                            'cuda_available': sys_info.get('cuda_available', False)
                        })
                    
                    # Статистика датасета  
                    if 'dataset_stats' in data:
                        ds_stats = data['dataset_stats']
                        info['dataset_stats'] = ds_stats
                    
                    # Hash для обнаружения дублей
                    info['content_hash'] = self._calculate_content_hash(data)
                    
                else:
                    info['data_type'] = type(data).__name__
                    info['valid_format'] = False
                    
            except Exception as e:
                info['load_error'] = str(e)
                info['valid_format'] = False
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path.name}: {e}")
            return None
    
    def _calculate_content_hash(self, data: Dict) -> str:
        """Вычисление хэша содержимого для обнаружения дублей"""
        try:
            # Используем форму тензоров и некоторые метаданные для хэша
            hash_data = {
                'input_shape': data['input_embeddings'].shape if 'input_embeddings' in data else None,
                'target_shape': data['target_embeddings'].shape if 'target_embeddings' in data else None,
                'sample_values': data['input_embeddings'][:5].sum().item() if 'input_embeddings' in data else 0
            }
            
            hash_str = json.dumps(hash_data, sort_keys=True)
            return hashlib.md5(hash_str.encode()).hexdigest()[:8]
            
        except Exception:
            return "unknown"
    
    def get_dataset_info(self, filename: str, detailed: bool = True) -> Optional[Dict[str, Any]]:
        """Получить детальную информацию о конкретном датасете"""
        # Ищем в активных
        active_path = self.active_dir / filename
        if active_path.exists():
            return self._analyze_dataset(active_path, "active", detailed)
        
        # Ищем в архиве
        for file_path in self.archive_dir.rglob(filename):
            if file_path.name == filename:
                return self._analyze_dataset(file_path, "archive", detailed)
        
        return None
    
    def find_duplicates(self) -> List[List[Dict[str, Any]]]:
        """Поиск дублирующихся датасетов по содержимому"""
        logger.info("🔍 Searching for duplicate datasets...")
        
        datasets = self.list_datasets(include_archive=True, detailed=True)
        
        # Группируем по хэшу содержимого
        hash_groups = {}
        for dataset in datasets:
            content_hash = dataset.get('content_hash', 'unknown')
            if content_hash != 'unknown':
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(dataset)
        
        # Возвращаем группы с дублями
        duplicates = [group for group in hash_groups.values() if len(group) > 1]
        
        logger.info(f"Found {len(duplicates)} groups of duplicates")
        return duplicates
    
    def cleanup_duplicates(self, keep_newest: bool = True, dry_run: bool = True) -> Dict[str, Any]:
        """Очистка дублирующихся файлов"""
        duplicates = self.find_duplicates()
        
        if not duplicates:
            return {'removed_count': 0, 'saved_space_mb': 0, 'errors': []}
        
        removed_files = []
        errors = []
        saved_space_mb = 0
        
        for group in duplicates:
            try:
                # Сортируем по времени создания
                sorted_group = sorted(group, key=lambda x: x['creation_time'], reverse=keep_newest)
                
                # Оставляем первый (самый новый/старый), удаляем остальные
                keep_file = sorted_group[0]
                remove_files = sorted_group[1:]
                
                logger.info(f"Keeping: {keep_file['filename']}")
                
                for file_info in remove_files:
                    file_path = Path(file_info['filepath'])
                    
                    if dry_run:
                        logger.info(f"Would remove: {file_info['filename']} ({file_info['file_size_mb']:.1f} MB)")
                        saved_space_mb += file_info['file_size_mb']
                    else:
                        try:
                            file_path.unlink()
                            removed_files.append(file_info['filename'])
                            saved_space_mb += file_info['file_size_mb']
                            logger.info(f"Removed: {file_info['filename']}")
                        except Exception as e:
                            error_msg = f"Failed to remove {file_info['filename']}: {e}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                            
            except Exception as e:
                error_msg = f"Error processing duplicate group: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            'removed_count': len(removed_files),
            'removed_files': removed_files,
            'saved_space_mb': saved_space_mb,
            'errors': errors,
            'dry_run': dry_run
        }
    
    def validate_datasets(self) -> Dict[str, Any]:
        """Валидация всех датасетов на целостность"""
        logger.info("🔍 Validating all datasets...")
        
        datasets = self.list_datasets(include_archive=True, detailed=True)
        
        valid_datasets = []
        invalid_datasets = []
        
        for dataset in datasets:
            if dataset.get('valid_format', True) and dataset.get('sample_count', 0) > 0:
                valid_datasets.append(dataset)
            else:
                invalid_datasets.append(dataset)
        
        return {
            'total_datasets': len(datasets),
            'valid_count': len(valid_datasets),
            'invalid_count': len(invalid_datasets),
            'invalid_datasets': invalid_datasets
        }
    
    def archive_by_pattern(self, pattern: str, max_age_days: Optional[int] = None) -> Dict[str, Any]:
        """Архивирование датасетов по шаблону имени или возрасту"""
        logger.info(f"📁 Archiving datasets matching pattern: {pattern}")
        
        archived_files = []
        errors = []
        
        # Создаем папку архива по дате
        archive_subdir = self.archive_dir / datetime.now().strftime("%Y-%m")
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60) if max_age_days else 0
        
        for file_path in self.active_dir.glob(pattern):
            try:
                # Проверяем возраст если указан
                if max_age_days and file_path.stat().st_mtime > cutoff_time:
                    continue
                
                # Перемещаем в архив
                archive_path = archive_subdir / file_path.name
                file_path.rename(archive_path)
                
                archived_files.append({
                    'filename': file_path.name,
                    'archive_path': str(archive_path)
                })
                
                logger.info(f"Archived: {file_path.name}")
                
            except Exception as e:
                error_msg = f"Failed to archive {file_path.name}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            'archived_count': len(archived_files),
            'archived_files': archived_files,
            'archive_directory': str(archive_subdir),
            'errors': errors
        }
    
    def generate_report(self) -> str:
        """Генерация подробного отчета о состоянии датасетов"""
        datasets = self.list_datasets(include_archive=True, detailed=True)
        
        # Общая статистика
        active_datasets = [d for d in datasets if d['category'] == 'active']
        archive_datasets = [d for d in datasets if d['category'] == 'archive']
        
        total_size_mb = sum(d['file_size_mb'] for d in datasets)
        active_size_mb = sum(d['file_size_mb'] for d in active_datasets)
        archive_size_mb = sum(d['file_size_mb'] for d in archive_datasets)
        
        # Статистика по режимам
        mode_stats = {}
        for dataset in datasets:
            mode = dataset.get('mode', 'unknown')
            if mode not in mode_stats:
                mode_stats[mode] = {'count': 0, 'size_mb': 0}
            mode_stats[mode]['count'] += 1
            mode_stats[mode]['size_mb'] += dataset['file_size_mb']
        
        # Статистика по источникам
        source_stats = {}
        for dataset in datasets:
            sources = dataset.get('sources', [])
            sources_key = '+'.join(sorted(sources)) if sources else 'unknown'
            if sources_key not in source_stats:
                source_stats[sources_key] = {'count': 0, 'size_mb': 0}
            source_stats[sources_key]['count'] += 1
            source_stats[sources_key]['size_mb'] += dataset['file_size_mb']
        
        # Формируем отчет
        report_lines = [
            "="*60,
            "ENERGY FLOW DATASETS MANAGEMENT REPORT",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 OVERVIEW:",
            f"  Total datasets: {len(datasets)}",
            f"  Active: {len(active_datasets)} ({active_size_mb:.1f} MB)",
            f"  Archive: {len(archive_datasets)} ({archive_size_mb:.1f} MB)",
            f"  Total size: {total_size_mb:.1f} MB",
            "",
            "🏷️ BY MODE:",
        ]
        
        for mode, stats in sorted(mode_stats.items()):
            report_lines.append(f"  {mode}: {stats['count']} files ({stats['size_mb']:.1f} MB)")
        
        report_lines.extend([
            "",
            "📋 BY SOURCES:",
        ])
        
        for sources, stats in sorted(source_stats.items()):
            report_lines.append(f"  {sources}: {stats['count']} files ({stats['size_mb']:.1f} MB)")
        
        # Последние активные датасеты
        if active_datasets:
            report_lines.extend([
                "",
                "🕒 RECENT ACTIVE DATASETS:",
            ])
            
            for dataset in active_datasets[:5]:
                report_lines.append(
                    f"  {dataset['filename']} "
                    f"({dataset.get('sample_count', 'N/A'):,} samples, "
                    f"{dataset['file_size_mb']:.1f} MB, "
                    f"{dataset['creation_date']})"
                )
        
        report_lines.extend([
            "",
            "="*60
        ])
        
        return "\n".join(report_lines)


def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Energy Flow Dataset Management Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--list-archive", action="store_true", help="Include archive in listing")
    parser.add_argument("--info", metavar="FILENAME", help="Show detailed info about specific dataset")
    parser.add_argument("--duplicates", action="store_true", help="Find duplicate datasets")
    parser.add_argument("--cleanup", action="store_true", help="Remove duplicate datasets")
    parser.add_argument("--validate", action="store_true", help="Validate all datasets")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--archive-pattern", metavar="PATTERN", help="Archive datasets matching pattern")
    parser.add_argument("--archive-older", type=int, metavar="DAYS", help="Archive datasets older than N days")
    parser.add_argument("--active-dir", default="data/energy_flow/active", help="Active datasets directory")
    parser.add_argument("--archive-dir", default="data/energy_flow/archive", help="Archive directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't actually delete files)")
    
    args = parser.parse_args()
    
    # Создаем менеджер
    manager = DatasetManager(args.active_dir, args.archive_dir)
    
    try:
        if args.list or args.list_archive:
            datasets = manager.list_datasets(include_archive=args.list_archive, detailed=True)
            
            if not datasets:
                print("📭 No datasets found")
                return
            
            print(f"📋 Found {len(datasets)} datasets:")
            print("-" * 50)
            
            for dataset in datasets:
                status = "🟢" if dataset['category'] == 'active' else "📦"
                print(f"{status} {dataset['filename']}")
                print(f"   📊 {dataset.get('sample_count', 'N/A'):,} samples, "
                      f"{dataset['file_size_mb']:.1f} MB")
                print(f"   🏷️ {dataset.get('mode', 'unknown')} / "
                      f"{', '.join(dataset.get('sources', []))}")
                print(f"   🕒 {dataset['creation_date']}")
                print()
        
        elif args.info:
            info = manager.get_dataset_info(args.info, detailed=True)
            if info:
                print(f"📋 Dataset Information: {args.info}")
                print("-" * 50)
                print(json.dumps(info, indent=2, default=str))
            else:
                print(f"❌ Dataset not found: {args.info}")
        
        elif args.duplicates:
            duplicates = manager.find_duplicates()
            if duplicates:
                print(f"🔍 Found {len(duplicates)} groups of duplicates:")
                for i, group in enumerate(duplicates, 1):
                    print(f"\nGroup {i}:")
                    for dataset in group:
                        print(f"  - {dataset['filename']} ({dataset['category']}, "
                              f"{dataset['file_size_mb']:.1f} MB)")
            else:
                print("✅ No duplicates found")
        
        elif args.cleanup:
            result = manager.cleanup_duplicates(dry_run=args.dry_run)
            
            if args.dry_run:
                print(f"🔍 Dry run - would remove {result['removed_count']} files")
                print(f"💾 Would save {result['saved_space_mb']:.1f} MB")
            else:
                print(f"🗑️ Removed {result['removed_count']} duplicate files")
                print(f"💾 Saved {result['saved_space_mb']:.1f} MB")
            
            if result['errors']:
                print(f"\n⚠️ Errors ({len(result['errors'])}):")
                for error in result['errors']:
                    print(f"  ❌ {error}")
        
        elif args.validate:
            result = manager.validate_datasets()
            print(f"🔍 Dataset Validation Results:")
            print(f"  Total: {result['total_datasets']}")
            print(f"  Valid: {result['valid_count']} ✅")
            print(f"  Invalid: {result['invalid_count']} ❌")
            
            if result['invalid_datasets']:
                print(f"\n❌ Invalid datasets:")
                for dataset in result['invalid_datasets']:
                    reason = dataset.get('load_error', 'Unknown error')
                    print(f"  - {dataset['filename']}: {reason}")
        
        elif args.report:
            report = manager.generate_report()
            print(report)
        
        elif args.archive_pattern:
            result = manager.archive_by_pattern(
                args.archive_pattern, 
                args.archive_older
            )
            
            print(f"📁 Archived {result['archived_count']} files")
            if result['archived_files']:
                for file_info in result['archived_files']:
                    print(f"  📁 {file_info['filename']}")
            
            if result['errors']:
                print(f"\n⚠️ Errors:")
                for error in result['errors']:
                    print(f"  ❌ {error}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error in dataset management: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()