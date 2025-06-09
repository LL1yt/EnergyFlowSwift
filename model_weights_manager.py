#!/usr/bin/env python3
"""
Менеджер весов модели - правильное сохранение/загрузка/отслеживание весов
"""

import torch
import os
import json
from datetime import datetime
from pathlib import Path
import shutil
import hashlib

class ModelWeightsManager:
    """Менеджер для работы с весами модели"""
    
    def __init__(self, base_dir="checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Создаем структуру папок
        self.latest_dir = self.base_dir / "latest"
        self.versioned_dir = self.base_dir / "versioned"
        self.backups_dir = self.base_dir / "backups"
        
        for dir_path in [self.latest_dir, self.versioned_dir, self.backups_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"📁 Model Weights Manager initialized: {self.base_dir}")
    
    def save_latest_weights(self, trainer, config, metadata=None):
        """Сохранить последние веса (перезаписывает предыдущие)"""
        
        # Резервируем предыдущие веса если существуют
        latest_path = self.latest_dir / "trainer_latest.pt"
        if latest_path.exists():
            backup_name = f"trainer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy2(latest_path, self.backups_dir / backup_name)
            print(f"   📦 Предыдущие веса сохранены в backup: {backup_name}")
        
        # Создаем полные metadata
        full_metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_params': sum(p.numel() for p in trainer.parameters()),
            'trainable_params': sum(p.numel() for p in trainer.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in trainer.parameters()) / (1024**2),
            'config_hash': self._hash_config(config),
            'custom_metadata': metadata or {}
        }
        
        # Сохраняем веса
        save_data = {
            'model_state_dict': trainer.state_dict(),
            'config': config,
            'metadata': full_metadata
        }
        
        torch.save(save_data, latest_path)
        
        # Сохраняем readable metadata
        metadata_path = self.latest_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Последние веса сохранены: {latest_path}")
        print(f"   Параметров: {full_metadata['trainable_params']:,}")
        print(f"   Размер: {full_metadata['model_size_mb']:.1f} MB")
        
        return latest_path
    
    def save_versioned_weights(self, trainer, config, version_name, metadata=None):
        """Сохранить версионированные веса (для milestone)"""
        
        version_dir = self.versioned_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        full_metadata = {
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'model_params': sum(p.numel() for p in trainer.parameters()),
            'trainable_params': sum(p.numel() for p in trainer.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in trainer.parameters()) / (1024**2),
            'config_hash': self._hash_config(config),
            'custom_metadata': metadata or {}
        }
        
        # Сохраняем веса
        weights_path = version_dir / f"trainer_{version_name}.pt"
        save_data = {
            'model_state_dict': trainer.state_dict(),
            'config': config,
            'metadata': full_metadata
        }
        
        torch.save(save_data, weights_path)
        
        # Сохраняем metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"🏷️ Версионированные веса сохранены: {weights_path}")
        print(f"   Версия: {version_name}")
        
        return weights_path
    
    def load_latest_weights(self, trainer):
        """Загрузить последние веса"""
        latest_path = self.latest_dir / "trainer_latest.pt"
        
        if not latest_path.exists():
            print(f"⚠️ Последние веса не найдены: {latest_path}")
            return None
        
        return self._load_weights(trainer, latest_path)
    
    def load_versioned_weights(self, trainer, version_name):
        """Загрузить версионированные веса"""
        weights_path = self.versioned_dir / version_name / f"trainer_{version_name}.pt"
        
        if not weights_path.exists():
            print(f"⚠️ Версионированные веса не найдены: {weights_path}")
            return None
        
        return self._load_weights(trainer, weights_path)
    
    def _load_weights(self, trainer, weights_path):
        """Внутренний метод загрузки весов"""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Проверяем совместимость
            current_state = trainer.state_dict()
            loaded_state = checkpoint['model_state_dict']
            
            # Проверяем совпадение ключей
            current_keys = set(current_state.keys())
            loaded_keys = set(loaded_state.keys())
            
            if current_keys != loaded_keys:
                missing = current_keys - loaded_keys
                extra = loaded_keys - current_keys
                
                print(f"⚠️ Несовместимость весов:")
                if missing:
                    print(f"   Отсутствующие ключи: {missing}")
                if extra:
                    print(f"   Лишние ключи: {extra}")
                
                return None
            
            # Загружаем веса
            trainer.load_state_dict(loaded_state)
            
            metadata = checkpoint.get('metadata', {})
            config = checkpoint.get('config', {})
            
            print(f"✅ Веса загружены: {weights_path}")
            print(f"   Timestamp: {metadata.get('timestamp', 'unknown')}")
            print(f"   Параметров: {metadata.get('trainable_params', 'unknown'):,}")
            
            return {
                'config': config,
                'metadata': metadata,
                'path': weights_path
            }
            
        except Exception as e:
            print(f"❌ Ошибка загрузки весов: {e}")
            return None
    
    def list_available_weights(self):
        """Список доступных весов"""
        print(f"\n📋 ДОСТУПНЫЕ ВЕСА:")
        
        # Latest weights
        latest_path = self.latest_dir / "trainer_latest.pt"
        if latest_path.exists():
            metadata_path = self.latest_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"   🔄 LATEST:")
                print(f"      Path: {latest_path}")
                print(f"      Timestamp: {metadata.get('timestamp', 'unknown')}")
                print(f"      Parameters: {metadata.get('trainable_params', 'unknown'):,}")
                print(f"      Size: {metadata.get('model_size_mb', 'unknown')} MB")
            else:
                print(f"   🔄 LATEST: {latest_path} (no metadata)")
        else:
            print(f"   🔄 LATEST: не найден")
        
        # Versioned weights
        print(f"\n   🏷️ VERSIONED:")
        version_dirs = [d for d in self.versioned_dir.iterdir() if d.is_dir()]
        
        if version_dirs:
            for version_dir in sorted(version_dirs):
                version_name = version_dir.name
                weights_path = version_dir / f"trainer_{version_name}.pt"
                metadata_path = version_dir / "metadata.json"
                
                if weights_path.exists():
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        print(f"      {version_name}:")
                        print(f"         Path: {weights_path}")
                        print(f"         Timestamp: {metadata.get('timestamp', 'unknown')}")
                        print(f"         Parameters: {metadata.get('trainable_params', 'unknown'):,}")
                    else:
                        print(f"      {version_name}: {weights_path} (no metadata)")
        else:
            print(f"      Нет версионированных весов")
        
        # Backups
        print(f"\n   📦 BACKUPS:")
        backup_files = list(self.backups_dir.glob("trainer_backup_*.pt"))
        if backup_files:
            for backup_file in sorted(backup_files, reverse=True)[:5]:  # Показываем последние 5
                print(f"      {backup_file.name}")
        else:
            print(f"      Нет backup файлов")
    
    def create_training_checkpoint(self, trainer, config, epoch, loss, similarity, metadata=None):
        """Создать checkpoint в процессе обучения"""
        checkpoint_name = f"epoch_{epoch:03d}_loss_{loss:.4f}_sim_{similarity:.3f}"
        
        training_metadata = {
            'epoch': epoch,
            'loss': loss,
            'similarity': similarity,
            'training_stage': 'in_progress'
        }
        
        if metadata:
            training_metadata.update(metadata)
        
        return self.save_versioned_weights(trainer, config, checkpoint_name, training_metadata)
    
    def create_milestone_checkpoint(self, trainer, config, milestone_name, results, metadata=None):
        """Создать checkpoint для важного milestone"""
        milestone_metadata = {
            'milestone': milestone_name,
            'results': results,
            'training_stage': 'milestone'
        }
        
        if metadata:
            milestone_metadata.update(metadata)
        
        return self.save_versioned_weights(trainer, config, f"milestone_{milestone_name}", milestone_metadata)
    
    def _hash_config(self, config):
        """Создать hash config для отслеживания изменений"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def cleanup_old_backups(self, keep_last=10):
        """Очистка старых backup файлов"""
        backup_files = list(self.backups_dir.glob("trainer_backup_*.pt"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(backup_files) > keep_last:
            old_backups = backup_files[keep_last:]
            
            for backup_file in old_backups:
                backup_file.unlink()
                print(f"🗑️ Удален старый backup: {backup_file.name}")
            
            print(f"🧹 Очищено {len(old_backups)} старых backup файлов")

def main():
    """Демонстрация работы с менеджером весов"""
    manager = ModelWeightsManager()
    manager.list_available_weights()
    
    print(f"\n💡 ИСПОЛЬЗОВАНИЕ:")
    print(f"1. manager.save_latest_weights(trainer, config) - сохранить текущие веса")
    print(f"2. manager.load_latest_weights(trainer) - загрузить последние веса")
    print(f"3. manager.create_training_checkpoint(trainer, config, epoch, loss, sim) - checkpoint")
    print(f"4. manager.list_available_weights() - показать все веса")

if __name__ == "__main__":
    main() 