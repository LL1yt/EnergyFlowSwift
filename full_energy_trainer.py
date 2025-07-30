#!/usr/bin/env python3
"""
Полноценный тренер для energy_flow архитектуры
==============================================

Основной тренировочный скрипт с experiment конфигурацией и умным управлением чекпоинтами.
Использует experiment датасет (5021 pairs) и поддерживает возобновление обучения.

Особенности:
- Experiment конфигурация (50x50x20 решетка)
- Умное именование и сохранение чекпоинтов
- Автоматическое возобновление с последнего чекпоинта
- Валидация и детальное логирование
- Управление через command line аргументы
"""

import sys
import argparse
from pathlib import Path
import torch
from datetime import datetime

# Добавляем корень проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_experiment_config, set_energy_config
from energy_flow.training import EnergyTrainer
from energy_flow.training.checkpoint_loader import create_checkpoint_loader
from energy_flow.utils.logging import DEBUG_ENERGY, get_logger, DEBUG_TRAINING, setup_logging
from energy_flow.utils.checkpoint_utils import list_checkpoints

logger = get_logger(__name__)

# Настройка логирования с convergence категорией
setup_logging(debug_mode=True, level="DEBUG_FORWARD")

# Путь к experiment датасету
EXPERIMENT_DATASET_PATH = "data/energy_flow/active/experiment_mixed_5021pairs_20250729_121801.pt"


def load_experiment_dataset(dataset_path: str):
    """Загрузка experiment датасета"""
    logger.info(f"📁 Loading experiment dataset from {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Experiment dataset not found: {dataset_path}")
    
    # Загружаем датасет
    dataset = torch.load(dataset_path, map_location='cuda', weights_only=False)
    
    total_pairs = len(dataset['text_pairs'])
    embedding_dim = dataset['input_embeddings'].shape[1]
    generation_time = dataset['generation_info']['generation_timestamp']
    sources = ', '.join(dataset['generation_info']['sources'])
    
    logger.info(f"✅ Experiment dataset loaded:")
    logger.info(f"   📊 Total pairs: {total_pairs:,}")
    logger.info(f"   🔢 Embedding dimension: {embedding_dim}")
    logger.info(f"   📅 Generated: {generation_time}")
    logger.info(f"   📚 Sources: {sources}")
    
    return dataset


def create_dataloader_from_experiment_dataset(dataset, batch_size: int = 16, shuffle: bool = True):
    """Создание DataLoader из experiment датасета"""
    from torch.utils.data import DataLoader
    
    class ExperimentDatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.length = len(dataset['text_pairs'])
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            input_text, target_text = self.dataset['text_pairs'][idx]
            return {
                'input_embedding': self.dataset['input_embeddings'][idx],
                'target_embedding': self.dataset['target_embeddings'][idx],
                'input_text': input_text,
                'target_text': target_text
            }
    
    wrapped_dataset = ExperimentDatasetWrapper(dataset)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device=torch.get_default_device()) if shuffle else None,
        collate_fn=lambda batch: {
            'input_embedding': torch.stack([item['input_embedding'] for item in batch]),
            'target_embedding': torch.stack([item['target_embedding'] for item in batch]),
            'input_text': [item['input_text'] for item in batch],
            'target_text': [item['target_text'] for item in batch]
        }
    )
    
    return dataloader


def setup_experiment_trainer(resume_from: str = None):
    """Настройка experiment тренера с возможностью возобновления"""
    logger.info("🔧 Setting up experiment trainer...")
    
    # Experiment конфигурация
    config = create_experiment_config()
    set_energy_config(config)
    
    logger.info(f"📐 Experiment config loaded:")
    logger.info(f"   🔲 Lattice: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    logger.info(f"   📦 Batch size: {config.batch_size}")
    logger.info(f"   📈 Learning rate: {config.learning_rate}")
    logger.info(f"   🌉 Text bridge: {config.text_bridge_enabled}")
    
    # Создаем тренер
    trainer = EnergyTrainer(config)
    
    # Возобновление обучения, если указано
    if resume_from:
        logger.info(f"🔄 Attempting to resume from: {resume_from}")
        
        if resume_from == "latest":
            success = trainer.load_smart_checkpoint(load_latest=True)
        elif resume_from == "best":
            success = trainer.load_smart_checkpoint(load_best=True)
        else:
            # Конкретный файл или паттерн
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                success = trainer.load_smart_checkpoint(checkpoint_path=checkpoint_path)
            else:
                # Попробуем как паттерн
                success = trainer.checkpoint_loader.load_checkpoint_by_pattern(resume_from)
                if success:
                    success = trainer.load_smart_checkpoint(load_latest=True)  # После загрузки по паттерну
        
        if success:
            logger.info("✅ Resume successful - continuing from loaded checkpoint")
        else:
            logger.warning("⚠️ Resume failed - starting fresh training")
    
    return config, trainer


def run_training_session(
    trainer: EnergyTrainer,
    dataloader,
    num_epochs: int = 10,
    validate_every: int = 5,
    save_every: int = 2
):
    """Запуск тренировочной сессии"""
    logger.info(f"🚀 Starting training session:")
    logger.info(f"   📊 Epochs: {num_epochs}")
    logger.info(f"   🔍 Validation every: {validate_every} epochs")
    logger.info(f"   💾 Save every: {save_every} epochs")
    logger.info(f"   📦 Batches per epoch: {len(dataloader)}")
    
    start_epoch = trainer.epoch
    session_start_time = datetime.now()
    
    try:
        for epoch in range(num_epochs):
            current_epoch = start_epoch + epoch
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 EPOCH {current_epoch + 1} (session {epoch + 1}/{num_epochs})")
            logger.info(f"{'='*60}")
            
            epoch_start_time = datetime.now()
            
            # Тренировочный цикл для эпохи
            epoch_metrics = {
                'total_loss': 0.0,
                'energy_loss': 0.0,
                'text_loss': 0.0,
                'step_time': 0.0,
                'flow_time': 0.0,
                'batches_processed': 0
            }
            
            # Проходим по батчам
            for batch_idx, batch in enumerate(dataloader):
                # Извлекаем данные
                input_texts = batch['input_text']
                target_texts = batch['target_text']
                input_embeddings = batch['input_embedding']
                target_embeddings = batch['target_embedding']
                
                # Один шаг обучения
                step_metrics = trainer.train_step(
                    input_texts=input_texts,
                    target_texts=target_texts,
                    teacher_input_embeddings=input_embeddings,
                    teacher_target_embeddings=target_embeddings
                )
                
                # Аккумулируем метрики
                for key in ['total_loss', 'energy_loss', 'text_loss', 'step_time', 'flow_time']:
                    if key in step_metrics:
                        epoch_metrics[key] += step_metrics[key]
                epoch_metrics['batches_processed'] += 1
                
                # Логирование прогресса батчей
                if batch_idx % 10 == 0:
                    progress = (batch_idx + 1) / len(dataloader) * 100
                    logger.info(f"  📈 Batch {batch_idx + 1}/{len(dataloader)} ({progress:.1f}%): "
                              f"loss={step_metrics.get('total_loss', 0):.4f}, "
                              f"time={step_metrics.get('step_time', 0):.2f}s")
                
                # Для debug режима - ограничиваем количество батчей
                if batch_idx >= 50:  # Максимум 50 батчей за эпоху для experiment
                    logger.info(f"  ⏹️ Limited to {batch_idx + 1} batches for experiment session")
                    break
            
            # Усредняем метрики по эпохе
            if epoch_metrics['batches_processed'] > 0:
                for key in ['total_loss', 'energy_loss', 'text_loss', 'step_time', 'flow_time']:
                    epoch_metrics[key] /= epoch_metrics['batches_processed']
            
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # Обновляем тренер
            trainer.epoch = current_epoch + 1
            
            # Планировщик learning rate
            trainer.scheduler.step(epoch_metrics['total_loss'])
            
            # Сохранение лучшей модели
            if epoch_metrics['total_loss'] < trainer.best_loss:
                trainer.best_loss = epoch_metrics['total_loss']
                trainer.save_smart_checkpoint(
                    current_loss=epoch_metrics['total_loss'],
                    is_best=True,
                    custom_suffix=f"session_{session_start_time.strftime('%H%M%S')}"
                )
            
            # Периодическое сохранение
            if (epoch + 1) % save_every == 0:
                trainer.save_smart_checkpoint(
                    current_loss=epoch_metrics['total_loss'],
                    is_best=False,
                    custom_suffix=f"session_{session_start_time.strftime('%H%M%S')}_periodic"
                )
            
            # Логирование эпохи
            logger.info(f"✅ Epoch {current_epoch + 1} completed:")
            logger.info(f"   📊 Total loss: {epoch_metrics['total_loss']:.4f}")
            logger.info(f"   ⚡ Energy loss: {epoch_metrics['energy_loss']:.4f}")
            logger.info(f"   📝 Text loss: {epoch_metrics['text_loss']:.4f}")
            logger.info(f"   ⏱️ Epoch time: {epoch_time:.1f}s")
            logger.info(f"   🔄 Learning rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"   📦 Batches processed: {epoch_metrics['batches_processed']}")
            
            # Валидация
            if (epoch + 1) % validate_every == 0:
                logger.info(f"\n🔍 Running validation...")
                
                # Берем несколько батчей для валидации
                val_batches = []
                val_dataloader_iter = iter(dataloader)
                for _ in range(min(3, len(dataloader))):  # 3 батча для валидации
                    try:
                        val_batches.append(next(val_dataloader_iter))
                    except StopIteration:
                        break
                
                if val_batches:
                    # Объединяем валидационные батчи
                    val_input_texts = []
                    val_target_texts = []
                    val_input_embeddings = []
                    val_target_embeddings = []
                    
                    for batch in val_batches:
                        val_input_texts.extend(batch['input_text'])
                        val_target_texts.extend(batch['target_text'])
                        val_input_embeddings.append(batch['input_embedding'])
                        val_target_embeddings.append(batch['target_embedding'])
                    
                    val_input_embeddings = torch.cat(val_input_embeddings, dim=0)
                    val_target_embeddings = torch.cat(val_target_embeddings, dim=0)
                    
                    # Ограничиваем до разумного размера
                    max_val_samples = 12
                    if len(val_input_texts) > max_val_samples:
                        val_input_texts = val_input_texts[:max_val_samples]
                        val_target_texts = val_target_texts[:max_val_samples]
                        val_input_embeddings = val_input_embeddings[:max_val_samples]
                        val_target_embeddings = val_target_embeddings[:max_val_samples]
                    
                    # Запускаем валидацию
                    val_results = trainer.validate(
                        input_texts=val_input_texts,
                        target_texts=val_target_texts,
                        teacher_input_embeddings=val_input_embeddings,
                        teacher_target_embeddings=val_target_embeddings
                    )
                    
                    logger.info(f"📊 Validation results:")
                    logger.info(f"   📉 Validation loss: {val_results.get('total_loss', 'N/A'):.4f}")
                    logger.info(f"   📝 Examples generated: {len(val_results.get('examples', []))}")
                    
                    # Показываем примеры
                    examples = val_results.get('examples', [])
                    if examples:
                        logger.info(f"📝 Sample predictions:")
                        for i, example in enumerate(examples[:2]):  # Первые 2 примера
                            logger.info(f"   {i+1}. Input: '{example['input'][:60]}...'")
                            logger.info(f"      Target: '{example['target'][:60]}...'")
                            logger.info(f"      Predicted: '{example['predicted'][:60]}...'")
        
        session_time = (datetime.now() - session_start_time).total_seconds()
        
        logger.info(f"\n🎉 Training session completed successfully!")
        logger.info(f"   ⏱️ Total session time: {session_time/60:.1f} minutes")
        logger.info(f"   📊 Final epoch: {trainer.epoch}")
        logger.info(f"   🎯 Best loss achieved: {trainer.best_loss:.4f}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Training interrupted by user")
        logger.info(f"   💾 Saving interruption checkpoint...")
        
        # Сохраняем чекпоинт прерывания
        trainer.save_smart_checkpoint(
            current_loss=epoch_metrics.get('total_loss', float('inf')),
            is_best=False,
            custom_suffix=f"interrupted_{datetime.now().strftime('%H%M%S')}"
        )
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Training session failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def list_available_checkpoints():
    """Показывает доступные чекпоинты"""
    logger.info("📋 Available checkpoints:")
    
    checkpoint_loader = create_checkpoint_loader()
    checkpoints = checkpoint_loader.list_available_checkpoints()
    
    if not checkpoints:
        logger.info("   (No checkpoints found)")
        return
    
    logger.info(f"\n📁 Found {len(checkpoints)} checkpoints:")
    for i, (path, metadata) in enumerate(checkpoints):
        status = "🏆 BEST" if metadata['is_best'] else "📄 REG"
        timestamp = metadata['timestamp'].strftime("%Y-%m-%d %H:%M")
        logger.info(f"   {i+1:2d}. {status} {path.name}")
        logger.info(f"       📅 {timestamp} | 📊 Epoch {metadata['epoch']} | 📉 Loss {metadata['loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Full Energy Flow Trainer")
    parser.add_argument("--dataset", type=str, default=EXPERIMENT_DATASET_PATH,
                       help="Path to experiment dataset file")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint: 'latest', 'best', filename, or pattern")
    parser.add_argument("--validate-every", type=int, default=5,
                       help="Run validation every N epochs")
    parser.add_argument("--save-every", type=int, default=2,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--list-checkpoints", action="store_true",
                       help="List available checkpoints and exit")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override default batch size")
    
    args = parser.parse_args()
    
    try:
        # Список чекпоинтов
        if args.list_checkpoints:
            list_available_checkpoints()
            return
        
        logger.info(f"🌟 Full Energy Flow Training Session")
        logger.info(f"{'='*80}")
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Dataset: {args.dataset}")
        logger.info(f"📊 Epochs: {args.epochs}")
        if args.resume:
            logger.info(f"🔄 Resume from: {args.resume}")
        
        # Загрузка датасета
        logger.info(f"\n1️⃣ Loading experiment dataset...")
        dataset = load_experiment_dataset(args.dataset)
        
        # Настройка тренера
        logger.info(f"\n2️⃣ Setting up experiment trainer...")
        config, trainer = setup_experiment_trainer(resume_from=args.resume)
        
        # Переопределяем batch_size если указан
        if args.batch_size:
            logger.info(f"🔧 Overriding batch size: {config.batch_size} -> {args.batch_size}")
            config.batch_size = args.batch_size
        
        # Создание DataLoader
        logger.info(f"\n3️⃣ Creating experiment DataLoader...")
        dataloader = create_dataloader_from_experiment_dataset(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        logger.info(f"✅ DataLoader ready: {len(dataloader)} batches, batch_size={config.batch_size}")
        
        # Запуск обучения
        logger.info(f"\n4️⃣ Starting experiment training session...")
        success = run_training_session(
            trainer=trainer,
            dataloader=dataloader,
            num_epochs=args.epochs,
            validate_every=args.validate_every,
            save_every=args.save_every
        )
        
        if success:
            logger.info(f"\n✨ Experiment training completed successfully!")
            logger.info(f"💾 Checkpoints saved to: checkpoints/energy_flow/active/")
            logger.info(f"🎯 Best loss achieved: {trainer.best_loss:.4f}")
        else:
            logger.info(f"\n⚠️ Training session ended early")
        
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()