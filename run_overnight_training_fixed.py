#!/usr/bin/env python3
"""
ИСПРАВЛЕННОЕ Overnight Training с Simple Fallback Embedding Loader
Обходит проблемы с конфигурацией и использует прямое подключение к transformers
"""

import torch
import torch.nn as nn
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os
import signal

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import EmergentCubeTrainer
from utils.config_manager.config_manager import ConfigManager
from simple_embedding_fallback import create_dialogue_dataset_simple_fallback
from model_weights_manager import ModelWeightsManager
from config_converter import convert_config_dict_to_object

# Исправление Unicode проблемы в Windows
import sys
import re

class EmojiFilter(logging.Filter):
    """Фильтр для удаления эмодзи из логов на Windows"""
    
    def filter(self, record):
        if sys.platform == 'win32':
            # Удаляем все эмодзи из сообщения
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"  # dingbats
                u"\U000024C2-\U0001F251"
                u"\U0001F900-\U0001F9FF"  # supplemental symbols
                u"\U00002600-\U000026FF"  # miscellaneous symbols
                u"\U00002700-\U000027BF"  # dingbats
                "]+", flags=re.UNICODE)
            
            # Заменяем эмодзи на текстовые эквиваленты
            emoji_replacements = {
                '🚀': '[START]',
                '⚙️': '[SETUP]',
                '📚': '[DATA]',
                '✅': '[OK]',
                '🎯': '[TARGET]',
                '🏆': '[BEST]',
                '🏁': '[DONE]',
                '❌': '[ERROR]',
                '📊': '[STATS]',
                '💾': '[SAVE]',
                '🔧': '[DEBUG]',
                '📈': '[PROGRESS]',
                '⏰': '[TIME]',
                '🧪': '[TEST]'
            }
            
            message = record.getMessage()
            for emoji, replacement in emoji_replacements.items():
                message = message.replace(emoji, replacement)
            
            # Удаляем оставшиеся эмодзи
            message = emoji_pattern.sub('', message)
            record.msg = message
            record.args = ()
        
        return True

# Настройка логирования
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/overnight_training_fixed.log', encoding='utf-8')

# Добавляем эмодзи фильтр к консольному обработчику на Windows
if sys.platform == 'win32':
    console_handler.addFilter(EmojiFilter())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Добавляем фильтр ко всем существующим логгерам
if sys.platform == 'win32':
    for name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(name)
        for handler in existing_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.addFilter(EmojiFilter())

class FixedOvernightTrainer:
    """Исправленный Overnight Trainer с простым embedding loader"""
    
    def __init__(self):
        self.trainer = None
        self.dataset = None
        self.config = None
        self.weights_manager = ModelWeightsManager()
        self.should_stop = False
        self.best_similarity = 0.0
        self.training_log = []
        
        # Обработчик сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 FixedOvernightTrainer initialized")
        
        # Применяем эмодзи фильтр к всем логгерам включая trainer логгеры
        if sys.platform == 'win32':
            self._apply_emoji_filter_to_all_loggers()
    
    def _apply_emoji_filter_to_all_loggers(self):
        """Применить эмодзи фильтр ко всем логгерам"""
        emoji_filter = EmojiFilter()
        
        # Получаем все существующие логгеры
        all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        all_loggers.append(logging.root)
        
        for logger_obj in all_loggers:
            for handler in logger_obj.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.addFilter(emoji_filter)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful остановки"""
        logger.info(f"📡 Received signal {signum}, stopping training gracefully...")
        self.should_stop = True
    
    def setup_training(self):
        """Настройка обучения"""
        logger.info("⚙️ Setting up training components...")
        
        # 1. Загружаем конфигурацию
        config_manager = ConfigManager()
        config_dict = config_manager.get_config()  # Получаем всю конфигурацию
        self.config = convert_config_dict_to_object(config_dict)  # Конвертируем в объект
        
        # 2. Создаем trainer
        self.trainer = EmergentCubeTrainer(self.config)
        self.trainer.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Повторно применяем эмодзи фильтр после создания trainer
        if sys.platform == 'win32':
            self._apply_emoji_filter_to_all_loggers()
        
        # 3. Создаем данные с ПРОСТЫМ fallback loader
        logger.info("📚 Creating dataset with SimpleFallbackEmbeddingLoader...")
        dialogue_pairs = [
            {"question": "What is artificial intelligence?", "answer": "AI is the simulation of human intelligence."},
            {"question": "How do neural networks work?", "answer": "Neural networks process data through interconnected layers."},
            {"question": "What is machine learning?", "answer": "ML enables computers to learn from data without explicit programming."},
            {"question": "Explain deep learning.", "answer": "Deep learning uses multi-layered neural networks for complex pattern recognition."},
            {"question": "What are transformers in AI?", "answer": "Transformers are attention-based models for sequence processing."},
            {"question": "How does natural language processing work?", "answer": "NLP enables computers to understand and generate human language."},
            {"question": "What is computer vision?", "answer": "Computer vision enables machines to interpret visual information."},
            {"question": "Explain reinforcement learning.", "answer": "RL trains agents through interaction with an environment."},
            {"question": "What is supervised learning?", "answer": "Supervised learning uses labeled data to train models."},
            {"question": "How do convolutional networks work?", "answer": "CNNs use filters to detect features in spatial data."}
        ]
        
        # Используем простой fallback вместо сложной системы
        self.dataset = create_dialogue_dataset_simple_fallback(
            dialogue_pairs,
            teacher_model="distilbert-base-uncased",
            normalize_embeddings=True
        )
        
        # 4. Проверяем что данные нормальные
        sample = self.dataset[0]
        q_emb, a_emb = sample
        logger.info(f"✅ Dataset created successfully:")
        logger.info(f"   Question embedding norm: {q_emb.norm().item():.6f}")
        logger.info(f"   Answer embedding norm: {a_emb.norm().item():.6f}")
        logger.info(f"   Dataset size: {len(self.dataset)}")
        
        if q_emb.norm().item() < 0.1 or a_emb.norm().item() < 0.1:
            raise ValueError("Dataset still contains zero embeddings!")
        
        logger.info("✅ Training setup completed successfully")
    
    def run_training(self, max_epochs: int = 999999, batch_size: int = 1024):
        """Запуск обучения"""
        logger.info(f"🎯 Starting overnight training:")
        logger.info(f"   Max epochs: {max_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Device: {next(self.trainer.parameters()).device}")
        
        # Создаем DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор
        optimizer = torch.optim.AdamW(self.trainer.parameters(), lr=0.0001)
        
        epoch = 0
        start_time = time.time()
        
        try:
            while epoch < max_epochs and not self.should_stop:
                epoch_start_time = time.time()
                
                # Training epoch
                total_loss = 0.0
                total_similarity = 0.0
                num_batches = 0
                
                for batch_idx, (question_emb, answer_emb) in enumerate(dataloader):
                    if self.should_stop:
                        break
                    
                    # Перемещаем на device
                    device = next(self.trainer.parameters()).device
                    question_emb = question_emb.to(device)
                    answer_emb = answer_emb.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.trainer.forward(question_emb)
                    
                    # КРИТИЧНО: Адаптируем target embedding к 225D через тот же адаптер
                    with torch.no_grad():
                        adapted_target = self.trainer.base_trainer.adapter(answer_emb)
                    
                    # Targets
                    targets = {
                        'target_embedding': adapted_target,  # Используем адаптированный target 225D
                        'target_surface': outputs['input_surface']
                    }
                    
                    # Loss computation
                    losses = self.trainer.compute_loss(outputs, targets)
                    
                    # Правильное суммирование loss'ов в скаляр
                    total_loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)
                    for loss_name, loss_value in losses.items():
                        if torch.is_tensor(loss_value) and loss_value.requires_grad:
                            # Убеждаемся что loss скаляр
                            if loss_value.dim() > 0:
                                loss_value = loss_value.mean()
                            total_loss_tensor = total_loss_tensor + loss_value
                    
                    # Backward pass
                    total_loss_tensor.backward()
                    optimizer.step()
                    
                    # Metrics - используем adapted target для правильной размерности
                    with torch.no_grad():
                        similarity = torch.cosine_similarity(
                            outputs['final_output'], 
                            adapted_target,  # Используем адаптированный target 225D
                            dim=-1
                        ).mean().item()
                    
                    total_loss += total_loss_tensor.item()
                    total_similarity += similarity
                    num_batches += 1
                
                # Epoch metrics
                avg_loss = total_loss / max(num_batches, 1)
                avg_similarity = total_similarity / max(num_batches, 1)
                epoch_time = time.time() - epoch_start_time
                
                # Logging
                epoch += 1
                
                # Детальный лог каждые 10 эпох или если loss изменился
                if epoch % 10 == 0 or epoch <= 5:
                    logger.info(f"Epoch {epoch:4d} | "
                              f"Loss: {avg_loss:.6f} | "
                              f"Similarity: {avg_similarity:.4f} | "
                              f"Time: {epoch_time:.1f}s")
                
                # Сохранение прогресса
                log_entry = {
                    'epoch': epoch,
                    'loss': avg_loss,
                    'similarity': avg_similarity,
                    'time': epoch_time,
                    'timestamp': datetime.now().isoformat()
                }
                self.training_log.append(log_entry)
                
                # Best model tracking
                if avg_similarity > self.best_similarity:
                    self.best_similarity = avg_similarity
                    logger.info(f"🏆 New best similarity: {avg_similarity:.4f}")
                    
                    # Сохраняем лучшую модель
                    self.weights_manager.save_latest_weights(
                        self.trainer, 
                        self.config.to_dict(),  # Конвертируем в dict для JSON
                        metadata={
                            'epoch': epoch,
                            'loss': avg_loss,
                            'similarity': avg_similarity,
                            'training_type': 'overnight_fixed'
                        }
                    )
                
                # Checkpoint каждые 50 эпох
                if epoch % 50 == 0:
                    self.weights_manager.create_training_checkpoint(
                        self.trainer, self.config.to_dict(), epoch, avg_loss, avg_similarity,
                        metadata={'training_type': 'overnight_fixed'}
                    )
                
                # Проверка на потрясающие результаты
                if avg_similarity > 0.6:
                    logger.info(f"🎉 EXCELLENT RESULTS! Similarity > 60%")
                elif avg_similarity > 0.45:
                    logger.info(f"🎯 GREAT PROGRESS! Similarity > 45%")
                
                # Сохранение лога каждые 100 эпох
                if epoch % 100 == 0:
                    self._save_training_log()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            self._finalize_training(epoch, time.time() - start_time)
    
    def _save_training_log(self):
        """Сохранение лога обучения"""
        log_path = f"logs/overnight_training_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
    
    def _finalize_training(self, final_epoch: int, total_time: float):
        """Финализация обучения"""
        logger.info(f"🏁 Training completed:")
        logger.info(f"   Final epoch: {final_epoch}")
        logger.info(f"   Total time: {total_time/3600:.1f} hours")
        logger.info(f"   Best similarity: {self.best_similarity:.4f}")
        
        # Финальное сохранение
        self._save_training_log()
        
        # Создаем milestone checkpoint
        self.weights_manager.create_milestone_checkpoint(
            self.trainer, self.config.to_dict(),  # Конвертируем в dict для JSON
            f"overnight_fixed_final_{final_epoch}",
            {
                'final_epoch': final_epoch,
                'total_time_hours': total_time/3600,
                'best_similarity': self.best_similarity,
                'total_batches': len(self.training_log)
            }
        )
        
        logger.info("✅ Training finalization completed")


def main():
    """Главная функция"""
    print("🌙 ИСПРАВЛЕННОЕ OVERNIGHT TRAINING")
    print("="*50)
    print("Using SimpleFallbackEmbeddingLoader")
    print("No early stopping - run until manually stopped")
    print("Optimal batch_size 1024 for RTX 5090")
    print("="*50)
    
    trainer = FixedOvernightTrainer()
    
    try:
        trainer.setup_training()
        trainer.run_training(
            max_epochs=999999,  # Unlimited
            batch_size=1024     # Optimal for RTX 5090
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 