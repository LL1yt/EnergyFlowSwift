#!/usr/bin/env python3
"""
🔧 ДИАГНОСТИКА И ИСПРАВЛЕНИЕ ИНИЦИАЛИЗАЦИИ ВЕСОВ

Проблема: Loss = 0.0000 сразу (должен быть 5-10 изначально)
Цель: Проверить и исправить инициализацию весов в gMLP системе
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

class WeightInitializationDiagnostics:
    """Диагностика и исправление инициализации весов"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run_diagnostics(self):
        """Полная диагностика проблемы с инициализацией"""
        
        print("🔧 ДИАГНОСТИКА ИНИЦИАЛИЗАЦИИ ВЕСОВ")
        print("="*50)
        
        # 1. Создаем trainer
        trainer = self._create_trainer()
        if trainer is None:
            return
            
        # 2. Проверяем начальные веса
        self._check_initial_weights(trainer)
        
        # 3. Проверяем forward pass
        self._check_forward_pass(trainer)
        
        # 4. Проверяем loss computation
        self._check_loss_computation(trainer)
        
        # 5. Применяем правильную инициализацию
        self._apply_proper_initialization(trainer)
        
        # 6. Проверяем после исправления
        self._verify_after_fix(trainer)
        
    def _create_trainer(self):
        """Создание trainer для диагностики"""
        try:
            config = EmergentTrainingConfig()
            config.teacher_model = "distilbert-base-uncased"
            config.cube_dimensions = (15, 15, 11)
            config.mixed_precision = False  # Отключаем для диагностики
            
            trainer = EmergentCubeTrainer(config, device=str(self.device))
            
            total_params = sum(p.numel() for p in trainer.parameters())
            print(f"✅ Trainer создан: {total_params:,} параметров")
            
            return trainer
            
        except Exception as e:
            print(f"❌ Ошибка создания trainer: {e}")
            return None
    
    def _check_initial_weights(self, trainer):
        """Проверка начальных весов"""
        print("\n🔍 ПРОВЕРКА НАЧАЛЬНЫХ ВЕСОВ:")
        
        weight_stats = {}
        zero_params = 0
        total_params = 0
        
        for name, param in trainer.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                total_params += param.numel()
                
                # Статистика
                mean_val = weight_data.mean().item()
                std_val = weight_data.std().item()
                min_val = weight_data.min().item()
                max_val = weight_data.max().item()
                zero_count = (weight_data == 0).sum().item()
                
                weight_stats[name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'zero_count': zero_count,
                    'total': param.numel(),
                    'zero_percentage': (zero_count / param.numel()) * 100
                }
                
                zero_params += zero_count
                
                # Выводим проблемные слои
                if abs(mean_val) < 1e-6 and std_val < 1e-6:
                    print(f"🚨 ПРОБЛЕМА: {name}")
                    print(f"   Mean: {mean_val:.8f}, Std: {std_val:.8f}")
                    print(f"   Все веса практически нули!")
                elif zero_count > param.numel() * 0.9:
                    print(f"⚠️ ПОДОЗРИТЕЛЬНО: {name}")
                    print(f"   {zero_count}/{param.numel()} ({weight_stats[name]['zero_percentage']:.1f}%) нулевых весов")
                else:
                    print(f"✅ НОРМАЛЬНО: {name}")
                    print(f"   Mean: {mean_val:.6f}, Std: {std_val:.6f}, Range: [{min_val:.6f}, {max_val:.6f}]")
        
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Всего параметров: {total_params:,}")
        print(f"   Нулевых параметров: {zero_params:,} ({(zero_params/total_params)*100:.1f}%)")
        
        return weight_stats
    
    def _check_forward_pass(self, trainer):
        """Проверка forward pass"""
        print("\n⚡ ПРОВЕРКА FORWARD PASS:")
        
        # Создаем тестовый input
        batch_size = 2
        input_dim = 768  # DistilBERT
        test_input = torch.randn(batch_size, input_dim).to(self.device)
        
        print(f"   Input: shape={test_input.shape}, mean={test_input.mean().item():.6f}")
        
        with torch.no_grad():
            outputs = trainer.forward(test_input)
        
        for key, tensor in outputs.items():
            if torch.is_tensor(tensor):
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                zero_count = (tensor == 0).sum().item()
                total_elements = tensor.numel()
                
                print(f"   {key}: shape={tensor.shape}")
                print(f"      Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                print(f"      Zeros: {zero_count}/{total_elements} ({(zero_count/total_elements)*100:.1f}%)")
                
                if abs(mean_val) < 1e-6 and std_val < 1e-6:
                    print(f"      🚨 ПРОБЛЕМА: Выход практически нулевой!")
                elif zero_count > total_elements * 0.9:
                    print(f"      ⚠️ ПОДОЗРИТЕЛЬНО: Слишком много нулей")
                else:
                    print(f"      ✅ НОРМАЛЬНО: Разумные значения")
        
        return outputs
    
    def _check_loss_computation(self, trainer):
        """Проверка вычисления loss"""
        print("\n💰 ПРОВЕРКА LOSS COMPUTATION:")
        
        # Создаем тестовые данные
        dialogue_pairs = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        ]
        
        dataset = create_dialogue_dataset(
            dialogue_pairs,
            teacher_model="distilbert-base-uncased",
            cache_embeddings=False,
            validation_split=0.0,
            normalize_embeddings=True
        )
        
        sample = dataset[0]
        # Правильная распаковка dataset - это может быть tuple
        if isinstance(sample, tuple):
            question_emb, answer_emb = sample
            question_emb = question_emb.unsqueeze(0).to(self.device)
            answer_emb = answer_emb.unsqueeze(0).to(self.device)
        else:
            question_emb = sample['question_embedding'].unsqueeze(0).to(self.device)
            answer_emb = sample['answer_embedding'].unsqueeze(0).to(self.device)
        
        print(f"   Question embedding: shape={question_emb.shape}, norm={question_emb.norm().item():.6f}")
        print(f"   Answer embedding: shape={answer_emb.shape}, norm={answer_emb.norm().item():.6f}")
        
        # Forward pass
        outputs = trainer.forward(question_emb)
        
        # Targets
        targets = {
            'target_embedding': answer_emb,
            'target_surface': outputs['input_surface']
        }
        
        # Compute loss
        losses = trainer.compute_loss(outputs, targets)
        
        print(f"   Loss components:")
        for key, loss_tensor in losses.items():
            if torch.is_tensor(loss_tensor):
                # Обрабатываем случай когда тензор не скалярный
                if loss_tensor.numel() == 1:
                    loss_val = loss_tensor.item()
                else:
                    loss_val = loss_tensor.mean().item()  # Берем среднее для многоэлементного тензора
                    print(f"      {key}: {loss_val:.6f} (multi-element tensor)")
                    continue
                
                print(f"      {key}: {loss_val:.6f}")
                
                if loss_val == 0.0:
                    print(f"         🚨 ПРОБЛЕМА: {key} = 0.0!")
                    self._debug_zero_loss_component(key, outputs, targets, trainer)
                else:
                    print(f"         ✅ НОРМАЛЬНО: {key} > 0")
        
        return losses
    
    def _debug_zero_loss_component(self, component: str, outputs, targets, trainer):
        """Детальная диагностика нулевого loss компонента"""
        print(f"         🔍 Диагностика {component}:")
        
        if component == 'dialogue_similarity_loss':
            final_output = outputs['final_output']
            target_embedding = targets['target_embedding']
            
            print(f"            final_output: shape={final_output.shape}, norm={final_output.norm().item():.6f}")
            print(f"            target_embedding: shape={target_embedding.shape}, norm={target_embedding.norm().item():.6f}")
            
            # Проверяем проекцию
            if hasattr(trainer.loss_function, 'embedding_to_surface'):
                projected_target = trainer.loss_function.embedding_to_surface(target_embedding)
                print(f"            projected_target: shape={projected_target.shape}, norm={projected_target.norm().item():.6f}")
                
                # Ручной расчет cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(final_output, projected_target, dim=-1)
                dialogue_loss = 1.0 - torch.mean(cos_sim)
                
                print(f"            cosine_similarity: {torch.mean(cos_sim).item():.6f}")
                print(f"            manual_dialogue_loss: {dialogue_loss.item():.6f}")
                
                if torch.mean(cos_sim).item() == 1.0:
                    print(f"            🚨 ПРОБЛЕМА: Perfect similarity - возможно идентичные тензоры!")
                    print(f"            final_output[0][:5]: {final_output[0][:5]}")
                    print(f"            projected_target[0][:5]: {projected_target[0][:5]}")
    
    def _apply_proper_initialization(self, trainer):
        """Применение правильной инициализации весов"""
        print("\n🔧 ПРИМЕНЕНИЕ ПРАВИЛЬНОЙ ИНИЦИАЛИЗАЦИИ:")
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization для Linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Небольшой bias
                print(f"   Инициализирован Linear: {module.weight.shape}")
                
            elif isinstance(module, nn.Conv3d):
                # Kaiming initialization для Conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
                print(f"   Инициализирован Conv3d: {module.weight.shape}")
                
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"   Инициализирован BatchNorm3d")
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"   Инициализирован LayerNorm")
        
        # Применяем инициализацию ко всем модулям
        trainer.apply(init_weights)
        
        print("✅ Инициализация применена ко всем слоям")
    
    def _verify_after_fix(self, trainer):
        """Проверка после исправления инициализации"""
        print("\n✅ ПРОВЕРКА ПОСЛЕ ИСПРАВЛЕНИЯ:")
        
        # Проверяем веса после инициализации
        print("   Проверка весов после инициализации:")
        
        sample_weights = {}
        for name, param in trainer.named_parameters():
            if 'weight' in name and param.requires_grad:
                mean_val = param.data.mean().item()
                std_val = param.data.std().item()
                sample_weights[name] = (mean_val, std_val)
                
                if len(sample_weights) <= 5:  # Показываем первые 5
                    print(f"      {name}: mean={mean_val:.6f}, std={std_val:.6f}")
        
        # Проверяем loss еще раз
        print("   Проверка loss после инициализации:")
        
        dialogue_pairs = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        ]
        
        dataset = create_dialogue_dataset(
            dialogue_pairs,
            teacher_model="distilbert-base-uncased",
            cache_embeddings=False,
            validation_split=0.0,
            normalize_embeddings=True
        )
        
        sample = dataset[0]
        # Правильная распаковка dataset - это может быть tuple
        if isinstance(sample, tuple):
            question_emb, answer_emb = sample
            question_emb = question_emb.unsqueeze(0).to(self.device)
            answer_emb = answer_emb.unsqueeze(0).to(self.device)
        else:
            question_emb = sample['question_embedding'].unsqueeze(0).to(self.device)
            answer_emb = sample['answer_embedding'].unsqueeze(0).to(self.device)
        
        outputs = trainer.forward(question_emb)
        targets = {
            'target_embedding': answer_emb,
            'target_surface': outputs['input_surface']
        }
        
        losses = trainer.compute_loss(outputs, targets)
        
        print("   Loss после исправления:")
        for key, loss_tensor in losses.items():
            if torch.is_tensor(loss_tensor):
                loss_val = loss_tensor.item()
                if loss_val > 0.01:
                    print(f"      ✅ {key}: {loss_val:.6f} (ХОРОШО - больше 0!)")
                elif loss_val > 0.0:
                    print(f"      ⚠️ {key}: {loss_val:.6f} (маленький но не 0)")
                else:
                    print(f"      ❌ {key}: {loss_val:.6f} (все еще 0)")
        
        # Сохраняем исправленную модель
        self._save_fixed_trainer(trainer)
        
    def _save_fixed_trainer(self, trainer):
        """Сохранение исправленной модели"""
        save_path = "checkpoints/fixed_initialization_trainer.pt"
        os.makedirs("checkpoints", exist_ok=True)
        
        torch.save({
            'model_state_dict': trainer.state_dict(),
            'config': trainer.config,
            'note': 'Fixed weight initialization - should have non-zero loss'
        }, save_path)
        
        print(f"\n💾 Исправленная модель сохранена: {save_path}")
        print("   Эта модель имеет правильную инициализацию весов")
        print("   Используйте её для overnight training")

def main():
    """Запуск диагностики инициализации"""
    diagnostics = WeightInitializationDiagnostics()
    diagnostics.run_diagnostics()
    
    print("\n🎯 РЕКОМЕНДАЦИИ:")
    print("1. Запустите этот скрипт для диагностики")
    print("2. Если найдены проблемы - они будут исправлены")
    print("3. Используйте исправленную модель для overnight training")
    print("4. Loss должен начаться с ~5-10 и постепенно снижаться")

if __name__ == "__main__":
    main() 