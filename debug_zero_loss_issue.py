#!/usr/bin/env python3
"""
[MAGNIFY] ДИАГНОСТИЧЕСКИЙ СКРИПТ: Проблема с нулевыми метриками

Цель: Систематически выявить источник проблемы с Loss: 0.0000 и Similarity: 0.0000

Проверяемые компоненты:
1. Forward pass и tensor flow
2. Loss function computation
3. Gradient flow через систему
4. Target/input data integrity
5. Dimension matching
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
from data.embedding_loader.format_handlers import create_llm_handler

class ZeroLossDiagnostics:
    """Комплексная диагностика проблемы с нулевыми метриками"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        print(f"[MAGNIFY] [ДИАГНОСТИКА] Инициализация на device: {self.device}")
        
    def run_full_diagnostics(self):
        """Запуск полного цикла диагностики"""
        
        print("\n" + "="*50)
        print("[TARGET] НАЧАЛО СИСТЕМАТИЧЕСКОЙ ДИАГНОСТИКИ")
        print("="*50)
        
        # 1. Проверка создания модели
        trainer = self._test_model_creation()
        if trainer is None:
            return
            
        # 2. Проверка данных
        dataset = self._test_data_creation()
        if not dataset:
            return
            
        # 3. Проверка forward pass
        outputs = self._test_forward_pass(trainer, dataset)
        if outputs is None:
            return
            
        # 4. Проверка loss computation
        loss_results = self._test_loss_computation(trainer, outputs, dataset)
        if loss_results is None:
            return
            
        # 5. Проверка gradient flow
        self._test_gradient_flow(trainer, outputs, dataset)
        
        # 6. Финальный анализ
        self._analyze_results()
        
    def _test_model_creation(self) -> 'EmergentCubeTrainer':
        """Тест 1: Создание модели"""
        print("\n[CONFIG] [ТЕСТ 1] Проверка создания модели...")
        
        try:
            config = EmergentTrainingConfig()
            config.teacher_model = "distilbert-base-uncased"
            config.cube_dimensions = (15, 15, 11)
            config.mixed_precision = False  # Отключаем для диагностики
            
            trainer = EmergentCubeTrainer(config, device=str(self.device))
            
            # Проверяем параметры
            total_params = sum(p.numel() for p in trainer.parameters())
            trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
            
            print(f"[OK] [ТЕСТ 1] Модель создана успешно")
            print(f"   - Всего параметров: {total_params:,}")
            print(f"   - Обучаемых параметров: {trainable_params:,}")
            print(f"   - Device: {trainer.device}")
            
            self.results['model_creation'] = {
                'status': 'success',
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            return trainer
            
        except Exception as e:
            print(f"[ERROR] [ТЕСТ 1] Ошибка создания модели: {e}")
            self.results['model_creation'] = {'status': 'failed', 'error': str(e)}
            return None

    def _test_data_creation(self) -> List:
        """Тест 2: Создание данных"""
        print("\n[DATA] [ТЕСТ 2] Проверка создания данных...")
        
        try:
            # Простой тестовый dataset
            dialogue_pairs = [
                {"question": "What is AI?", "answer": "AI is artificial intelligence."},
                {"question": "How do neural networks work?", "answer": "Neural networks process information."}
            ]
            
            dataset = create_dialogue_dataset(
                dialogue_pairs,
                teacher_model="distilbert-base-uncased",
                cache_embeddings=False,  # Отключаем кэш для диагностики
                validation_split=0.0,
                normalize_embeddings=True
            )
            
            print(f"[OK] [ТЕСТ 2] Dataset создан успешно")
            print(f"   - Количество примеров: {len(dataset)}")
            
            # Проверяем структуру данных
            if dataset:
                sample = dataset[0]
                print(f"   - Ключи в примере: {list(sample.keys())}")
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   - {key}: type={type(value)}")
            
            self.results['data_creation'] = {
                'status': 'success',
                'dataset_size': len(dataset),
                'sample_keys': list(dataset[0].keys()) if dataset else []
            }
            
            return dataset
            
        except Exception as e:
            print(f"[ERROR] [ТЕСТ 2] Ошибка создания данных: {e}")
            self.results['data_creation'] = {'status': 'failed', 'error': str(e)}
            return []

    def _test_forward_pass(self, trainer: 'EmergentCubeTrainer', dataset: List) -> Dict:
        """Тест 3: Forward pass"""
        print("\n[FAST] [ТЕСТ 3] Проверка forward pass...")
        
        try:
            # Берем первый пример
            sample = dataset[0]
            question_embedding = sample['question_embedding'].unsqueeze(0).to(self.device)
            answer_embedding = sample['answer_embedding'].unsqueeze(0).to(self.device)
            
            print(f"   - Input shapes:")
            print(f"     Question: {question_embedding.shape}")
            print(f"     Answer: {question_embedding.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = trainer.forward(question_embedding)
            
            print(f"[OK] [ТЕСТ 3] Forward pass выполнен успешно")
            print(f"   - Output keys: {list(outputs.keys())}")
            
            for key, tensor in outputs.items():
                if torch.is_tensor(tensor):
                    print(f"   - {key}: shape={tensor.shape}, requires_grad={tensor.requires_grad}")
                    
                    # Проверяем на нули
                    zero_count = (tensor == 0).sum().item()
                    total_elements = tensor.numel()
                    zero_percentage = (zero_count / total_elements) * 100
                    print(f"     Нулевых элементов: {zero_count}/{total_elements} ({zero_percentage:.1f}%)")
                    
                    # Статистика
                    if tensor.numel() > 0:
                        print(f"     Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
                        print(f"     Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
                        
            self.results['forward_pass'] = {
                'status': 'success',
                'output_keys': list(outputs.keys()),
                'output_shapes': {k: list(v.shape) if torch.is_tensor(v) else None 
                                 for k, v in outputs.items()}
            }
            
            return outputs
            
        except Exception as e:
            print(f"[ERROR] [ТЕСТ 3] Ошибка forward pass: {e}")
            self.results['forward_pass'] = {'status': 'failed', 'error': str(e)}
            return None

    def _test_loss_computation(self, trainer: 'EmergentCubeTrainer', outputs: Dict, dataset: List) -> Dict:
        """Тест 4: Loss computation"""
        print("\n💰 [ТЕСТ 4] Проверка loss computation...")
        
        try:
            # Подготавливаем targets
            sample = dataset[0]
            answer_embedding = sample['answer_embedding'].unsqueeze(0).to(self.device)
            
            targets = {
                'target_embedding': answer_embedding,
                'target_surface': outputs['input_surface']
            }
            
            print(f"   - Target shapes:")
            for key, tensor in targets.items():
                print(f"     {key}: {tensor.shape}")
            
            # Compute loss
            losses = trainer.compute_loss(outputs, targets)
            
            print(f"[OK] [ТЕСТ 4] Loss computation выполнен успешно")
            print(f"   - Loss components: {list(losses.keys())}")
            
            for key, loss_tensor in losses.items():
                if torch.is_tensor(loss_tensor):
                    loss_value = loss_tensor.item()
                    print(f"   - {key}: {loss_value:.6f}")
                    print(f"     requires_grad: {loss_tensor.requires_grad}")
                    print(f"     grad_fn: {loss_tensor.grad_fn}")
                    
                    # Критическая проверка: почему loss = 0?
                    if loss_value == 0.0:
                        print(f"     [ALERT] КРИТИЧНО: {key} = 0.0!")
                        self._debug_zero_loss_component(key, outputs, targets, trainer)
                        
            self.results['loss_computation'] = {
                'status': 'success',
                'loss_values': {k: v.item() if torch.is_tensor(v) else v 
                               for k, v in losses.items()}
            }
            
            return losses
            
        except Exception as e:
            print(f"[ERROR] [ТЕСТ 4] Ошибка loss computation: {e}")
            self.results['loss_computation'] = {'status': 'failed', 'error': str(e)}
            return None

    def _debug_zero_loss_component(self, component: str, outputs: Dict, targets: Dict, trainer):
        """Детальная диагностика нулевого loss компонента"""
        print(f"\n[MAGNIFY] [ДЕТАЛЬНАЯ ДИАГНОСТИКА] {component}")
        
        if component == 'surface_reconstruction_loss':
            # Проверяем surface reconstruction
            if 'output_surface' in outputs and 'input_surface' in outputs:
                output_surface = outputs['output_surface']
                input_surface = outputs['input_surface']
                
                print(f"   - output_surface: shape={output_surface.shape}")
                print(f"   - input_surface: shape={input_surface.shape}")
                
                # Проверяем projection
                if hasattr(trainer.loss_function, 'embedding_to_surface'):
                    projected_input = trainer.loss_function.embedding_to_surface(input_surface)
                    print(f"   - projected_input: shape={projected_input.shape}")
                    
                    # Ручной расчет MSE
                    mse_manual = torch.mean((output_surface - projected_input) ** 2)
                    print(f"   - Ручной MSE: {mse_manual.item():.6f}")
                    
                    # Проверяем идентичность
                    if torch.allclose(output_surface, projected_input):
                        print("   - [ALERT] output_surface идентичен projected_input!")
                    else:
                        print(f"   - Различие найдено: max_diff={torch.max(torch.abs(output_surface - projected_input)).item():.6f}")
                        
        elif component == 'dialogue_similarity_loss':
            # Проверяем dialogue similarity
            if 'final_output' in outputs and 'target_embedding' in targets:
                final_output = outputs['final_output']
                target_embedding = targets['target_embedding']
                
                print(f"   - final_output: shape={final_output.shape}")
                print(f"   - target_embedding: shape={target_embedding.shape}")
                
                # Проверяем projection
                if hasattr(trainer.loss_function, 'embedding_to_surface'):
                    projected_target = trainer.loss_function.embedding_to_surface(target_embedding)
                    print(f"   - projected_target: shape={projected_target.shape}")
                    
                    # Ручной расчет cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(final_output, projected_target, dim=-1)
                    dialogue_loss_manual = 1.0 - torch.mean(cos_sim)
                    print(f"   - Ручной cosine similarity: {torch.mean(cos_sim).item():.6f}")
                    print(f"   - Ручной dialogue loss: {dialogue_loss_manual.item():.6f}")

    def _test_gradient_flow(self, trainer: 'EmergentCubeTrainer', outputs: Dict, dataset: List):
        """Тест 5: Gradient flow"""
        print("\n🌊 [ТЕСТ 5] Проверка gradient flow...")
        
        try:
            # Подготавливаем данные с requires_grad
            sample = dataset[0]
            question_embedding = sample['question_embedding'].unsqueeze(0).to(self.device).requires_grad_(True)
            answer_embedding = sample['answer_embedding'].unsqueeze(0).to(self.device)
            
            # Forward pass с градиентами
            trainer.train()
            outputs = trainer.forward(question_embedding)
            
            targets = {
                'target_embedding': answer_embedding,
                'target_surface': outputs['input_surface']
            }
            
            losses = trainer.compute_loss(outputs, targets)
            total_loss = losses['total_loss']
            
            print(f"   - Total loss для backward: {total_loss.item():.6f}")
            print(f"   - requires_grad: {total_loss.requires_grad}")
            print(f"   - grad_fn: {total_loss.grad_fn}")
            
            # Backward pass
            total_loss.backward()
            
            # Проверяем градиенты
            grad_stats = {}
            for name, param in trainer.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_stats[name] = grad_norm
                    if grad_norm == 0:
                        print(f"   - [ALERT] Zero gradient: {name}")
                    elif grad_norm > 0:
                        print(f"   - [OK] Non-zero gradient: {name} = {grad_norm:.6f}")
                else:
                    print(f"   - [ERROR] No gradient: {name}")
            
            # Общая статистика градиентов
            non_zero_grads = sum(1 for g in grad_stats.values() if g > 0)
            zero_grads = sum(1 for g in grad_stats.values() if g == 0)
            no_grads = len([p for p in trainer.parameters() if p.grad is None])
            
            print(f"[OK] [ТЕСТ 5] Gradient analysis завершен")
            print(f"   - Non-zero gradients: {non_zero_grads}")
            print(f"   - Zero gradients: {zero_grads}")
            print(f"   - No gradients: {no_grads}")
            
            self.results['gradient_flow'] = {
                'status': 'success',
                'non_zero_grads': non_zero_grads,
                'zero_grads': zero_grads,
                'no_grads': no_grads,
                'total_params': len(list(trainer.parameters()))
            }
            
        except Exception as e:
            print(f"[ERROR] [ТЕСТ 5] Ошибка gradient flow: {e}")
            self.results['gradient_flow'] = {'status': 'failed', 'error': str(e)}

    def _analyze_results(self):
        """Финальный анализ результатов диагностики"""
        print("\n" + "="*50)
        print("[DATA] АНАЛИЗ РЕЗУЛЬТАТОВ ДИАГНОСТИКИ")
        print("="*50)
        
        # Выводим статус каждого теста
        for test_name, result in self.results.items():
            status = result.get('status', 'unknown')
            print(f"\n[MAGNIFY] {test_name.upper()}:")
            print(f"   Status: {'[OK] УСПЕХ' if status == 'success' else '[ERROR] ОШИБКА'}")
            
            if status == 'failed':
                print(f"   Error: {result.get('error', 'Unknown')}")
            else:
                # Показываем ключевые метрики
                for key, value in result.items():
                    if key != 'status':
                        print(f"   {key}: {value}")
        
        # Выявляем проблемы
        print("\n[TARGET] ВЕРОЯТНЫЕ ПРИЧИНЫ ПРОБЛЕМЫ:")
        self._identify_root_causes()
        
        # Рекомендации по исправлению
        print("\n[IDEA] РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ:")
        self._provide_recommendations()

    def _identify_root_causes(self):
        """Выявление корневых причин проблемы"""
        
        # Проверяем loss computation
        if 'loss_computation' in self.results and self.results['loss_computation']['status'] == 'success':
            loss_values = self.results['loss_computation']['loss_values']
            
            if loss_values.get('total_loss', 0) == 0.0:
                print("   [ALERT] КРИТИЧНО: total_loss = 0.0")
                
                if loss_values.get('surface_reconstruction_loss', 0) == 0.0:
                    print("     - Surface reconstruction loss = 0 (возможно идентичные input/output)")
                    
                if loss_values.get('dialogue_similarity_loss', 0) == 0.0:
                    print("     - Dialogue similarity loss = 0 (возможно perfect similarity или dimension error)")
                    
                if loss_values.get('internal_consistency_loss', 0) == 0.0:
                    print("     - Internal consistency loss = 0 (возможно нет internal states)")
        
        # Проверяем градиенты
        if 'gradient_flow' in self.results and self.results['gradient_flow']['status'] == 'success':
            grad_info = self.results['gradient_flow']
            if grad_info['non_zero_grads'] == 0:
                print("   [ALERT] КРИТИЧНО: Нет ненулевых градиентов")
                print("     - Backward pass не работает или loss не связан с параметрами")

    def _provide_recommendations(self):
        """Рекомендации по исправлению"""
        
        print("   1. [CONFIG] Проверить loss function implementation:")
        print("      - Убедиться что loss components не возвращают константы")
        print("      - Проверить dimension matching в cosine similarity")
        print("      - Добавить epsilon для numerical stability")
        
        print("   2. [MAGNIFY] Проверить forward pass:")
        print("      - Убедиться что model не возвращает идентичные outputs")
        print("      - Проверить что parameters обновляются")
        
        print("   3. 🌊 Проверить gradient flow:")
        print("      - Добавить gradient debugging в train_step")
        print("      - Проверить retain_graph usage")
        print("      - Убедиться что optimizer.step() вызывается")

def main():
    """Запуск диагностики"""
    diagnostics = ZeroLossDiagnostics()
    diagnostics.run_full_diagnostics()
    
    print("\n[TARGET] Диагностика завершена. Проверьте результаты выше.")
    print("Следующий шаг: исправить выявленные проблемы и повторно запустить обучение.")

if __name__ == "__main__":
    main()