#!/usr/bin/env python3
"""
Специальная диагностика Loss Function - почему loss = 0.0000 вместо 5-10
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from training.embedding_trainer.emergent_cube_trainer import EmergentCubeTrainer
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
from config.config_manager import ConfigManager

class LossFunctionDiagnostics:
    """Детальная диагностика почему loss = 0.0000"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CONFIG] Диагностика Loss Function (устройство: {self.device})")
    
    def run_diagnostics(self):
        """Запуск полной диагностики loss function"""
        print("\n" + "="*60)
        print("[MAGNIFY] ДИАГНОСТИКА LOSS FUNCTION")
        print("="*60)
        
        # 1. Создаем trainer
        trainer = self._create_trainer()
        
        # 2. Создаем тестовые данные
        test_data = self._create_test_data()
        
        # 3. Проверяем каждый компонент loss отдельно
        self._test_individual_loss_components(trainer, test_data)
        
        # 4. Проверяем полный loss
        self._test_full_loss_computation(trainer, test_data)
        
        # 5. Тестируем разные scenarios
        self._test_different_scenarios(trainer)
        
        print("\n" + "="*60)
        print("[OK] ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("="*60)
    
    def _create_trainer(self):
        """Создание trainer для тестирования"""
        config_manager = ConfigManager()
        config = config_manager.get_full_config()
        
        trainer = EmergentCubeTrainer(config)
        trainer.to(self.device)
        trainer.eval()  # Evaluation mode для консистентности
        
        return trainer
    
    def _create_test_data(self):
        """Создание тестовых данных"""
        # Разные scenarios для тестирования
        test_cases = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."},
            {"question": "How does learning work?", "answer": "Learning involves acquiring knowledge."},
            {"question": "Explain neural networks.", "answer": "Neural networks process information."}
        ]
        
        dataset = create_dialogue_dataset(
            test_cases,
            teacher_model="distilbert-base-uncased",
            cache_embeddings=False,
            validation_split=0.0,
            normalize_embeddings=True
        )
        
        return dataset
    
    def _test_individual_loss_components(self, trainer, dataset):
        """Тестирование каждого компонента loss отдельно"""
        print("\n🧩 ТЕСТИРОВАНИЕ КОМПОНЕНТОВ LOSS:")
        
        # Получаем sample
        sample = dataset[0]
        if isinstance(sample, tuple):
            question_emb, answer_emb = sample
            question_emb = question_emb.unsqueeze(0).to(self.device)
            answer_emb = answer_emb.unsqueeze(0).to(self.device)
        else:
            question_emb = sample['question_embedding'].unsqueeze(0).to(self.device)
            answer_emb = sample['answer_embedding'].unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = trainer.forward(question_emb)
        
        # Targets
        targets = {
            'target_embedding': answer_emb,
            'target_surface': outputs['input_surface']
        }
        
        print(f"   Input shapes:")
        print(f"      question_emb: {question_emb.shape}")
        print(f"      answer_emb: {answer_emb.shape}")
        print(f"      final_output: {outputs['final_output'].shape}")
        
        # Тестируем dialogue_similarity_loss
        self._test_dialogue_similarity_loss(trainer, outputs, targets)
        
        # Тестируем surface_consistency_loss  
        self._test_surface_consistency_loss(trainer, outputs, targets)
        
        # Тестируем internal_dynamics_loss
        self._test_internal_dynamics_loss(trainer, outputs, targets)
    
    def _test_dialogue_similarity_loss(self, trainer, outputs, targets):
        """Детальное тестирование dialogue similarity loss"""
        print(f"\n   [DATA] DIALOGUE SIMILARITY LOSS:")
        
        final_output = outputs['final_output']
        target_embedding = targets['target_embedding']
        
        print(f"      final_output: shape={final_output.shape}, norm={final_output.norm().item():.6f}")
        print(f"      target_embedding: shape={target_embedding.shape}, norm={target_embedding.norm().item():.6f}")
        
        # Проверяем projection к surface размерности
        if hasattr(trainer.loss_function, 'embedding_to_surface'):
            projected_target = trainer.loss_function.embedding_to_surface(target_embedding)
            print(f"      projected_target: shape={projected_target.shape}, norm={projected_target.norm().item():.6f}")
            
            # Manual cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(final_output, projected_target, dim=-1)
            dialogue_loss = 1.0 - torch.mean(cos_sim)
            
            print(f"      cosine_similarity: {torch.mean(cos_sim).item():.6f}")
            print(f"      dialogue_loss (1-cos): {dialogue_loss.item():.6f}")
            
            # Проверяем на идентичность
            if torch.mean(cos_sim).item() > 0.999:
                print(f"      [ALERT] ПРОБЛЕМА: Почти идентичные тензоры!")
                print(f"         final_output[:5]: {final_output[0][:5]}")
                print(f"         projected_target[:5]: {projected_target[0][:5]}")
                print(f"         difference: {(final_output[0][:5] - projected_target[0][:5]).abs()}")
                
                # Возможная причина: projection возвращает тот же тензор
                print(f"      [MAGNIFY] Проверка projection layer:")
                print(f"         Input to projection: {target_embedding[0][:5]}")
                print(f"         Output from projection: {projected_target[0][:5]}")
                
                # Проверяем веса projection layer
                if hasattr(trainer.loss_function.embedding_to_surface, 'weight'):
                    proj_weights = trainer.loss_function.embedding_to_surface.weight
                    print(f"         Projection weights: shape={proj_weights.shape}")
                    print(f"         Projection weights mean: {proj_weights.mean().item():.6f}")
                    print(f"         Projection weights std: {proj_weights.std().item():.6f}")
            
            # Тестируем разные векторы
            print(f"      [TEST] Тест с random векторами:")
            random_output = torch.randn_like(final_output)
            random_target = torch.randn_like(projected_target)
            random_cos = torch.nn.functional.cosine_similarity(random_output, random_target, dim=-1)
            random_loss = 1.0 - torch.mean(random_cos)
            print(f"         random cosine_similarity: {torch.mean(random_cos).item():.6f}")
            print(f"         random dialogue_loss: {random_loss.item():.6f}")
            
            if random_loss.item() > 0.5:
                print(f"         [OK] Random vectors дают нормальный loss > 0.5")
            else:
                print(f"         [ALERT] Даже random vectors дают низкий loss!")
    
    def _test_surface_consistency_loss(self, trainer, outputs, targets):
        """Тестирование surface consistency loss"""
        print(f"\n   [HOME] SURFACE CONSISTENCY LOSS:")
        
        input_surface = outputs['input_surface']
        output_surface = outputs['final_output']  # Это должно быть output_surface
        
        print(f"      input_surface: shape={input_surface.shape}, norm={input_surface.norm().item():.6f}")
        print(f"      output_surface: shape={output_surface.shape}, norm={output_surface.norm().item():.6f}")
        
        # Manual surface consistency loss
        surface_loss = torch.nn.functional.mse_loss(input_surface, output_surface)
        print(f"      surface_consistency_loss (MSE): {surface_loss.item():.6f}")
        
        if surface_loss.item() < 0.001:
            print(f"      [ALERT] ПРОБЛЕМА: Слишком маленький surface loss!")
            print(f"         Разность: {(input_surface - output_surface).abs().mean().item():.8f}")
            print(f"         Возможно input_surface == output_surface")
        else:
            print(f"      [OK] Surface loss выглядит нормально")
    
    def _test_internal_dynamics_loss(self, trainer, outputs, targets):
        """Тестирование internal dynamics loss"""
        print(f"\n   [GEAR] INTERNAL DYNAMICS LOSS:")
        
        if 'internal_state' in outputs:
            internal_state = outputs['internal_state']
            print(f"      internal_state: shape={internal_state.shape}, norm={internal_state.norm().item():.6f}")
            
            # Должен быть какой-то regularization loss
            # Обычно это energy функция или stability loss
            internal_loss = torch.mean(internal_state**2)  # L2 regularization
            print(f"      internal_dynamics_loss (L2): {internal_loss.item():.6f}")
            
            if internal_loss.item() < 0.001:
                print(f"      [ALERT] Очень маленький internal loss")
            else:
                print(f"      [OK] Internal loss выглядит разумно")
        else:
            print(f"      [WARNING] Нет internal_state в outputs")
    
    def _test_full_loss_computation(self, trainer, dataset):
        """Тестирование полного loss computation"""
        print(f"\n[TARGET] ПОЛНЫЙ LOSS COMPUTATION:")
        
        # Получаем sample
        sample = dataset[0]
        if isinstance(sample, tuple):
            question_emb, answer_emb = sample
            question_emb = question_emb.unsqueeze(0).to(self.device)
            answer_emb = answer_emb.unsqueeze(0).to(self.device)
        else:
            question_emb = sample['question_embedding'].unsqueeze(0).to(self.device)
            answer_emb = sample['answer_embedding'].unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = trainer.forward(question_emb)
            
        targets = {
            'target_embedding': answer_emb,
            'target_surface': outputs['input_surface']
        }
        
        # Compute loss через trainer
        losses = trainer.compute_loss(outputs, targets)
        
        print(f"   Компоненты loss от trainer.compute_loss:")
        total_loss = 0
        for key, loss_tensor in losses.items():
            if torch.is_tensor(loss_tensor):
                loss_val = loss_tensor.item()
                total_loss += loss_val
                print(f"      {key}: {loss_val:.6f}")
                
                if loss_val == 0.0:
                    print(f"         [ALERT] {key} = 0.0 - ПРОБЛЕМА!")
                elif loss_val < 0.01:
                    print(f"         [WARNING] {key} очень маленький")
                else:
                    print(f"         [OK] {key} выглядит нормально")
        
        print(f"   Total loss: {total_loss:.6f}")
        
        if total_loss == 0.0:
            print(f"   [ALERT] КРИТИЧЕСКАЯ ПРОБЛЕМА: Total loss = 0.0!")
            print(f"      Все компоненты loss равны нулю")
            print(f"      Обучение невозможно в таком состоянии")
        elif total_loss < 0.1:
            print(f"   [WARNING] Total loss очень маленький - может быть проблемой")
        else:
            print(f"   [OK] Total loss выглядит разумно для начала обучения")
    
    def _test_different_scenarios(self, trainer):
        """Тестирование разных scenarios"""
        print(f"\n[MASK] ТЕСТИРОВАНИЕ РАЗНЫХ SCENARIOS:")
        
        # Scenario 1: Очень разные embeddings
        print(f"   Scenario 1: Противоположные векторы")
        question_emb = torch.ones(1, 768, device=self.device)
        answer_emb = -torch.ones(1, 768, device=self.device)
        
        with torch.no_grad():
            outputs = trainer.forward(question_emb)
            
        targets = {'target_embedding': answer_emb, 'target_surface': outputs['input_surface']}
        losses = trainer.compute_loss(outputs, targets)
        
        scenario1_loss = sum(loss.item() for loss in losses.values() if torch.is_tensor(loss))
        print(f"      Total loss с противоположными векторами: {scenario1_loss:.6f}")
        
        # Scenario 2: Random embeddings
        print(f"   Scenario 2: Random векторы")
        question_emb = torch.randn(1, 768, device=self.device)
        answer_emb = torch.randn(1, 768, device=self.device)
        
        with torch.no_grad():
            outputs = trainer.forward(question_emb)
            
        targets = {'target_embedding': answer_emb, 'target_surface': outputs['input_surface']}
        losses = trainer.compute_loss(outputs, targets)
        
        scenario2_loss = sum(loss.item() for loss in losses.values() if torch.is_tensor(loss))
        print(f"      Total loss с random векторами: {scenario2_loss:.6f}")
        
        # Scenario 3: Нулевые embeddings
        print(f"   Scenario 3: Нулевые векторы")
        question_emb = torch.zeros(1, 768, device=self.device)
        answer_emb = torch.zeros(1, 768, device=self.device)
        
        with torch.no_grad():
            outputs = trainer.forward(question_emb)
            
        targets = {'target_embedding': answer_emb, 'target_surface': outputs['input_surface']}
        losses = trainer.compute_loss(outputs, targets)
        
        scenario3_loss = sum(loss.item() for loss in losses.values() if torch.is_tensor(loss))
        print(f"      Total loss с нулевыми векторами: {scenario3_loss:.6f}")
        
        # Анализ результатов
        print(f"\n   [DATA] Анализ scenarios:")
        if scenario1_loss > 1.0 and scenario2_loss > 0.5:
            print(f"      [OK] Loss function работает - разные inputs дают разные losses")
        elif all(loss < 0.01 for loss in [scenario1_loss, scenario2_loss, scenario3_loss]):
            print(f"      [ALERT] ПРОБЛЕМА: Все scenarios дают нулевой loss!")
            print(f"         Loss function не работает правильно")
        else:
            print(f"      [WARNING] Частичная проблема - некоторые scenarios дают нулевой loss")

def main():
    """Запуск диагностики loss function"""
    diagnostics = LossFunctionDiagnostics()
    diagnostics.run_diagnostics()
    
    print("\n[TARGET] СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Если loss function работает правильно - проблема в данных")
    print("2. Если loss = 0 во всех scenarios - проблема в реализации loss")
    print("3. Если только real data дает 0 - проблема в embeddings или projection")

if __name__ == "__main__":
    main() 