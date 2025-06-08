#!/usr/bin/env python3
"""
🔬 МИНИМАЛЬНЫЙ ТЕСТ TENSOR VERSION CONFLICTS
Изолированная проверка без лишнего кода
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.cell_prototype.architectures.gmlp_cell import GatedMLPCell

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalEmergentCell(nn.Module):
    """Минимальная версия EmergentGMLPCell для изоляции проблемы"""
    
    def __init__(self, state_size=32, memory_dim=16):
        super().__init__()
        self.state_size = state_size
        self.memory_dim = memory_dim
        
        # Создаем только базовый gMLP
        self.base_gmlp = GatedMLPCell(
            state_size=state_size,
            memory_dim=memory_dim,
            hidden_dim=128,
            neighbor_count=6,
            external_input_size=12
        )
        
        # Простейший projection layer (GatedMLPCell уже выводит state_size)
        self.output_projection = nn.Linear(state_size, state_size)
        
    def forward(self, input_state):
        """Простейший forward pass"""
        batch_size = input_state.shape[0]
        
        # Создаем фиктивные neighbor states (нулевые)
        neighbor_states = torch.zeros(batch_size, 6, self.state_size, device=input_state.device)
        
        # Base gMLP processing - правильные аргументы
        gmlp_output = self.base_gmlp(
            neighbor_states=neighbor_states,
            own_state=input_state,
            external_input=None
        )
        
        # Simple projection
        output = self.output_projection(gmlp_output)
        
        return output

class MinimalTrainer(nn.Module):
    """Минимальный trainer для изоляции tensor version conflict"""
    
    def __init__(self):
        super().__init__()
        
        # Один cell для тестирования
        self.cell = MinimalEmergentCell()
        
        # Простая loss function
        self.loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        
    def train_step(self, input_data, target_data):
        """Один шаг обучения"""
        
        # Forward pass
        output = self.cell(input_data)
        
        # Loss computation
        loss = self.loss_fn(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def debug_tensor_versions(self, step_name):
        """Детальная диагностика tensor versions"""
        logger.debug(f"\n🔍 [DEBUG] {step_name} - Tensor Versions:")
        
        for name, param in self.named_parameters():
            if hasattr(param, '_version'):
                logger.debug(f"  {name}: version {param._version}, shape {param.shape}")
        
        # Проверяем buffers
        for name, buffer in self.named_buffers():
            if hasattr(buffer, '_version'):
                logger.debug(f"  {name} (buffer): version {buffer._version}, shape {buffer.shape}")

def test_tensor_version_conflict():
    """Тест на tensor version conflicts в минимальной обстановке"""
    
    logger.info("🚀 [TEST] Starting minimal tensor version conflict test...")
    
    # Создаем trainer
    trainer = MinimalTrainer()
    
    # Создаем тестовые данные
    input_data = torch.randn(1, 32, requires_grad=True)  # batch=1, state_size=32
    target_data = torch.randn(1, 32)
    
    logger.info("📊 [TEST] Initial tensor versions:")
    trainer.debug_tensor_versions("INITIAL")
    
    # Первый training step
    logger.info("\n🔄 [TEST] Step 1...")
    try:
        loss_1 = trainer.train_step(input_data.clone(), target_data)
        logger.info(f"✅ [TEST] Step 1 successful, loss: {loss_1:.6f}")
        
        trainer.debug_tensor_versions("AFTER_STEP_1")
        
    except Exception as e:
        logger.error(f"❌ [TEST] Step 1 failed: {e}")
        return False
    
    # Второй training step - тут должна быть проблема
    logger.info("\n🔄 [TEST] Step 2...")
    try:
        loss_2 = trainer.train_step(input_data.clone(), target_data)
        logger.info(f"✅ [TEST] Step 2 successful, loss: {loss_2:.6f}")
        
        trainer.debug_tensor_versions("AFTER_STEP_2")
        
    except Exception as e:
        logger.error(f"❌ [TEST] Step 2 failed: {e}")
        logger.error(f"🔍 [DEBUG] Exception details: {type(e).__name__}: {e}")
        
        # Попытка понять, что именно вызвало ошибку
        trainer.debug_tensor_versions("ERROR_STATE")
        return False
    
    # Третий step для уверенности
    logger.info("\n🔄 [TEST] Step 3...")
    try:
        loss_3 = trainer.train_step(input_data.clone(), target_data)
        logger.info(f"✅ [TEST] Step 3 successful, loss: {loss_3:.6f}")
        
    except Exception as e:
        logger.error(f"❌ [TEST] Step 3 failed: {e}")
        return False
    
    logger.info("🎉 [TEST] All steps successful - no tensor version conflicts detected!")
    return True

if __name__ == "__main__":
    success = test_tensor_version_conflict()
    
    if success:
        print("\n✅ РЕЗУЛЬТАТ: Tensor version conflicts не обнаружены в минимальном тесте")
        print("   → Проблема может быть в более сложных компонентах (spatial propagation, multi-objective loss)")
    else:
        print("\n❌ РЕЗУЛЬТАТ: Tensor version conflicts воспроизведены в минимальном тесте")
        print("   → Проблема в базовых компонентах (GatedMLPCell или простые операции)") 