#!/usr/bin/env python3
"""
🔬 ДИАГНОСТИЧЕСКИЙ ТЕСТ: Анализ tensor version conflicts между training steps

Цель: Понять почему первый train_step проходит, а второй падает с version error.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from training.embedding_trainer.emergent_training_stage_3_1_4_1_no_st import EmergentCubeTrainer, EmergentTrainingConfig

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tensor_version_debug.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_tensor_versions():
    """Тест версий тензоров между training steps"""
    
    print("🔬 === TENSOR VERSION DIAGNOSTIC TEST ===")
    
    # Создаем минимальную конфигурацию
    config = EmergentTrainingConfig()
    config.cube_dimensions = (3, 3, 3)  # Минимальный куб для быстрой диагностики
    config.mixed_precision = False  # Отключаем для простоты
    config.gradient_checkpointing = False
    
    # Создаем trainer
    trainer = EmergentCubeTrainer(config=config, device="cpu")
    print(f"✅ Trainer created with {trainer.get_system_info()['total_system_params']} parameters")
    
    # Создаем test data
    batch_size = 4
    question_embeddings = torch.randn(batch_size, 225, requires_grad=True)
    answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    
    print("\n🔍 === TRAINING STEP 1 ===")
    try:
        # Первый training step
        loss_1 = trainer.train_step(question_embeddings, answer_embeddings)
        print(f"✅ Step 1 SUCCESS: loss = {loss_1['total_loss']:.6f}")
        
        # Диагностика состояния после первого шага
        trainer._debug_tensor_versions("AFTER Step 1")
        
    except Exception as e:
        print(f"❌ Step 1 FAILED: {e}")
        return False
    
    print("\n🔍 === TRAINING STEP 2 ===")
    try:
        # Второй training step (здесь должна быть ошибка)
        loss_2 = trainer.train_step(question_embeddings, answer_embeddings)
        print(f"✅ Step 2 SUCCESS: loss = {loss_2['total_loss']:.6f}")
        
    except RuntimeError as e:
        if "is at version" in str(e) and "expected version" in str(e):
            print(f"❌ Step 2 FAILED with VERSION ERROR: {e}")
            
            # Детальный анализ версий проблемных тензоров
            print("\n🔍 === DETAILED VERSION ANALYSIS ===")
            trainer._debug_tensor_versions("DURING ERROR Step 2")
            
            return False
        else:
            print(f"❌ Step 2 FAILED with OTHER ERROR: {e}")
            return False
    
    print("\n✅ === ALL TESTS PASSED ===")
    return True

def test_specialization_tracker_isolation():
    """Тест изоляции specialization_tracker между вызовами"""
    
    print("\n🧪 === SPECIALIZATION TRACKER ISOLATION TEST ===")
    
    # Создаем минимальную gMLP cell
    from training.embedding_trainer.emergent_training_stage_3_1_4_1_no_st import EmergentGMLPCell
    
    cell = EmergentGMLPCell(state_size=8, neighbor_count=6, hidden_dim=8)
    
    # Проверяем версию tracker'а до и после forward pass
    initial_version = cell.specialization_tracker._version if hasattr(cell.specialization_tracker, '_version') else 'N/A'
    print(f"🔍 Initial tracker version: {initial_version}")
    
    # Первый forward pass
    neighbor_states = torch.randn(1, 6, 8)
    own_state = torch.randn(1, 8)
    
    output_1 = cell(neighbor_states, own_state)
    version_after_1 = cell.specialization_tracker._version if hasattr(cell.specialization_tracker, '_version') else 'N/A'
    print(f"🔍 After forward 1 version: {version_after_1}")
    
    # Второй forward pass (проблемное место)
    output_2 = cell(neighbor_states, own_state)
    version_after_2 = cell.specialization_tracker._version if hasattr(cell.specialization_tracker, '_version') else 'N/A'
    print(f"🔍 After forward 2 version: {version_after_2}")
    
    print(f"✅ Specialization tracker isolation test completed")

if __name__ == "__main__":
    print("🚀 Starting tensor version diagnostic tests...")
    
    # Тест 1: Полный training steps
    success = test_tensor_versions()
    
    # Тест 2: Изоляция specialization tracker
    test_specialization_tracker_isolation()
    
    if success:
        print("\n🎉 ALL DIAGNOSTIC TESTS PASSED")
    else:
        print("\n⚠️ DIAGNOSTIC TESTS REVEALED ISSUES - check logs for details") 