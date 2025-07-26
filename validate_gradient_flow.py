#!/usr/bin/env python3
"""
Диагностический скрипт для исследования gradient flow в energy_flow training
===========================================================================

Изолированное тестирование компонентов для выявления проблем с градиентами:
- FlowProcessor gradient flow
- Individual components testing  
- Loss computation analysis
- In-place operations detection
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import traceback

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.core import FlowProcessor, EnergyLattice, SimpleNeuron, EnergyCarrier
from energy_flow.text_bridge import TextToCubeEncoder, CubeToTextDecoder
from energy_flow.utils.logging import get_logger, DEBUG_TRAINING

logger = get_logger(__name__)

def setup_test_environment():
    """Настройка тестового окружения"""
    print("🔧 Setting up test environment...")
    
    # Debug конфигурация для быстрого тестирования
    config = create_debug_config()
    set_energy_config(config)
    
    # Устанавливаем CUDA если доступно
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    return config

def test_flow_processor_gradients(config):
    """Детальное тестирование gradient flow через FlowProcessor"""
    print("\n🔬 Testing FlowProcessor gradient flow...")
    
    batch_size = 2  # Маленький batch для детальной диагностики
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Создаем FlowProcessor
        flow_processor = FlowProcessor(config).to(device)
        
        # Входные данные с градиентами
        input_embeddings = torch.randn(batch_size, 768, device=device, requires_grad=True)
        print(f"📊 Input embeddings: {input_embeddings.shape}, requires_grad={input_embeddings.requires_grad}")
        
        # Прямой проход
        print("🔄 Forward pass through FlowProcessor...")
        output_surface = flow_processor.forward(input_embeddings, max_steps=5)  # Короткий проход
        
        print(f"📊 Output surface: {output_surface.shape}, requires_grad={output_surface.requires_grad}")
        print(f"📊 Output stats: mean={output_surface.mean():.4f}, std={output_surface.std():.4f}")
        
        # Простой loss для проверки градиентов
        loss = output_surface.sum()
        print(f"📊 Loss: {loss:.4f}, requires_grad={loss.requires_grad}")
        
        # Проверяем что loss имеет grad_fn
        if loss.grad_fn is not None:
            print(f"✅ Loss has grad_fn: {type(loss.grad_fn).__name__}")
        else:
            print("❌ Loss has no grad_fn!")
            return {"FlowProcessor": "❌ No grad_fn in loss"}
        
        # Обратное распространение
        print("🔄 Backward pass...")
        loss.backward()
        
        # Проверяем градиенты в компонентах
        components_with_grads = {}
        
        # FlowProcessor parameters
        fp_params = sum(1 for p in flow_processor.parameters() if p.grad is not None)
        fp_total = sum(1 for p in flow_processor.parameters())
        components_with_grads['FlowProcessor'] = f"{fp_params}/{fp_total}"
        
        # Neuron parameters
        neuron_params = sum(1 for p in flow_processor.neuron.parameters() if p.grad is not None)
        neuron_total = sum(1 for p in flow_processor.neuron.parameters())
        components_with_grads['SimpleNeuron'] = f"{neuron_params}/{neuron_total}"
        
        # Carrier parameters
        carrier_params = sum(1 for p in flow_processor.carrier.parameters() if p.grad is not None)
        carrier_total = sum(1 for p in flow_processor.carrier.parameters())
        components_with_grads['EnergyCarrier'] = f"{carrier_params}/{carrier_total}"
        
        # Mapper parameters
        mapper_params = sum(1 for p in flow_processor.mapper.parameters() if p.grad is not None)
        mapper_total = sum(1 for p in flow_processor.mapper.parameters())
        components_with_grads['EmbeddingMapper'] = f"{mapper_params}/{mapper_total}"
        
        print("📊 Components with gradients:")
        for comp, ratio in components_with_grads.items():
            print(f"  {comp}: {ratio}")
        
        # Проверяем input gradients
        input_grad_norm = 0
        if input_embeddings.grad is not None:
            input_grad_norm = input_embeddings.grad.norm().item()
            print(f"✅ Input gradients: norm={input_grad_norm:.6f}")
        else:
            print("❌ No gradients in input_embeddings")
        
        return {
            "FlowProcessor": "✅ Gradient flow successful", 
            "components": components_with_grads,
            "input_grad_norm": input_grad_norm
        }
        
    except Exception as e:
        print(f"❌ FlowProcessor gradient test failed: {e}")
        traceback.print_exc()
        return {"FlowProcessor": f"❌ Error: {e}"}

def debug_gradient_flow(config):
    """Подробная диагностика gradient flow в реальном training setup"""
    print("\n🐛 Debugging real training setup...")
    
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Создаем компоненты как в реальном training
        flow_processor = FlowProcessor(config).to(device)
        text_encoder = TextToCubeEncoder(config).to(device)
        
        # Teacher embeddings (как в реальном training)
        teacher_input = torch.randn(batch_size, 768, device=device, requires_grad=False)  # Teacher без градиентов!
        teacher_target = torch.randn(batch_size, 768, device=device, requires_grad=False)
        
        # Тексты
        input_texts = ["Test input text", "Another input"]
        target_texts = ["Target output text", "Another target"]
        
        print(f"📊 Teacher embeddings: input={teacher_input.requires_grad}, target={teacher_target.requires_grad}")
        
        # 1. Основной energy flow
        print("🔄 Energy flow forward pass...")
        cube_output = flow_processor.forward(teacher_input, max_steps=3)
        print(f"📊 Cube output: {cube_output.shape}, requires_grad={cube_output.requires_grad}")
        
        # ⚠️ КРИТИЧЕСКАЯ ПРОВЕРКА: Почему cube_output НЕ имеет градиентов?
        if not cube_output.requires_grad:
            print("🚨 ПРОБЛЕМА НАЙДЕНА: FlowProcessor.forward() не сохраняет градиенты!")
            print("🔍 Это объясняет ошибку 'element 0 of tensors does not require grad'")
            
            # Дополнительная диагностика
            print("🔍 Диагностика внутри FlowProcessor:")
            
            # Проверим mapper
            target_surface = flow_processor.mapper.input_mapper.forward(teacher_target)
            print(f"  📊 Mapper output: requires_grad={target_surface.requires_grad}")
            
            # Проверим lattice
            print("  📊 Checking EnergyLattice operations...")
            
            return {
                "status": "🚨 CRITICAL: FlowProcessor breaks gradient flow",
                "cube_output_has_grad": cube_output.requires_grad,
                "investigation": "FlowProcessor.forward() не сохраняет градиенты"
            }
        
        # 2. Target mapping 
        print("🔄 Target mapping...")
        target_surface = flow_processor.mapper.input_mapper.forward(teacher_target)
        if target_surface.dim() == 3:
            target_surface = target_surface.view(batch_size, -1)
        print(f"📊 Target surface: {target_surface.shape}, requires_grad={target_surface.requires_grad}")
        
        # 3. Energy loss
        print("🔄 Computing energy loss...")
        energy_loss = nn.functional.mse_loss(cube_output, target_surface)
        print(f"📊 Energy loss: {energy_loss:.4f}, requires_grad={energy_loss.requires_grad}")
        
        # 4. Text encoding
        print("🔄 Text encoding...")
        encoder_outputs = text_encoder.encode_text(input_texts)
        print(f"📊 Encoder outputs: {encoder_outputs.shape}, requires_grad={encoder_outputs.requires_grad}")
        
        # 5. Text loss
        print("🔄 Computing text loss...")
        text_loss = nn.functional.mse_loss(encoder_outputs, target_surface)
        print(f"📊 Text loss: {text_loss:.4f}, requires_grad={text_loss.requires_grad}")
        
        # 6. Combined loss
        print("🔄 Computing combined loss...")
        total_loss = energy_loss + 0.1 * text_loss
        print(f"📊 Total loss: {total_loss:.4f}, requires_grad={total_loss.requires_grad}")
        
        # 7. Проверяем grad_fn цепочку
        print("🔗 Gradient function chain:")
        if total_loss.grad_fn:
            print(f"  total_loss.grad_fn: {type(total_loss.grad_fn).__name__}")
        else:
            print("  ❌ No grad_fn in total_loss!")
        
        # 8. Backward pass
        print("🔄 Backward pass...")
        total_loss.backward()
        
        return {
            "status": "✅ Debug successful",
            "energy_loss": energy_loss.item(),
            "text_loss": text_loss.item(),
            "total_loss": total_loss.item()
        }
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()
        return {"status": f"❌ Error: {e}"}

def investigate_flow_processor_internals(config):
    """Исследование внутренних операций FlowProcessor"""
    print("\n🔍 Investigating FlowProcessor internals...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    
    try:
        flow_processor = FlowProcessor(config).to(device)
        input_embeddings = torch.randn(batch_size, 768, device=device, requires_grad=True)
        
        print(f"📊 Input: requires_grad={input_embeddings.requires_grad}")
        
        # Пошаговая диагностика
        print("🔄 Step 1: mapper.map_to_surface")
        cell_energies = flow_processor.mapper.map_to_surface(input_embeddings)
        print(f"  📊 Cell energies: {len(cell_energies)} cells")
        for i, (pos, energy, batch_idx) in enumerate(cell_energies[:3]):  # Первые 3
            print(f"    Cell {i}: energy.requires_grad={energy.requires_grad}")
        
        print("🔄 Step 2: lattice.place_initial_energy")
        flow_ids = flow_processor.lattice.place_initial_energy(input_embeddings, flow_processor.mapper)
        print(f"  📊 Created {len(flow_ids)} flows")
        
        print("🔄 Step 3: _collect_final_surface_output")
        output_surface, completed_flows = flow_processor._collect_final_surface_output()
        print(f"  📊 Output surface: requires_grad={output_surface.requires_grad}")
        
        if not output_surface.requires_grad:
            print("🚨 ПРОБЛЕМА: _collect_final_surface_output не сохраняет градиенты!")
            print("🔍 Нужно исследовать collect_buffered_surface_energy")
        
        return {
            "input_has_grad": input_embeddings.requires_grad,
            "output_has_grad": output_surface.requires_grad,
            "problem_location": "_collect_final_surface_output" if not output_surface.requires_grad else "unknown"
        }
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """Главная функция запуска всех тестов"""
    print("🔬 Energy Flow Gradient Diagnostics")
    print("=" * 50)
    
    config = setup_test_environment()
    
    # Запускаем все тесты
    tests = [
        ("FlowProcessor Gradients", lambda: test_flow_processor_gradients(config)),
        ("Gradient Flow Debug", lambda: debug_gradient_flow(config)),
        ("FlowProcessor Internals", lambda: investigate_flow_processor_internals(config))
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = {"error": str(e)}
    
    # Итоговый отчет
    print(f"\n{'='*20} SUMMARY {'='*20}")
    for test_name, result in results.items():
        print(f"📋 {test_name}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {result}")
    
    print(f"\n🏁 Diagnostics completed!")

if __name__ == "__main__":
    main()