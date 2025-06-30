#!/usr/bin/env python3
"""
Простой отладочный тест без сложного логирования
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("🔄 Starting debug test...")

try:
    from new_rebuild.config import SimpleProjectConfig
    print("✅ Config imported")
    
    config = SimpleProjectConfig()
    print(f"✅ Config created")
    print(f"   fallback_cpu: {config.device.fallback_cpu}")
    print(f"   max_total_samples: {config.training_embedding.max_total_samples}")
    
    # Принудительно установим малый лимит
    config.training_embedding.max_total_samples = 10
    print(f"✅ Set max_total_samples to: {config.training_embedding.max_total_samples}")
    
    from new_rebuild.utils.device_manager import get_device_manager
    print("✅ DeviceManager imported")
    
    device_manager = get_device_manager()
    print(f"✅ GPU available: {device_manager.is_cuda()}")
    
    if device_manager.is_cuda():
        print(f"✅ GPU memory: {device_manager.get_available_memory_gb():.1f}GB")
    
    print("\n🔄 Testing dataset creation...")
    
    from new_rebuild.core.training.utils import create_training_dataloader
    print("✅ Import successful")
    
    print("🔄 Creating dataloader...")
    dataloader, stats = create_training_dataloader(
        config=config,
        max_total_samples=5,  # Крошечный лимит для теста
        shuffle=False,
        num_workers=0  # Без воркеров для Windows
    )
    print(f"✅ DataLoader created!")
    print(f"   Total samples: {stats.total_samples}")
    print(f"   Sources: {stats.source_distribution}")
    
    print("\n🔄 Testing one batch...")
    for i, batch in enumerate(dataloader):
        embeddings = batch['embedding']
        print(f"✅ Batch {i}: {embeddings.shape}, device: {embeddings.device}")
        break
    
    print("\n✅ DEBUG TEST COMPLETED!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()