#!/usr/bin/env python3
"""
Простой отладочный тест без сложного логирования
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("[SYNC] Starting debug test...")

try:
    from new_rebuild.config import SimpleProjectConfig
    print("[OK] Config imported")
    
    config = SimpleProjectConfig()
    print(f"[OK] Config created")
    print(f"   fallback_cpu: {config.device.fallback_cpu}")
    print(f"   max_total_samples: {config.training_embedding.max_total_samples}")
    
    # Принудительно установим малый лимит
    config.training_embedding.max_total_samples = 10
    print(f"[OK] Set max_total_samples to: {config.training_embedding.max_total_samples}")
    
    from new_rebuild.utils.device_manager import get_device_manager
    print("[OK] DeviceManager imported")
    
    device_manager = get_device_manager()
    print(f"[OK] GPU available: {device_manager.is_cuda()}")
    
    if device_manager.is_cuda():
        print(f"[OK] GPU memory: {device_manager.get_available_memory_gb():.1f}GB")
    
    print("\n[SYNC] Testing dataset creation...")
    
    from new_rebuild.core.training.utils import create_training_dataloader
    print("[OK] Import successful")
    
    print("[SYNC] Creating dataloader...")
    dataloader, stats = create_training_dataloader(
        config=config,
        max_total_samples=5,  # Крошечный лимит для теста
        shuffle=False,
        num_workers=0  # Без воркеров для Windows
    )
    print(f"[OK] DataLoader created!")
    print(f"   Total samples: {stats.total_samples}")
    print(f"   Sources: {stats.source_distribution}")
    
    print("\n[SYNC] Testing one batch...")
    for i, batch in enumerate(dataloader):
        embeddings = batch['embedding']
        print(f"[OK] Batch {i}: {embeddings.shape}, device: {embeddings.device}")
        break
    
    print("\n[OK] DEBUG TEST COMPLETED!")
    
except Exception as e:
    print(f"[ERROR] ERROR: {e}")
    import traceback
    traceback.print_exc()