#!/usr/bin/env python3
"""
Performance Optimization Script
==============================

Quick fixes for GPU optimization issues.
"""

import torch
import os
from pathlib import Path

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 for Ampere GPUs (30xx series)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Memory optimizations
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Environment variables for maximum performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['DISABLE_MKL'] = '1'  # Force GPU usage

print("✅ GPU optimizations applied")
print("🚀 Ready for high-performance testing")