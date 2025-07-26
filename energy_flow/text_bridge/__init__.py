"""
Text Bridge модуль для energy_flow архитектуры
==============================================

Двунаправленное преобразование между текстом и эмбеддингами поверхности куба:
- TextToCubeEncoder: текст → эмбеддинги поверхности куба (surface_dim)
- CubeToTextDecoder: эмбеддинги поверхности куба (surface_dim) → текст  
- TextCache: кэширование известных пар
- BridgeTrainer: обучение преобразователей

Интеграция с основной архитектурой energy_flow для контроля
качества обучения и возможности общения с моделью на естественном языке.
"""

from .text_to_cube_encoder import TextToCubeEncoder, create_text_to_cube_encoder
from .cube_to_text_decoder import CubeToTextDecoder, create_cube_to_text_decoder
from .text_cache import TextCache, create_text_cache, CachedTextToCubeEncoder, CachedCubeToTextDecoder
# from .bridge_trainer import BridgeTrainer

__all__ = [
    'TextToCubeEncoder',
    'create_text_to_cube_encoder',
    'CubeToTextDecoder',
    'create_cube_to_text_decoder', 
    'TextCache',
    'create_text_cache',
    'CachedTextToCubeEncoder',
    'CachedCubeToTextDecoder',
    # 'BridgeTrainer'
]