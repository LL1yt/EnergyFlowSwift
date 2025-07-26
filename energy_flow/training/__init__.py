"""
Training <>4C;L 4;O energy_flow 0@E8B5:BC@K
==========================================

>;=>F5==K9 B@5=8@>2>G=K9 ?09?;09= A 8=B53@0F859 text_bridge:
- EnergyTrainer: >A=>2=>9 :;0AA 4;O >1CG5=8O
- ><18=8@>20==K9 loss (energy + text)
- GPU >?B8<870F8O 8 <>=8B>@8=3
- '5:?>8=B8=3 8 20;840F8O
"""

from .energy_trainer import EnergyTrainer, create_energy_trainer

__all__ = [
    'EnergyTrainer',
    'create_energy_trainer'
]