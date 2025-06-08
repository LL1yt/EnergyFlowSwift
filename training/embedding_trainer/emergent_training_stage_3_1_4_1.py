#!/usr/bin/env python3
"""
🧠 Stage 3.1.4.1: Emergent Training Infrastructure для 3D Cellular Neural Network

Реализация emergent processing концепции:
TRAINING MODE: 4096D LLaMA → 225D Surface → FULL CUBE INFLUENCE → 225D Surface → Learning
INFERENCE MODE: Question → 225D Front → [EMERGENT PROCESSING] → 225D Back → Answer

Ключевые компоненты:
1. EmergentTrainingConfig - конфигурация emergent training
2. EmergentMultiObjectiveLoss - multi-objective loss (surface + internal + dialogue)
3. EmergentSpatialPropagation - spatial propagation система 
4. EmergentGMLPCell - enhanced gMLP с spatial connectivity
5. EmergentCubeTrainer - основной trainer с full cube gradient flow
"""

import sys
import os

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import yaml
import json

# Импорты нашей системы
from .cube_trainer import CubeTrainer, TrainingConfig, EmbeddingMetrics
from .adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from data.embedding_adapter.universal_adapter import KNOWN_MODELS
from core.lattice_3d import Lattice3D, LatticeConfig
from core.embedding_processor import EmbeddingProcessor, ProcessingMode
from core.cell_prototype.architectures.gmlp_cell import GatedMLPCell

logger = logging.getLogger(__name__)


@dataclass
class EmergentTrainingConfig:
    """Конфигурация для emergent training с full cube влиянием"""
    
    # Base configuration
    teacher_model: str = "Meta-Llama-3-8B"
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11)
    
    # Emergent processing settings
    enable_full_cube_gradient: bool = True
    spatial_propagation_depth: int = 11  # All layers
    emergent_specialization: bool = True
    
    # gMLP cell configuration для 25K params target
    gmlp_config: Dict[str, Any] = None
    
    # Multi-objective loss configuration
    loss_weights: Dict[str, float] = None
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 8
    epochs: int = 15
    warmup_epochs: int = 3
    
    # Optimization settings
    gradient_balancing: bool = True
    adaptive_loss_weighting: bool = True
    
    def __post_init__(self):
        if self.gmlp_config is None:
            # OPTIMIZED configuration для точного 25K params target
            self.gmlp_config = {
                'state_size': 32,             # OPTIMIZED from parameter analysis
                'neighbor_count': 6,          # Standard 6-connectivity
                'hidden_dim': 32,             # OPTIMIZED from 128 → 32
                'external_input_size': 12,    # OPTIMIZED input dimension
                'memory_dim': 16,             # OPTIMIZED from 32 → 16
                'use_memory': True,
                'activation': 'gelu',
                'dropout': 0.1,
                'spatial_connections': True   # EMERGENT FEATURE - spatial connectivity
            }
        
        if self.loss_weights is None:
            self.loss_weights = {
                'surface_reconstruction': 0.3,  # Surface input → output consistency
                'internal_consistency': 0.3,    # Internal layer coherence
                'dialogue_similarity': 0.4      # Final Q→A similarity
            }


class EmergentMultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss function для emergent training
    
    Компоненты:
    1. Surface Reconstruction Loss - consistency между input/output surfaces
    2. Internal Consistency Loss - coherence внутренних layers
    3. Dialogue Similarity Loss - final Q→A semantic similarity
    """
    
    def __init__(self, config: EmergentTrainingConfig):
        super().__init__()
        self.config = config
        self.weights = config.loss_weights
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
        # Dimension projection для dialogue similarity (225D ↔ 4096D)
        self.surface_to_embedding = nn.Linear(225, 4096, bias=False)  # 225D → 4096D
        self.embedding_to_surface = nn.Linear(4096, 225, bias=False)  # 4096D → 225D
        
        # Adaptive weighting (if enabled)
        if config.adaptive_loss_weighting:
            self.weight_adaptation = nn.Parameter(
                torch.tensor([0.3, 0.3, 0.4], requires_grad=True)
            )
        else:
            self.register_buffer(
                'weight_adaptation',
                torch.tensor([self.weights['surface_reconstruction'],
                            self.weights['internal_consistency'], 
                            self.weights['dialogue_similarity']])
            )
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                internal_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss
        
        Args:
            outputs: Model outputs including surface and internal representations
            targets: Target embeddings and states
            internal_states: Internal layer states for consistency loss
            
        Returns:
            Dict with individual and total losses
        """
        
        # 1. Surface Reconstruction Loss
        if 'input_surface' in outputs and 'output_surface' in outputs:
            surface_loss = self.mse_loss(outputs['output_surface'], targets.get('target_surface', outputs['input_surface']))
        else:
            surface_loss = torch.tensor(0.0, device=outputs['final_output'].device)
        
        # 2. Internal Consistency Loss
        if internal_states is not None and internal_states.numel() > 0:
            # Consistency между соседними layers
            internal_loss = 0.0
            for i in range(internal_states.size(1) - 1):  # Across depth dimension
                layer_diff = internal_states[:, i+1] - internal_states[:, i]
                internal_loss += torch.mean(layer_diff ** 2)
            internal_loss = internal_loss / (internal_states.size(1) - 1)
        else:
            internal_loss = torch.tensor(0.0, device=outputs['final_output'].device)
        
        # 3. Dialogue Similarity Loss (с dimension matching)
        if 'final_output' in outputs and 'target_embedding' in targets:
            final_output = outputs['final_output']  # [batch, 225]
            target_embedding = targets['target_embedding']  # [batch, 4096]
            
            # Strategy: project target_embedding down to 225D (более efficient)
            if target_embedding.shape[-1] == 4096 and final_output.shape[-1] == 225:
                projected_target = self.embedding_to_surface(target_embedding)  # [batch, 225]
                cos_sim = self.cosine_similarity(final_output, projected_target)
            elif target_embedding.shape[-1] == 225 and final_output.shape[-1] == 225:
                # Already same dimension
                cos_sim = self.cosine_similarity(final_output, target_embedding)
            else:
                # Fallback: project final_output up to 4096D
                projected_output = self.surface_to_embedding(final_output)  # [batch, 4096]
                cos_sim = self.cosine_similarity(projected_output, target_embedding)
            
            dialogue_loss = 1.0 - torch.mean(cos_sim)
        else:
            dialogue_loss = torch.tensor(0.0, device=outputs['final_output'].device)
        
        # Normalize weights
        normalized_weights = torch.softmax(self.weight_adaptation, dim=0)
        
        # Total weighted loss
        total_loss = (normalized_weights[0] * surface_loss + 
                     normalized_weights[1] * internal_loss + 
                     normalized_weights[2] * dialogue_loss)
        
        return {
            'total_loss': total_loss,
            'surface_reconstruction_loss': surface_loss,
            'internal_consistency_loss': internal_loss,
            'dialogue_similarity_loss': dialogue_loss,
            'loss_weights': normalized_weights.detach()  # Detach для избежания grad issues
        }


class EmergentSpatialPropagation(nn.Module):
    """
    Spatial propagation система для full cube gradient flow
    
    Обеспечивает:
    - Signal propagation через все 11 layers depth
    - Cross-layer influence между соседними cells
    - Emergent specialization поддержка
    """
    
    def __init__(self, cube_dimensions: Tuple[int, int, int], 
                 cell_state_size: int = 32):
        super().__init__()
        self.cube_dims = cube_dimensions  # (15, 15, 11) = [width, height, depth]
        self.width, self.height, self.depth = cube_dimensions  # (15, 15, 11)
        self.state_size = cell_state_size
        
        # Cross-layer connection weights
        self.layer_connections = nn.Parameter(
            torch.randn(self.depth-1, cell_state_size, cell_state_size) * 0.01
        )
        
        # Propagation gating
        self.propagation_gate = nn.Sequential(
            nn.Linear(cell_state_size * 2, cell_state_size),
            nn.Sigmoid()
        )
        
        # Debug tracking
        self.propagation_count = 0
        self.last_layer_states = {}
    
    def forward(self, cube_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cube_states: [batch, depth, height, width, state_size]
            
        Returns:
            Enhanced states with spatial propagation
        """
        batch_size, depth, height, width, state_size = cube_states.shape
        
        self.propagation_count += 1
        
        logger.debug(f"🔍 [SpatialPropagation] Propagation #{self.propagation_count}")
        logger.debug(f"🔍 [SpatialPropagation] Input cube state: shape={cube_states.shape}, requires_grad={cube_states.requires_grad}")
        
        # Check for layer state reuse
        if self.last_layer_states:
            for layer_idx, last_state in self.last_layer_states.items():
                current_id = id(cube_states)
                last_id = id(last_state)
                if current_id == last_id:
                    logger.warning(f"⚠️ [SpatialPropagation] Tensor reuse detected! Layer {layer_idx}: current_id={current_id}")
        
        # Cross-layer propagation
        enhanced_states = cube_states.clone()
        
        for layer_idx in range(depth - 1):
            current_layer = cube_states[:, layer_idx]  # [batch, height, width, state_size]
            next_layer = cube_states[:, layer_idx + 1]
            
            # Flatten spatial dimensions for processing
            current_flat = current_layer.view(batch_size, -1, state_size)  # [batch, height*width, state_size]
            next_flat = next_layer.view(batch_size, -1, state_size)
            
            # Cross-layer influence
            connection_weights = self.layer_connections[layer_idx]  # [state_size, state_size]
            influenced_next = torch.bmm(
                current_flat, 
                connection_weights.unsqueeze(0).expand(batch_size, -1, -1)
            )
            
            # Gating mechanism
            combined = torch.cat([next_flat, influenced_next], dim=-1)  # [batch, height*width, state_size*2]
            gate = self.propagation_gate(combined)
            
            # Apply influence
            enhanced_next = next_flat + gate * influenced_next
            
            # Reshape back and store
            enhanced_states[:, layer_idx + 1] = enhanced_next.view(batch_size, height, width, state_size)
            
            # Store state для debugging
            self.last_layer_states[layer_idx] = enhanced_next.detach().clone()
        
        logger.debug(f"🔍 [SpatialPropagation] Final propagated state: shape={enhanced_states.shape}, requires_grad={enhanced_states.requires_grad}")
        
        return enhanced_states


class EmergentGMLPCell(nn.Module):
    """
    Enhanced gMLP Cell с spatial connectivity для emergent training
    
    Расширяет базовый GatedMLPCell:
    1. Spatial connections между соседними клетками
    2. Cross-layer influence mechanisms
    3. Emergent specialization support
    4. Enhanced gradient flow для full cube training
    """
    
    def __init__(self,
                 state_size: int = 32,
                 neighbor_count: int = 6,
                 hidden_dim: int = 32,
                 external_input_size: int = 12,
                 memory_dim: int = 16,
                 use_memory: bool = True,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 spatial_connections: bool = True):
        
        super().__init__()
        
        # === BASE gMLP CELL ===
        self.base_gmlp = GatedMLPCell(
            state_size=state_size,
            neighbor_count=neighbor_count,
            hidden_dim=hidden_dim,
            external_input_size=external_input_size,
            memory_dim=memory_dim,
            use_memory=use_memory,
            activation=activation,
            dropout=dropout
        )
        
        # === SPATIAL CONNECTIVITY ENHANCEMENTS ===
        self.spatial_connections = spatial_connections
        
        # Логирование параметров только один раз для всех cells
        if spatial_connections and not hasattr(EmergentGMLPCell, '_param_count_logged'):
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"🧠 EmergentGMLPCell: {total_params:,} params (target: ~25K)")
            EmergentGMLPCell._param_count_logged = True
        
        if spatial_connections:
            # Cross-cell influence mechanism
            self.spatial_weight_generator = nn.Sequential(
                nn.Linear(state_size * 2, hidden_dim),  # current + neighbor states
                nn.ReLU(),
                nn.Linear(hidden_dim, neighbor_count),  # weight for each neighbor
                nn.Softmax(dim=-1)
            )
            
            # Cross-layer influence projection
            self.cross_layer_projection = nn.Linear(state_size, state_size)
            
            # Emergent specialization tracking
            self.specialization_tracker = nn.Parameter(
                torch.zeros(1, state_size), requires_grad=False
            )
            
        # Debug tracking
        self.forward_count = 0
        self.last_output_id = None
        
        logger.debug(f"🔧 [EmergentGMLPCell] Created with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total parameters в cell"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self,
                neighbor_states: torch.Tensor,
                own_state: torch.Tensor,
                external_input: Optional[torch.Tensor] = None,
                layer_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass с spatial connectivity
        
        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size] (optional)
            layer_context: [batch, state_size] from other layers (optional)
            
        Returns:
            new_state: [batch, state_size] - enhanced state
        """
        
        self.forward_count += 1
        
        # Debug logging for tensor reuse detection
        state_id = id(own_state)
        logger.debug(f"🔍 [EmergentGMLPCell] Forward #{self.forward_count}, state_id={state_id}")
        
        if self.last_output_id and hasattr(own_state, 'grad_fn'):
            if own_state.grad_fn and hasattr(own_state.grad_fn, 'next_functions'):
                logger.debug(f"🔍 [EmergentGMLPCell] State grad_fn: {own_state.grad_fn}")
        
        # Check for memory state reuse
        if hasattr(self.base_gmlp, 'memory_state') and self.base_gmlp.memory_state is not None:
            mem_id = id(self.base_gmlp.memory_state)
            logger.debug(f"🔍 [EmergentGMLPCell] Memory state id={mem_id}, requires_grad={self.base_gmlp.memory_state.requires_grad}")
        
        # === ЭТАП 1: Base gMLP Processing ===
        base_output = self.base_gmlp(neighbor_states, own_state, external_input)
        
        if not self.spatial_connections:
            return base_output
            
        # === ЭТАП 2: Spatial Connectivity Enhancement ===
        
        # Spatial weighting для neighbor influence
        if neighbor_states.numel() > 0:
            batch_size = own_state.shape[0]
            
            # Compute spatial weights для каждого neighbor
            spatial_weights_input = []
            for i in range(neighbor_states.shape[1]):  # For each neighbor
                neighbor_state = neighbor_states[:, i]  # [batch, state_size]
                combined = torch.cat([own_state, neighbor_state], dim=-1)  # [batch, state_size*2]
                spatial_weights_input.append(combined)
            
            if spatial_weights_input:
                spatial_weights_input = torch.stack(spatial_weights_input, dim=1)  # [batch, neighbor_count, state_size*2]
                
                # Generate adaptive weights для каждого neighbor
                spatial_weights = []
                for i in range(neighbor_states.shape[1]):
                    weight = self.spatial_weight_generator(spatial_weights_input[:, i])  # [batch, neighbor_count]
                    spatial_weights.append(weight[:, i:i+1])  # Take weight for this neighbor
                
                spatial_weights = torch.cat(spatial_weights, dim=-1)  # [batch, neighbor_count]
                
                # Apply spatial weighting
                weighted_neighbors = neighbor_states * spatial_weights.unsqueeze(-1)  # [batch, neighbor_count, state_size]
                spatial_influence = torch.mean(weighted_neighbors, dim=1)  # [batch, state_size]
                
                # Combine с base output
                base_output = base_output + 0.1 * spatial_influence
        
        # === ЭТАП 3: Cross-layer Influence ===
        if layer_context is not None:
            cross_layer_influence = self.cross_layer_projection(layer_context)
            base_output = base_output + 0.05 * cross_layer_influence
            
        # === ЭТАП 4: Emergent Specialization Tracking ===
        with torch.no_grad():
            # Update specialization tracker (running average активации)
            current_activation = torch.mean(torch.abs(base_output), dim=0, keepdim=True)
            self.specialization_tracker.data = (
                0.99 * self.specialization_tracker.data + 
                0.01 * current_activation
            )
        
        # Store output id for debugging
        self.last_output_id = id(base_output)
        logger.debug(f"🔍 [EmergentGMLPCell] Output id={self.last_output_id}, requires_grad={base_output.requires_grad}")
        
        return base_output
    
    def get_specialization_score(self) -> float:
        """Получить score emergent specialization"""
        if self.spatial_connections:
            # Измеряем variance в specialization tracker
            variance = torch.var(self.specialization_tracker).item()
            return variance
        return 0.0


class EmergentCubeTrainer(nn.Module):
    """
    Enhanced trainer для Stage 3.1.4.1 Emergent Training Infrastructure
    
    Ключевые особенности:
    - Full cube gradient flow через все 2,475 клеток
    - gMLP neurons с 25K параметрами каждый
    - Multi-objective loss function
    - Spatial propagation система
    - Emergent behavior поддержка
    """
    
    def __init__(self, config: Optional[EmergentTrainingConfig] = None, device: str = "cpu"):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or EmergentTrainingConfig()
        self._device = torch.device(device)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        self.logger.info("🧠 EmergentCubeTrainer initialized for Stage 3.1.4.1")
        self.logger.info(f"   Cube: {self.config.cube_dimensions}")
        self.logger.info(f"   Full gradient flow: {self.config.enable_full_cube_gradient}")
        self.logger.info(f"   Spatial propagation depth: {self.config.spatial_propagation_depth}")
    
    def _initialize_components(self):
        """Initialize all emergent training components"""
        
        # 1. Base adapter integration (existing working component)
        adapter_config = AdapterIntegrationConfig(
            teacher_model=self.config.teacher_model,
            cube_dimensions=self.config.cube_dimensions,
            adapter_strategy="hierarchical",  # Optimal from Stage 3.1.3
            joint_training=True
        )
        
        self.base_trainer = AdapterCubeTrainer(adapter_config, str(self._device))
        self.base_trainer.initialize_components()
        
        # 2. Enhanced lattice с gMLP cells
        self._setup_enhanced_lattice()
        
        # 3. Spatial propagation system
        self.spatial_propagation = EmergentSpatialPropagation(
            self.config.cube_dimensions,
            cell_state_size=32  # gMLP state size
        ).to(self._device)
        
        # 4. Multi-objective loss
        self.loss_function = EmergentMultiObjectiveLoss(self.config).to(self._device)
        
        # 5. Optimizer for full system
        self._setup_optimizer()
    
    def _setup_enhanced_lattice(self):
        """Setup enhanced lattice с gMLP cells"""
        
        # Create enhanced lattice config
        lattice_config = LatticeConfig(
            dimensions=self.config.cube_dimensions,
            boundary_conditions="walls"
        )
        
        # Create lattice с enhanced cells
        self.enhanced_lattice = Lattice3D(lattice_config).to(self._device)
        
        # Replace cells с gMLP neurons
        total_cells = self.config.cube_dimensions[0] * self.config.cube_dimensions[1] * self.config.cube_dimensions[2]
        
        self.gmlp_cells = nn.ModuleList([
            EmergentGMLPCell(**self.config.gmlp_config) for _ in range(total_cells)
        ])
        
        self.logger.info(f"✅ Enhanced lattice created: {total_cells} gMLP cells")
        
        # Log parameter count
        total_params = sum(sum(p.numel() for p in cell.parameters()) for cell in self.gmlp_cells)
        avg_params = total_params / total_cells
        self.logger.info(f"   Average parameters per cell: {avg_params:.0f}")
        self.logger.info(f"   Total lattice parameters: {total_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer for full emergent system"""
        
        # Collect all parameters
        params = []
        
        # Base adapter parameters
        params.extend(self.base_trainer.adapter.parameters())
        
        # gMLP cell parameters
        for cell in self.gmlp_cells:
            params.extend(cell.parameters())
        
        # Spatial propagation parameters
        params.extend(self.spatial_propagation.parameters())
        
        # Loss function parameters (if adaptive)
        if self.config.adaptive_loss_weighting:
            params.extend(self.loss_function.parameters())
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        total_params = sum(p.numel() for p in params)
        self.logger.info(f"✅ Optimizer setup: {total_params:,} total parameters")
    
    def forward(self, surface_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass через emergent cube processing + debugging"""
        
        batch_size = surface_embeddings.shape[0]
        logger.debug(f"🔍 [FORWARD] Starting forward pass, batch_size={batch_size}")
        logger.debug(f"🔍 [FORWARD] Input surface: shape={surface_embeddings.shape}, requires_grad={surface_embeddings.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Input tensor id={id(surface_embeddings)}")
        
        # Step 1: Surface injection into cube
        logger.debug("🔍 [FORWARD] Step 1: Surface injection...")
        cube_states = self._inject_surface_to_cube(surface_embeddings)
        logger.debug(f"🔍 [FORWARD] Cube states created: shape={cube_states.shape}, requires_grad={cube_states.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Cube states id={id(cube_states)}")
        
        # Step 2: Full cube processing (emergent behavior)
        logger.debug("🔍 [FORWARD] Step 2: Full cube processing...")
        
        # Track tensor IDs before processing
        cube_id_before = id(cube_states)
        
        processed_cube = self._process_full_cube(cube_states)
        
        cube_id_after = id(processed_cube)
        logger.debug(f"🔍 [FORWARD] Cube processing complete: shape={processed_cube.shape}, requires_grad={processed_cube.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Cube ID before={cube_id_before}, after={cube_id_after}, changed={cube_id_before != cube_id_after}")
        
        # Step 3: Spatial propagation
        logger.debug("🔍 [FORWARD] Step 3: Spatial propagation...")
        
        propagated_cube = self.spatial_propagation(processed_cube)
        logger.debug(f"🔍 [FORWARD] Spatial propagation complete: shape={propagated_cube.shape}, requires_grad={propagated_cube.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Propagated cube id={id(propagated_cube)}")
        
        # Step 4: Output extraction
        logger.debug("🔍 [FORWARD] Step 4: Output extraction...")
        
        final_output = self._extract_output_surface(propagated_cube)
        logger.debug(f"🔍 [FORWARD] Final output extracted: shape={final_output.shape}, requires_grad={final_output.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Final output id={id(final_output)}")
        
        # Check for computational graph connections
        if hasattr(final_output, 'grad_fn') and final_output.grad_fn:
            logger.debug(f"🔍 [FORWARD] Final output grad_fn: {final_output.grad_fn}")
        
        # Step 5: Internal state analysis
        logger.debug("🔍 [FORWARD] Step 5: Internal state analysis...")
        
        internal_state = self._analyze_internal_state(propagated_cube)
        logger.debug(f"🔍 [FORWARD] Internal state: shape={internal_state.shape}, requires_grad={internal_state.requires_grad}")
        logger.debug(f"🔍 [FORWARD] Internal state id={id(internal_state)}")
        
        # Prepare outputs
        outputs = {
            'input_surface': surface_embeddings,  # Keep original input
            'cube_states': propagated_cube,
            'final_output': final_output,
            'internal_state': internal_state
        }
        
        # Log output tensor relationships
        logger.debug("🔍 [FORWARD] Output tensor analysis:")
        for key, tensor in outputs.items():
            if torch.is_tensor(tensor):
                logger.debug(f"   {key}: id={id(tensor)}, requires_grad={tensor.requires_grad}")
                if hasattr(tensor, 'grad_fn') and tensor.grad_fn:
                    logger.debug(f"   {key} grad_fn: {tensor.grad_fn}")
        
        logger.debug("🔍 [FORWARD] Forward pass completed")
        
        return outputs
    
    def _process_full_cube(self, cube_states: torch.Tensor) -> torch.Tensor:
        """Process entire cube through gMLP cells с debugging"""
        
        logger.debug("🔍 [PROCESS_CUBE] Starting full cube processing...")
        logger.debug(f"🔍 [PROCESS_CUBE] Input cube: shape={cube_states.shape}, id={id(cube_states)}")
        
        batch_size, depth, height, width, state_size = cube_states.shape
        
        # Flatten для cell processing
        flattened_states = cube_states.view(batch_size, -1, state_size)
        total_cells = flattened_states.shape[1]
        
        logger.debug(f"🔍 [PROCESS_CUBE] Processing {total_cells} cells...")
        
        # Process each cell
        processed_states = []
        
        for cell_idx in range(total_cells):
            if cell_idx < 5:  # Log first 5 cells
                logger.debug(f"🔍 [PROCESS_CUBE] Processing cell {cell_idx}")
            
            cell_state = flattened_states[:, cell_idx, :]  # [batch, state_size]
            
            # Get cell neighbors
            neighbors = self._get_cell_neighbors(cell_idx, flattened_states, batch_size, depth, height, width)
            
            # External input (zero for internal cells)
            external_input = torch.zeros(batch_size, self.config.external_input_size, 
                                       device=cell_state.device)
            
            # Process через соответствующую cell
            gmlp_cell = self.gmlp_cells[cell_idx % len(self.gmlp_cells)]
            
            # Track tensor reuse
            if cell_idx < 5:
                logger.debug(f"🔍 [PROCESS_CUBE] Cell {cell_idx} input id={id(cell_state)}")
            
            cell_output = gmlp_cell(cell_state, neighbors, external_input)
            
            if cell_idx < 5:
                logger.debug(f"🔍 [PROCESS_CUBE] Cell {cell_idx} output id={id(cell_output)}")
            
            processed_states.append(cell_output)
        
        # Stack processed states
        processed_flattened = torch.stack(processed_states, dim=1)
        
        # Reshape back to cube
        processed_cube = processed_flattened.view(batch_size, depth, height, width, state_size)
        
        logger.debug(f"🔍 [PROCESS_CUBE] Cube processing complete: shape={processed_cube.shape}, id={id(processed_cube)}")
        
        return processed_cube
    
    def _get_cell_neighbors(self, cell_idx: int, flattened_states: torch.Tensor, 
                           batch_size: int, depth: int, height: int, width: int) -> torch.Tensor:
        """Get neighbor states for a cell (6-connectivity в 3D)"""
        neighbors = []
        
        # Calculate 3D coordinates from flat index
        d = cell_idx // (height * width)
        h = (cell_idx % (height * width)) // width
        w = cell_idx % width
        
        # 6-connectivity directions: ±d, ±h, ±w
        directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        for dd, dh, dw in directions:
            nd, nh, nw = d + dd, h + dh, w + dw
            
            # Boundary check
            if 0 <= nd < depth and 0 <= nh < height and 0 <= nw < width:
                neighbor_idx = nd * height * width + nh * width + nw
                neighbor_state = flattened_states[:, neighbor_idx]
            else:
                # Boundary condition: zero state
                neighbor_state = torch.zeros(
                    batch_size, self.config.state_size, 
                    device=flattened_states.device
                )
            
            neighbors.append(neighbor_state)
        
        return torch.stack(neighbors, dim=1)  # [batch, 6, state_size]
    
    def _inject_surface_to_cube(self, surface_embeddings: torch.Tensor) -> torch.Tensor:
        """Inject 225D surface embeddings into 3D cube structure"""
        
        logger.debug("🔍 [INJECT_SURFACE] Starting surface injection...")
        
        batch_size = surface_embeddings.shape[0]
        width, height, depth = self.config.cube_dimensions  # [15, 15, 11]
        state_size = self.config.state_size  # 32
        
        logger.debug(f"🔍 [INJECT_SURFACE] Cube dims: {width}×{height}×{depth}, state_size: {state_size}")
        
        # Initialize cube with zeros
        cube_states = torch.zeros(
            batch_size, depth, height, width, state_size,
            device=surface_embeddings.device, dtype=surface_embeddings.dtype
        )
        
        # Surface embeddings should be 225D (15×15)
        if surface_embeddings.shape[-1] == 225:
            # Reshape to front face: [batch, 225] → [batch, 15, 15]
            front_face = surface_embeddings.view(batch_size, height, width)
        else:
            # Project to 225D if different size
            if not hasattr(self, 'surface_projection'):
                self.surface_projection = nn.Linear(
                    surface_embeddings.shape[-1], 225, 
                    device=surface_embeddings.device
                )
            projected = self.surface_projection(surface_embeddings)
            front_face = projected.view(batch_size, height, width)
        
        # Inject into front layer (depth=0)
        for h in range(height):
            for w in range(width):
                surface_value = front_face[:, h, w]  # [batch]
                
                # Distribute surface value across state dimensions
                cube_states[:, 0, h, w, 0] = surface_value  # Primary channel
                
                # Add structured noise to other state dimensions
                cube_states[:, 0, h, w, 1:] = torch.randn(
                    batch_size, state_size - 1, 
                    device=surface_embeddings.device
                ) * 0.01
        
        # Initialize other depth layers with decay
        for d in range(1, depth):
            decay_factor = 0.9 ** d
            cube_states[:, d] = cube_states[:, 0] * decay_factor
        
        logger.debug(f"🔍 [INJECT_SURFACE] Cube injection complete: {cube_states.shape}")
        
        return cube_states
    
    def _extract_output_surface(self, cube_states: torch.Tensor) -> torch.Tensor:
        """Extract 225D surface output from processed cube"""
        
        logger.debug("🔍 [EXTRACT_OUTPUT] Starting output extraction...")
        
        batch_size, depth, height, width, state_size = cube_states.shape
        
        # Extract from back layer (depth=-1)
        back_layer = cube_states[:, -1]  # [batch, height, width, state_size]
        
        # Average across state dimensions to get surface values
        surface_values = torch.mean(back_layer, dim=-1)  # [batch, height, width]
        
        # Flatten to 225D
        output_surface = surface_values.view(batch_size, -1)  # [batch, 225]
        
        logger.debug(f"🔍 [EXTRACT_OUTPUT] Output extraction complete: {output_surface.shape}")
        
        return output_surface
    
    def _analyze_internal_state(self, cube_states: torch.Tensor) -> torch.Tensor:
        """Analyze internal cube state for consistency loss"""
        
        logger.debug("🔍 [ANALYZE_INTERNAL] Starting internal analysis...")
        
        batch_size, depth, height, width, state_size = cube_states.shape
        
        # Internal layers (exclude front and back)
        if depth > 2:
            internal_layers = cube_states[:, 1:-1]  # [batch, depth-2, height, width, state_size]
            
            # Average across internal layers
            internal_state = torch.mean(internal_layers, dim=1)  # [batch, height, width, state_size]
            
            # Flatten for loss computation
            internal_flattened = internal_state.view(batch_size, -1)  # [batch, height*width*state_size]
        else:
            # If only 2 layers, use middle representation
            middle_layer = cube_states[:, depth//2]
            internal_flattened = middle_layer.view(batch_size, -1)
        
        logger.debug(f"🔍 [ANALYZE_INTERNAL] Internal analysis complete: {internal_flattened.shape}")
        
        return internal_flattened
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss"""
        return self.loss_function(outputs, targets, outputs.get('internal_state'))
    
    def train_step(self, question_embeddings: torch.Tensor, 
                   answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """Single training step с emergent processing + detailed debugging"""
        
        logger.debug("🔍 [TRAIN_STEP] Starting train_step...")
        logger.debug(f"🔍 [TRAIN_STEP] Input shapes: Q={question_embeddings.shape}, A={answer_embeddings.shape}")
        logger.debug(f"🔍 [TRAIN_STEP] Input requires_grad: Q={question_embeddings.requires_grad}, A={answer_embeddings.requires_grad}")
        
        self.train()
        
        # Clear gradients first
        logger.debug("🔍 [TRAIN_STEP] Clearing gradients...")
        self.optimizer.zero_grad()
        
        # Check for retained gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                logger.warning(f"⚠️ [TRAIN_STEP] Parameter {name} still has gradients after zero_grad()!")
        
        # Forward pass (create new computation graph)
        logger.debug("🔍 [TRAIN_STEP] Starting forward pass...")
        try:
            question_outputs = self.forward(question_embeddings)
            logger.debug(f"🔍 [TRAIN_STEP] Forward completed. Output keys: {list(question_outputs.keys())}")
            
            # Check computational graph attachment
            for key, tensor in question_outputs.items():
                if torch.is_tensor(tensor):
                    logger.debug(f"🔍 [TRAIN_STEP] {key}: shape={tensor.shape}, requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}")
            
        except Exception as e:
            logger.error(f"❌ [TRAIN_STEP] Forward pass failed: {e}")
            raise
        
        # Prepare targets
        logger.debug("🔍 [TRAIN_STEP] Preparing targets...")
        targets = {
            'target_embedding': answer_embeddings.detach(),  # Detach targets
            'target_surface': question_outputs['input_surface'].detach()  # Detach reconstruction target
        }
        
        # Compute loss
        logger.debug("🔍 [TRAIN_STEP] Computing loss...")
        try:
            losses = self.compute_loss(question_outputs, targets)
            logger.debug(f"🔍 [TRAIN_STEP] Loss computed. Components: {list(losses.keys())}")
            
            # Check loss computational graph
            for key, loss_tensor in losses.items():
                if torch.is_tensor(loss_tensor) and loss_tensor.requires_grad:
                    logger.debug(f"🔍 [TRAIN_STEP] Loss {key}: value={loss_tensor.item():.6f}, grad_fn={loss_tensor.grad_fn}")
            
        except Exception as e:
            logger.error(f"❌ [TRAIN_STEP] Loss computation failed: {e}")
            raise
        
        # Backward pass
        logger.debug("🔍 [TRAIN_STEP] Starting backward pass...")
        try:
            total_loss = losses['total_loss']
            logger.debug(f"🔍 [TRAIN_STEP] Total loss: {total_loss.item():.6f}, grad_fn: {total_loss.grad_fn}")
            
            # Critical: Check if this tensor has already been backwarded through
            if hasattr(total_loss, '_backward_hooks') and total_loss._backward_hooks:
                logger.warning(f"⚠️ [TRAIN_STEP] Total loss already has backward hooks: {total_loss._backward_hooks}")
            
            total_loss.backward()
            logger.debug("🔍 [TRAIN_STEP] Backward completed successfully")
            
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                logger.error("❌ [TRAIN_STEP] COMPUTATIONAL GRAPH REUSE DETECTED!")
                logger.error("🔍 [DEBUG] Analyzing computational graph...")
                
                # Detailed graph analysis
                self._debug_computational_graph(question_outputs, losses, targets)
                logger.error(f"❌ [TRAIN_STEP] Full error: {e}")
                raise
            else:
                logger.error(f"❌ [TRAIN_STEP] Other backward error: {e}")
                raise
        
        # Gradient clipping
        if self.config.gradient_balancing:
            logger.debug("🔍 [TRAIN_STEP] Applying gradient clipping...")
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step
        logger.debug("🔍 [TRAIN_STEP] Taking optimizer step...")
        self.optimizer.step()
        
        # Clear gradients after step to free graph
        logger.debug("🔍 [TRAIN_STEP] Final gradient clearing...")
        self.optimizer.zero_grad()
        
        # Return metrics (detach all tensors для избежания graph retention)
        with torch.no_grad():
            # Cosine similarity для dialogue metric (с dimension matching)
            final_output = question_outputs['final_output'].detach()  # [batch, 225]
            answer_embeddings_detached = answer_embeddings.detach()
            
            # Project answer_embeddings down to 225D для comparison
            if hasattr(self.loss_function, 'embedding_to_surface'):
                projected_answers = self.loss_function.embedding_to_surface(answer_embeddings_detached)
                cos_sim = torch.nn.functional.cosine_similarity(
                    final_output, projected_answers, dim=-1
                ).mean().item()
            else:
                # Fallback: simple similarity on first 225 dimensions
                cos_sim = torch.nn.functional.cosine_similarity(
                    final_output, answer_embeddings_detached[:, :225], dim=-1
                ).mean().item()
            
            # Detach all loss values for return
            losses_detached = {k: v.detach() if torch.is_tensor(v) else v for k, v in losses.items()}
        
        logger.debug("🔍 [TRAIN_STEP] Train step completed successfully")
        
        return {
            'total_loss': losses_detached['total_loss'].item(),
            'surface_loss': losses_detached['surface_reconstruction_loss'].item(),
            'internal_loss': losses_detached['internal_consistency_loss'].item(),
            'dialogue_loss': losses_detached['dialogue_similarity_loss'].item(),
            'cosine_similarity': cos_sim,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _debug_computational_graph(self, outputs: Dict[str, torch.Tensor], 
                                  losses: Dict[str, torch.Tensor],
                                  targets: Dict[str, torch.Tensor]):
        """Debug computational graph для finding reuse issues"""
        
        logger.error("🔍 [DEBUG_GRAPH] === COMPUTATIONAL GRAPH ANALYSIS ===")
        
        # 1. Check parameter states
        logger.error("🔍 [DEBUG_GRAPH] Parameter analysis:")
        
        # gMLP cells memory states
        for i, cell in enumerate(self.gmlp_cells[:3]):  # First 3 cells
            if hasattr(cell.base_gmlp, 'memory_state') and cell.base_gmlp.memory_state is not None:
                mem_state = cell.base_gmlp.memory_state
                logger.error(f"   Cell {i} memory_state: shape={mem_state.shape}, requires_grad={mem_state.requires_grad}")
        
        # Spatial propagation connections
        if hasattr(self.spatial_propagation, 'layer_connections'):
            conn = self.spatial_propagation.layer_connections
            logger.error(f"   Spatial connections: shape={conn.shape}, requires_grad={conn.requires_grad}")
        
        # Loss function projection layers
        if hasattr(self.loss_function, 'surface_to_embedding'):
            proj = self.loss_function.surface_to_embedding
            for name, param in proj.named_parameters():
                logger.error(f"   Loss projection {name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")
        
        # 2. Check dynamic layer creation
        logger.error("🔍 [DEBUG_GRAPH] Dynamic layer analysis:")
        
        if hasattr(self, 'surface_projection'):
            logger.error(f"   Surface projection exists: {self.surface_projection}")
            for name, param in self.surface_projection.named_parameters():
                logger.error(f"   Surface proj {name}: requires_grad={param.requires_grad}")
        
        # 3. Check outputs computational graph
        logger.error("🔍 [DEBUG_GRAPH] Output tensor analysis:")
        
        for key, tensor in outputs.items():
            if torch.is_tensor(tensor) and tensor.requires_grad:
                logger.error(f"   Output {key}: grad_fn={tensor.grad_fn}")
                
                # Check if tensor has backward hooks (indication of previous backward)
                if hasattr(tensor, '_backward_hooks') and tensor._backward_hooks:
                    logger.error(f"   ⚠️ Output {key} has backward hooks: {tensor._backward_hooks}")
        
        # 4. Check base trainer state
        logger.error("🔍 [DEBUG_GRAPH] Base trainer analysis:")
        
        if hasattr(self, 'base_trainer'):
            logger.error(f"   Base trainer mode: {self.base_trainer.training}")
            if hasattr(self.base_trainer, 'adapter'):
                adapter_params = sum(1 for _ in self.base_trainer.adapter.parameters())
                logger.error(f"   Adapter parameters: {adapter_params}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        # gMLP cells info
        total_cells = len(self.gmlp_cells)
        total_params = sum(sum(p.numel() for p in cell.parameters()) for cell in self.gmlp_cells)
        avg_params = total_params / total_cells
        
        # Full system parameters
        all_params = sum(p.numel() for p in self.parameters())
        
        return {
            'stage': '3.1.4.1',
            'architecture': 'EmergentCubeTrainer',
            'cube_dimensions': self.config.cube_dimensions,
            'total_cells': total_cells,
            'avg_params_per_cell': avg_params,
            'total_lattice_params': total_params,
            'total_system_params': all_params,
            'teacher_model': self.config.teacher_model,
            'full_cube_gradient': self.config.enable_full_cube_gradient,
            'spatial_propagation': self.config.spatial_propagation_depth,
            'loss_components': list(self.config.loss_weights.keys()),
            'current_epoch': self.current_epoch
        }

# Helper functions

def create_emergent_trainer(cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                           teacher_model: str = "Meta-Llama-3-8B",
                           device: str = "cpu") -> EmergentCubeTrainer:
    """Create configured emergent trainer for Stage 3.1.4.1"""
    
    config = EmergentTrainingConfig(
        teacher_model=teacher_model,
        cube_dimensions=cube_dimensions,
        enable_full_cube_gradient=True,
        spatial_propagation_depth=cube_dimensions[2]
    )
    
    trainer = EmergentCubeTrainer(config, device)
    
    logger.info("🚀 Emergent trainer created for Stage 3.1.4.1")
    info = trainer.get_system_info()
    logger.info(f"   System: {info['total_system_params']:,} parameters")
    logger.info(f"   Lattice: {info['total_cells']} cells × {info['avg_params_per_cell']:.0f} params")
    
    return trainer


def test_emergent_training_basic() -> bool:
    """Basic test для emergent training system"""
    logger.info("🧪 Testing EmergentCubeTrainer...")
    
    try:
        # Create trainer
        trainer = create_emergent_trainer(device="cpu")
        
        # Test data
        batch_size = 2
        question_embeddings = torch.randn(batch_size, 4096)  # LLaMA-3-8B embeddings
        answer_embeddings = torch.randn(batch_size, 4096)
        
        # Forward pass
        outputs = trainer.forward(question_embeddings)
        
        # Check outputs
        assert 'final_output' in outputs
        assert outputs['final_output'].shape[0] == batch_size
        assert not torch.isnan(outputs['final_output']).any()
        
        # Training step
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        # Check metrics
        assert 'total_loss' in metrics
        assert 'cosine_similarity' in metrics
        
        logger.info("✅ EmergentCubeTrainer basic test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ EmergentCubeTrainer test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_emergent_training_basic()
    if success:
        print("🎯 Stage 3.1.4.1 Emergent Training Infrastructure ready!")
    else:
        print("❌ Issues detected in emergent training system")