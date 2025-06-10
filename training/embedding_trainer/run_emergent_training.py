"""
[START] Stage 3.1.4.1: Emergent Training Execution Script
====================================================

Production-ready emergent training с:
- Real LLaMA-3-8B integration
- Full cube gradient flow (2,475 gMLP neurons)
- Multi-objective loss optimization
- Comprehensive monitoring и analysis
- Emergent behavior detection
"""

import torch
import logging
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
    EmergentTrainingConfig,
    create_emergent_trainer
)
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/emergent_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EmergentTrainingRunner:
    """
    Production runner for emergent training
    
    Features:
    - Real LLaMA-3-8B embeddings
    - Comprehensive monitoring
    - Emergent behavior analysis
    - Results saving и visualization
    """
    
    def __init__(self, config_path: str = "config/emergent_training_3_1_4_1.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.trainer = None
        self.training_history = []
        self.emergent_analysis = []
        
        # Create results directory
        self.results_dir = Path(self.config['output']['save_directory'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[BRAIN] EmergentTrainingRunner initialized")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Results: {self.results_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"[OK] Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"[ERROR] Failed to load config: {e}")
            raise
    
    def setup_trainer(self):
        """Setup emergent trainer from configuration"""
        logger.info("[CONFIG] Setting up EmergentCubeTrainer...")
        
        # Extract configuration
        emergent_config = self.config['emergent_training']
        
        # Create trainer config
        trainer_config = EmergentTrainingConfig(
            teacher_model=emergent_config['teacher_model'],
            cube_dimensions=tuple(emergent_config['cube_dimensions']),
            enable_full_cube_gradient=emergent_config['enable_full_cube_gradient'],
            spatial_propagation_depth=emergent_config['spatial_propagation_depth'],
            emergent_specialization=emergent_config['emergent_specialization'],
            gmlp_config=emergent_config['gmlp_config'],
            loss_weights=emergent_config['loss_weights'],
            learning_rate=emergent_config['learning_rate'],
            batch_size=emergent_config['batch_size'],
            epochs=emergent_config['epochs'],
            warmup_epochs=emergent_config['warmup_epochs'],
            gradient_balancing=emergent_config['gradient_balancing'],
            adaptive_loss_weighting=emergent_config['adaptive_loss_weighting']
        )
        
        # Device selection
        device = self.config['training_optimization']['device']
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create trainer
        self.trainer = EmergentCubeTrainer(trainer_config, device=device)
        
        # Log system info
        info = self.trainer.get_system_info()
        logger.info("[OK] Trainer setup complete:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
    
    def load_training_data(self) -> Dict[str, torch.Tensor]:
        """Load real training data с LLaMA-3-8B embeddings"""
        logger.info("[BOOKS] Loading training data...")
        
        # Use existing dialogue dataset infrastructure
        dataset_config = {
            'batch_size': self.config['emergent_training']['batch_size'],
            'max_pairs': 50,  # Start с manageable size
            'embedding_source': 'llama',  # Use LLaMA embeddings
            'quality_threshold': 0.3
        }
        
        try:
            # Create dialogue dataset
            dialogue_dataset = create_dialogue_dataset(dataset_config)
            
            # Extract question/answer pairs
            questions = []
            answers = []
            
            for batch in dialogue_dataset:
                if 'question_embedding' in batch and 'answer_embedding' in batch:
                    questions.append(batch['question_embedding'])
                    answers.append(batch['answer_embedding'])
                
                if len(questions) >= 10:  # Sufficient для testing
                    break
            
            if not questions:
                # Fallback: synthetic data для testing
                logger.warning("No real data found, using synthetic embeddings")
                batch_size = dataset_config['batch_size']
                questions = [torch.randn(batch_size, 4096) for _ in range(5)]
                answers = [torch.randn(batch_size, 4096) for _ in range(5)]
            
            training_data = {
                'questions': questions,
                'answers': answers,
                'num_batches': len(questions)
            }
            
            logger.info(f"[OK] Training data loaded: {len(questions)} batches")
            return training_data
            
        except Exception as e:
            logger.warning(f"[WARNING]  Failed to load real data: {e}")
            logger.info("[WRITE] Using synthetic data для testing...")
            
            # Synthetic fallback
            batch_size = self.config['emergent_training']['batch_size']
            num_batches = 5
            
            questions = [torch.randn(batch_size, 4096) for _ in range(num_batches)]
            answers = [torch.randn(batch_size, 4096) for _ in range(num_batches)]
            
            return {
                'questions': questions,
                'answers': answers,
                'num_batches': num_batches
            }
    
    def analyze_emergent_behavior(self, epoch: int) -> Dict[str, Any]:
        """Analyze emergent behavior patterns"""
        logger.info(f"🔬 Analyzing emergent behavior (epoch {epoch})...")
        
        analysis = {
            'epoch': epoch,
            'timestamp': time.time(),
            'layer_analysis': {},
            'spatial_patterns': {},
            'information_flow': {}
        }
        
        try:
            # Test input для analysis
            test_input = torch.randn(1, 4096)
            
            with torch.no_grad():
                outputs = self.trainer.forward(test_input)
                
                # Layer specialization analysis
                enhanced_states = outputs['enhanced_states']  # [1, 11, 15, 15, 32]
                
                layer_variances = []
                layer_norms = []
                
                for layer in range(11):
                    layer_states = enhanced_states[0, layer].flatten()
                    variance = torch.var(layer_states).item()
                    norm = torch.norm(layer_states).item()
                    
                    layer_variances.append(variance)
                    layer_norms.append(norm)
                
                analysis['layer_analysis'] = {
                    'variances': layer_variances,
                    'norms': layer_norms,
                    'specialization_index': max(layer_variances) / (sum(layer_variances) / len(layer_variances))
                }
                
                # Information flow analysis
                input_surface = outputs['input_surface']
                output_surface = outputs['output_surface']
                
                if input_surface is not None and output_surface is not None:
                    info_preservation = torch.nn.functional.cosine_similarity(
                        input_surface, output_surface, dim=-1
                    ).item()
                    
                    analysis['information_flow'] = {
                        'preservation_ratio': info_preservation,
                        'input_norm': torch.norm(input_surface).item(),
                        'output_norm': torch.norm(output_surface).item()
                    }
                
        except Exception as e:
            logger.warning(f"[WARNING]  Emergent analysis failed: {e}")
            analysis['error'] = str(e)
        
        self.emergent_analysis.append(analysis)
        return analysis
    
    def save_results(self, epoch: int, metrics: Dict[str, float]):
        """Save training results и analysis"""
        
        # Save training metrics
        metrics_file = self.results_dir / f"training_metrics_epoch_{epoch}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save emergent analysis
        if self.emergent_analysis:
            analysis_file = self.results_dir / f"emergent_analysis_epoch_{epoch}.json"
            with open(analysis_file, 'w') as f:
                json.dump(self.emergent_analysis[-1], f, indent=2)
        
        # Save system info
        if self.trainer:
            info = self.trainer.get_system_info()
            info_file = self.results_dir / f"system_info_epoch_{epoch}.yaml"
            with open(info_file, 'w') as f:
                yaml.dump(info, f, default_flow_style=False)
        
        # Save checkpoint
        if self.config['output']['save_checkpoints']:
            checkpoint_file = self.results_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'trainer_state_dict': self.trainer.state_dict() if self.trainer else None,
                'optimizer_state_dict': self.trainer.optimizer.state_dict() if self.trainer else None,
                'metrics': metrics,
                'config': self.config
            }, checkpoint_file)
    
    def run_training(self):
        """Run complete emergent training pipeline"""
        logger.info("[START] Starting emergent training...")
        
        # Setup
        self.setup_trainer()
        training_data = self.load_training_data()
        
        # Training parameters
        epochs = self.config['emergent_training']['epochs']
        warmup_epochs = self.config['emergent_training']['warmup_epochs']
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"\n[TARGET] EPOCH {epoch+1}/{epochs}")
            
            epoch_metrics = []
            epoch_start_time = time.time()
            
            # Training steps
            for batch_idx in range(training_data['num_batches']):
                questions = training_data['questions'][batch_idx]
                answers = training_data['answers'][batch_idx]
                
                # Training step
                step_start_time = time.time()
                metrics = self.trainer.train_step(questions, answers)
                step_time = time.time() - step_start_time
                
                metrics['step_time'] = step_time
                metrics['epoch'] = epoch
                metrics['batch'] = batch_idx
                
                epoch_metrics.append(metrics)
                
                # Log progress
                if batch_idx % self.config['monitoring']['log_interval'] == 0:
                    logger.info(f"   Step {batch_idx}: loss={metrics['total_loss']:.6f}, "
                               f"similarity={metrics['cosine_similarity']:.6f}, "
                               f"time={step_time:.3f}s")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_metrics = self._average_metrics(epoch_metrics)
            avg_metrics['epoch_time'] = epoch_time
            
            logger.info(f"[DATA] EPOCH {epoch+1} SUMMARY:")
            logger.info(f"   Average loss: {avg_metrics['total_loss']:.6f}")
            logger.info(f"   Average similarity: {avg_metrics['cosine_similarity']:.6f}")
            logger.info(f"   Epoch time: {epoch_time:.1f}s")
            
            # Emergent behavior analysis
            if epoch >= warmup_epochs:
                emergent_analysis = self.analyze_emergent_behavior(epoch)
                if 'specialization_index' in emergent_analysis.get('layer_analysis', {}):
                    spec_index = emergent_analysis['layer_analysis']['specialization_index']
                    logger.info(f"   Specialization index: {spec_index:.3f}")
            
            # Save results
            self.save_results(epoch, avg_metrics)
            
            # Learning rate scheduler step
            if hasattr(self.trainer, 'scheduler'):
                self.trainer.scheduler.step(avg_metrics['total_loss'])
            
            self.training_history.append(avg_metrics)
        
        # Final summary
        self._print_final_summary()
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        
        return avg_metrics
    
    def _print_final_summary(self):
        """Print final training summary"""
        logger.info("\n" + "="*60)
        logger.info("[TARGET] EMERGENT TRAINING COMPLETE!")
        logger.info("="*60)
        
        if self.training_history:
            initial_loss = self.training_history[0]['total_loss']
            final_loss = self.training_history[-1]['total_loss']
            loss_improvement = initial_loss - final_loss
            
            initial_sim = self.training_history[0]['cosine_similarity']
            final_sim = self.training_history[-1]['cosine_similarity']
            sim_improvement = final_sim - initial_sim
            
            logger.info(f"[DATA] TRAINING METRICS:")
            logger.info(f"   Initial loss: {initial_loss:.6f}")
            logger.info(f"   Final loss: {final_loss:.6f}")
            logger.info(f"   Loss improvement: {loss_improvement:.6f}")
            logger.info(f"   Initial similarity: {initial_sim:.6f}")
            logger.info(f"   Final similarity: {final_sim:.6f}")
            logger.info(f"   Similarity improvement: {sim_improvement:.6f}")
        
        if self.emergent_analysis:
            final_analysis = self.emergent_analysis[-1]
            if 'layer_analysis' in final_analysis:
                spec_index = final_analysis['layer_analysis'].get('specialization_index', 0)
                logger.info(f"[BRAIN] EMERGENT BEHAVIOR:")
                logger.info(f"   Final specialization index: {spec_index:.3f}")
        
        # System performance
        if self.trainer:
            info = self.trainer.get_system_info()
            logger.info(f"[PC]  SYSTEM INFO:")
            logger.info(f"   Total parameters: {info['total_system_params']:,}")
            logger.info(f"   Cells: {info['total_cells']}")
            logger.info(f"   Avg params per cell: {info['avg_params_per_cell']:.0f}")
        
        logger.info(f"[SAVE] Results saved to: {self.results_dir}")
        logger.info("[SUCCESS] Stage 3.1.4.1 Emergent Training Infrastructure COMPLETE!")


def main():
    """Main execution function"""
    print("[BRAIN] Stage 3.1.4.1: Emergent Training Infrastructure")
    print("="*60)
    
    try:
        # Create and run trainer
        runner = EmergentTrainingRunner()
        runner.run_training()
        
    except KeyboardInterrupt:
        print("\n[WARNING]  Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 