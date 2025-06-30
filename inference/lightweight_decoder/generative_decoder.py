"""
[BRAIN] GENERATIVE DECODER - Advanced Embedding-to-Text Generation

Основной генеративный декодер, интегрирующий революционные архитектуры:
- 🥇 Resource-Efficient Transformer v2.1 (ULTRA-COMPACT, 722K params)
- 🥈 Hybrid CCT+Mamba architecture (планируется)
( CCT+Mamba превосходит по:
  1. Spatial Intelligence (🔥 КРИТИЧНО для 15×15×11 lattice)
  2. Natural 3D Processing (поверхности → объем → поверхности)
  3. Emergent Pattern Detection (CNN отлично для spatial patterns)
  4. Biological Alignment (ближе к cortical column аналогии))
- 🥉 Enhanced CCT baseline (fallback)

CRITICAL SUCCESS METRICS:
- Parameters: ≤800K ✅ ACHIEVED (722K)
- Memory reduction: ≥60% target
- Speed: maintain 50% improvement
- BLEU score: >0.45 target
- RTX 5090: full compatibility

Phase 2.7 Stage 2.1 - Integration Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List
import logging
import time
from pathlib import Path
import yaml
from dataclasses import dataclass
import json

# Import our RET v2.1 architecture (handle both relative and absolute imports)
try:
    from .resource_efficient_decoder_v2_1 import (
        ResourceEfficientDecoderV21,
        RETConfigV21,
        create_ultra_compact_decoder
    )
except ImportError:
    from resource_efficient_decoder_v2_1 import (
        ResourceEfficientDecoderV21,
        RETConfigV21,
        create_ultra_compact_decoder
    )

# Import other architectures (when available)
try:
    from .resource_efficient_decoder_v2 import ResourceEfficientDecoderV2, RETConfigV2
except ImportError:
    try:
        from resource_efficient_decoder_v2 import ResourceEfficientDecoderV2, RETConfigV2
    except ImportError:
        ResourceEfficientDecoderV2 = None
        RETConfigV2 = None

logger = logging.getLogger(__name__)


@dataclass
class GenerativeConfig:
    """Unified configuration for GenerativeDecoder"""
    
    # Architecture selection
    architecture_type: str = "resource_efficient_v21"  # v21, v2, hybrid_cct, enhanced_cct
    
    # Input/Output specifications
    embedding_dim: int = 768           # Input от EmbeddingProcessor
    max_length: int = 128              # Maximum generation length
    vocab_size: int = 32000            # Target vocabulary (adaptive)
    
    # Generation parameters
    temperature: float = 0.8           # Sampling temperature
    top_k: int = 50                    # Top-k sampling
    top_p: float = 0.9                 # Nucleus sampling
    repetition_penalty: float = 1.1    # Repetition penalty
    length_penalty: float = 1.0        # Length penalty
    
    # Performance targets
    target_parameters: int = 800_000    # STRICT parameter limit
    memory_reduction_target: float = 0.60  # 60% memory reduction target
    speed_target: float = 0.50         # 50% speed improvement target
    
    # RTX 5090 optimizations
    mixed_precision: bool = True       # Enable mixed precision
    gradient_checkpointing: bool = True # Memory optimization
    edge_optimization: bool = True     # RTX 5090 edge optimizations
    
    # Quality parameters
    enable_quality_filter: bool = True  # Enable output quality filtering
    min_quality_score: float = 0.3     # Minimum quality threshold
    coherence_weight: float = 0.4      # Coherence importance
    fluency_weight: float = 0.6        # Fluency importance
    
    # Integration parameters
    tokenizer_path: Optional[str] = None     # Path to tokenizer (if available)
    fallback_enabled: bool = True           # Enable fallback strategies
    verbose_logging: bool = False           # Detailed logging


class AdvancedTokenizer:
    """
    Simple tokenizer for development/testing
    In production, this will be replaced with proper tokenizer integration
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Simple vocabulary mapping для proof-of-concept
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Basic word tokens (expandable)
        self.word_tokens = {
            'the': 4, 'a': 5, 'an': 6, 'and': 7, 'or': 8, 'but': 9,
            'is': 10, 'are': 11, 'was': 12, 'were': 13, 'be': 14,
            'have': 15, 'has': 16, 'had': 17, 'do': 18, 'does': 19,
            'this': 20, 'that': 21, 'these': 22, 'those': 23,
            'I': 24, 'you': 25, 'he': 26, 'she': 27, 'it': 28, 'we': 29, 'they': 30,
            'text': 31, 'generated': 32, 'using': 33, 'decoder': 34, 'model': 35,
            'embedding': 36, 'transformer': 37, 'neural': 38, 'network': 39,
            'efficient': 40, 'compact': 41, 'resource': 42, 'optimized': 43
        }
        
        self.token_to_id = {**self.special_tokens, **self.word_tokens}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        logger.info(f"🔤 AdvancedTokenizer initialized with {len(self.token_to_id)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Simple text encoding"""
        words = text.lower().split()
        tokens = [self.special_tokens['<BOS>']]
        
        for word in words:
            tokens.append(self.token_to_id.get(word, self.special_tokens['<UNK>']))
        
        tokens.append(self.special_tokens['<EOS>'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Simple token decoding"""
        words = []
        
        for token_id in token_ids:
            if token_id in [self.special_tokens['<PAD>'], self.special_tokens['<BOS>']]:
                continue
            elif token_id == self.special_tokens['<EOS>']:
                break
            else:
                word = self.id_to_token.get(token_id, '<UNK>')
                words.append(word)
        
        return ' '.join(words)


class QualityAssessment:
    """Advanced quality assessment for generated text"""
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self.coherence_weight = config.coherence_weight
        self.fluency_weight = config.fluency_weight
    
    def assess_quality(self, tokens: List[int], text: str) -> Dict[str, float]:
        """Comprehensive quality assessment"""
        
        # Diversity metrics
        unique_tokens = len(set(tokens)) if tokens else 0
        total_tokens = len(tokens) if tokens else 1
        diversity_score = unique_tokens / total_tokens
        
        # Length appropriateness
        length_score = min(total_tokens / 20, 1.0)  # Prefer 10-20 tokens
        
        # Repetition penalty
        repetition_score = 1.0
        if total_tokens > 3:
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            unique_bigrams = len(set(bigrams))
            repetition_score = unique_bigrams / len(bigrams) if bigrams else 1.0
        
        # Coherence estimation (simple heuristics)
        coherence_score = self._estimate_coherence(text)
        
        # Fluency estimation
        fluency_score = self._estimate_fluency(text)
        
        # Combined quality score
        quality_score = (
            self.coherence_weight * coherence_score +
            self.fluency_weight * fluency_score +
            0.2 * diversity_score +
            0.2 * length_score +
            0.2 * repetition_score
        )
        
        return {
            'overall_quality': quality_score,
            'coherence': coherence_score,
            'fluency': fluency_score,
            'diversity': diversity_score,
            'length_appropriateness': length_score,
            'repetition_quality': repetition_score
        }
    
    def _estimate_coherence(self, text: str) -> float:
        """Simple coherence estimation"""
        if not text or len(text) < 5:
            return 0.1
        
        # Basic coherence indicators
        words = text.split()
        if not words:
            return 0.1
        
        # Check for basic sentence structure indicators
        coherence_indicators = 0
        total_checks = 0
        
        # Has reasonable length
        if 5 <= len(words) <= 50:
            coherence_indicators += 1
        total_checks += 1
        
        # Contains common function words
        function_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were'}
        if any(word in function_words for word in words):
            coherence_indicators += 1
        total_checks += 1
        
        # Not too many repetitions
        if len(set(words)) / len(words) > 0.5:
            coherence_indicators += 1
        total_checks += 1
        
        return coherence_indicators / total_checks if total_checks > 0 else 0.1
    
    def _estimate_fluency(self, text: str) -> float:
        """Simple fluency estimation"""
        if not text:
            return 0.1
        
        words = text.split()
        if not words:
            return 0.1
        
        # Basic fluency indicators
        fluency_score = 0.5  # Base score
        
        # Length appropriateness (neither too short nor too long)
        if 3 <= len(words) <= 30:
            fluency_score += 0.2
        
        # No obvious repetition patterns
        if len(words) >= 3:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition <= len(words) // 3:  # No word appears too frequently
                fluency_score += 0.2
        
        # Contains common English patterns
        common_patterns = ['the ', ' and ', ' is ', ' are ', ' was ', ' were ']
        text_lower = ' ' + text.lower() + ' '
        if any(pattern in text_lower for pattern in common_patterns):
            fluency_score += 0.1
        
        return min(fluency_score, 1.0)


class GenerativeDecoder(nn.Module):
    """
    [BRAIN] GENERATIVE DECODER - Advanced Embedding-to-Text Generation
    
    Unified interface integrating multiple revolutionary architectures:
    - RET v2.1 (ULTRA-COMPACT, 722K parameters) ✅ READY
    - RET v2.0 (Resource-efficient baseline) 
    - Hybrid CCT+Mamba (bio-inspired) 🔜 PLANNED
    - Enhanced CCT (fallback) 🔜 PLANNED
    
    Target Performance:
    - Parameters: ≤800K ✅ ACHIEVED
    - Memory: 60% reduction target
    - Speed: 50% improvement maintained
    - Quality: BLEU >0.45
    """
    
    def __init__(self, config: Optional[GenerativeConfig] = None):
        super().__init__()
        
        self.config = config or GenerativeConfig()
        
        # Initialize tokenizer
        self.tokenizer = AdvancedTokenizer(vocab_size=self.config.vocab_size)
        
        # Initialize quality assessment
        self.quality_assessor = QualityAssessment(self.config)
        
        # Load architecture based on config
        self.decoder_model = self._load_architecture()
        
        # Performance tracking
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_quality': 0.0,
            'average_time': 0.0,
            'parameter_count': self._count_parameters()
        }
        
        logger.info(f"🚀 GenerativeDecoder initialized:")
        logger.info(f"   Architecture: {self.config.architecture_type}")
        logger.info(f"   Parameters: {self.generation_stats['parameter_count']:,}")
        logger.info(f"   Target: ≤{self.config.target_parameters:,}")
        logger.info(f"   Memory optimization: {self.config.mixed_precision}")
        logger.info(f"   RTX 5090 ready: {self.config.edge_optimization}")
    
    def _load_architecture(self) -> nn.Module:
        """Load the selected architecture"""
        
        if self.config.architecture_type == "resource_efficient_v21":
            # RET v2.1 ULTRA-COMPACT (722K parameters)
            ret_config = RETConfigV21(
                embedding_dim=self.config.embedding_dim,
                mixed_precision=self.config.mixed_precision,
                gradient_checkpointing=self.config.gradient_checkpointing
            )
            
            model = ResourceEfficientDecoderV21(ret_config)
            logger.info("🥇 Loaded RET v2.1 ULTRA-COMPACT architecture")
            
        elif self.config.architecture_type == "resource_efficient_v2" and ResourceEfficientDecoderV2:
            # RET v2.0 (fallback)
            ret_config = RETConfigV2(
                embedding_dim=self.config.embedding_dim,
                mixed_precision=self.config.mixed_precision
            )
            
            model = ResourceEfficientDecoderV2(ret_config)
            logger.info("🥈 Loaded RET v2.0 baseline architecture")
            
        else:
            # Fallback to RET v2.1 if other architectures not available
            logger.warning(f"Architecture '{self.config.architecture_type}' not available, using RET v2.1")
            ret_config = RETConfigV21(embedding_dim=self.config.embedding_dim)
            model = ResourceEfficientDecoderV21(ret_config)
        
        return model
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        if hasattr(self.decoder_model, '_count_parameters'):
            return self.decoder_model._count_parameters()
        else:
            return sum(p.numel() for p in self.decoder_model.parameters() if p.requires_grad)
    
    def generate(self, 
                embedding: torch.Tensor,
                max_length: Optional[int] = None,
                temperature: Optional[float] = None,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Advanced generation with comprehensive metrics and quality control
        
        Args:
            embedding: Input embedding от Module 2 (768D)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generated text, tokens, quality metrics, and performance data
        """
        
        # Parameter defaults
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        start_time = time.time()
        self.generation_stats['total_generations'] += 1
        
        try:
            # Validate input
            if embedding.dim() == 1 and embedding.size(0) != self.config.embedding_dim:
                raise ValueError(f"Expected embedding dim {self.config.embedding_dim}, got {embedding.size(0)}")
            
            # Generate with underlying model
            generation_result = self.decoder_model.forward(
                embedding=embedding,
                max_length=max_length,
                temperature=temperature
            )
            
            # Extract tokens and convert to text
            tokens = generation_result.get('tokens', [])
            text = self.tokenizer.decode(tokens) if tokens else "Empty generation"
            
            # Quality assessment
            quality_metrics = self.quality_assessor.assess_quality(tokens, text)
            
            # Performance metrics
            generation_time = time.time() - start_time
            
            # Check quality threshold
            quality_passed = quality_metrics['overall_quality'] >= self.config.min_quality_score
            
            if quality_passed:
                self.generation_stats['successful_generations'] += 1
            
            # Update running averages
            self._update_statistics(quality_metrics['overall_quality'], generation_time)
            
            # Comprehensive result
            result = {
                'text': text,
                'tokens': tokens,
                'quality_metrics': quality_metrics,
                'generation_time': generation_time,
                'quality_passed': quality_passed,
                'model_metrics': generation_result.get('metrics', {}),
                'parameters_used': self.generation_stats['parameter_count'],
                'architecture': self.config.architecture_type
            }
            
            if self.config.verbose_logging:
                logger.info(f"🎯 Generation completed:")
                logger.info(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                logger.info(f"   Quality: {quality_metrics['overall_quality']:.3f}")
                logger.info(f"   Time: {generation_time:.3f}s")
                logger.info(f"   Tokens: {len(tokens)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            
            if self.config.fallback_enabled:
                # Simple fallback generation
                fallback_text = f"GenerativeDecoder fallback response (embedding processed)"
                return {
                    'text': fallback_text,
                    'tokens': self.tokenizer.encode(fallback_text),
                    'quality_metrics': {'overall_quality': 0.2},
                    'generation_time': time.time() - start_time,
                    'quality_passed': False,
                    'error': str(e),
                    'fallback_used': True
                }
            else:
                raise
    
    def decode(self, embedding: torch.Tensor, **kwargs) -> str:
        """
        Simple decode method for API compatibility with PhraseBankDecoder
        
        Args:
            embedding: Input embedding от Module 2 (768D)
            **kwargs: Generation parameters
            
        Returns:
            Generated text string
        """
        
        result = self.generate(embedding, **kwargs)
        return result['text']
    
    def batch_generate(self, embeddings: torch.Tensor, **kwargs) -> List[Dict[str, Any]]:
        """Batch generation with session handling"""
        
        if embeddings.dim() != 2 or embeddings.size(1) != self.config.embedding_dim:
            raise ValueError(f"Expected embeddings shape (N, {self.config.embedding_dim}), got {embeddings.shape}")
        
        results = []
        
        for i, embedding in enumerate(embeddings):
            try:
                result = self.generate(embedding, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Batch generation failed for item {i}: {e}")
                if self.config.fallback_enabled:
                    results.append({
                        'text': f"Batch generation error for item {i}",
                        'error': str(e),
                        'fallback_used': True
                    })
                else:
                    raise
        
        return results
    
    def _update_statistics(self, quality: float, time_taken: float):
        """Update running statistics"""
        
        total = self.generation_stats['total_generations']
        
        # Running average quality
        if total == 1:
            self.generation_stats['average_quality'] = quality
            self.generation_stats['average_time'] = time_taken
        else:
            # Exponential moving average
            alpha = 2 / (total + 1)
            self.generation_stats['average_quality'] = (
                alpha * quality + (1 - alpha) * self.generation_stats['average_quality']
            )
            self.generation_stats['average_time'] = (
                alpha * time_taken + (1 - alpha) * self.generation_stats['average_time']
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Comprehensive performance report"""
        
        success_rate = (
            self.generation_stats['successful_generations'] / 
            max(self.generation_stats['total_generations'], 1)
        )
        
        return {
            'architecture': self.config.architecture_type,
            'parameter_count': self.generation_stats['parameter_count'],
            'parameter_target': self.config.target_parameters,
            'parameter_efficiency': self.config.target_parameters / self.generation_stats['parameter_count'],
            'total_generations': self.generation_stats['total_generations'],
            'successful_generations': self.generation_stats['successful_generations'],
            'success_rate': success_rate,
            'average_quality': self.generation_stats['average_quality'],
            'average_generation_time': self.generation_stats['average_time'],
            'memory_optimization': self.config.mixed_precision,
            'rtx5090_ready': self.config.edge_optimization
        }
    
    def save_model(self, path: Union[str, Path]):
        """Save model state"""
        
        save_dict = {
            'model_state_dict': self.decoder_model.state_dict(),
            'config': self.config.__dict__,
            'generation_stats': self.generation_stats,
            'architecture_type': self.config.architecture_type
        }
        
        torch.save(save_dict, path)
        logger.info(f"[SAVE] Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load model state"""
        
        checkpoint = torch.load(path, map_location='cpu')
        
        self.decoder_model.load_state_dict(checkpoint['model_state_dict'])
        self.generation_stats = checkpoint.get('generation_stats', self.generation_stats)
        
        logger.info(f"📂 Model loaded from {path}")


def create_generative_decoder(
    architecture: str = "resource_efficient_v21",
    embedding_dim: int = 768,
    target_parameters: int = 800000,
    **kwargs
) -> GenerativeDecoder:
    """
    Factory function for creating GenerativeDecoder with optimal configuration
    
    Args:
        architecture: Architecture type ('resource_efficient_v21', 'resource_efficient_v2', etc.)
        embedding_dim: Input embedding dimension
        target_parameters: Target parameter count
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GenerativeDecoder instance
    """
    
    config = GenerativeConfig(
        architecture_type=architecture,
        embedding_dim=embedding_dim,
        target_parameters=target_parameters,
        **kwargs
    )
    
    return GenerativeDecoder(config)


# Example usage для testing
if __name__ == "__main__":
    # Create decoder
    decoder = create_generative_decoder()
    
    # Test generation
    test_embedding = torch.randn(768)
    result = decoder.generate(test_embedding)
    
    print(f"Generated: {result['text']}")
    print(f"Quality: {result['quality_metrics']['overall_quality']:.3f}")
    print(f"Time: {result['generation_time']:.3f}s")
    
    # Performance report
    report = decoder.get_performance_report()
    print(f"Model parameters: {report['parameter_count']:,}")
    print(f"Success rate: {report['success_rate']:.1%}") 