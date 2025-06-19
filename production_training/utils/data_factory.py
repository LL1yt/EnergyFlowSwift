"""
Data Factory for creating datasets for different training stages.
"""

import logging
from typing import List, Dict, Any

from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
from production_training.config.training_stages import TrainingStage

logger = logging.getLogger(__name__)


def get_dataset_for_stage(
    stage: TrainingStage, model_name: str, dataset_limit: int = -1
):
    """
    Creates and returns a dataset suitable for the given training stage.

    Args:
        stage: The current training stage configuration.
        model_name: The name of the teacher model for embedding generation.
        dataset_limit: Optional limit for the dataset size.

    Returns:
        A DialogueDataset instance.
    """
    logger.info(f"[DATA FACTORY] Creating dataset for stage: {stage.name}")

    dialogue_pairs = _get_dialogue_pairs_for_stage(stage.name)

    effective_limit = stage.dataset_limit if dataset_limit == -1 else dataset_limit

    dataset = create_dialogue_dataset(
        dialogue_pairs=dialogue_pairs,
        teacher_model=model_name,
        cache_embeddings=(
            stage.name != "validation"
        ),  # Don't cache for quick validation
        validation_split=0.0,
        normalize_embeddings=True,
        dataset_limit=effective_limit,
    )
    logger.info(f"[DATA FACTORY] Dataset created with {len(dataset)} pairs.")
    return dataset


def _get_dialogue_pairs_for_stage(stage_name: str) -> List[Dict[str, Any]]:
    """Returns a list of dialogue pairs based on the stage name."""
    if stage_name == "validation":
        return [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence technology.",
            },
            {
                "question": "How do neural networks work?",
                "answer": "Neural networks process data through connected layers.",
            },
        ]
    elif stage_name == "convergence":
        return _get_medium_dialogue_dataset()
    else:  # production, optimization
        return _get_full_dialogue_dataset()


def _get_medium_dialogue_dataset() -> List[Dict[str, str]]:
    """Provides a medium-sized dataset for convergence testing."""
    return [
        {
            "question": "What is a neural network?",
            "answer": "A computational model inspired by biological neural networks with interconnected processing nodes.",
        },
        {
            "question": "How does backpropagation work?",
            "answer": "An algorithm that calculates gradients by propagating errors backward through network layers.",
        },
        {
            "question": "What is overfitting?",
            "answer": "When a model learns training data too well, including noise, reducing generalization ability.",
        },
        {
            "question": "What are activation functions?",
            "answer": "Mathematical functions that determine neuron output, introducing non-linearity to networks.",
        },
        {
            "question": "How do convolutional layers work?",
            "answer": "They apply filters to detect local features in input data through convolution operations.",
        },
        {
            "question": "What are cellular neural networks?",
            "answer": "Arrays of locally connected processing units performing complex computations through emergent behavior.",
        },
        {
            "question": "How do 3D neural networks differ?",
            "answer": "They process volumetric data with depth information for spatial reasoning in three dimensions.",
        },
        {
            "question": "What is emergent behavior?",
            "answer": "Complex patterns arising from simple interactions between individual network components.",
        },
        {
            "question": "What is spatial reasoning?",
            "answer": "The ability to understand and manipulate spatial relationships between objects.",
        },
        {
            "question": "How do cellular automata work?",
            "answer": "Discrete models with cells evolving based on local rules and neighbor states.",
        },
    ]


def _get_full_dialogue_dataset() -> List[Dict[str, str]]:
    """Provides a large, comprehensive dataset for production training."""
    advanced_pairs = [
        {
            "question": "What are attention mechanisms?",
            "answer": "Neural network components that focus on relevant input parts, enabling better sequence processing.",
        },
        {
            "question": "How do transformers work?",
            "answer": "Models using self-attention to process sequences in parallel, capturing long-range dependencies.",
        },
        {
            "question": "What is transfer learning?",
            "answer": "Using pre-trained models as starting points for new tasks, leveraging learned features.",
        },
        {
            "question": "What are generative models?",
            "answer": "AI systems that learn to create new data samples similar to training data.",
        },
        {
            "question": "How does reinforcement learning work?",
            "answer": "Learning through interaction with environment, receiving rewards and penalties for actions.",
        },
        {
            "question": "What is meta-learning?",
            "answer": "Learning algorithms that learn how to learn, adapting quickly to new tasks.",
        },
        {
            "question": "What are neural architecture search?",
            "answer": "Automated methods for discovering optimal neural network architectures.",
        },
        {
            "question": "How do graph neural networks work?",
            "answer": "Networks processing graph-structured data by passing messages between connected nodes.",
        },
        {
            "question": "What is continual learning?",
            "answer": "Learning new tasks without forgetting previously learned knowledge, avoiding catastrophic forgetting.",
        },
        {
            "question": "What are foundation models?",
            "answer": "Large pre-trained models serving as basis for various downstream applications.",
        },
    ]
    return _get_medium_dialogue_dataset() + advanced_pairs
