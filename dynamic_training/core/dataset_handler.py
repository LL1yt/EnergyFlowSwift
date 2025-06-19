"""
Handles dataset preparation for dynamic training.
"""

import logging
from typing import Optional, Dict

# Assuming the DialogueDataset is in this path
from training.embedding_trainer.dialogue_dataset.dialogue_dataset import DialogueDataset

logger = logging.getLogger(__name__)


class DatasetHandler:
    """Prepares the dataset for training."""

    def __init__(self, config: Dict):
        self.config = config

    def prepare_dataset(
        self, limit: Optional[int] = None, fixed_sampling: bool = False
    ) -> DialogueDataset:
        """
        Initializes and prepares the DialogueDataset.

        Args:
            limit: Optional limit for the number of dialogue pairs.
            fixed_sampling: Whether to use fixed sampling for reproducibility.

        Returns:
            An instance of the prepared DialogueDataset.
        """
        embeddings_config = self.config.get("embeddings", {})
        teacher_model = embeddings_config.get("teacher_model")
        if not teacher_model:
            raise ValueError("teacher_model not specified in embeddings config.")

        dataset_limit = limit or embeddings_config.get("dataset_limit")

        logger.info(f"Preparing dataset with teacher model: {teacher_model}")
        logger.info(
            f"Dataset limit: {'No limit' if dataset_limit is None else dataset_limit}"
        )
        logger.info(f"Fixed sampling: {fixed_sampling}")

        try:
            dataset = DialogueDataset(
                teacher_model_name=teacher_model,
                limit=dataset_limit,
                fixed_sampling=fixed_sampling,
            )
            logger.info("Dataset prepared successfully.")
            return dataset
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}", exc_info=True)
            raise
