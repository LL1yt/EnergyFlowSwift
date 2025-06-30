"""
System Validator for Production Training.
"""

import torch
import logging
from emergent_training import EmergentCubeTrainer, EmergentTrainingConfig
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

logger = logging.getLogger(__name__)


def validate_system(model_name: str, device: str) -> bool:
    """
    Validates the entire system before starting production training.
    """
    logger.info("[VALIDATOR] Starting system validation...")

    try:
        logger.info("1️⃣ Testing embedding loading...")
        embedding_loader = EmbeddingLoader()
        test_embeddings = embedding_loader.get_embeddings(["hello", "world"])
        assert len(test_embeddings) == 2, "Failed to load test embeddings."
        logger.info("✅ Embedding loading successful.")

        logger.info("2️⃣ Testing dataset creation...")
        test_dataset = ["Question1?", "Answer1", "Question2?", "Answer2"]
        dataset = DialogueDataset(
            model_name=model_name, data=test_dataset, batch_size=2
        )
        assert len(dataset) > 0, "Failed to create dataset."
        logger.info("✅ Dataset creation successful.")

        logger.info("3️⃣ Testing EmergentCubeTrainer initialization...")

        # Load main config for proper NCA configuration
        try:
            import yaml
            from pathlib import Path

            main_config_path = Path("config/main_config.yaml")
            if main_config_path.exists():
                with open(main_config_path, "r") as f:
                    main_config = yaml.safe_load(f)
                # Use the new from_main_config method
                config = EmergentTrainingConfig.from_main_config(main_config)
                config.teacher_model = model_name  # Override with test model
            else:
                # Fallback to default config
                config = EmergentTrainingConfig(teacher_model=model_name)
                logger.warning(
                    "Main config not found, using default EmergentTrainingConfig"
                )
        except Exception as e:
            logger.warning(f"Failed to load main config: {e}, using default")
            config = EmergentTrainingConfig(teacher_model=model_name)

        trainer = EmergentCubeTrainer(config, device=device)
        logger.info("✅ Trainer initialization successful.")

        logger.info("4️⃣ Testing training step...")
        batch = next(iter(dataset.get_dataloader()))
        input_emb, target_emb = batch["input_embeddings"], batch["target_embeddings"]

        input_emb = input_emb.unsqueeze(0).to(device)
        target_emb = target_emb.unsqueeze(0).to(device)
        metrics = trainer.train_step(input_emb, target_emb)
        assert "total_loss" in metrics, "Training step did not return a loss."
        logger.info(
            f"✅ Training step successful: loss = {metrics.get('total_loss', 'N/A'):.4f}"
        )

        logger.info("[SUCCESS] System validation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ System validation failed: {e}", exc_info=True)
        return False
