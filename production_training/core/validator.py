"""
System Validator for Production Training.
Ensures that all components are correctly initialized and functional before starting
the main training pipeline.
"""

import torch
import logging
from datetime import datetime
import json

from emergent_training import EmergentCubeTrainer, EmergentTrainingConfig
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

logger = logging.getLogger(__name__)


def validate_system(model_name: str, device: str) -> bool:
    """
    Performs a series of checks to validate the system readiness.

    Args:
        model_name: The name of the teacher model being used.
        device: The device to run tests on.

    Returns:
        True if all checks pass, False otherwise.
    """
    try:
        logger.info("[VALIDATOR] Starting system validation...")

        # 1. Check CUDA availability
        logger.info(f"1️⃣ Using device: {device}")
        if "cuda" in device and not torch.cuda.is_available():
            logger.error("CUDA device specified but not available.")
            return False

        # 2. Test Universal Adapter
        logger.info("2️⃣ Testing Universal Adapter...")
        test_embedding = torch.randn(1, 768)  # DistilBERT-like embedding
        adapter = UniversalEmbeddingAdapter(
            input_dim=768, output_dim=256, strategy="hierarchical"
        )
        test_surface = adapter.forward(test_embedding)
        assert test_surface.shape == (1, 256), "Adapter output shape is incorrect."
        logger.info("[OK] Universal Adapter works as expected.")

        # 3. Test EmergentCubeTrainer initialization
        logger.info("3️⃣ Testing EmergentCubeTrainer initialization...")
        config = EmergentTrainingConfig(
            teacher_model=model_name,
            cube_dimensions=(16, 16, 16),  # PHASE 4: 16x16x16 вместо 15x15x1
            batch_size=1,
            epochs=1,
        )

        # --- Enhanced Logging ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = {"error": "Failed to serialize config"}

        logger.info(
            f"VALIDATOR is creating EmergentCubeTrainer @ {timestamp}\n"
            f"     WITH_CONFIG: {json.dumps(config_dict, indent=2, default=str)}"
        )
        # --- End of Logging ---

        trainer = EmergentCubeTrainer(config, device=device)
        logger.info(f"[OK] EmergentCubeTrainer initialized on {trainer.device}")

        # 4. Test dataset creation
        logger.info("4️⃣ Testing dataset creation...")
        test_dialogue = [{"question": "Q", "answer": "A"}]
        dataset = create_dialogue_dataset(
            dialogue_pairs=test_dialogue,
            teacher_model=model_name,
            cache_embeddings=False,
            validation_split=0.0,
        )
        assert len(dataset) > 0, "Dataset creation failed."
        logger.info(f"[OK] Dataset created with {len(dataset)} pairs.")

        # 5. Test a single training step
        logger.info("5️⃣ Testing a single training step...")
        input_emb, target_emb = dataset[0]
        input_emb = input_emb.unsqueeze(0).to(device)
        target_emb = target_emb.unsqueeze(0).to(device)
        metrics = trainer.train_step(input_emb, target_emb)
        assert "total_loss" in metrics, "Training step did not return a loss."
        logger.info(
            f"[OK] Training step successful: loss = {metrics.get('total_loss', 'N/A'):.4f}"
        )

        logger.info("[SUCCESS] System validation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"[ERROR] System validation failed: {e}", exc_info=True)
        return False
