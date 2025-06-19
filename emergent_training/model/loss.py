"""
Multi-objective loss function for emergent training.
"""

import torch
import torch.nn as nn
import logging

from emergent_training.config.config import EmergentTrainingConfig

logger = logging.getLogger(__name__)


class EmergentMultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss function.
    Components:
    1. Surface Reconstruction Loss
    2. Internal Consistency Loss
    3. Dialogue Similarity Loss
    """

    def __init__(self, config: EmergentTrainingConfig):
        super().__init__()
        self.config = config
        self.weights = config.loss_weights

        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        surface_size = config.cube_dimensions[0] * config.cube_dimensions[1]
        embedding_dim = config.teacher_embedding_dim

        # Linear layers for projecting between surface and embedding dimensions
        self.surface_to_embedding = nn.Linear(surface_size, embedding_dim, bias=False)
        self.embedding_to_surface = nn.Linear(embedding_dim, surface_size, bias=False)

        if config.adaptive_loss_weighting:
            self.log_vars = nn.Parameter(
                torch.zeros(3)
            )  # For surface, internal, dialogue
            logger.info("Using adaptive loss weighting.")
        else:
            self.register_buffer(
                "static_weights", torch.tensor(list(self.weights.values()))
            )

    def forward(
        self, outputs: dict, targets: dict, internal_states: torch.Tensor
    ) -> dict:
        """Computes the multi-objective loss."""

        # 1. Surface Reconstruction Loss
        # This part has some complex projection logic. I'll simplify it to be more robust.
        output_surface = outputs.get("output_surface")
        target_surface = targets.get(
            "target_surface_embedding"
        )  # Assuming this is the teacher embedding

        surface_loss = torch.tensor(0.0, device=output_surface.device)
        if output_surface is not None and target_surface is not None:
            projected_target = self.embedding_to_surface(target_surface)
            surface_loss = self.mse_loss(output_surface, projected_target)

        # 2. Internal Consistency Loss
        internal_loss = torch.tensor(0.0, device=internal_states.device)
        if internal_states is not None and internal_states.shape[0] > 1:
            # Measures the similarity between consecutive internal state snapshots
            mean_internal_state_diff = (
                (internal_states[1:] - internal_states[:-1]).abs().mean()
            )
            internal_loss = mean_internal_state_diff

        # 3. Dialogue Similarity Loss
        question_embedding = outputs.get("final_embedding")
        answer_embedding = targets.get("answer_embedding")
        dialogue_loss = torch.tensor(0.0, device=question_embedding.device)
        if question_embedding is not None and answer_embedding is not None:
            similarity = self.cosine_similarity(question_embedding, answer_embedding)
            dialogue_loss = 1.0 - similarity.mean()  # Convert similarity to a loss

        # Combine losses
        if hasattr(self, "log_vars"):
            # Adaptive weighting: loss = sum(loss_i * exp(-log_var_i) + log_var_i)
            total_loss = (
                surface_loss * torch.exp(-self.log_vars[0])
                + self.log_vars[0]
                + internal_loss * torch.exp(-self.log_vars[1])
                + self.log_vars[1]
                + dialogue_loss * torch.exp(-self.log_vars[2])
                + self.log_vars[2]
            )
        else:
            total_loss = (
                surface_loss * self.static_weights[0]
                + internal_loss * self.static_weights[1]
                + dialogue_loss * self.static_weights[2]
            )

        return {
            "total_loss": total_loss,
            "surface_loss": surface_loss.detach(),
            "internal_loss": internal_loss.detach(),
            "dialogue_loss": dialogue_loss.detach(),
        }
