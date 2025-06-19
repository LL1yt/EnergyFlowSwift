"""
Checkpoint Manager for finding and analyzing compatible checkpoints.
"""

import logging
import torch
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages finding and analyzing compatible checkpoints."""

    def __init__(self, config: Dict):
        if not config:
            raise ValueError("Configuration must be provided to CheckpointManager.")
        self.current_config = config
        self.current_signature = self._get_config_signature(self.current_config)

    def find_compatible_checkpoints(
        self, checkpoints_dir: str = "checkpoints"
    ) -> List[Dict[str, Any]]:
        """
        Finds compatible checkpoints based on the current configuration.
        Returns a sorted list of compatible checkpoints.
        """
        logger.info(f"Searching for compatible checkpoints in '{checkpoints_dir}'")
        checkpoints_path = Path(checkpoints_dir)
        if not checkpoints_path.exists():
            logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
            return []

        compatible_checkpoints = []
        search_patterns = [
            checkpoints_path / "latest" / "*.pt",
            checkpoints_path / "versioned" / "**" / "*.pt",
            checkpoints_path / "*.pt",
        ]

        for pattern in search_patterns:
            for checkpoint_file in glob.glob(str(pattern), recursive=True):
                checkpoint_info = self._analyze_checkpoint(checkpoint_file)
                if (
                    checkpoint_info
                    and checkpoint_info.get("compatibility_score", 0) >= 0.5
                ):
                    compatible_checkpoints.append(checkpoint_info)

        compatible_checkpoints.sort(
            key=lambda x: (
                -x.get("compatibility_score", 0),
                -x.get("timestamp_score", 0),
            )
        )

        logger.info(f"Found {len(compatible_checkpoints)} compatible checkpoints.")
        for i, cp in enumerate(compatible_checkpoints[:3]):
            logger.info(
                f"  Top-{i+1}: {cp['name']} (Score: {cp['compatibility_score']:.2f})"
            )
        return compatible_checkpoints

    def _analyze_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Analyzes a single checkpoint for compatibility."""
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_config = checkpoint_data.get("config")
            if not checkpoint_config:
                return None

            checkpoint_signature = self._get_config_signature(checkpoint_config)
            compatibility_score = self._calculate_compatibility(
                self.current_signature, checkpoint_signature
            )

            metadata = checkpoint_data.get("metadata", {})
            timestamp_str = metadata.get("timestamp", "1970-01-01T00:00:00Z")
            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_score = ts.timestamp()
            except ValueError:
                timestamp_score = 0

            return {
                "path": checkpoint_path,
                "name": Path(checkpoint_path).name,
                "config": checkpoint_config,
                "metadata": metadata,
                "compatibility_score": compatibility_score,
                "timestamp_score": timestamp_score,
                "timestamp": timestamp_str,
                "description": self._generate_checkpoint_description(metadata),
            }
        except Exception as e:
            logger.debug(f"Could not analyze checkpoint {checkpoint_path}: {e}")
            return None

    def _get_config_signature(self, config: Dict) -> Dict:
        """Creates a configuration signature for comparison."""
        sig = {}
        if "lattice" in config:
            sig["lattice_shape"] = tuple(
                config["lattice"].get(k) for k in ["xs", "ys", "zs"]
            )
        if "embeddings" in config:
            sig["embedding_dim"] = config["embeddings"].get("embedding_dim")
        if "gmlp" in config:
            gmlp = config["gmlp"]
            sig["gmlp_params"] = (
                gmlp.get("state_size"),
                gmlp.get("hidden_dim"),
                gmlp.get("memory_dim"),
            )
        if "emergent_training" in config:
            sig["spatial_depth"] = config["emergent_training"].get(
                "spatial_propagation_depth"
            )
        return sig

    def _calculate_compatibility(self, current: Dict, checkpoint: Dict) -> float:
        """Calculates a compatibility score between two configuration signatures."""
        score = 0.0
        total_possible_score = 0.0

        if "lattice_shape" in current and "lattice_shape" in checkpoint:
            total_possible_score += 1.0
            if current["lattice_shape"] == checkpoint["lattice_shape"]:
                score += 1.0

        if "embedding_dim" in current and "embedding_dim" in checkpoint:
            total_possible_score += 1.0
            if current["embedding_dim"] == checkpoint["embedding_dim"]:
                score += 1.0

        if "gmlp_params" in current and "gmlp_params" in checkpoint:
            total_possible_score += 1.0
            score += self._tuple_similarity(
                current["gmlp_params"], checkpoint["gmlp_params"]
            )

        if "spatial_depth" in current and "spatial_depth" in checkpoint:
            total_possible_score += 1.0
            if current["spatial_depth"] == checkpoint["spatial_depth"]:
                score += 1.0

        return score / total_possible_score if total_possible_score > 0 else 0

    def _tuple_similarity(self, t1: Optional[tuple], t2: Optional[tuple]) -> float:
        """Calculates similarity for tuples of numbers."""
        if t1 is None or t2 is None or len(t1) != len(t2):
            return 0.0

        sims = []
        for v1, v2 in zip(t1, t2):
            if v1 is None or v2 is None or v1 == 0 or v2 == 0:
                sims.append(0.0)
            else:
                sims.append(1.0 - abs(v1 - v2) / max(v1, v2))

        return sum(sims) / len(sims) if sims else 0.0

    def _generate_checkpoint_description(self, metadata: Dict) -> str:
        """Generates a human-readable description for a checkpoint."""
        mode = metadata.get("mode", "N/A")
        sim = metadata.get("final_similarity")
        sim_str = f"Sim: {sim:.3f}" if sim else "No sim"
        ts = metadata.get("timestamp", "No date").split("T")[0]
        return f"Mode: {mode}, {sim_str}, Date: {ts}"
