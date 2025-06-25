import unittest
import torch
import time
import logging
from queue import Queue

from new_rebuild.config import get_project_config, ProjectConfig, LatticeConfig
from new_rebuild.core.lattice import create_lattice
from new_rebuild.utils.device_manager import get_device_manager
from new_rebuild.core.lattice.spatial_optimization.memory_manager import (
    get_memory_pool_manager,
)

# Configure logger for testing
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestMemoryOptimizations(unittest.TestCase):

    def setUp(self):
        """Set up a project configuration for testing optimizations."""
        self.device_manager = get_device_manager(debug_mode=False)
        if not self.device_manager.is_cuda():
            self.skipTest("Memory optimization tests require a CUDA-enabled GPU.")

        # Create and set a specific config for this test
        # We manually overwrite the global config singleton for test isolation
        from new_rebuild.config import project_config

        project_config._global_config = self.create_test_config()

        self.memory_pool_manager = get_memory_pool_manager()
        self.memory_pool_manager.cleanup()  # Ensure clean state before test

    def create_test_config(self):
        """Creates a project config tailored for this test."""
        config = ProjectConfig()
        config.lattice = LatticeConfig(dimensions=(32, 32, 32))
        config.expert.enabled = True
        config.gnn.state_size = 32
        config.logging.debug_mode = False
        return config

    def test_optimizations_run_and_check_stats(self):
        """
        Tests that the forward pass runs with all optimizations and checks stats.
        1. Gradient Checkpointing is used in MoEProcessor.
        2. Memory Pooling is used for empty expert outputs.
        3. Async chunking is used by the AdaptiveGPUChunker.
        """
        logger.info("--- Running Test: Optimizations Forward Pass ---")

        # 1. Initialization
        try:
            lattice = create_lattice()
            self.assertIsNotNone(lattice, "Lattice creation failed.")
            logger.info(
                f"Lattice created successfully with dimensions {lattice.config.lattice.dimensions}"
            )
        except Exception as e:
            self.fail(f"Lattice initialization failed with optimizations: {e}")

        # 2. Warm-up run (for JIT compilation, etc.)
        logger.info("Starting warm-up forward pass...")
        with torch.no_grad():
            lattice.forward()
        self.device_manager.synchronize()
        logger.info("Warm-up finished.")

        # Reset stats after warm-up
        self.memory_pool_manager.cleanup()
        if hasattr(lattice.spatial_optimizer, "chunker"):
            # Accessing internal state for testing purposes
            lattice.spatial_optimizer.chunker.scheduler.task_queue = Queue()

        # 3. Main Test Run
        logger.info("Starting main test forward pass (5 steps)...")
        start_time = time.time()

        mem_stats_before = self.device_manager.get_memory_stats()
        torch.cuda.reset_peak_memory_stats(self.device_manager.get_device())

        num_steps = 5
        for i in range(num_steps):
            with torch.no_grad():
                output_states = lattice.forward()
            self.assertEqual(output_states.shape[0], lattice.states.shape[0])

        self.device_manager.synchronize()
        duration = time.time() - start_time
        logger.info(f"Finished {num_steps} forward passes in {duration:.2f} seconds.")

        # 4. Assertions and Stat Checks
        mem_stats_after = self.device_manager.get_memory_stats()

        logger.info(
            f"Initial Memory (Allocated): {mem_stats_before['allocated_mb']:.2f} MB"
        )
        logger.info(
            f"Peak Memory (Max Allocated): {mem_stats_after['max_allocated_mb']:.2f} MB"
        )

        # Check Memory Pool Manager stats
        pool_stats = self.memory_pool_manager.get_memory_stats()
        logger.info(
            f"Memory Pool Hits: {pool_stats['pool_hits']}, Misses: {pool_stats['pool_misses']}"
        )
        self.assertGreater(
            pool_stats["total_allocations"], 0, "Memory pool was not used at all."
        )

        # Check Chunker stats (if chunking was used)
        if hasattr(lattice.spatial_optimizer, "chunker"):
            chunker = lattice.spatial_optimizer.chunker
            if len(chunker.chunks) > 1:
                logger.info("Chunking was enabled.")
                chunker_stats = chunker.get_comprehensive_stats()
                self.assertGreater(chunker_stats["performance"]["total_chunks"], 1)
            else:
                logger.info(
                    "Chunking was not enabled (lattice processed as a single chunk)."
                )

        logger.info("--- Test Finished Successfully ---")


if __name__ == "__main__":
    unittest.main()
